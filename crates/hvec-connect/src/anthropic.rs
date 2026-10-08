//! Anthropic Messages API over raw HTTP. There is no official Rust SDK.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::{ChatProfile, api_key};
use crate::traits::{ChatModel, ChatRequest, ChatResponse, Message, Usage};
use crate::{ConnectError, Result};

const PROVIDER: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
/// Beta header for the scalar `fallbacks: "default"` form.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Models known to accept `fallbacks: "default"` on the Claude API.
const FALLBACK_MODELS: &[&str] = &[
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
];

/// Chat connector for Claude models.
#[derive(Debug, Clone)]
pub struct AnthropicChat {
    profile: String,
    model: String,
    base_url: String,
    api_key: String,
    max_tokens: u32,
    effort: Option<String>,
    fallbacks: bool,
    client: Client,
}

impl AnthropicChat {
    pub fn from_profile(name: &str, profile: &ChatProfile) -> Result<Self> {
        let var = profile.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
        let api_key = api_key(name, Some(var))?.expect("api_key returns Some when var is Some");
        Ok(Self {
            profile: name.to_owned(),
            model: profile.model.clone(),
            base_url: profile.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            api_key,
            max_tokens: profile.max_tokens,
            effort: profile.effort.clone(),
            fallbacks: profile.fallbacks,
            client: Client::new(),
        })
    }

    fn use_fallbacks(&self) -> bool {
        self.fallbacks && FALLBACK_MODELS.iter().any(|m| self.model == *m)
    }
}

#[derive(Serialize)]
struct Body<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    fallbacks: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig<'a>>,
}

#[derive(Serialize)]
struct OutputConfig<'a> {
    effort: &'a str,
}

#[derive(Deserialize)]
struct Response {
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    usage: ResponseUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct StopDetails {
    #[serde(default)]
    category: Option<String>,
}

#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[async_trait::async_trait]
impl ChatModel for AnthropicChat {
    fn profile(&self) -> &str {
        &self.profile
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse> {
        let body = Body {
            model: &self.model,
            max_tokens: request.max_tokens.unwrap_or(self.max_tokens),
            system: request.system.as_deref(),
            messages: &request.messages,
            fallbacks: self.use_fallbacks().then_some("default"),
            output_config: self.effort.as_deref().map(|effort| OutputConfig { effort }),
        };

        let mut req = self
            .client
            .post(format!("{}/v1/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body);
        if body.fallbacks.is_some() {
            req = req.header("anthropic-beta", FALLBACK_BETA);
        }

        let resp = req.send().await.map_err(|source| ConnectError::Http {
            provider: PROVIDER,
            source,
        })?;
        let status = resp.status();
        let text = resp.text().await.map_err(|source| ConnectError::Http {
            provider: PROVIDER,
            source,
        })?;
        if !status.is_success() {
            return Err(ConnectError::Api {
                provider: PROVIDER,
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: Response = serde_json::from_str(&text).map_err(|e| ConnectError::MalformedResponse {
            provider: PROVIDER,
            detail: e.to_string(),
        })?;

        if parsed.stop_reason.as_deref() == Some("refusal") {
            let category = parsed
                .stop_details
                .and_then(|d| d.category)
                .unwrap_or_else(|| "unknown".to_owned());
            return Err(ConnectError::Refused {
                provider: PROVIDER,
                category,
            });
        }

        let text = parsed
            .content
            .into_iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text),
                ContentBlock::Other => None,
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(ChatResponse {
            text,
            model: parsed.model,
            usage: Usage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            },
            stop_reason: parsed.stop_reason,
        })
    }
}
