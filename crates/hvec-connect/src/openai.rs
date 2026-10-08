//! OpenAI-compatible chat and embedding endpoints.
//!
//! Covers OpenAI itself plus Ollama, vLLM, LM Studio and most hosted providers.

use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};

use crate::config::{ChatProfile, EmbedProfile, api_key};
use crate::traits::{ChatModel, ChatRequest, ChatResponse, Embedder, Usage};
use crate::{ConnectError, Result};

const PROVIDER: &str = "openai-compatible";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

fn authed(req: RequestBuilder, key: Option<&str>) -> RequestBuilder {
    match key {
        Some(k) => req.bearer_auth(k),
        None => req,
    }
}

async fn send_json<T: for<'de> Deserialize<'de>>(req: RequestBuilder) -> Result<T> {
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
    serde_json::from_str(&text).map_err(|e| ConnectError::MalformedResponse {
        provider: PROVIDER,
        detail: e.to_string(),
    })
}

/// Chat connector for `/chat/completions`.
#[derive(Debug, Clone)]
pub struct OpenAiChat {
    profile: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    max_tokens: u32,
    client: Client,
}

impl OpenAiChat {
    pub fn from_profile(name: &str, profile: &ChatProfile) -> Result<Self> {
        Ok(Self {
            profile: name.to_owned(),
            model: profile.model.clone(),
            base_url: profile.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            api_key: api_key(name, profile.api_key_env.as_deref())?,
            max_tokens: profile.max_tokens,
            client: Client::new(),
        })
    }
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatBody<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    max_tokens: u32,
}

#[derive(Deserialize)]
struct ChatCompletion {
    #[serde(default)]
    model: Option<String>,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[async_trait::async_trait]
impl ChatModel for OpenAiChat {
    fn profile(&self) -> &str {
        &self.profile
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut messages = Vec::with_capacity(request.messages.len() + 1);
        if let Some(system) = request.system.as_deref() {
            messages.push(WireMessage {
                role: "system",
                content: system,
            });
        }
        messages.extend(request.messages.iter().map(|m| WireMessage {
            role: m.role.as_str(),
            content: &m.content,
        }));

        let body = ChatBody {
            model: &self.model,
            messages,
            max_tokens: request.max_tokens.unwrap_or(self.max_tokens),
        };
        let req = authed(
            self.client
                .post(format!("{}/chat/completions", self.base_url.trim_end_matches('/')))
                .json(&body),
            self.api_key.as_deref(),
        );
        let parsed: ChatCompletion = send_json(req).await?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ConnectError::MalformedResponse {
                provider: PROVIDER,
                detail: "no choices".into(),
            })?;
        let usage = parsed.usage.unwrap_or_default();
        Ok(ChatResponse {
            text: choice.message.content.unwrap_or_default(),
            model: parsed.model.unwrap_or_else(|| self.model.clone()),
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            },
            stop_reason: choice.finish_reason,
        })
    }
}

/// Embedding connector for `/embeddings`.
#[derive(Debug, Clone)]
pub struct OpenAiEmbedder {
    profile: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    dimension: usize,
    client: Client,
}

#[derive(Serialize)]
struct EmbedBody<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    index: usize,
    embedding: Vec<f32>,
}

impl OpenAiEmbedder {
    /// Build from a profile. Probes the endpoint once when `dimension` is not configured.
    pub async fn from_profile(name: &str, profile: &EmbedProfile) -> Result<Self> {
        let mut this = Self {
            profile: name.to_owned(),
            model: profile.model.clone(),
            base_url: profile.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            api_key: api_key(name, profile.api_key_env.as_deref())?,
            dimension: profile.dimension.unwrap_or(0),
            client: Client::new(),
        };
        if this.dimension == 0 {
            tracing::info!(profile = name, "probing embedding dimension");
            let probe = this.embed_raw(&["dimension probe".to_owned()]).await?;
            this.dimension = probe.first().map(Vec::len).unwrap_or(0);
        }
        Ok(this)
    }

    async fn embed_raw(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedBody {
            model: &self.model,
            input: texts,
        };
        let req = authed(
            self.client
                .post(format!("{}/embeddings", self.base_url.trim_end_matches('/')))
                .json(&body),
            self.api_key.as_deref(),
        );
        let mut parsed: EmbedResponse = send_json(req).await?;
        if parsed.data.len() != texts.len() {
            return Err(ConnectError::EmbeddingCount {
                expected: texts.len(),
                got: parsed.data.len(),
            });
        }
        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait::async_trait]
impl Embedder for OpenAiEmbedder {
    fn profile(&self) -> &str {
        &self.profile
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embed_raw(texts).await
    }
}
