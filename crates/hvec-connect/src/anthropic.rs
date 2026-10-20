//! Anthropic Messages API over raw HTTP. There is no official Rust SDK.
//!
//! Each completion is tried up to `max_attempts` times. Transport errors and
//! HTTP 408, 409, 429 and 5xx replies are retried after an exponential
//! backoff (or the server's `retry-after`); anything else ends the request.
//! Every attempt's raw reply is kept on the [`ChatResponse`], or on the final
//! [`ConnectError`], so the run log shows exactly what came back each time.

use std::time::{Duration, Instant};

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::config::{ChatProfile, api_key};
use crate::retry;
use crate::traits::{Attempt, AttemptOutcome, ChatModel, ChatRequest, ChatResponse, Message, Usage};
use crate::{ConnectError, Result};

const PROVIDER: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_WORKSPACE_ID_ENV: &str = "ANTHROPIC_WORKSPACE_ID";
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
    /// Sent as `anthropic-workspace-id`; required for keys not scoped to a workspace.
    workspace_id: Option<String>,
    max_tokens: u32,
    effort: Option<String>,
    fallbacks: bool,
    /// Attempts per request, at least 1.
    max_attempts: u32,
    client: Client,
}

impl AnthropicChat {
    pub fn from_profile(name: &str, profile: &ChatProfile) -> Result<Self> {
        let var = profile.api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
        let api_key = api_key(name, Some(var))?.expect("api_key returns Some when var is Some");
        let workspace_var = profile.workspace_id_env.as_deref().unwrap_or(DEFAULT_WORKSPACE_ID_ENV);
        let workspace_id = std::env::var(workspace_var).ok().filter(|v| !v.trim().is_empty());
        Ok(Self {
            profile: name.to_owned(),
            model: profile.model.clone(),
            base_url: profile.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
            api_key,
            workspace_id,
            max_tokens: profile.max_tokens,
            effort: profile.effort.clone(),
            fallbacks: profile.fallbacks,
            max_attempts: profile.max_attempts.max(1),
            client: Client::new(),
        })
    }

    fn use_fallbacks(&self) -> bool {
        self.fallbacks && FALLBACK_MODELS.iter().any(|m| self.model == *m)
    }

    /// One round trip. Returns whatever came back, success or not; only
    /// transport failures are errors here.
    async fn send_once(&self, body: &Body<'_>) -> std::result::Result<Raw, reqwest::Error> {
        let mut req = self
            .client
            .post(format!("{}/v1/messages", self.base_url.trim_end_matches('/')))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(body);
        if body.fallbacks.is_some() {
            req = req.header("anthropic-beta", FALLBACK_BETA);
        }
        if let Some(workspace) = &self.workspace_id {
            req = req.header("anthropic-workspace-id", workspace);
        }
        let resp = req.send().await?;
        let status = resp.status();
        let retry_after = retry::retry_after(resp.headers());
        let text = resp.text().await?;
        Ok(Raw {
            status,
            retry_after,
            text,
        })
    }
}

/// A reply from the endpoint before interpretation.
struct Raw {
    status: StatusCode,
    retry_after: Option<Duration>,
    text: String,
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

/// Turn a 2xx body into a response. Refusals and unreadable bodies are errors
/// and are never retried: the same request would get the same answer.
fn interpret(text: &str) -> Result<ChatResponse> {
    let parsed: Response = serde_json::from_str(text).map_err(|e| ConnectError::MalformedResponse {
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
        attempts: Vec::new(),
    })
}

fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
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

        let mut attempts: Vec<Attempt> = Vec::with_capacity(self.max_attempts as usize);
        let mut number = 1;
        loop {
            let started = Instant::now();
            let sent = self.send_once(&body).await;
            let elapsed_ms = millis(started.elapsed());

            let (status, text, failure, retry_after, retryable) = match sent {
                Ok(raw) if raw.status.is_success() => {
                    attempts.push(Attempt {
                        number,
                        status: Some(raw.status.as_u16()),
                        body: raw.text.clone(),
                        elapsed_ms,
                        backoff_ms: None,
                        outcome: AttemptOutcome::Succeeded,
                    });
                    return match interpret(&raw.text) {
                        Ok(mut response) => {
                            response.attempts = attempts;
                            Ok(response)
                        }
                        Err(e) => Err(retry::after_retries(PROVIDER, e, attempts)),
                    };
                }
                Ok(raw) => {
                    let code = raw.status.as_u16();
                    let failure = ConnectError::Api {
                        provider: PROVIDER,
                        status: code,
                        body: raw.text.clone(),
                    };
                    (
                        Some(code),
                        raw.text,
                        failure,
                        raw.retry_after,
                        retry::retryable_status(code),
                    )
                }
                Err(source) => {
                    let retryable = retry::retryable_transport(&source);
                    let text = retry::error_chain(&source);
                    let failure = ConnectError::Http {
                        provider: PROVIDER,
                        source,
                    };
                    (None, text, failure, None, retryable)
                }
            };

            if retryable && number < self.max_attempts {
                let delay = retry::backoff(number, retry_after);
                tracing::warn!(
                    provider = PROVIDER,
                    attempt = number,
                    max_attempts = self.max_attempts,
                    status = ?status,
                    delay_ms = millis(delay),
                    "request failed, retrying"
                );
                attempts.push(Attempt {
                    number,
                    status,
                    body: text,
                    elapsed_ms,
                    backoff_ms: Some(millis(delay)),
                    outcome: AttemptOutcome::Retried,
                });
                tokio::time::sleep(delay).await;
                number += 1;
                continue;
            }

            attempts.push(Attempt {
                number,
                status,
                body: text,
                elapsed_ms,
                backoff_ms: None,
                outcome: AttemptOutcome::Failed,
            });
            return Err(retry::after_retries(PROVIDER, failure, attempts));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    use super::*;

    /// Serve canned `(status, headers, body)` replies in order on a local port,
    /// one connection per reply. Returns the base URL.
    fn serve(replies: Vec<(u16, &'static str, String)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for (status, extra_headers, body) in replies {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap();
                    }
                    if line == "\r\n" {
                        break;
                    }
                }
                let mut request_body = vec![0; content_length];
                reader.read_exact(&mut request_body).unwrap();
                let mut stream = reader.into_inner();
                write!(
                    stream,
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n{extra_headers}\r\n{body}",
                    body.len()
                )
                .unwrap();
                stream.flush().unwrap();
            }
        });
        base
    }

    fn chat(base_url: String, max_attempts: u32) -> AnthropicChat {
        AnthropicChat {
            profile: "test".into(),
            model: "claude-opus-5-5".into(),
            base_url,
            api_key: "sk-test".into(),
            workspace_id: None,
            max_tokens: 64,
            effort: None,
            fallbacks: false,
            max_attempts,
            client: Client::new(),
        }
    }

    fn ok_body() -> String {
        serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1}
        })
        .to_string()
    }

    fn request() -> ChatRequest {
        ChatRequest {
            system: None,
            messages: vec![Message::user("hello")],
            max_tokens: None,
        }
    }

    #[tokio::test]
    async fn retries_overload_then_records_every_attempt() {
        let base = serve(vec![
            (529, "retry-after: 0\r\n", r#"{"error":"overloaded"}"#.into()),
            (500, "retry-after: 0\r\n", "boom".into()),
            (200, "", ok_body()),
        ]);
        let resp = chat(base, 3).complete(request()).await.unwrap();
        assert_eq!(resp.text, "hi");
        assert_eq!(resp.attempts.len(), 3);
        assert_eq!(resp.attempts[0].status, Some(529));
        assert_eq!(resp.attempts[0].body, r#"{"error":"overloaded"}"#);
        assert_eq!(resp.attempts[0].outcome, AttemptOutcome::Retried);
        assert_eq!(resp.attempts[0].backoff_ms, Some(0));
        assert_eq!(resp.attempts[1].status, Some(500));
        assert_eq!(resp.attempts[1].body, "boom");
        assert_eq!(resp.attempts[2].status, Some(200));
        assert_eq!(resp.attempts[2].outcome, AttemptOutcome::Succeeded);
        assert_eq!(resp.attempts[2].backoff_ms, None);
        assert!(resp.attempts[2].body.contains("end_turn"));
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let base = serve(vec![
            (529, "retry-after: 0\r\n", "a".into()),
            (529, "retry-after: 0\r\n", "b".into()),
        ]);
        let err = chat(base, 2).complete(request()).await.unwrap_err();
        let ConnectError::AfterRetries { attempts, source, .. } = &err else {
            panic!("expected AfterRetries, got {err:?}");
        };
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].outcome, AttemptOutcome::Retried);
        assert_eq!(attempts[1].outcome, AttemptOutcome::Failed);
        assert_eq!(attempts[1].body, "b");
        assert!(matches!(**source, ConnectError::Api { status: 529, .. }));
    }

    #[tokio::test]
    async fn client_errors_are_not_retried() {
        let base = serve(vec![(400, "", r#"{"error":"bad request"}"#.into())]);
        let err = chat(base, 3).complete(request()).await.unwrap_err();
        assert!(matches!(err, ConnectError::Api { status: 400, .. }), "{err:?}");
    }

    #[tokio::test]
    async fn refusal_is_not_retried() {
        let body = serde_json::json!({
            "content": [],
            "model": "claude-opus-5-5",
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber"},
            "usage": {"input_tokens": 3, "output_tokens": 0}
        })
        .to_string();
        let base = serve(vec![(200, "", body)]);
        let err = chat(base, 3).complete(request()).await.unwrap_err();
        assert!(matches!(err, ConnectError::Refused { .. }), "{err:?}");
    }
}
