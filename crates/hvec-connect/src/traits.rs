//! Provider-agnostic request and response shapes.

use serde::{Deserialize, Serialize};

use crate::Result;

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A chat completion request.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    /// Operator instructions sent out of band from the conversation.
    pub system: Option<String>,
    /// Conversation so far, ending with a user turn.
    pub messages: Vec<Message>,
    /// Override the profile's maximum output tokens.
    pub max_tokens: Option<u32>,
}

/// Token accounting reported by the provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A completed chat turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    /// The model id the provider reports it actually used.
    pub model: String,
    pub usage: Usage,
    pub stop_reason: Option<String>,
}

/// A chat completion backend.
#[async_trait::async_trait]
pub trait ChatModel: Send + Sync + std::fmt::Debug {
    /// Profile name this instance was built from.
    fn profile(&self) -> &str;
    /// Model id as configured.
    fn model_id(&self) -> &str;
    /// Run one completion.
    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse>;
}

/// A text embedding backend.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug {
    /// Profile name this instance was built from.
    fn profile(&self) -> &str;
    /// Model id as configured.
    fn model_id(&self) -> &str;
    /// Output dimensionality. Known up front for every supported provider.
    fn dimension(&self) -> usize;
    /// Embed a batch of texts, preserving order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
