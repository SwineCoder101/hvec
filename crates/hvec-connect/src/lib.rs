//! Configurable connectors to chat models and embedding models.
//!
//! Two traits, [`ChatModel`] and [`Embedder`], sit behind named profiles in
//! a TOML [`Config`]. Switching providers is a flag, not a code change.

pub mod anthropic;
pub mod config;
pub mod error;
#[cfg(feature = "local")]
pub mod local;
pub mod openai;
pub mod traits;

pub use config::{ChatProfile, ChatProvider, Config, EmbedProfile, EmbedProvider};
pub use error::{ConnectError, Result};
pub use traits::{ChatModel, ChatRequest, ChatResponse, Embedder, Message, Role, Usage};

use std::sync::Arc;

/// Build a chat model from a named profile in the config.
pub fn chat_model(config: &Config, profile_name: &str) -> Result<Arc<dyn ChatModel>> {
    let profile = config.chat_profile(profile_name)?;
    let model: Arc<dyn ChatModel> = match profile.provider {
        ChatProvider::Anthropic => Arc::new(anthropic::AnthropicChat::from_profile(profile_name, profile)?),
        ChatProvider::Openai => Arc::new(openai::OpenAiChat::from_profile(profile_name, profile)?),
    };
    Ok(model)
}

/// Build an embedder from a named profile in the config.
pub async fn embedder(config: &Config, profile_name: &str) -> Result<Arc<dyn Embedder>> {
    let profile = config.embed_profile(profile_name)?;
    let embedder: Arc<dyn Embedder> = match profile.provider {
        EmbedProvider::Openai => Arc::new(openai::OpenAiEmbedder::from_profile(profile_name, profile).await?),
        #[cfg(feature = "local")]
        EmbedProvider::Local => Arc::new(local::LocalEmbedder::from_profile(profile_name, profile).await?),
        #[cfg(not(feature = "local"))]
        EmbedProvider::Local => {
            return Err(ConnectError::FeatureDisabled {
                profile: profile_name.to_owned(),
                feature: "local",
            });
        }
    };
    Ok(embedder)
}
