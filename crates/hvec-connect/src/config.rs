//! TOML configuration with named chat and embedding profiles.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{ConnectError, Result};

/// Environment variable that overrides the config file location.
pub const CONFIG_ENV: &str = "HVEC_CONFIG";

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// SQLite file holding collections, sessions and the run log.
    #[serde(default)]
    pub db_path: Option<PathBuf>,
    /// Chat profile used when `--chat` is not given.
    pub default_chat: String,
    /// Embedding profile used when `--embedder` is not given.
    pub default_embedder: String,
    /// Metric used for retrieval: `cosine`, `dot` or `l2`.
    #[serde(default = "default_metric")]
    pub metric: String,
    /// Named chat profiles.
    #[serde(default)]
    pub chat: BTreeMap<String, ChatProfile>,
    /// Named embedding profiles.
    #[serde(default)]
    pub embed: BTreeMap<String, EmbedProfile>,
}

fn default_metric() -> String {
    "cosine".to_owned()
}

/// Which wire protocol a chat profile speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatProvider {
    /// Anthropic Messages API.
    Anthropic,
    /// Any OpenAI-compatible `/chat/completions` endpoint: OpenAI, Ollama, vLLM, and others.
    Openai,
}

/// A chat model profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatProfile {
    pub provider: ChatProvider,
    pub model: String,
    /// Override the provider's default endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Name of the environment variable holding the API key. Optional for local servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Maximum output tokens per completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Anthropic only: `low`, `medium`, `high`, `xhigh` or `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Anthropic only: enable server-side refusal fallbacks on models that support them.
    #[serde(default = "default_true")]
    pub fallbacks: bool,
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_true() -> bool {
    true
}

/// Which backend an embedding profile uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedProvider {
    /// In-process ONNX model via fastembed. Free, offline, reproducible.
    Local,
    /// Any OpenAI-compatible `/embeddings` endpoint.
    Openai,
}

/// An embedding model profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedProfile {
    pub provider: EmbedProvider,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Output dimension. Required for remote providers unless hvec may probe for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,
    /// Local only: where downloaded model files are cached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
}

impl Config {
    /// The config path: `$HVEC_CONFIG`, else the platform config dir.
    #[must_use]
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var(CONFIG_ENV) {
            return PathBuf::from(p);
        }
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hvec")
            .join("config.toml")
    }

    /// Default location of the SQLite database.
    #[must_use]
    pub fn default_db_path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hvec")
            .join("hvec.db")
    }

    /// Load from a path.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                ConnectError::ConfigMissing(path.to_path_buf())
            } else {
                ConnectError::ConfigRead {
                    path: path.to_path_buf(),
                    source,
                }
            }
        })?;
        Self::parse(&text).map_err(|source| ConnectError::ConfigParse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Parse from TOML text.
    pub fn parse(text: &str) -> std::result::Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Write the starter config to `path`, creating parent directories.
    pub fn write_starter(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ConnectError::ConfigWrite {
                path: path.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(path, Self::starter_toml()).map_err(|source| ConnectError::ConfigWrite {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The commented starter config written by `hvec config init`.
    #[must_use]
    pub fn starter_toml() -> &'static str {
        include_str!("starter-config.toml")
    }

    /// Resolve the database path, expanding a leading `~`.
    #[must_use]
    pub fn db_path(&self) -> PathBuf {
        match &self.db_path {
            Some(p) => expand_tilde(p),
            None => Self::default_db_path(),
        }
    }

    pub fn chat_profile(&self, name: &str) -> Result<&ChatProfile> {
        self.chat.get(name).ok_or_else(|| ConnectError::UnknownProfile {
            kind: "chat",
            name: name.to_owned(),
        })
    }

    pub fn embed_profile(&self, name: &str) -> Result<&EmbedProfile> {
        self.embed.get(name).ok_or_else(|| ConnectError::UnknownProfile {
            kind: "embed",
            name: name.to_owned(),
        })
    }
}

/// Expand a leading `~/` to the home directory.
#[must_use]
pub fn expand_tilde(path: &Path) -> PathBuf {
    let Some(rest) = path.strip_prefix("~").ok() else {
        return path.to_path_buf();
    };
    match dirs::home_dir() {
        Some(home) => home.join(rest),
        None => path.to_path_buf(),
    }
}

/// Read an API key from the environment variable named in a profile.
pub(crate) fn api_key(profile: &str, var: Option<&str>) -> Result<Option<String>> {
    let Some(var) = var else { return Ok(None) };
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Ok(Some(v)),
        _ => Err(ConnectError::MissingApiKey {
            profile: profile.to_owned(),
            var: var.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_config_parses() {
        let cfg = Config::parse(Config::starter_toml()).expect("starter config must parse");
        assert!(cfg.chat.contains_key(&cfg.default_chat));
        assert!(cfg.embed.contains_key(&cfg.default_embedder));
        assert_eq!(cfg.chat["anthropic"].provider, ChatProvider::Anthropic);
        assert_eq!(cfg.embed["local"].provider, EmbedProvider::Local);
    }

    #[test]
    fn unknown_profile_is_an_error() {
        let cfg = Config::parse(Config::starter_toml()).unwrap();
        assert!(matches!(
            cfg.chat_profile("nope"),
            Err(ConnectError::UnknownProfile { kind: "chat", .. })
        ));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let text = "default_chat='a'\ndefault_embedder='b'\nbogus=1\n";
        assert!(Config::parse(text).is_err());
    }

    #[test]
    fn tilde_expands_to_home() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde(Path::new("~/x.db")), home.join("x.db"));
        assert_eq!(expand_tilde(Path::new("/abs/x.db")), PathBuf::from("/abs/x.db"));
    }
}
