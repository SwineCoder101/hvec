//! Error type shared by every connector.

/// Convenience alias.
pub type Result<T, E = ConnectError> = std::result::Result<T, E>;

/// Errors from loading config or talking to a provider.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConnectError {
    #[error("config file not found at {0}; run `hvec config init`")]
    ConfigMissing(std::path::PathBuf),
    #[error("could not read config at {path}")]
    ConfigRead {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse config at {path}")]
    ConfigParse {
        path: std::path::PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("could not write config at {path}")]
    ConfigWrite {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("no {kind} profile named `{name}` in config")]
    UnknownProfile { kind: &'static str, name: String },
    #[error("profile `{profile}` needs environment variable {var} to be set")]
    MissingApiKey { profile: String, var: String },
    #[error("profile `{profile}` requires the `{feature}` cargo feature, which is disabled in this build")]
    FeatureDisabled { profile: String, feature: &'static str },
    #[error("HTTP request to {provider} failed")]
    Http {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("{provider} returned HTTP {status}: {body}")]
    Api {
        provider: &'static str,
        status: u16,
        body: String,
    },
    #[error("{provider} returned a response hvec could not interpret: {detail}")]
    MalformedResponse { provider: &'static str, detail: String },
    #[error("{provider} declined the request (category: {category})")]
    Refused { provider: &'static str, category: String },
    #[error("local embedding model error: {0}")]
    LocalModel(String),
    #[error("unknown local embedding model `{0}`; run `hvec embedders` to list supported names")]
    UnknownLocalModel(String),
    #[error("embedding batch returned {got} vectors for {expected} inputs")]
    EmbeddingCount { expected: usize, got: usize },
    #[error("blocking task failed: {0}")]
    Join(String),
}
