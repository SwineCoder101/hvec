//! Error type shared by every connector.

use crate::traits::Attempt;

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
    #[error("{provider} gave up after {} attempts ({})", attempts.len(), summarise(attempts))]
    AfterRetries {
        provider: &'static str,
        /// Every attempt, oldest first; the last one carries the error in `source`.
        attempts: Vec<Attempt>,
        #[source]
        source: Box<ConnectError>,
    },
    #[error("local embedding model error: {0}")]
    LocalModel(String),
    #[error("unknown local embedding model `{0}`; run `hvec embedders` to list supported names")]
    UnknownLocalModel(String),
    #[error("embedding batch returned {got} vectors for {expected} inputs")]
    EmbeddingCount { expected: usize, got: usize },
    #[error("blocking task failed: {0}")]
    Join(String),
}

/// One line per attempt: status, time taken, and the wait that followed.
fn summarise(attempts: &[Attempt]) -> String {
    attempts
        .iter()
        .map(|a| {
            let status = match a.status {
                Some(s) => format!("HTTP {s}"),
                None => "transport error".to_owned(),
            };
            match a.backoff_ms {
                Some(wait) => format!(
                    "attempt {}: {status} after {}ms, waited {wait}ms",
                    a.number, a.elapsed_ms
                ),
                None => format!("attempt {}: {status} after {}ms", a.number, a.elapsed_ms),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}
