//! Core types for hvec: the [`Codec`] trait, distance kernels and metrics.
//!
//! A codec is *homomorphic* with respect to a distance when the distance can be
//! computed on the compressed representation without decoding. Every codec in
//! hvec declares which [`Metric`]s it preserves and how it computes them.

pub mod codecs;
pub mod distance;

use serde::{Deserialize, Serialize};

/// A dense embedding vector in its uncompressed form.
pub type Vector = Vec<f32>;

/// The similarity or distance function used for ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Metric {
    /// Inner product. Higher is more similar.
    Dot,
    /// Cosine similarity. Higher is more similar.
    Cosine,
    /// Squared Euclidean distance. Lower is more similar.
    L2,
}

impl Metric {
    /// Whether a larger score means a closer match.
    #[must_use]
    pub fn higher_is_better(self) -> bool {
        !matches!(self, Metric::L2)
    }

    /// Parse from the short names used on the command line and in config.
    pub fn parse(name: &str) -> Result<Self, CodecError> {
        match name {
            "dot" => Ok(Metric::Dot),
            "cosine" => Ok(Metric::Cosine),
            "l2" => Ok(Metric::L2),
            other => Err(CodecError::UnknownMetric(other.to_owned())),
        }
    }

    /// The short name used in config and the run log.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Metric::Dot => "dot",
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Errors raised by codecs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    #[error("vector has dimension {got}, codec expects {expected}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("encoded payload has {got} bytes, codec expects {expected}")]
    PayloadSize { expected: usize, got: usize },
    #[error("codec {codec} does not support metric {metric}")]
    UnsupportedMetric { codec: String, metric: Metric },
    #[error("unknown metric: {0}")]
    UnknownMetric(String),
    #[error("unknown codec: {0}")]
    UnknownCodec(String),
}

/// A compressed representation of a vector, as stored on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded(pub Vec<u8>);

/// A compression scheme whose compressed form supports distance computation.
///
/// Implementations must document their error model: exact, bounded, or
/// approximate for each supported metric.
pub trait Codec: Send + Sync + std::fmt::Debug {
    /// Stable identifier recorded in the run log, for example `f32` or `int8`.
    fn name(&self) -> &'static str;

    /// Dimensionality of the vectors this codec instance accepts.
    fn dimension(&self) -> usize;

    /// Size of one encoded vector in bytes.
    fn encoded_size(&self) -> usize;

    /// Metrics that can be computed in the compressed domain.
    fn supported_metrics(&self) -> &'static [Metric];

    /// Compress a single vector.
    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError>;

    /// Reconstruct an approximation of the original vector.
    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError>;

    /// Score an uncompressed query against a compressed stored vector.
    ///
    /// This is the asymmetric form. It must not decode `encoded` into a full
    /// `Vector` unless the codec documents that it is a baseline.
    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError>;

    /// Bytes per vector relative to uncompressed f32.
    fn compression_ratio(&self) -> f32 {
        (self.dimension() * 4) as f32 / self.encoded_size() as f32
    }
}

/// Instantiate a codec by its stable name.
pub fn codec_by_name(name: &str, dimension: usize) -> Result<Box<dyn Codec>, CodecError> {
    match name {
        "f32" => Ok(Box::new(codecs::F32Codec::new(dimension))),
        other => Err(CodecError::UnknownCodec(other.to_owned())),
    }
}

/// Names of every codec this build knows about.
#[must_use]
pub fn codec_names() -> &'static [&'static str] {
    &["f32"]
}
