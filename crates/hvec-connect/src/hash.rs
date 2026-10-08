//! A deterministic, offline embedder for tests and CI.
//!
//! Hashes each lowercase word into one of `dimension` buckets with a sign bit
//! and normalises the result. Lexically similar texts get similar vectors.
//! It has no semantics and must never be used for a published benchmark;
//! its job is to make the pipeline testable without a model download or an
//! API key.

use crate::config::EmbedProfile;
use crate::traits::Embedder;
use crate::{ConnectError, Result};

/// Default dimension when the profile does not set one.
pub const DEFAULT_DIMENSION: usize = 256;

#[derive(Debug, Clone)]
pub struct HashEmbedder {
    profile: String,
    model: String,
    dimension: usize,
}

impl HashEmbedder {
    pub fn from_profile(name: &str, profile: &EmbedProfile) -> Result<Self> {
        let dimension = profile.dimension.unwrap_or(DEFAULT_DIMENSION);
        if dimension < 8 {
            return Err(ConnectError::LocalModel(format!(
                "hash embedder needs dimension >= 8, got {dimension}"
            )));
        }
        Ok(Self {
            profile: name.to_owned(),
            model: profile.model.clone(),
            dimension,
        })
    }

    /// Embed one text. Public so tests can call it synchronously.
    #[must_use]
    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dimension];
        for word in text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            let h = fnv1a(word.to_lowercase().as_bytes());
            let idx = (h % self.dimension as u64) as usize;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        hvec_core::distance::normalize(&mut v);
        v
    }
}

/// FNV-1a, stable across Rust versions unlike `DefaultHasher`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[async_trait::async_trait]
impl Embedder for HashEmbedder {
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
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hvec_core::distance::cosine;

    fn emb() -> HashEmbedder {
        HashEmbedder {
            profile: "t".into(),
            model: "hash".into(),
            dimension: 64,
        }
    }

    #[test]
    fn deterministic_and_unit_length() {
        let e = emb();
        let a = e.embed_one("Refunds within 30 days");
        let b = e.embed_one("Refunds within 30 days");
        assert_eq!(a, b);
        let n: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn overlapping_words_score_higher() {
        let e = emb();
        let q = e.embed_one("refund window");
        let near = e.embed_one("the refund window is thirty days");
        let far = e.embed_one("shipping takes three business days");
        assert!(cosine(&q, &near) > cosine(&q, &far));
    }

    #[test]
    fn empty_text_is_zero_vector() {
        assert!(emb().embed_one("   ").iter().all(|x| *x == 0.0));
    }
}
