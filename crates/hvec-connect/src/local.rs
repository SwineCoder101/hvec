//! In-process embeddings with fastembed (ONNX Runtime).

use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding, TextInitOptions};

use crate::config::{EmbedProfile, expand_tilde};
use crate::traits::Embedder;
use crate::{ConnectError, Result};

/// Local embedder. The model is loaded once and shared.
#[derive(Clone)]
pub struct LocalEmbedder {
    profile: String,
    model_code: String,
    dimension: usize,
    // fastembed's `embed` takes `&mut self`, so the session sits behind a mutex.
    // Calls run on the blocking pool and the lock is never held across an await.
    model: Arc<Mutex<TextEmbedding>>,
}

impl std::fmt::Debug for LocalEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEmbedder")
            .field("profile", &self.profile)
            .field("model_code", &self.model_code)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

/// A supported local model, for `hvec embedders`.
#[derive(Debug, Clone)]
pub struct LocalModelInfo {
    /// Short name accepted in config, for example `BGESmallENV15Q`.
    pub name: String,
    /// Hugging Face repository the files come from.
    pub code: String,
    pub dimension: usize,
    pub description: String,
}

/// Every model fastembed can download. Either `name` or `code` is accepted in config.
#[must_use]
pub fn supported_models() -> Vec<LocalModelInfo> {
    let mut v: Vec<_> = TextEmbedding::list_supported_models()
        .into_iter()
        .map(|m| LocalModelInfo {
            name: format!("{:?}", m.model),
            code: m.model_code,
            dimension: m.dim,
            description: m.description,
        })
        .collect();
    v.sort_by(|a, b| a.code.cmp(&b.code).then_with(|| a.name.cmp(&b.name)));
    v
}

fn find_model(name: &str) -> Result<ModelInfo<EmbeddingModel>> {
    TextEmbedding::list_supported_models()
        .into_iter()
        .find(|m| m.model_code.eq_ignore_ascii_case(name) || format!("{:?}", m.model).eq_ignore_ascii_case(name))
        .ok_or_else(|| ConnectError::UnknownLocalModel(name.to_owned()))
}

impl LocalEmbedder {
    /// Load (and on first use download) the configured model.
    pub async fn from_profile(name: &str, profile: &EmbedProfile) -> Result<Self> {
        let info = find_model(&profile.model)?;
        let cache_dir = profile.cache_dir.as_deref().map(expand_tilde).unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| ".".into())
                .join("hvec")
                .join("models")
        });
        let model_enum = info.model.clone();
        let model_code = info.model_code.clone();
        let dimension = info.dim;

        tracing::info!(model = %model_code, cache = %cache_dir.display(), "loading local embedding model");
        let model = tokio::task::spawn_blocking(move || {
            let opts = TextInitOptions::new(model_enum)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true);
            TextEmbedding::try_new(opts)
        })
        .await
        .map_err(|e| ConnectError::Join(e.to_string()))?
        .map_err(|e| ConnectError::LocalModel(e.to_string()))?;

        Ok(Self {
            profile: name.to_owned(),
            model_code,
            dimension,
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[async_trait::async_trait]
impl Embedder for LocalEmbedder {
    fn profile(&self) -> &str {
        &self.profile
    }

    fn model_id(&self) -> &str {
        &self.model_code
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let model = Arc::clone(&self.model);
        let texts = texts.to_vec();
        let expected = texts.len();
        let out = tokio::task::spawn_blocking(move || {
            let mut guard = model.lock().unwrap_or_else(|p| p.into_inner());
            guard.embed(texts, None)
        })
        .await
        .map_err(|e| ConnectError::Join(e.to_string()))?
        .map_err(|e| ConnectError::LocalModel(e.to_string()))?;
        if out.len() != expected {
            return Err(ConnectError::EmbeddingCount {
                expected,
                got: out.len(),
            });
        }
        Ok(out)
    }
}
