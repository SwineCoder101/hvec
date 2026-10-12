//! Task prefixes for embedding models that expect them.
//!
//! Several retrieval models are trained with a marker in front of the text:
//! nomic-embed uses `search_query: ` and `search_document: `, E5 uses
//! `query: ` and `passage: `. The wrapper adds the configured prefix on the
//! way in; the raw [`Embedder::embed`] stays prefix-free.

use std::sync::Arc;

use crate::Result;
use crate::traits::Embedder;

/// An embedder whose document and query inputs get a fixed prefix.
#[derive(Debug, Clone)]
pub struct PrefixedEmbedder {
    inner: Arc<dyn Embedder>,
    query_prefix: String,
    document_prefix: String,
}

impl PrefixedEmbedder {
    #[must_use]
    pub fn new(inner: Arc<dyn Embedder>, query_prefix: Option<&str>, document_prefix: Option<&str>) -> Self {
        Self {
            inner,
            query_prefix: query_prefix.unwrap_or("").to_owned(),
            document_prefix: document_prefix.unwrap_or("").to_owned(),
        }
    }

    fn with_prefix(prefix: &str, texts: &[String]) -> Vec<String> {
        texts.iter().map(|t| format!("{prefix}{t}")).collect()
    }
}

#[async_trait::async_trait]
impl Embedder for PrefixedEmbedder {
    fn profile(&self) -> &str {
        self.inner.profile()
    }

    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed(texts).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.document_prefix.is_empty() {
            return self.inner.embed(texts).await;
        }
        self.inner.embed(&Self::with_prefix(&self.document_prefix, texts)).await
    }

    async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.query_prefix.is_empty() {
            return self.inner.embed(texts).await;
        }
        self.inner.embed(&Self::with_prefix(&self.query_prefix, texts)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbedProfile, EmbedProvider};
    use crate::hash::HashEmbedder;

    #[tokio::test]
    async fn prefixes_change_only_the_prefixed_side() {
        let profile = EmbedProfile {
            provider: EmbedProvider::Hash,
            model: "hash".into(),
            base_url: None,
            api_key_env: None,
            dimension: Some(64),
            cache_dir: None,
            query_prefix: None,
            document_prefix: None,
        };
        let raw = HashEmbedder::from_profile("t", &profile).unwrap();
        let wrapped = PrefixedEmbedder::new(Arc::new(raw.clone()), Some("search_query: "), None);
        let text = vec!["refund window".to_owned()];
        assert_eq!(wrapped.embed(&text).await.unwrap(), raw.embed(&text).await.unwrap());
        assert_eq!(
            wrapped.embed_documents(&text).await.unwrap(),
            raw.embed(&text).await.unwrap()
        );
        let q = wrapped.embed_queries(&text).await.unwrap();
        assert_eq!(q, raw.embed(&["search_query: refund window".to_owned()]).await.unwrap());
        assert_ne!(q, raw.embed(&text).await.unwrap());
        assert_eq!(wrapped.dimension(), 64);
        assert_eq!(wrapped.profile(), "t");
    }
}
