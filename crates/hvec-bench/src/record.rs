//! Typed run records that serialise into the store's generic run log.

use hvec_store::{Hit, RunRow};
use serde::{Deserialize, Serialize};

use crate::prompt::PROMPT_VERSION;

/// A retrieved chunk as recorded, without the full text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedChunk {
    pub chunk_id: i64,
    pub source: String,
    pub ordinal: u32,
    pub score: f32,
}

impl From<&Hit> for RetrievedChunk {
    fn from(h: &Hit) -> Self {
        Self {
            chunk_id: h.chunk_id,
            source: h.source.clone(),
            ordinal: h.ordinal,
            score: h.score,
        }
    }
}

/// Outcome of one query or chat turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub k: usize,
    pub embed_ms: u64,
    pub retrieval_ms: u64,
    pub generation_ms: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub retrieved: Vec<RetrievedChunk>,
}

/// Setup and outcome of one query, ready to be written as a `RunRow`.
#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub kind: &'static str,
    pub collection: String,
    pub chat_profile: String,
    pub chat_model_configured: String,
    pub chat_model_reported: String,
    pub embed_profile: String,
    pub embed_model: String,
    pub dimension: usize,
    pub codec: String,
    pub metric: String,
    pub session: Option<String>,
    pub question: String,
    pub answer: String,
    pub stop_reason: Option<String>,
    pub metrics: QueryMetrics,
}

impl QueryRecord {
    /// Convert into the generic row. Never fails for these field types, so the
    /// JSON conversion panics are unreachable in practice.
    #[must_use]
    pub fn into_row(self) -> RunRow {
        let mut row = RunRow::new(self.kind);
        row.collection = Some(self.collection);
        row.chat_profile = Some(self.chat_profile);
        row.chat_model = Some(self.chat_model_reported);
        row.embed_profile = Some(self.embed_profile);
        row.embed_model = Some(self.embed_model);
        row.codec = Some(self.codec);
        row.metric = Some(self.metric);
        row.metadata = serde_json::json!({
            "prompt_version": PROMPT_VERSION,
            "chat_model_configured": self.chat_model_configured,
            "dimension": self.dimension,
            "session": self.session,
            "question": self.question,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
        });
        row.metrics = serde_json::to_value(&self.metrics).expect("QueryMetrics serialises to JSON");
        row
    }
}
