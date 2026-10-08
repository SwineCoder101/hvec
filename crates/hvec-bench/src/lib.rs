//! Pieces of the RAG pipeline that must stay fixed across benchmark runs:
//! chunking, prompt construction, metrics and the run record.

pub mod chunk;
pub mod metrics;
pub mod prompt;
pub mod recall;
pub mod record;

pub use chunk::{ChunkOptions, chunk_text};
pub use prompt::{RAG_SYSTEM_PROMPT, build_user_prompt};
pub use recall::{RecallQuery, RecallReport};
pub use record::{QueryMetrics, QueryRecord, RetrievedChunk};
