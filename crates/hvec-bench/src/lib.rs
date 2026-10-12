//! Pieces of the RAG pipeline that must stay fixed across benchmark runs:
//! chunking, prompt construction, metrics and the run record.

pub mod beir;
pub mod chunk;
pub mod judge;
pub mod memory;
pub mod metrics;
pub mod payload;
pub mod prompt;
pub mod questions;
pub mod recall;
pub mod record;
pub mod report;

pub use chunk::{ChunkOptions, chunk_text};
pub use judge::{JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, Verdict, build_judge_prompt, parse_verdict};
pub use payload::{Payload, PayloadStats};
pub use prompt::{RAG_SYSTEM_PROMPT, build_user_prompt};
pub use questions::{AnswerScore, Question, RelevanceScore};
pub use recall::{RecallQuery, RecallReport};
pub use record::{QueryMetrics, QueryRecord, RetrievedChunk};
pub use report::{Cell, CellKey};
