pub mod config;
pub mod ingest;
pub mod list;
pub mod query;
pub mod shell;

/// Shared retrieve-then-answer step used by `query` and `chat`.
pub mod rag {
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::Result;
    use hvec_bench::{QueryMetrics, RAG_SYSTEM_PROMPT, RetrievedChunk, build_user_prompt};
    use hvec_connect::{ChatModel, ChatRequest, ChatResponse, Embedder, Message};
    use hvec_core::Codec;
    use hvec_store::{Collection, Hit, Store};

    pub struct TurnOutcome {
        pub hits: Vec<Hit>,
        pub response: ChatResponse,
        pub metrics: QueryMetrics,
    }

    /// Everything a retrieve-then-answer turn needs, wired once per command.
    pub struct Pipeline<'a> {
        pub store: &'a Store,
        pub collection: &'a Collection,
        pub codec: &'a dyn Codec,
        pub embedder: Arc<dyn Embedder>,
        pub chat: Arc<dyn ChatModel>,
    }

    impl Pipeline<'_> {
        /// Embed the question, retrieve `k` passages, and ask the model with
        /// the prior `history` prepended. Measures each stage.
        pub async fn answer(&self, history: Vec<Message>, question: &str, k: usize) -> Result<TurnOutcome> {
            let t = Instant::now();
            let mut qv = self.embedder.embed(std::slice::from_ref(&question.to_owned())).await?;
            let query = qv.pop().ok_or_else(|| anyhow::anyhow!("embedder returned no vector"))?;
            let embed_ms = t.elapsed().as_millis() as u64;

            let t = Instant::now();
            let hits = self
                .store
                .search(&self.collection.name, &query, k, self.codec, self.collection.metric)?;
            let retrieval_ms = t.elapsed().as_millis() as u64;

            let mut messages = history;
            messages.push(Message::user(build_user_prompt(&hits, question)));

            let t = Instant::now();
            let response = self
                .chat
                .complete(ChatRequest {
                    system: Some(RAG_SYSTEM_PROMPT.to_owned()),
                    messages,
                    max_tokens: None,
                })
                .await?;
            let generation_ms = t.elapsed().as_millis() as u64;

            let metrics = QueryMetrics {
                k,
                embed_ms,
                retrieval_ms,
                generation_ms,
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
                retrieved: hits.iter().map(RetrievedChunk::from).collect(),
            };
            Ok(TurnOutcome {
                hits,
                response,
                metrics,
            })
        }
    }

    pub fn print_context(hits: &[Hit]) {
        for (i, h) in hits.iter().enumerate() {
            println!("--- [{}] {} #{} (score {:.4})", i + 1, h.source, h.ordinal, h.score);
            println!("{}", h.text.trim());
        }
        println!("---");
    }
}
