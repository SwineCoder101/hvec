pub mod bench;
pub mod config;
pub mod ingest;
pub mod judge;
pub mod list;
pub mod memory;
pub mod query;
pub mod questions;
pub mod shell;

/// Shared retrieve-then-answer step used by `query` and `chat`.
pub mod rag {
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::{Context as _, Result};
    use hvec_bench::payload::{self, Payload, PayloadStats};
    use hvec_bench::{QueryMetrics, RAG_SYSTEM_PROMPT, RetrievedChunk, build_user_prompt};
    use hvec_connect::{ChatModel, ChatRequest, ChatResponse, Embedder, Message};
    use hvec_core::Codec;
    use hvec_store::{Collection, Hit, Store};

    pub struct TurnOutcome {
        /// The passages as retrieved, before the payload stage.
        pub hits: Vec<Hit>,
        /// The passages as sent to the model.
        pub sent: Vec<Hit>,
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
        pub payload: Payload,
    }

    /// Run the payload stage over the retrieved passages.
    pub async fn apply_payload(stage: &Payload, question: &str, hits: &[Hit]) -> Result<Vec<Hit>> {
        match stage {
            Payload::Http { url } => {
                let body = payload::HttpRequest::new(question, hits);
                let resp = reqwest::Client::new()
                    .post(url)
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("payload compressor at {url}"))?;
                let status = resp.status();
                let text = resp
                    .text()
                    .await
                    .with_context(|| format!("payload compressor at {url}"))?;
                if !status.is_success() {
                    anyhow::bail!("payload compressor at {url} returned {status}: {text}");
                }
                let parsed: payload::HttpResponse =
                    serde_json::from_str(&text).with_context(|| format!("payload compressor reply: {text}"))?;
                payload::apply_response(hits, parsed).map_err(|e| anyhow::anyhow!(e))
            }
            local => Ok(payload::apply_local(local, hits)),
        }
    }

    impl Pipeline<'_> {
        /// Embed the question, retrieve `k` passages, and ask the model with
        /// the prior `history` prepended. Measures each stage.
        pub async fn answer(&self, history: Vec<Message>, question: &str, k: usize) -> Result<TurnOutcome> {
            self.answer_with_query(history, question, question, k).await
        }

        /// Like [`Pipeline::answer`], but embeds `query` for retrieval while
        /// the model is asked `question`. A question set's `query` field uses
        /// this to phrase the prompt without changing the retrieval text.
        pub async fn answer_with_query(
            &self,
            history: Vec<Message>,
            query: &str,
            question: &str,
            k: usize,
        ) -> Result<TurnOutcome> {
            let t = Instant::now();
            let mut qv = self
                .embedder
                .embed_queries(std::slice::from_ref(&query.to_owned()))
                .await?;
            let query = qv.pop().ok_or_else(|| anyhow::anyhow!("embedder returned no vector"))?;
            let embed_ms = t.elapsed().as_millis() as u64;

            let t = Instant::now();
            let hits = self
                .store
                .search(&self.collection.name, &query, k, self.codec, self.collection.metric)?;
            let retrieval_ms = t.elapsed().as_millis() as u64;

            let t = Instant::now();
            let sent = apply_payload(&self.payload, question, &hits).await?;
            let payload_ms = t.elapsed().as_millis() as u64;
            let stats = PayloadStats::measure(&hits, &sent);

            let mut messages = history;
            messages.push(Message::user(build_user_prompt(&sent, question)));

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
                payload_ms,
                context_chars_before: stats.context_chars_before,
                context_chars_after: stats.context_chars_after,
                context_words_before: stats.context_words_before,
                context_words_after: stats.context_words_after,
            };
            Ok(TurnOutcome {
                hits,
                sent,
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
