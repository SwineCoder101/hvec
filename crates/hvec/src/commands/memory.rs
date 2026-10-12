use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use hvec_bench::memory::{self, MemorySummary, Step};
use hvec_bench::payload::Payload;
use hvec_bench::{ChunkOptions, chunk_text, questions};
use hvec_core::codec_by_name;
use hvec_store::{NewChunk, NewCollection, RunRow};

use crate::cli::MemoryArgs;
use crate::commands::rag::Pipeline;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, args: MemoryArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let metric = ctx.metric()?;
    let mut store = ctx.open_store()?;
    let steps: Vec<Step> = memory::load(&args.script)?;
    let script_name = args
        .script
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("script")
        .to_owned();
    let embed_name = ctx.embed_profile_name(args.embedder.as_deref()).to_owned();
    let chat_name = ctx.chat_profile_name(args.chat.as_deref()).to_owned();
    let embedder = hvec_connect::embedder(&ctx.config, &embed_name).await?;
    let chat = hvec_connect::chat_model(&ctx.config, &chat_name)?;
    let dimension = embedder.dimension();
    let batch = RunRow::new("x").id[..8].to_owned();
    let opts = ChunkOptions {
        words: args.chunk_words,
        overlap: args.overlap_words,
    };
    let asks = steps.iter().filter(|s| !s.is_write()).count();
    eprintln!(
        "batch {batch}: {} steps ({} writes, {asks} asks) × {} codecs, embedder {} ({}), chat {} ({}), k={}",
        steps.len(),
        steps.len() - asks,
        args.codecs.len(),
        embed_name,
        embedder.model_id(),
        chat_name,
        chat.model_id(),
        args.k
    );

    // Trained codecs need corpus statistics before the first write. A real
    // agent would fit on a warm-up sample; here the script's own writes
    // stand in, and the row says so.
    let write_texts: Vec<String> = steps.iter().filter_map(|s| s.write.clone()).collect();
    let mut warmup: Option<Vec<Vec<f32>>> = None;
    let mut all_rows: Vec<RunRow> = Vec::new();

    for codec_name in &args.codecs {
        let mut codec = codec_by_name(codec_name, dimension)?;
        let mut fitted_on = serde_json::Value::Null;
        if codec.needs_fit() {
            if warmup.is_none() {
                let mut vs = Vec::new();
                for group in write_texts.chunks(32) {
                    vs.extend(embedder.embed_documents(group).await?);
                }
                warmup = Some(vs);
            }
            let sample = warmup.as_ref().expect("warm-up embedded");
            codec
                .fit(sample)
                .with_context(|| format!("fitting {} on {} script writes", codec.name(), sample.len()))?;
            fitted_on = serde_json::json!({ "source": "script-writes", "vectors": sample.len() });
        }
        let collection_name = format!("memory-{batch}-{}", codec.name());
        store.create_collection(&NewCollection {
            name: &collection_name,
            embed_profile: &embed_name,
            embed_model: embedder.model_id(),
            dimension,
            codec: codec.name(),
            metric,
            codec_params: &codec.params(),
        })?;
        let collection = store.require_collection(&collection_name)?;

        let mut cumulative_bytes = 0u64;
        let mut cumulative_chunks = 0u64;
        for (index, step) in steps.iter().enumerate() {
            let mut row = RunRow::new("memory");
            row.collection = Some(collection_name.clone());
            row.chat_profile = Some(chat_name.clone());
            row.chat_model = Some(chat.model_id().to_owned());
            row.embed_profile = Some(embed_name.clone());
            row.embed_model = Some(embedder.model_id().to_owned());
            row.codec = Some(codec.name().to_owned());
            row.metric = Some(metric.name().to_owned());
            let mut metadata = serde_json::json!({
                "batch": batch,
                "label": args.label,
                "script": script_name,
                "step_id": step.id,
                "step_index": index,
                "step": if step.is_write() { "write" } else { "ask" },
                "source": step.source,
                "k": args.k,
                "fitted_on": fitted_on,
                "prompt_version": hvec_bench::prompt::PROMPT_VERSION,
            });

            if let Some(text) = &step.write {
                let t = Instant::now();
                let pieces = chunk_text(text, opts);
                let vectors = embedder.embed_documents(&pieces).await?;
                let embed_ms = t.elapsed().as_millis() as u64;
                let t = Instant::now();
                let source = step.source.clone().unwrap_or_else(|| format!("step:{}", step.id));
                let chunks = pieces
                    .iter()
                    .zip(&vectors)
                    .enumerate()
                    .map(|(i, (text, v))| {
                        Ok(NewChunk {
                            source: source.clone(),
                            ordinal: i as u32,
                            text: text.clone(),
                            vector: codec.encode(v)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let bytes: u64 = chunks.iter().map(|c| c.vector.0.len() as u64).sum();
                let n = store.insert_chunks(&collection_name, &chunks)?;
                let store_ms = t.elapsed().as_millis() as u64;
                cumulative_bytes += bytes;
                cumulative_chunks += n as u64;
                row.metrics = serde_json::json!({
                    "embed_ms": embed_ms,
                    "store_ms": store_ms,
                    "write_ms": embed_ms + store_ms,
                    "chunks": n,
                    "bytes": bytes,
                    "cumulative_bytes": cumulative_bytes,
                    "cumulative_chunks": cumulative_chunks,
                });
                eprintln!(
                    "[{}] {:<15} write {:<8} {n} chunk(s), {bytes} B, {cumulative_bytes} B stored",
                    index + 1,
                    codec.name(),
                    step.id
                );
            } else if let Some(question) = &step.ask {
                metadata["question"] = serde_json::json!(question);
                metadata["answers"] = serde_json::json!(step.answers);
                let pipeline = Pipeline {
                    store: &store,
                    collection: &collection,
                    codec: codec.as_ref(),
                    embedder: Arc::clone(&embedder),
                    chat: Arc::clone(&chat),
                    payload: Payload::None,
                };
                match pipeline.answer(Vec::new(), question, args.k).await {
                    Ok(outcome) => {
                        let score = questions::score_answer(&outcome.response.text, &step.answers);
                        let sources: Vec<String> = outcome.hits.iter().map(|h| h.source.clone()).collect();
                        let hit = questions::source_hit(step.source.as_deref(), &[], &sources);
                        metadata["answer"] = serde_json::json!(outcome.response.text);
                        metadata["chat_model_served"] = serde_json::json!(outcome.response.model);
                        let mut metrics = serde_json::to_value(&outcome.metrics)?;
                        metrics["exact"] = serde_json::json!(score.exact);
                        metrics["contains"] = serde_json::json!(score.contains);
                        metrics["source_hit"] = serde_json::json!(hit);
                        metrics["cumulative_bytes"] = serde_json::json!(cumulative_bytes);
                        metrics["cumulative_chunks"] = serde_json::json!(cumulative_chunks);
                        row.metrics = metrics;
                        eprintln!(
                            "[{}] {:<15} ask   {:<8} {}{}",
                            index + 1,
                            codec.name(),
                            step.id,
                            if step.answers.is_empty() {
                                "answered".to_owned()
                            } else if score.contains {
                                "ok".to_owned()
                            } else {
                                "miss".to_owned()
                            },
                            match hit {
                                Some(true) => " (source hit)",
                                Some(false) => " (source miss)",
                                None => "",
                            }
                        );
                    }
                    Err(e) => {
                        metadata["error"] = serde_json::json!(format!("{e:#}"));
                        eprintln!("[{}] {:<15} ask   {:<8} ERROR {e:#}", index + 1, codec.name(), step.id);
                        if args.fail_fast {
                            row.metadata = metadata;
                            store.insert_run(&row)?;
                            bail!("stopping after first error (--fail-fast)");
                        }
                    }
                }
            }
            row.metadata = metadata;
            if !args.no_record {
                store.insert_run(&row)?;
            }
            all_rows.push(row);
        }
        if !args.keep {
            store.delete_collection(&collection_name)?;
        }
    }

    let summaries = memory::summarize(&all_rows);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
        return Ok(());
    }
    print_summaries(&summaries);
    if !args.no_record {
        eprintln!(
            "\n{} step rows recorded in batch {batch} (kind: memory)",
            all_rows.len()
        );
    }
    Ok(())
}

fn print_summaries(summaries: &[MemorySummary]) {
    println!(
        "{:<15} {:>6} {:>5} {:>9} {:>7} {:>6} {:>8} {:>7} {:>8} {:>7} {:>6} {:>4}",
        "codec", "writes", "asks", "bytes", "chunks", "exact", "contains", "src hit", "write ms", "ret ms", "in", "err"
    );
    for s in summaries {
        println!(
            "{:<15} {:>6} {:>5} {:>9} {:>7} {:>6.3} {:>8.3} {:>7} {:>8.1} {:>7.1} {:>6.0} {:>4}",
            s.codec,
            s.writes,
            s.asks,
            s.total_bytes,
            s.total_chunks,
            s.exact,
            s.contains,
            s.source_hit.map_or("-".to_owned(), |v| format!("{v:.3}")),
            s.mean_write_ms,
            s.mean_retrieval_ms,
            s.mean_input_tokens,
            s.errors
        );
    }
}
