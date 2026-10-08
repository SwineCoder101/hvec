use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use hvec_bench::recall::{self, RecallQuery, RecallReport};
use hvec_core::codecs::F32Codec;
use hvec_core::{Codec, codec_by_name};
use hvec_store::RunRow;

use std::sync::Arc;

use hvec_bench::questions::{self, Question};
use hvec_bench::report::{self as rpt, Cell};
use hvec_connect::{ChatModel, Embedder};

use crate::cli::{BenchCommand, BenchRunArgs, RecallArgs, ReportArgs};
use crate::commands::rag::Pipeline;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, cmd: BenchCommand) -> Result<()> {
    match cmd {
        BenchCommand::Recall(args) => recall_cmd(config, args).await,
        BenchCommand::Run(args) => run_cmd(config, args).await,
    }
}

async fn run_cmd(config: &Option<PathBuf>, args: BenchRunArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let mut qs: Vec<Question> = questions::load(&args.questions)?;
    if let Some(n) = args.limit {
        qs.truncate(n);
    }
    let set_name = args
        .questions
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("questions")
        .to_owned();
    let chat_names: Vec<String> = if args.chat.is_empty() {
        vec![ctx.config.default_chat.clone()]
    } else {
        args.chat.clone()
    };
    let batch = uuid_short();

    // Resolve everything up front so a typo fails before any model is called.
    let collections = args
        .collections
        .iter()
        .map(|c| store.require_collection(c))
        .collect::<Result<Vec<_>, _>>()?;
    let chats: Vec<Arc<dyn ChatModel>> = chat_names
        .iter()
        .map(|n| hvec_connect::chat_model(&ctx.config, n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut embedders: std::collections::HashMap<String, Arc<dyn Embedder>> = Default::default();
    for c in &collections {
        if !embedders.contains_key(&c.embed_profile) {
            embedders.insert(
                c.embed_profile.clone(),
                hvec_connect::embedder(&ctx.config, &c.embed_profile).await?,
            );
        }
    }

    let total = qs.len() * collections.len() * chats.len();
    eprintln!(
        "batch {batch}: {} questions × {} collections × {} chat profiles = {total} runs (k={})",
        qs.len(),
        collections.len(),
        chats.len(),
        args.k
    );

    let mut done = 0usize;
    let mut failures = 0usize;
    for collection in &collections {
        let codec = codec_by_name(&collection.codec, collection.dimension)?;
        let embedder = Arc::clone(&embedders[&collection.embed_profile]);
        for (chat_name, chat) in chat_names.iter().zip(&chats) {
            let pipeline = Pipeline {
                store: &store,
                collection,
                codec: codec.as_ref(),
                embedder: Arc::clone(&embedder),
                chat: Arc::clone(chat),
            };
            for q in &qs {
                done += 1;
                let mut row = RunRow::new("bench");
                row.collection = Some(collection.name.clone());
                row.chat_profile = Some(chat_name.clone());
                row.chat_model = Some(chat.model_id().to_owned());
                row.embed_profile = Some(collection.embed_profile.clone());
                row.embed_model = Some(collection.embed_model.clone());
                row.codec = Some(collection.codec.clone());
                row.metric = Some(collection.metric.name().to_owned());
                let mut metadata = serde_json::json!({
                    "batch": batch,
                    "label": args.label,
                    "question_set": set_name,
                    "question_id": q.id,
                    "question": q.question,
                    "answers": q.answers,
                    "source": q.source,
                    "prompt_version": hvec_bench::prompt::PROMPT_VERSION,
                    "dimension": collection.dimension,
                    "k": args.k,
                });

                match pipeline.answer(Vec::new(), &q.question, args.k).await {
                    Ok(outcome) => {
                        let score = questions::score_answer(&outcome.response.text, &q.answers);
                        let sources: Vec<String> = outcome.hits.iter().map(|h| h.source.clone()).collect();
                        let hit = questions::source_hit(q.source.as_deref(), &sources);
                        metadata["chat_model_served"] = serde_json::json!(outcome.response.model);
                        metadata["answer"] = serde_json::json!(outcome.response.text);
                        metadata["stop_reason"] = serde_json::json!(outcome.response.stop_reason);
                        let mut metrics = serde_json::to_value(&outcome.metrics)?;
                        metrics["exact"] = serde_json::json!(score.exact);
                        metrics["contains"] = serde_json::json!(score.contains);
                        metrics["source_hit"] = serde_json::json!(hit);
                        row.metrics = metrics;
                        eprintln!(
                            "[{done}/{total}] {} · {} · {} → {}{}",
                            collection.name,
                            chat_name,
                            q.id,
                            if score.contains { "ok" } else { "miss" },
                            match hit {
                                Some(true) => " (source hit)",
                                Some(false) => " (source miss)",
                                None => "",
                            }
                        );
                    }
                    Err(e) => {
                        failures += 1;
                        metadata["error"] = serde_json::json!(format!("{e:#}"));
                        eprintln!(
                            "[{done}/{total}] {} · {} · {} → ERROR {e:#}",
                            collection.name, chat_name, q.id
                        );
                        if args.fail_fast {
                            row.metadata = metadata;
                            store.insert_run(&row)?;
                            bail!("stopping after first error (--fail-fast)");
                        }
                    }
                }
                row.metadata = metadata;
                store.insert_run(&row)?;
            }
        }
    }

    eprintln!(
        "\n{done} runs recorded in batch {batch}{}",
        if failures > 0 {
            format!(", {failures} errors")
        } else {
            String::new()
        }
    );
    let rows = store.runs_by_kind("bench", Some(&batch))?;
    print_cells(&rpt::aggregate(&rows));
    eprintln!("\nreport again with: hvec report --batch {batch}");
    Ok(())
}

pub fn report(config: &Option<PathBuf>, args: &ReportArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let rows = store.runs_by_kind("bench", args.batch.as_deref())?;
    let mut cells = rpt::aggregate(&rows);
    if let Some(set) = &args.set {
        cells.retain(|c| &c.key.question_set == set);
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&cells)?);
        return Ok(());
    }
    if cells.is_empty() {
        println!("no bench runs match; run `hvec bench run` first");
        return Ok(());
    }
    print_cells(&cells);
    Ok(())
}

fn print_cells(cells: &[Cell]) {
    let mut rerouted = 0usize;
    println!(
        "{:<12} {:<18} {:<26} {:<8} {:<24} {:>4} {:>6} {:>8} {:>7} {:>7} {:>7} {:>6} {:>6} {:>4}",
        "set",
        "collection",
        "embed model",
        "codec",
        "chat model",
        "n",
        "exact",
        "contains",
        "src hit",
        "ret ms",
        "gen ms",
        "in",
        "out",
        "err"
    );
    for c in cells {
        let embed = c.key.embed_model.rsplit('/').next().unwrap_or(&c.key.embed_model);
        println!(
            "{:<12} {:<18} {:<26} {:<8} {:<24} {:>4} {:>6.3} {:>8.3} {:>7} {:>7.0} {:>7.0} {:>6.0} {:>6.0} {:>4}",
            trunc(&c.key.question_set, 12),
            trunc(&c.key.collection, 18),
            trunc(embed, 26),
            c.key.codec,
            trunc(&c.key.chat_model, 24),
            c.n,
            c.exact,
            c.contains,
            c.source_hit.map_or("-".to_owned(), |v| format!("{v:.3}")),
            c.mean_retrieval_ms,
            c.mean_generation_ms,
            c.mean_input_tokens,
            c.mean_output_tokens,
            c.errors
        );
        rerouted += c.served_other;
    }
    if rerouted > 0 {
        println!(
            "\n{rerouted} run(s) were served by a different model than configured (fallback routing); see metadata.chat_model_served"
        );
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

fn uuid_short() -> String {
    RunRow::new("x").id[..8].to_owned()
}

async fn recall_cmd(config: &Option<PathBuf>, args: RecallArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let collection = store.require_collection(&args.collection)?;
    if collection.codec != "f32" {
        bail!(
            "`bench recall` needs an f32 collection as ground truth; `{}` is stored as {}",
            collection.name,
            collection.codec
        );
    }
    if collection.chunk_count == 0 {
        bail!("collection `{}` is empty", collection.name);
    }

    let baseline = F32Codec::new(collection.dimension);
    let (ids, vectors): (Vec<i64>, Vec<Vec<f32>>) = store
        .vectors(&collection.name)?
        .into_iter()
        .map(|(id, enc)| baseline.decode(&enc).map(|v| (id, v)))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();

    let (queries, query_source) = match &args.queries {
        Some(path) => {
            let text = tokio::fs::read_to_string(path)
                .await
                .with_context(|| format!("reading {}", path.display()))?;
            let lines: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect();
            if lines.is_empty() {
                bail!("{} has no queries", path.display());
            }
            let embedder = hvec_connect::embedder(&ctx.config, &collection.embed_profile).await?;
            let qs = embedder
                .embed(&lines)
                .await?
                .into_iter()
                .map(|vector| RecallQuery { vector, exclude: None })
                .collect();
            (qs, format!("file:{}", path.display()))
        }
        None => (
            recall::self_queries(&ids, &vectors, args.sample),
            format!("self-queries:{}", args.sample.min(vectors.len())),
        ),
    };

    let mut names: Vec<String> = vec!["f32".to_owned()];
    names.extend(args.codecs.iter().filter(|c| c.as_str() != "f32").cloned());

    let mut reports: Vec<RecallReport> = Vec::new();
    for name in &names {
        let codec = codec_by_name(name, collection.dimension)?;
        let report = recall::evaluate(&ids, &vectors, &queries, codec.as_ref(), collection.metric, args.k)?;
        if !args.no_record {
            let mut row = RunRow::new("recall");
            row.collection = Some(collection.name.clone());
            row.embed_profile = Some(collection.embed_profile.clone());
            row.embed_model = Some(collection.embed_model.clone());
            row.codec = Some(report.codec.clone());
            row.metric = Some(collection.metric.name().to_owned());
            row.metadata = serde_json::json!({
                "dimension": collection.dimension,
                "queries": query_source,
                "k": args.k,
            });
            row.metrics = serde_json::to_value(&report)?;
            store.insert_run(&row)?;
        }
        reports.push(report);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }
    println!(
        "collection `{}` · {} vectors · {} dims · {} · {} · k={}",
        collection.name,
        vectors.len(),
        collection.dimension,
        collection.embed_model,
        collection.metric.name(),
        args.k
    );
    println!("queries: {query_source}");
    println!();
    println!(
        "{:<8} {:>9} {:>7} {:>10} {:>10} {:>8} {:>7} {:>9} {:>8}",
        "codec", "recall@k", "top1", "mean err", "max err", "bytes", "ratio", "encode ms", "scan ms"
    );
    for r in &reports {
        println!(
            "{:<8} {:>9.4} {:>7.3} {:>10.5} {:>10.5} {:>8} {:>6.1}x {:>9} {:>8}",
            r.codec,
            r.recall_at_k,
            r.top1_agreement,
            r.mean_abs_score_error,
            r.max_abs_score_error,
            r.bytes_per_vector,
            r.compression_ratio,
            r.encode_ms,
            r.scan_ms
        );
    }
    if !args.no_record {
        eprintln!("\n{} runs recorded (kind: recall)", reports.len());
    }
    Ok(())
}
