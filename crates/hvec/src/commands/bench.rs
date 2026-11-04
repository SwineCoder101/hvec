use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use hvec_bench::recall::{self, RecallQuery, RecallReport};
use hvec_bench::rescore;
use hvec_core::codecs::F32Codec;
use hvec_core::{Codec, codec_by_name};
use hvec_store::RunRow;

use std::sync::Arc;

use hvec_bench::payload::Payload;
use hvec_bench::questions::{self, Question};
use hvec_bench::report::{self as rpt, Cell};
use hvec_connect::{ChatModel, ConnectError, Embedder};

use crate::cli::{BenchCommand, BenchRunArgs, RecallArgs, ReportArgs, RescoreArgs};
use crate::commands::rag::Pipeline;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, cmd: BenchCommand) -> Result<()> {
    match cmd {
        BenchCommand::Recall(args) => recall_cmd(config, args).await,
        BenchCommand::Rescore(args) => rescore_cmd(config, args).await,
        BenchCommand::Run(args) => run_cmd(config, args).await,
        BenchCommand::Memory(args) => crate::commands::memory::run(config, args).await,
        BenchCommand::Scan(args) => crate::commands::scan::run(config, &args),
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
    let payload = Payload::parse(&args.payload)?;

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
        "batch {batch}: {} questions × {} collections × {} chat profiles = {total} runs (k={}, payload {})",
        qs.len(),
        collections.len(),
        chats.len(),
        args.k,
        payload.name()
    );

    let mut done = 0usize;
    let mut failures = 0usize;
    for collection in &collections {
        let codec = collection.codec()?;
        let embedder = Arc::clone(&embedders[&collection.embed_profile]);
        for (chat_name, chat) in chat_names.iter().zip(&chats) {
            let pipeline = Pipeline {
                store: &store,
                collection,
                codec: codec.as_ref(),
                embedder: Arc::clone(&embedder),
                chat: Arc::clone(chat),
                payload: payload.clone(),
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
                    "relevant": q.relevant,
                    "query": q.query,
                    "prompt_version": hvec_bench::prompt::PROMPT_VERSION,
                    "dimension": collection.dimension,
                    "k": args.k,
                    "payload": payload.name(),
                });

                match pipeline
                    .answer_with_query(Vec::new(), q.retrieval_text(), &q.question, args.k)
                    .await
                {
                    Ok(outcome) => {
                        let score = questions::score_answer(&outcome.response.text, &q.answers);
                        let sources: Vec<String> = outcome.hits.iter().map(|h| h.source.clone()).collect();
                        let hit = questions::source_hit(q.source.as_deref(), &q.relevant, &sources);
                        let rel = questions::relevance(&q.relevant, &sources);
                        metadata["chat_model_served"] = serde_json::json!(outcome.response.model);
                        metadata["answer"] = serde_json::json!(outcome.response.text);
                        metadata["stop_reason"] = serde_json::json!(outcome.response.stop_reason);
                        metadata["attempts"] = serde_json::to_value(&outcome.response.attempts)?;
                        let mut metrics = serde_json::to_value(&outcome.metrics)?;
                        metrics["exact"] = serde_json::json!(score.exact);
                        metrics["contains"] = serde_json::json!(score.contains);
                        metrics["source_hit"] = serde_json::json!(hit);
                        metrics["doc_recall"] = serde_json::json!(rel.map(|r| r.doc_recall));
                        metrics["ndcg"] = serde_json::json!(rel.map(|r| r.ndcg));
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
                        if let Some(ConnectError::AfterRetries { attempts, .. }) = e.downcast_ref::<ConnectError>() {
                            metadata["attempts"] = serde_json::json!(attempts);
                        }
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
    let rows = store.runs_by_kinds(&["bench", "judge"], Some(&batch))?;
    print_cells(&rpt::aggregate(&rows));
    eprintln!("\nreport again with: hvec report --batch {batch}");
    Ok(())
}

pub fn report(config: &Option<PathBuf>, args: &ReportArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let rows = store.runs_by_kinds(&["bench", "judge"], args.batch.as_deref())?;
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
    let judged = cells.iter().any(|c| c.judged > 0);
    let opt = |v: Option<f64>| v.map_or("-".to_owned(), |v| format!("{v:.3}"));
    println!(
        "{:<12} {:<18} {:<26} {:<15} {:<24} {:<12} {:>4} {:>5} {:>6} {:>8} {:>7} {:>6}{} {:>7} {:>7} {:>6} {:>6} {:>4}",
        "set",
        "collection",
        "embed model",
        "codec",
        "chat model",
        "payload",
        "n",
        "gold",
        "exact",
        "contains",
        "src hit",
        "ndcg",
        if judged { "  judged faithf" } else { "" },
        "ret ms",
        "gen ms",
        "in",
        "out",
        "err"
    );
    for c in cells {
        let embed = c.key.embed_model.rsplit('/').next().unwrap_or(&c.key.embed_model);
        println!(
            "{:<12} {:<18} {:<26} {:<15} {:<24} {:<12} {:>4} {:>5} {:>6.3} {:>8.3} {:>7} {:>6}{} {:>7.0} {:>7.0} {:>6.0} {:>6.0} {:>4}",
            trunc(&c.key.question_set, 12),
            trunc(&c.key.collection, 18),
            trunc(embed, 26),
            c.key.codec,
            trunc(&c.key.chat_model, 24),
            trunc(&c.key.payload, 12),
            c.n,
            c.scored,
            c.exact,
            c.contains,
            opt(c.source_hit),
            opt(c.ndcg),
            if judged {
                format!("  {:>6} {:>6}", opt(c.judge_correct), opt(c.judge_faithful))
            } else {
                String::new()
            },
            c.mean_retrieval_ms,
            c.mean_generation_ms,
            c.mean_input_tokens,
            c.mean_output_tokens,
            c.errors
        );
        rerouted += c.served_other;
    }
    if judged {
        println!(
            "\njudged = fraction the judge marked correct (over judged questions with gold); faithf = fraction judged faithful to the retrieved passages"
        );
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
    let mut ids: Vec<i64> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for (id, source, enc) in store.vectors_with_sources(&collection.name)? {
        ids.push(id);
        sources.push(source);
        vectors.push(baseline.decode(&enc)?);
    }

    let (queries, query_source) = load_queries(
        &ctx,
        &collection.embed_profile,
        args.queries.as_ref(),
        args.questions.as_ref(),
        args.sample,
        &ids,
        &vectors,
    )
    .await?;

    let mut names: Vec<String> = vec!["f32".to_owned()];
    names.extend(args.codecs.iter().filter(|c| c.as_str() != "f32").cloned());

    let mut reports: Vec<RecallReport> = Vec::new();
    for name in &names {
        let mut codec = codec_by_name(name, collection.dimension)?;
        // Trained codecs learn their statistics from the collection itself,
        // exactly as `ingest --codec` would.
        let fitted = codec.needs_fit();
        if fitted {
            codec.fit(&vectors)?;
        }
        let report = recall::evaluate(
            &ids,
            &vectors,
            &queries,
            codec.as_ref(),
            collection.metric,
            args.k,
            Some(&sources),
        )?;
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
                "fitted_on_collection": fitted,
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
    let labelled = reports.first().is_some_and(|r| r.labelled_queries > 0);
    println!(
        "{:<15} {:>9} {:>7} {:>10} {:>10} {:>9} {:>8} {:>7} {:>9} {:>8}{}",
        "codec",
        "recall@k",
        "top1",
        "mean err",
        "max err",
        "rank noise",
        "bytes",
        "ratio",
        "encode ms",
        "scan ms",
        if labelled { "   ndcg@k  doc rec" } else { "" }
    );
    for r in &reports {
        println!(
            "{:<15} {:>9.4} {:>7.3} {:>10.5} {:>10.5} {:>9.5} {:>8} {:>6.1}x {:>9} {:>8}{}",
            r.codec,
            r.recall_at_k,
            r.top1_agreement,
            r.mean_abs_score_error,
            r.max_abs_score_error,
            r.rank_noise,
            r.bytes_per_vector,
            r.compression_ratio,
            r.encode_ms,
            r.scan_ms,
            match (r.ndcg_at_k, r.doc_recall_at_k) {
                (Some(n), Some(d)) => format!("  {n:>7.4} {d:>8.4}"),
                _ => String::new(),
            }
        );
    }
    if let Some(r) = reports.first() {
        println!(
            "\nneighbour spacing (exact f32, median over queries): rank k to k+1 {:.5}, rank k to 2k {:.5}; rank noise is the codec's score error std over the exact top-2k, per-query bias removed",
            r.boundary_gap, r.neighbour_band
        );
    }
    if labelled {
        println!(
            "\nndcg@k and doc rec are against the question set's relevance judgments ({} labelled queries); the f32 row is the ceiling the codec is measured against",
            reports[0].labelled_queries
        );
    }
    if !args.no_record {
        eprintln!("\n{} runs recorded (kind: recall)", reports.len());
    }
    Ok(())
}

/// The queries for `bench recall` and `bench rescore`: a question set, a
/// file of query lines, or self-queries sampled from the collection.
async fn load_queries(
    ctx: &Ctx,
    embed_profile: &str,
    queries: Option<&PathBuf>,
    questions: Option<&PathBuf>,
    sample: usize,
    ids: &[i64],
    vectors: &[Vec<f32>],
) -> Result<(Vec<RecallQuery>, String)> {
    Ok(match (queries, questions) {
        (_, Some(path)) => {
            let qs = questions::load(path)?;
            let texts: Vec<String> = qs.iter().map(|q| q.retrieval_text().to_owned()).collect();
            let embedder = hvec_connect::embedder(&ctx.config, embed_profile).await?;
            let vectors = embedder.embed_queries(&texts).await?;
            let labelled = qs.iter().filter(|q| !q.relevant.is_empty()).count();
            let queries: Vec<RecallQuery> = vectors
                .into_iter()
                .zip(qs)
                .map(|(vector, q)| RecallQuery {
                    vector,
                    exclude: None,
                    relevant: q.relevant,
                })
                .collect();
            let n = queries.len();
            (
                queries,
                format!(
                    "questions:{} ({n} questions, {labelled} with judgments)",
                    path.display()
                ),
            )
        }
        (Some(path), None) => {
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
            let embedder = hvec_connect::embedder(&ctx.config, embed_profile).await?;
            let qs = embedder
                .embed_queries(&lines)
                .await?
                .into_iter()
                .map(|vector| RecallQuery {
                    vector,
                    exclude: None,
                    relevant: Vec::new(),
                })
                .collect();
            (qs, format!("file:{}", path.display()))
        }
        (None, None) => (
            recall::self_queries(ids, vectors, sample),
            format!("self-queries:{}", sample.min(vectors.len())),
        ),
    })
}

async fn rescore_cmd(config: &Option<PathBuf>, args: RescoreArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let collection = store.require_collection(&args.collection)?;
    if collection.codec != "f32" {
        bail!(
            "`bench rescore` needs an f32 collection as ground truth; `{}` is stored as {}",
            collection.name,
            collection.codec
        );
    }
    if collection.chunk_count == 0 {
        bail!("collection `{}` is empty", collection.name);
    }
    let baseline = F32Codec::new(collection.dimension);
    let mut ids: Vec<i64> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for (id, source, enc) in store.vectors_with_sources(&collection.name)? {
        ids.push(id);
        sources.push(source);
        vectors.push(baseline.decode(&enc)?);
    }
    let (queries, query_source) = load_queries(
        &ctx,
        &collection.embed_profile,
        args.queries.as_ref(),
        args.questions.as_ref(),
        args.sample,
        &ids,
        &vectors,
    )
    .await?;

    let fitted = |name: &str| -> Result<Box<dyn Codec>> {
        let mut codec = codec_by_name(name, collection.dimension)?;
        if codec.needs_fit() {
            codec.fit(&vectors)?;
        }
        Ok(codec)
    };
    let seconds: Vec<Box<dyn Codec>> = args.rescore.iter().map(|n| fitted(n)).collect::<Result<_>>()?;
    let second_refs: Vec<&dyn Codec> = seconds.iter().map(AsRef::as_ref).collect();

    // Calibration for the adaptive policy: self-queries only, so the test
    // queries' exact scores never inform their own depth.
    let calibration = recall::self_queries(&ids, &vectors, args.calibration_sample);

    // Margin grid for per-query traces: fine enough to trace the adaptive
    // policy's whole recall-against-depth curve offline.
    let trace_margins: Vec<f64> = (1..=24).map(|i| f64::from(i) * 0.25).collect();
    let mut trace_out = match &args.trace {
        Some(path) => Some(std::io::BufWriter::new(
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => None,
    };

    let mut profiles = Vec::new();
    let mut cells = Vec::new();
    for name in &args.codecs {
        let first = fitted(name)?;
        let noise = recall::evaluate(
            &ids,
            &vectors,
            &calibration,
            first.as_ref(),
            collection.metric,
            args.k,
            None,
        )?
        .rank_noise;
        let plan = rescore::Plan {
            seconds: &second_refs,
            depths: &args.depths,
            margins: &args.margins,
            noise,
            max_depth: args.max_depth,
            trace_margins: if args.trace.is_some() { &trace_margins } else { &[] },
        };
        let rescore::Evaluation {
            profile,
            reports,
            traces,
        } = rescore::evaluate(
            &ids,
            &vectors,
            &queries,
            first.as_ref(),
            collection.metric,
            args.k,
            &plan,
            Some(&sources),
        )?;
        if !args.no_record {
            for r in &reports {
                let mut row = RunRow::new("rescore");
                row.collection = Some(collection.name.clone());
                row.embed_profile = Some(collection.embed_profile.clone());
                row.embed_model = Some(collection.embed_model.clone());
                row.codec = Some(match r.policy {
                    rescore::Policy::None => r.first.clone(),
                    rescore::Policy::Fixed { depth } => format!("{}>{}@{depth}", r.first, r.second),
                    rescore::Policy::Margin { sigmas } => format!("{}>{}@{sigmas}σ", r.first, r.second),
                });
                row.metric = Some(collection.metric.name().to_owned());
                row.metadata = serde_json::json!({
                    "dimension": collection.dimension,
                    "queries": query_source,
                    "k": args.k,
                    "first_stage": serde_json::to_value(&profile)?,
                    "calibrated_noise": noise,
                });
                row.metrics = serde_json::to_value(r)?;
                store.insert_run(&row)?;
            }
        }
        if let Some(out) = trace_out.as_mut() {
            use std::io::Write as _;
            let line = serde_json::json!({
                "collection": collection.name,
                "first": profile.first,
                "first_bytes": profile.first_bytes,
                "vectors": vectors.len(),
                "k": args.k,
                "calibrated_noise": noise,
                "trace_margins": trace_margins,
                "ranks": traces.iter().map(|t| &t.ranks).collect::<Vec<_>>(),
                "band": traces.iter().map(|t| t.band).collect::<Vec<_>>(),
                "band_hat": traces.iter().map(|t| t.band_hat).collect::<Vec<_>>(),
                "margin_depths": traces.iter().map(|t| &t.margin_depths).collect::<Vec<_>>(),
            });
            writeln!(out, "{line}")?;
        }
        profiles.push(profile);
        cells.extend(reports);
    }

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "profiles": profiles, "cells": cells }))?
        );
        return Ok(());
    }
    let f32_scan = vectors.len() * baseline.encoded_size();
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
    for p in &profiles {
        println!();
        let depths: Vec<String> = p
            .depth_for_recall
            .iter()
            .map(|(t, d)| format!("{:.0}%: {d}", t * 100.0))
            .collect();
        println!(
            "first stage {} ({} B): recall@k {:.4}, noise/band {:.3}; depth an exact re-score needs for recall {}; median query complete at {}",
            p.first,
            p.first_bytes,
            p.recall_at_k,
            if p.neighbour_band > 0.0 {
                p.rank_noise / p.neighbour_band
            } else {
                f64::NAN
            },
            depths.join(", "),
            p.median_full_depth
        );
        println!(
            "  {:<8} {:<8} {:>7} {:>6} {:>9} {:>7} {:>7} {:>8} {:>8} {:>10} {:>8}",
            "rescore", "policy", "depth", "p90", "recall@k", "p10", "top1", "ndcg@k", "doc rec", "KB/query", "vs f32"
        );
        for r in cells.iter().filter(|r| r.first == p.first) {
            let policy = match r.policy {
                rescore::Policy::None => "-".to_owned(),
                rescore::Policy::Fixed { .. } => "fixed".to_owned(),
                rescore::Policy::Margin { sigmas } => format!("{sigmas}σ"),
            };
            println!(
                "  {:<8} {:<8} {:>7.1} {:>6} {:>9.4} {:>7.3} {:>7.3} {:>8} {:>8} {:>10.1} {:>7.3}x",
                r.second,
                policy,
                r.mean_depth,
                r.p90_depth,
                r.recall_at_k,
                r.p10_recall_at_k,
                r.top1_agreement,
                r.ndcg_at_k.map_or("-".into(), |v| format!("{v:.4}")),
                r.doc_recall_at_k.map_or("-".into(), |v| format!("{v:.4}")),
                r.bytes_read_per_query / 1024.0,
                r.bytes_read_per_query / f32_scan as f64,
            );
        }
    }
    println!(
        "\ndepth is mean candidates re-scored per query, p90 its 90th percentile, p10 the 10th-percentile query's recall; Nσ re-scores every candidate within N calibrated noise units of the k-th first-stage score; KB/query is the first-stage scan plus the re-scored candidates; vs f32 compares it with an exact f32 scan of every vector"
    );
    if !args.no_record {
        eprintln!("\n{} runs recorded (kind: rescore)", cells.len());
    }
    Ok(())
}
