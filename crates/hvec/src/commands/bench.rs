use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use hvec_bench::recall::{self, RecallQuery, RecallReport};
use hvec_core::codecs::F32Codec;
use hvec_core::{Codec, codec_by_name};
use hvec_store::RunRow;

use crate::cli::{BenchCommand, RecallArgs};
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, cmd: BenchCommand) -> Result<()> {
    match cmd {
        BenchCommand::Recall(args) => recall_cmd(config, args).await,
    }
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
