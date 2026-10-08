use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use hvec_bench::{ChunkOptions, chunk_text};
use hvec_core::codec_by_name;
use hvec_store::{NewChunk, NewCollection};
use tracing::info;

use crate::cli::IngestArgs;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, args: IngestArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let metric = ctx.metric()?;
    let embed_name = ctx.embed_profile_name(args.embedder.as_deref()).to_owned();

    let files = collect_files(&args.paths, &args.ext)?;
    if files.is_empty() {
        bail!("no files matched (extensions: {})", args.ext.join(","));
    }
    info!(files = files.len(), "collected input files");

    let embedder = hvec_connect::embedder(&ctx.config, &embed_name).await?;
    let codec = codec_by_name(&args.codec, embedder.dimension())?;

    let mut store = ctx.open_store()?;
    let collection = store.ensure_collection(&NewCollection {
        name: &args.collection,
        embed_profile: &embed_name,
        embed_model: embedder.model_id(),
        dimension: embedder.dimension(),
        codec: codec.name(),
        metric,
    })?;
    if collection.codec != codec.name() {
        bail!(
            "collection `{}` stores {} vectors; re-ingest into a new collection to use {}",
            collection.name,
            collection.codec,
            codec.name()
        );
    }

    let opts = ChunkOptions {
        words: args.chunk_words,
        overlap: args.overlap_words,
    };
    let started = Instant::now();
    let mut total_chunks = 0usize;

    for file in &files {
        let text = tokio::fs::read_to_string(file)
            .await
            .with_context(|| format!("reading {}", file.display()))?;
        let pieces = chunk_text(&text, opts);
        if pieces.is_empty() {
            continue;
        }
        let source = file.to_string_lossy().into_owned();
        let mut pending: Vec<NewChunk> = Vec::with_capacity(pieces.len());

        for (batch_idx, batch) in pieces.chunks(args.batch.max(1)).enumerate() {
            let vectors = embedder.embed(batch).await?;
            for (i, (text, vector)) in batch.iter().zip(vectors).enumerate() {
                let ordinal = (batch_idx * args.batch.max(1) + i) as u32;
                pending.push(NewChunk {
                    source: source.clone(),
                    ordinal,
                    text: text.clone(),
                    vector: codec.encode(&vector)?,
                });
            }
        }
        total_chunks += store.insert_chunks(&args.collection, &pending)?;
        info!(file = %file.display(), chunks = pending.len(), "ingested");
    }

    let secs = started.elapsed().as_secs_f64();
    let after = store.require_collection(&args.collection)?;
    println!(
        "ingested {} chunks from {} files into `{}` in {:.1}s ({} dims, codec {}, {} chunks total)",
        total_chunks,
        files.len(),
        args.collection,
        secs,
        after.dimension,
        after.codec,
        after.chunk_count
    );
    Ok(())
}

fn collect_files(paths: &[PathBuf], exts: &[String]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_file() {
            out.push(p.clone());
        } else if p.is_dir() {
            for entry in walkdir::WalkDir::new(p)
                .follow_links(false)
                .into_iter()
                .filter_map(Result::ok)
            {
                if entry.file_type().is_file() && has_ext(entry.path(), exts) {
                    out.push(entry.into_path());
                }
            }
        } else {
            bail!("{} does not exist", p.display());
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn has_ext(path: &Path, exts: &[String]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
}
