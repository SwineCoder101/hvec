use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use hvec_bench::{ChunkOptions, chunk_text};
use hvec_connect::Embedder;
use hvec_core::{Metric, codec_by_name};
use hvec_store::{NewChunk, NewCollection};
use tracing::info;

use crate::cli::IngestArgs;
use crate::context::Ctx;

/// Below this many vectors a fitted mean is unlikely to represent the corpus.
const SMALL_FIT_SAMPLE: usize = 100;

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
    let dimension = embedder.dimension();
    let mut codec = codec_by_name(&args.codec, dimension)?;

    let codec_name = codec.name();

    let mut store = ctx.open_store()?;
    fn describe<'a>(
        args: &'a IngestArgs,
        embed_name: &'a str,
        embedder: &'a dyn Embedder,
        codec_name: &'a str,
        metric: Metric,
        codec_params: &'a [u8],
    ) -> NewCollection<'a> {
        NewCollection {
            name: &args.collection,
            embed_profile: embed_name,
            embed_model: embedder.model_id(),
            dimension: embedder.dimension(),
            codec: codec_name,
            metric,
            codec_params,
        }
    }
    if let Some(existing) = store.get_collection(&args.collection)? {
        if existing.codec != codec.name() {
            bail!(
                "collection `{}` stores {} vectors; re-ingest into a new collection to use {}",
                existing.name,
                existing.codec,
                codec.name()
            );
        }
        // Refuses a different embedding model before any work is done.
        store.ensure_collection(&describe(
            &args,
            &embed_name,
            embedder.as_ref(),
            codec_name,
            metric,
            &[],
        ))?;
        // A trained codec keeps the parameters it was created with.
        codec = existing.codec()?;
    }

    let opts = ChunkOptions {
        words: args.chunk_words,
        overlap: args.overlap_words,
    };
    let batch = args.batch.max(1);
    let started = Instant::now();
    let mut total_chunks = 0usize;

    if codec.needs_fit() {
        // A trained codec needs corpus statistics before it can encode
        // anything, so embed everything first, fit, then encode and store.
        let mut pending: Vec<(String, u32, String)> = Vec::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for file in &files {
            let pieces = read_chunks(file, opts).await?;
            if pieces.is_empty() {
                continue;
            }
            let source = file.to_string_lossy().into_owned();
            vectors.extend(embed_all(embedder.as_ref(), &pieces, batch).await?);
            pending.extend(
                pieces
                    .into_iter()
                    .enumerate()
                    .map(|(i, text)| (source.clone(), i as u32, text)),
            );
            info!(file = %file.display(), "embedded");
        }
        if vectors.is_empty() {
            bail!("no text to ingest");
        }
        codec
            .fit(&vectors)
            .with_context(|| format!("fitting codec {} on {} vectors", codec.name(), vectors.len()))?;
        info!(codec = codec.name(), sample = vectors.len(), "fitted codec");
        eprintln!(
            "fitted {} on {} vectors; these statistics are frozen for `{}`, so later ingests reuse them",
            codec.name(),
            vectors.len(),
            args.collection
        );
        if vectors.len() < SMALL_FIT_SAMPLE {
            eprintln!(
                "warning: {} vectors is a small sample for a trained codec; ingest a representative corpus in the first call",
                vectors.len()
            );
        }
        store.ensure_collection(&describe(
            &args,
            &embed_name,
            embedder.as_ref(),
            codec_name,
            metric,
            &codec.params(),
        ))?;
        let chunks = pending
            .into_iter()
            .zip(&vectors)
            .map(|((source, ordinal, text), vector)| {
                Ok(NewChunk {
                    source,
                    ordinal,
                    text,
                    vector: codec.encode(vector)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        total_chunks += store.insert_chunks(&args.collection, &chunks)?;
    } else {
        store.ensure_collection(&describe(
            &args,
            &embed_name,
            embedder.as_ref(),
            codec_name,
            metric,
            &codec.params(),
        ))?;
        for file in &files {
            let pieces = read_chunks(file, opts).await?;
            if pieces.is_empty() {
                continue;
            }
            let source = file.to_string_lossy().into_owned();
            let vectors = embed_all(embedder.as_ref(), &pieces, batch).await?;
            let chunks = pieces
                .iter()
                .zip(&vectors)
                .enumerate()
                .map(|(i, (text, vector))| {
                    Ok(NewChunk {
                        source: source.clone(),
                        ordinal: i as u32,
                        text: text.clone(),
                        vector: codec.encode(vector)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            total_chunks += store.insert_chunks(&args.collection, &chunks)?;
            info!(file = %file.display(), chunks = chunks.len(), "ingested");
        }
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

async fn read_chunks(file: &Path, opts: ChunkOptions) -> Result<Vec<String>> {
    let text = tokio::fs::read_to_string(file)
        .await
        .with_context(|| format!("reading {}", file.display()))?;
    Ok(chunk_text(&text, opts))
}

async fn embed_all(embedder: &dyn Embedder, pieces: &[String], batch: usize) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(pieces.len());
    for group in pieces.chunks(batch) {
        out.extend(embedder.embed(group).await?);
    }
    Ok(out)
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
