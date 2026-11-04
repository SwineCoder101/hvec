//! `hvec bench scan`: how long the scan itself takes, and where the time goes.
//!
//! Three questions, one table each:
//! 1. Storage path. The same f32 search through every way bytes can reach the
//!    scorer: the original row scan (every row and its text copied out of
//!    SQLite on every query), SQLite rows with borrowed blobs (with SQLite's
//!    memory map off and on), and the cached contiguous buffer. Also what it
//!    costs to build the buffer from SQLite, or from a flat code file.
//! 2. Codec. Each codec scanned in place over its own buffer at each thread
//!    count: ms per query, GB/s of codes read, recall@k against exact f32.
//! 3. The ceiling. A plain sequential read of a large buffer at the same
//!    thread counts, so each scan's GB/s can be read as a fraction of what
//!    the machine's memory can deliver.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use hvec_core::codecs::F32Codec;
use hvec_core::{Codec, codec_by_name};
use hvec_store::CodeBuffer;
use serde::Serialize;

use crate::cli::ScanArgs;
use crate::context::Ctx;

#[derive(Debug, Serialize)]
struct PathRow {
    path: String,
    ms_per_query: f64,
    note: String,
}

#[derive(Debug, Serialize)]
struct LoadRow {
    source: String,
    ms: f64,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct CodecRow {
    codec: String,
    bytes_per_vector: usize,
    threads: usize,
    ms_per_query: f64,
    gb_per_s: f64,
    recall_at_k: f64,
}

#[derive(Debug, Serialize)]
struct BandwidthRow {
    threads: usize,
    gb_per_s: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    collection: String,
    vectors: usize,
    dimension: usize,
    queries: usize,
    k: usize,
    repeat: usize,
    paths: Vec<PathRow>,
    loads: Vec<LoadRow>,
    codecs: Vec<CodecRow>,
    bandwidth: Vec<BandwidthRow>,
}

/// Median over `repeat` passes of the mean ms per query.
fn time_queries(repeat: usize, queries: &[Vec<f32>], mut f: impl FnMut(&[f32]) -> Result<()>) -> Result<f64> {
    // One untimed pass warms caches and builds anything lazily cached.
    for q in queries.iter().take(1) {
        f(q)?;
    }
    let mut passes = Vec::with_capacity(repeat);
    for _ in 0..repeat.max(1) {
        let t = Instant::now();
        for q in queries {
            f(q)?;
        }
        passes.push(t.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64);
    }
    passes.sort_by(f64::total_cmp);
    Ok(passes[passes.len() / 2])
}

/// Sequential read bandwidth: sum a large buffer of u64, split over threads.
fn bandwidth(mib: usize, threads: usize, repeat: usize) -> f64 {
    let words = mib * 1024 * 1024 / 8;
    let data: Vec<u64> = (0..words as u64).collect();
    let per = words.div_ceil(threads.max(1));
    let mut best = f64::MAX;
    for _ in 0..repeat.max(1) + 1 {
        let t = Instant::now();
        let total: u64 = std::thread::scope(|s| {
            let handles: Vec<_> = data
                .chunks(per)
                .map(|c| s.spawn(move || c.iter().fold(0u64, |a, &x| a.wrapping_add(x))))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or(0))
                .fold(0, u64::wrapping_add)
        });
        std::hint::black_box(total);
        best = best.min(t.elapsed().as_secs_f64());
    }
    (words * 8) as f64 / best / 1e9
}

pub fn run(config: &Option<PathBuf>, args: &ScanArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let collection = store.require_collection(&args.collection)?;
    if collection.codec != "f32" {
        bail!(
            "`bench scan` needs an f32 collection; `{}` is stored as {}",
            collection.name,
            collection.codec
        );
    }
    let dim = collection.dimension;
    let metric = collection.metric;
    let f32c = F32Codec::new(dim);
    let k = args.k;

    // The collection as one buffer; this is also the exact ground truth.
    let t = Instant::now();
    let exact_buf = store.load_codes(&collection.name)?;
    let load_sqlite_ms = t.elapsed().as_secs_f64() * 1000.0;
    let n = exact_buf.len();
    if n == 0 {
        bail!("collection `{}` is empty", collection.name);
    }
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| f32c.decode(exact_buf.code(i)))
        .collect::<Result<_, _>>()?;
    let step = n as f64 / args.sample.clamp(1, n) as f64;
    let picked: Vec<usize> = (0..args.sample.clamp(1, n))
        .map(|i| ((i as f64 * step) as usize).min(n - 1))
        .collect();
    let queries: Vec<Vec<f32>> = picked.iter().map(|&i| vectors[i].clone()).collect();
    // Self-queries: each query is a stored vector, which is dropped from both
    // rankings before recall is counted, as `bench recall` does.
    let top_excluding = |buf: &CodeBuffer, codec: &dyn Codec, q: &[f32], skip: usize| -> Result<Vec<usize>> {
        Ok(buf
            .search(q, k + 1, codec, metric)?
            .iter()
            .map(|s| s.index)
            .filter(|&i| i != skip)
            .take(k)
            .collect())
    };
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .zip(&picked)
        .map(|(q, &i)| top_excluding(&exact_buf, &f32c, q, i))
        .collect::<Result<_>>()?;

    // 1. Storage paths, f32 throughout.
    let mut paths = Vec::new();
    if !args.skip_sqlite {
        let ms = time_queries(args.repeat, &queries, |q| {
            std::hint::black_box(store.search_by_rows(&collection.name, q, k, &f32c, metric)?);
            Ok(())
        })?;
        paths.push(PathRow {
            path: "sqlite rows, copied".into(),
            ms_per_query: ms,
            note: "the original search: every row and its text copied out on every query".into(),
        });
        for (label, mmap) in [
            ("sqlite rows, borrowed", 0u64),
            ("sqlite rows, borrowed, mmap", 1 << 36),
        ] {
            store.set_mmap_size(mmap)?;
            let ms = time_queries(args.repeat, &queries, |q| {
                std::hint::black_box(store.search_borrowed(&collection.name, q, k, &f32c, metric)?);
                Ok(())
            })?;
            paths.push(PathRow {
                path: label.into(),
                ms_per_query: ms,
                note: if mmap == 0 {
                    "vector column only, each blob scored in place in SQLite's page buffer".into()
                } else {
                    "as above, with SQLite reading the file through its memory map".into()
                },
            });
        }
        store.set_mmap_size(0)?;
    }
    let ms = time_queries(args.repeat, &queries, |q| {
        std::hint::black_box(store.search(&collection.name, q, k, &f32c, metric)?);
        Ok(())
    })?;
    paths.push(PathRow {
        path: "cached buffer (Store::search)".into(),
        ms_per_query: ms,
        note: "codes scanned in place in one allocation; text read for the hits only".into(),
    });

    // Building the buffer: from SQLite, and from a flat code file.
    let mut loads = vec![LoadRow {
        source: "sqlite (borrowed blobs, one copy)".into(),
        ms: load_sqlite_ms,
        bytes: exact_buf.bytes().len(),
    }];
    let file = std::env::temp_dir().join(format!("hvec-scan-{}-{}.codes", std::process::id(), collection.name));
    exact_buf.write_file(&file)?;
    let t = Instant::now();
    let back = CodeBuffer::read_file(&file);
    let _ = std::fs::remove_file(&file);
    let back = back?;
    loads.push(LoadRow {
        source: "flat code file (one read)".into(),
        ms: t.elapsed().as_secs_f64() * 1000.0,
        bytes: back.bytes().len(),
    });
    drop(back);

    // 2. Codecs, each scanned in place over its own buffer.
    let ids = exact_buf.ids().to_vec();
    let sources = exact_buf.sources().to_vec();
    let mut codecs = Vec::new();
    for name in &args.codecs {
        let mut codec: Box<dyn Codec> = codec_by_name(name, dim)?;
        if codec.needs_fit() {
            codec.fit(&vectors)?;
        }
        let buf = CodeBuffer::encode(codec.as_ref(), &ids, &sources, &vectors)?;
        let mut hits = 0usize;
        for ((q, t), &i) in queries.iter().zip(&truth).zip(&picked) {
            let got = top_excluding(&buf, codec.as_ref(), q, i)?;
            hits += got.iter().filter(|g| t.contains(g)).count();
        }
        let recall = hits as f64 / (queries.len() * k) as f64;
        for &threads in &args.threads {
            let ms = time_queries(args.repeat, &queries, |q| {
                std::hint::black_box(buf.search_threads(q, k, codec.as_ref(), metric, threads)?);
                Ok(())
            })?;
            codecs.push(CodecRow {
                codec: codec.name().to_owned(),
                bytes_per_vector: codec.encoded_size(),
                threads,
                ms_per_query: ms,
                gb_per_s: buf.bytes().len() as f64 / (ms / 1000.0) / 1e9,
                recall_at_k: recall,
            });
        }
    }

    // 3. The memory-bandwidth ceiling.
    let bandwidth = args
        .threads
        .iter()
        .map(|&t| BandwidthRow {
            threads: t,
            gb_per_s: bandwidth(args.bandwidth_mib, t, args.repeat),
        })
        .collect();

    let report = Report {
        collection: collection.name.clone(),
        vectors: n,
        dimension: dim,
        queries: queries.len(),
        k,
        repeat: args.repeat,
        paths,
        loads,
        codecs,
        bandwidth,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "collection `{}` · {} vectors · {} dims · {} queries · k={} · median of {} passes",
        report.collection, report.vectors, report.dimension, report.queries, k, report.repeat
    );
    println!("\nstorage path (f32)                 ms/query");
    for p in &report.paths {
        println!("  {:<32} {:>8.2}   {}", p.path, p.ms_per_query, p.note);
    }
    println!("\nbuilding the buffer                ms        MB");
    for l in &report.loads {
        println!("  {:<32} {:>8.1} {:>8.1}", l.source, l.ms, l.bytes as f64 / 1e6);
    }
    println!("\ncodec, in place       bytes  threads  ms/query     GB/s  % of bandwidth  recall@{k}");
    for c in &report.codecs {
        let bw = report
            .bandwidth
            .iter()
            .find(|b| b.threads == c.threads)
            .map_or(f64::NAN, |b| b.gb_per_s);
        println!(
            "  {:<18} {:>6} {:>8} {:>9.2} {:>8.2} {:>14.1}% {:>9.4}",
            c.codec,
            c.bytes_per_vector,
            c.threads,
            c.ms_per_query,
            c.gb_per_s,
            100.0 * c.gb_per_s / bw,
            c.recall_at_k
        );
    }
    println!("\nmemory bandwidth (sequential u64 read of {} MiB)", args.bandwidth_mib);
    for b in &report.bandwidth {
        println!("  {:>2} thread(s): {:>6.1} GB/s", b.threads, b.gb_per_s);
    }
    Ok(())
}
