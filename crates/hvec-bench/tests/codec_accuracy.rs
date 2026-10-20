//! A small, readable demonstration of how hvec measures a codec.
//!
//! A codec stores a vector compressed and scores a query against the stored
//! bytes directly; nothing is decompressed on the search path. Accuracy is
//! then the gap between that compressed-domain score and the exact score on
//! the original f32 vector, measured at two levels:
//!
//! 1. Score error, per (query, stored vector) pair:
//!    `|codec.score(q, bytes) - cosine(q, v)|`, reported as a mean and a max.
//! 2. Ranking: the codec's top-k against the exact f32 top-k for each query.
//!    recall@k is the overlap divided by k; top-1 agreement is how often the
//!    best hit is the same.
//!
//! `hvec bench recall` prints exactly these numbers for a real collection, and
//! with a labelled question set adds nDCG@k against human relevance judgments.
//! Both levels are computed here by `hvec_bench::recall::evaluate`, the same
//! function the CLI calls.
//!
//! Run `cargo test -p hvec-bench --test codec_accuracy -- --nocapture` to see
//! the table.

use hvec_bench::metrics::recall_at_k;
use hvec_bench::recall::{self, RecallReport};
use hvec_core::{Codec, Metric, codec_by_name, distance};

const DIM: usize = 64;
const CORPUS: usize = 400;
const K: usize = 10;

/// Deterministic generator, so the numbers below are the same on every run.
struct XorShift(u64);

impl XorShift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A corpus shaped like real embeddings: a dozen topics, each vector a noisy
/// copy of its topic's centre, unit length, and every vector sharing one
/// common component. Models such as BGE have that shared component, and it is
/// what `binary-centred` exists to remove; on a zero-mean corpus centring
/// changes nothing.
fn corpus() -> Vec<Vec<f32>> {
    let mut rng = XorShift(0x5EED_1234_ABCD);
    let shared: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let centres: Vec<Vec<f32>> = (0..12).map(|_| (0..DIM).map(|_| rng.next_f32()).collect()).collect();
    (0..CORPUS)
        .map(|i| {
            let mut v: Vec<f32> = centres[i % centres.len()]
                .iter()
                .zip(&shared)
                .map(|(c, s)| 1.5 * s + c + 0.4 * rng.next_f32())
                .collect();
            distance::normalize(&mut v);
            v
        })
        .collect()
}

/// Every codec family, fitted on the corpus where the codec needs it, exactly
/// as `hvec ingest --codec` and `hvec bench recall` do.
fn codecs(vectors: &[Vec<f32>]) -> Vec<Box<dyn Codec>> {
    ["f32", "int8", "mrl32+int8", "pq8", "binary-centred", "binary"]
        .iter()
        .map(|name| {
            let mut c = codec_by_name(name, DIM).expect(name);
            if c.needs_fit() {
                c.fit(vectors).expect(name);
            }
            c
        })
        .collect()
}

#[test]
fn a_score_computed_on_the_bytes_matches_the_exact_score() {
    // One stored vector, one query. The codec never sees the f32 vector again
    // after `encode`; `score` works on the bytes it produced.
    let vectors = corpus();
    let stored = &vectors[0];
    let query = &vectors[12]; // same topic as vectors[0]
    let exact = distance::cosine(query, stored);

    println!("\nexact cosine(query, stored) = {exact:.5}\n");
    println!("{:<15} {:>6} {:>12} {:>10}", "codec", "bytes", "score", "error");
    for codec in codecs(&vectors) {
        let bytes = codec.encode(stored).unwrap();
        let on_bytes = codec.score(query, &bytes, Metric::Cosine).unwrap();
        let error = (on_bytes - exact).abs();
        println!(
            "{:<15} {:>6} {:>12.5} {:>10.5}",
            codec.name(),
            bytes.0.len(),
            on_bytes,
            error
        );

        // The homomorphic property: scoring the bytes equals scoring what
        // the bytes decode to. The error above is purely what the
        // compression threw away, not an artefact of the fast path. The
        // Matryoshka codecs compare prefixes: `mrl32` scores the first 32
        // dimensions of the query against the first 32 it kept, which is the
        // Matryoshka convention, so the reference is the cosine of the two
        // prefixes rather than of the zero-padded decode.
        let decoded = codec.decode(&bytes).unwrap();
        let kept = if codec.name().starts_with("mrl32") { 32 } else { DIM };
        let on_decoded = distance::cosine(&query[..kept], &decoded[..kept]);
        assert!(
            (on_bytes - on_decoded).abs() < 1e-4,
            "{}: {on_bytes} on bytes vs {on_decoded} on decoded",
            codec.name()
        );

        match codec.name() {
            "f32" => assert_eq!(error, 0.0),
            // int8 keeps each component within half a quantization step.
            "int8" => assert!(error < 0.01, "int8 error {error}"),
            // One bit per dimension: the sign is kept, the magnitude is gone.
            "binary" | "binary-centred" => assert!(error < 0.5, "{} error {error}", codec.name()),
            _ => {}
        }
    }
}

#[test]
fn ranking_accuracy_against_the_f32_baseline() {
    let vectors = corpus();
    let ids: Vec<i64> = (0..vectors.len() as i64).collect();
    let codecs = codecs(&vectors);

    // --- By hand, for one query and one codec, to show what recall@k is. ---
    let int8 = &codecs[1];
    let query = &vectors[7];
    let rank = |scores: Vec<(i64, f32)>| -> Vec<i64> {
        let mut s = scores;
        s.sort_by(|a, b| b.1.total_cmp(&a.1));
        s.into_iter().skip(1).take(K).map(|(id, _)| id).collect() // skip(1): itself
    };
    let exact_top = rank(
        ids.iter()
            .zip(&vectors)
            .map(|(id, v)| (*id, distance::cosine(query, v)))
            .collect(),
    );
    let int8_top = rank(
        ids.iter()
            .zip(&vectors)
            .map(|(id, v)| {
                (
                    *id,
                    int8.score(query, &int8.encode(v).unwrap(), Metric::Cosine).unwrap(),
                )
            })
            .collect(),
    );
    let r = recall_at_k(&exact_top, &int8_top, K);
    println!("\nquery #7, top-{K} by exact f32: {exact_top:?}");
    println!("query #7, top-{K} by int8 bytes: {int8_top:?}");
    println!("recall@{K} = overlap / {K} = {r:.2}\n");
    assert!(r >= 0.8, "int8 recall on one query {r}");

    // --- The harness's measurement over many queries, one row per codec. ---
    // Self-queries: 100 stored vectors used as queries, each excluding itself.
    let queries = recall::self_queries(&ids, &vectors, 100);
    let reports: Vec<RecallReport> = codecs
        .iter()
        .map(|c| recall::evaluate(&ids, &vectors, &queries, c.as_ref(), Metric::Cosine, K, None).unwrap())
        .collect();

    println!(
        "{:<15} {:>9} {:>7} {:>10} {:>10} {:>6} {:>7}",
        "codec", "recall@k", "top1", "mean err", "max err", "bytes", "ratio"
    );
    for r in &reports {
        println!(
            "{:<15} {:>9.4} {:>7.3} {:>10.5} {:>10.5} {:>6} {:>6.1}x",
            r.codec,
            r.recall_at_k,
            r.top1_agreement,
            r.mean_abs_score_error,
            r.max_abs_score_error,
            r.bytes_per_vector,
            r.compression_ratio
        );
    }
    println!();

    let by_name = |name: &str| reports.iter().find(|r| r.codec == name).expect(name);
    let f32 = by_name("f32");
    let int8 = by_name("int8");
    let binary = by_name("binary");
    let centred = by_name("binary-centred");

    // The baseline is exact by definition.
    assert_eq!(f32.recall_at_k, 1.0);
    assert_eq!(f32.mean_abs_score_error, 0.0);
    assert_eq!(f32.bytes_per_vector, DIM * 4);

    // int8: a quarter of the bytes, the ranking nearly intact.
    assert_eq!(int8.bytes_per_vector, DIM + 8);
    assert!(int8.recall_at_k > 0.95, "int8 recall {}", int8.recall_at_k);
    assert!(
        int8.mean_abs_score_error < 0.005,
        "int8 mean err {}",
        int8.mean_abs_score_error
    );

    // binary: 32x smaller, and the score error shows the cost of one bit per
    // dimension. The corpus shares a common component, so plain sign bits
    // agree across most of it and the Hamming-style score sits far from the
    // exact cosine. Subtracting the corpus mean first does not add
    // information (same 8 bytes) but brings the scores and the ranking back.
    assert_eq!(binary.bytes_per_vector, DIM / 8);
    assert_eq!(centred.bytes_per_vector, DIM / 8);
    assert!(centred.mean_abs_score_error < binary.mean_abs_score_error);
    assert!(centred.recall_at_k > binary.recall_at_k);
    assert!(binary.recall_at_k < int8.recall_at_k);

    // mrl32+int8 keeps half the dimensions. These vectors were not trained
    // Matryoshka-style, so this is a plain lossy projection and the ranking
    // suffers; on a model trained for it the leading half carries most of the
    // information. The harness measures, it does not assume.
    let mrl = by_name("mrl32+int8");
    assert_eq!(mrl.bytes_per_vector, DIM / 2 + 8);
    assert!(mrl.recall_at_k < int8.recall_at_k);

    // Every row is measured on the same queries against the same truth.
    assert!(
        reports
            .iter()
            .all(|r| r.queries == 100 && r.vectors == CORPUS && r.k == K)
    );
}
