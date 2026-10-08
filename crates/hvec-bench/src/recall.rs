//! Retrieval-only evaluation of a codec against the f32 baseline.
//!
//! No chat model is involved, so this is free and fast. For each query the
//! exact top-k under f32 is the ground truth; the codec's top-k is compared
//! to it. Score error is measured over every (query, vector) pair.

use std::time::Instant;

use hvec_core::{Codec, CodecError, Metric, distance};
use serde::{Deserialize, Serialize};

use crate::metrics::recall_at_k;

/// A query vector, optionally excluding one stored id (itself) from ranking.
#[derive(Debug, Clone)]
pub struct RecallQuery {
    pub vector: Vec<f32>,
    pub exclude: Option<i64>,
}

/// What one codec did on one collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallReport {
    pub codec: String,
    pub metric: Metric,
    pub k: usize,
    pub vectors: usize,
    pub queries: usize,
    /// Mean over queries of |codec top-k ∩ f32 top-k| / k.
    pub recall_at_k: f64,
    /// Fraction of queries whose codec top-1 equals the f32 top-1.
    pub top1_agreement: f64,
    pub mean_abs_score_error: f64,
    pub max_abs_score_error: f64,
    pub bytes_per_vector: usize,
    pub compression_ratio: f32,
    pub encode_ms: u64,
    /// Time to score every query against every vector through the codec.
    pub scan_ms: u64,
    /// The same scan with exact f32 arithmetic.
    pub baseline_scan_ms: u64,
}

fn top_k(scores: &[(i64, f32)], k: usize, metric: Metric) -> Vec<i64> {
    let mut v: Vec<(i64, f32)> = scores.to_vec();
    if metric.higher_is_better() {
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    } else {
        v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    }
    v.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Evaluate `codec` on `(id, f32 vector)` pairs with the given queries.
pub fn evaluate(
    ids: &[i64],
    vectors: &[Vec<f32>],
    queries: &[RecallQuery],
    codec: &dyn Codec,
    metric: Metric,
    k: usize,
) -> Result<RecallReport, CodecError> {
    assert_eq!(ids.len(), vectors.len(), "ids and vectors must align");

    let t = Instant::now();
    let encoded = vectors.iter().map(|v| codec.encode(v)).collect::<Result<Vec<_>, _>>()?;
    let encode_ms = t.elapsed().as_millis() as u64;

    let mut recall_sum = 0f64;
    let mut top1_hits = 0usize;
    let mut err_sum = 0f64;
    let mut err_max = 0f64;
    let mut pairs = 0usize;
    let mut scan_ms = 0u64;
    let mut baseline_scan_ms = 0u64;

    for q in queries {
        let t = Instant::now();
        let exact: Vec<(i64, f32)> = ids
            .iter()
            .zip(vectors)
            .filter(|(id, _)| Some(**id) != q.exclude)
            .map(|(id, v)| (*id, distance::score(&q.vector, v, metric)))
            .collect();
        baseline_scan_ms += t.elapsed().as_millis() as u64;

        let t = Instant::now();
        let approx: Vec<(i64, f32)> = ids
            .iter()
            .zip(&encoded)
            .filter(|(id, _)| Some(**id) != q.exclude)
            .map(|(id, e)| codec.score(&q.vector, e, metric).map(|s| (*id, s)))
            .collect::<Result<_, _>>()?;
        scan_ms += t.elapsed().as_millis() as u64;

        for ((_, a), (_, b)) in exact.iter().zip(&approx) {
            let e = f64::from((a - b).abs());
            err_sum += e;
            err_max = err_max.max(e);
            pairs += 1;
        }
        let truth = top_k(&exact, k, metric);
        let got = top_k(&approx, k, metric);
        recall_sum += recall_at_k(&truth, &got, k);
        if !truth.is_empty() && truth.first() == got.first() {
            top1_hits += 1;
        }
    }

    let nq = queries.len().max(1) as f64;
    Ok(RecallReport {
        codec: codec.name().to_owned(),
        metric,
        k,
        vectors: vectors.len(),
        queries: queries.len(),
        recall_at_k: recall_sum / nq,
        top1_agreement: top1_hits as f64 / nq,
        mean_abs_score_error: if pairs == 0 { 0.0 } else { err_sum / pairs as f64 },
        max_abs_score_error: err_max,
        bytes_per_vector: codec.encoded_size(),
        compression_ratio: codec.compression_ratio(),
        encode_ms,
        scan_ms,
        baseline_scan_ms,
    })
}

/// Pick `n` stored vectors as self-queries, spread evenly through the set,
/// each excluding itself from its own ranking.
#[must_use]
pub fn self_queries(ids: &[i64], vectors: &[Vec<f32>], n: usize) -> Vec<RecallQuery> {
    if vectors.is_empty() || n == 0 {
        return Vec::new();
    }
    let n = n.min(vectors.len());
    let step = vectors.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let idx = ((i as f64 * step) as usize).min(vectors.len() - 1);
            RecallQuery {
                vector: vectors[idx].clone(),
                exclude: Some(ids[idx]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hvec_core::codecs::{BinaryCodec, F32Codec, Int8Codec};

    struct XorShift(u64);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// 8 clusters × 25 members in 64 dims, unit length.
    fn clustered() -> (Vec<i64>, Vec<Vec<f32>>) {
        let mut rng = XorShift(2024);
        let dim = 64;
        let centres: Vec<Vec<f32>> = (0..8).map(|_| (0..dim).map(|_| rng.next_f32()).collect()).collect();
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for (c, centre) in centres.iter().enumerate() {
            for m in 0..25 {
                let mut v: Vec<f32> = centre.iter().map(|x| x + 0.15 * rng.next_f32()).collect();
                distance::normalize(&mut v);
                ids.push((c * 100 + m) as i64);
                vecs.push(v);
            }
        }
        (ids, vecs)
    }

    #[test]
    fn f32_codec_is_perfect() {
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 20);
        let r = evaluate(&ids, &vecs, &queries, &F32Codec::new(64), Metric::Cosine, 10).unwrap();
        assert_eq!(r.recall_at_k, 1.0);
        assert_eq!(r.top1_agreement, 1.0);
        assert_eq!(r.mean_abs_score_error, 0.0);
        assert_eq!(r.queries, 20);
        assert_eq!(r.vectors, 200);
    }

    #[test]
    fn int8_is_near_perfect_and_binary_is_decent_on_clusters() {
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 40);
        let i8 = evaluate(&ids, &vecs, &queries, &Int8Codec::new(64), Metric::Cosine, 10).unwrap();
        assert!(i8.recall_at_k >= 0.95, "int8 recall {}", i8.recall_at_k);
        assert!(i8.mean_abs_score_error < 0.01, "int8 err {}", i8.mean_abs_score_error);
        let bin = evaluate(&ids, &vecs, &queries, &BinaryCodec::new(64), Metric::Cosine, 10).unwrap();
        assert!(bin.recall_at_k >= 0.5, "binary recall {}", bin.recall_at_k);
        assert!(bin.recall_at_k <= i8.recall_at_k);
        assert_eq!(bin.bytes_per_vector, 8);
        assert_eq!(bin.compression_ratio, 32.0);
    }

    #[test]
    fn self_queries_exclude_themselves() {
        let (ids, vecs) = clustered();
        let q = self_queries(&ids, &vecs, 5);
        assert_eq!(q.len(), 5);
        assert!(q.iter().all(|x| x.exclude.is_some()));
        let r = evaluate(&ids, &vecs, &q, &F32Codec::new(64), Metric::L2, 3).unwrap();
        assert_eq!(r.recall_at_k, 1.0);
        assert!(self_queries(&ids, &vecs, 0).is_empty());
        assert_eq!(self_queries(&ids, &vecs, 10_000).len(), vecs.len());
    }

    #[test]
    fn top_k_breaks_ties_by_id() {
        let s = vec![(5, 1.0), (2, 1.0), (9, 0.5)];
        assert_eq!(top_k(&s, 2, Metric::Dot), vec![2, 5]);
        assert_eq!(top_k(&s, 1, Metric::L2), vec![9]);
    }
}
