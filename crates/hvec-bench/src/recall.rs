//! Retrieval-only evaluation of a codec against the f32 baseline.
//!
//! No chat model is involved, so this is free and fast. For each query the
//! exact top-k under f32 is the ground truth; the codec's top-k is compared
//! to it. Score error is measured over every (query, vector) pair.
//!
//! When queries carry relevance judgments and the stored vectors carry their
//! source paths, the same scan also reports nDCG@k and document recall@k
//! against the labels, for the codec and for the f32 baseline alike.

use std::time::Instant;

use hvec_core::{Codec, CodecError, Metric, distance};
use serde::{Deserialize, Serialize};

use crate::metrics::recall_at_k;
use crate::questions::relevance;

/// A query vector, optionally excluding one stored id (itself) from ranking,
/// optionally with the ids of the documents judged relevant to it.
#[derive(Debug, Clone, Default)]
pub struct RecallQuery {
    pub vector: Vec<f32>,
    pub exclude: Option<i64>,
    pub relevant: Vec<String>,
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
    /// Queries that carried relevance judgments.
    pub labelled_queries: usize,
    /// Mean nDCG@k against the judgments, over labelled queries. `None` when
    /// no query is labelled or sources were not supplied.
    pub ndcg_at_k: Option<f64>,
    /// Mean fraction of relevant documents among the codec's top-k.
    pub doc_recall_at_k: Option<f64>,
    /// Median over queries of the exact score gap between rank k and rank
    /// k + 1: how far the last true neighbour sits above the first outsider.
    /// A property of the corpus and embedding model; the same for every codec.
    #[serde(default)]
    pub boundary_gap: f64,
    /// Median over queries of the exact score gap between rank k and rank
    /// 2k: the width of the score window holding the next k competitors.
    /// Codec-independent, like `boundary_gap`.
    #[serde(default)]
    pub neighbour_band: f64,
    /// Mean over queries of the standard deviation of the codec's score error
    /// over the exact top-2k. Per-query bias is removed, since a constant
    /// shift does not change a ranking: this is the noise that reorders.
    #[serde(default)]
    pub rank_noise: f64,
}

/// Indices into `scores`, best first, ties broken by id.
pub(crate) fn ranked(scores: &[(i64, f32)], metric: Metric) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    let by_id = |a: usize, b: usize| scores[a].0.cmp(&scores[b].0);
    if metric.higher_is_better() {
        order.sort_by(|&a, &b| scores[b].1.total_cmp(&scores[a].1).then(by_id(a, b)));
    } else {
        order.sort_by(|&a, &b| scores[a].1.total_cmp(&scores[b].1).then(by_id(a, b)));
    }
    order
}

fn top_k(scores: &[(i64, f32)], k: usize, metric: Metric) -> Vec<i64> {
    ranked(scores, metric)
        .into_iter()
        .take(k)
        .map(|i| scores[i].0)
        .collect()
}

pub(crate) fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// Standard deviation of `approx − exact` over the given indices.
pub(crate) fn centred_noise(exact: &[(i64, f32)], approx: &[(i64, f32)], idx: &[usize]) -> f64 {
    if idx.len() < 2 {
        return 0.0;
    }
    let errs: Vec<f64> = idx.iter().map(|&i| f64::from(approx[i].1 - exact[i].1)).collect();
    let mean = errs.iter().sum::<f64>() / errs.len() as f64;
    (errs.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errs.len() as f64).sqrt()
}

/// Evaluate `codec` on `(id, f32 vector)` pairs with the given queries.
///
/// `sources`, when given, aligns with `ids` and holds each vector's source
/// path; it is what the label metrics need to map a chunk to its document.
pub fn evaluate(
    ids: &[i64],
    vectors: &[Vec<f32>],
    queries: &[RecallQuery],
    codec: &dyn Codec,
    metric: Metric,
    k: usize,
    sources: Option<&[String]>,
) -> Result<RecallReport, CodecError> {
    assert_eq!(ids.len(), vectors.len(), "ids and vectors must align");
    if let Some(s) = sources {
        assert_eq!(s.len(), ids.len(), "ids and sources must align");
    }
    let source_of = |id: i64| -> Option<&String> {
        let s = sources?;
        ids.iter().position(|x| *x == id).map(|i| &s[i])
    };

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
    let mut labelled = 0usize;
    let mut ndcg_sum = 0f64;
    let mut doc_recall_sum = 0f64;
    let mut boundary_gaps: Vec<f64> = Vec::with_capacity(queries.len());
    let mut bands: Vec<f64> = Vec::with_capacity(queries.len());
    let mut noise_sum = 0f64;

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
        let prepared = codec.prepare_query(&q.vector)?;
        let approx: Vec<(i64, f32)> = ids
            .iter()
            .zip(&encoded)
            .filter(|(id, _)| Some(**id) != q.exclude)
            .map(|(id, e)| codec.score_prepared(&prepared, e, metric).map(|s| (*id, s)))
            .collect::<Result<_, _>>()?;
        scan_ms += t.elapsed().as_millis() as u64;

        for ((_, a), (_, b)) in exact.iter().zip(&approx) {
            let e = f64::from((a - b).abs());
            err_sum += e;
            err_max = err_max.max(e);
            pairs += 1;
        }
        // `exact` and `approx` are index-aligned: both walk `ids` in order.
        let order = ranked(&exact, metric);
        let at = |rank: usize| order.get(rank - 1).map(|&i| f64::from(exact[i].1));
        if let (Some(sk), Some(sk1)) = (at(k), at(k + 1)) {
            boundary_gaps.push((sk - sk1).abs());
        }
        if let (Some(sk), Some(s2k)) = (at(k), at(2 * k)) {
            bands.push((sk - s2k).abs());
        }
        noise_sum += centred_noise(&exact, &approx, &order[..order.len().min(2 * k)]);
        let truth: Vec<i64> = order.iter().take(k).map(|&i| exact[i].0).collect();
        let got = top_k(&approx, k, metric);
        recall_sum += recall_at_k(&truth, &got, k);
        if !truth.is_empty() && truth.first() == got.first() {
            top1_hits += 1;
        }
        if sources.is_some() && !q.relevant.is_empty() {
            let retrieved: Vec<String> = got.iter().filter_map(|id| source_of(*id).cloned()).collect();
            if let Some(r) = relevance(&q.relevant, &retrieved) {
                labelled += 1;
                ndcg_sum += r.ndcg;
                doc_recall_sum += r.doc_recall;
            }
        }
    }

    let nq = queries.len().max(1) as f64;
    let labelled_mean = |sum: f64| {
        if labelled == 0 {
            None
        } else {
            Some(sum / labelled as f64)
        }
    };
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
        labelled_queries: labelled,
        ndcg_at_k: labelled_mean(ndcg_sum),
        doc_recall_at_k: labelled_mean(doc_recall_sum),
        boundary_gap: median(boundary_gaps),
        neighbour_band: median(bands),
        rank_noise: noise_sum / nq,
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
                relevant: Vec::new(),
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
        let r = evaluate(&ids, &vecs, &queries, &F32Codec::new(64), Metric::Cosine, 10, None).unwrap();
        assert_eq!(r.recall_at_k, 1.0);
        assert_eq!(r.top1_agreement, 1.0);
        assert_eq!(r.mean_abs_score_error, 0.0);
        assert_eq!(r.queries, 20);
        assert_eq!(r.vectors, 200);
        assert_eq!(r.labelled_queries, 0);
        assert_eq!(r.ndcg_at_k, None);
        assert_eq!(r.rank_noise, 0.0);
        assert!(r.neighbour_band >= r.boundary_gap && r.boundary_gap > 0.0);
    }

    #[test]
    fn spacing_is_codec_independent_and_noise_is_not() {
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 20);
        let i8 = evaluate(&ids, &vecs, &queries, &Int8Codec::new(64), Metric::Cosine, 10, None).unwrap();
        let bin = evaluate(&ids, &vecs, &queries, &BinaryCodec::new(64), Metric::Cosine, 10, None).unwrap();
        assert_eq!(i8.boundary_gap, bin.boundary_gap);
        assert_eq!(i8.neighbour_band, bin.neighbour_band);
        assert!(
            bin.rank_noise > 10.0 * i8.rank_noise,
            "{} vs {}",
            bin.rank_noise,
            i8.rank_noise
        );
    }

    #[test]
    fn centred_noise_ignores_a_constant_shift() {
        let exact = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let shifted: Vec<(i64, f32)> = exact.iter().map(|(i, s)| (*i, s - 0.25)).collect();
        assert!(centred_noise(&exact, &shifted, &[0, 1, 2]) < 1e-6);
        let noisy = vec![(1, 0.8), (2, 0.9), (3, 0.7)];
        assert!(centred_noise(&exact, &noisy, &[0, 1, 2]) > 0.05);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn label_metrics_follow_cluster_membership() {
        // Each cluster is one "document"; a self-query's relevant document is
        // its own cluster, so f32 scores a perfect nDCG and binary a lower one.
        let (ids, vecs) = clustered();
        let sources: Vec<String> = ids.iter().map(|id| format!("corpus/c{}.txt", id / 100)).collect();
        let mut queries = self_queries(&ids, &vecs, 16);
        for q in &mut queries {
            let own = q.exclude.unwrap() / 100;
            q.relevant = vec![format!("c{own}")];
        }
        let f = evaluate(
            &ids,
            &vecs,
            &queries,
            &F32Codec::new(64),
            Metric::Cosine,
            5,
            Some(&sources),
        )
        .unwrap();
        assert_eq!(f.labelled_queries, 16);
        assert_eq!(f.ndcg_at_k, Some(1.0));
        assert_eq!(f.doc_recall_at_k, Some(1.0));
        let b = evaluate(
            &ids,
            &vecs,
            &queries,
            &BinaryCodec::new(64),
            Metric::Cosine,
            5,
            Some(&sources),
        )
        .unwrap();
        assert!(b.ndcg_at_k.unwrap() <= 1.0 && b.ndcg_at_k.unwrap() > 0.0);
        // Without sources the labels cannot be applied.
        let none = evaluate(&ids, &vecs, &queries, &F32Codec::new(64), Metric::Cosine, 5, None).unwrap();
        assert_eq!(none.labelled_queries, 0);
    }

    #[test]
    fn int8_is_near_perfect_and_binary_is_decent_on_clusters() {
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 40);
        let i8 = evaluate(&ids, &vecs, &queries, &Int8Codec::new(64), Metric::Cosine, 10, None).unwrap();
        assert!(i8.recall_at_k >= 0.95, "int8 recall {}", i8.recall_at_k);
        assert!(i8.mean_abs_score_error < 0.01, "int8 err {}", i8.mean_abs_score_error);
        let bin = evaluate(&ids, &vecs, &queries, &BinaryCodec::new(64), Metric::Cosine, 10, None).unwrap();
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
        let r = evaluate(&ids, &vecs, &q, &F32Codec::new(64), Metric::L2, 3, None).unwrap();
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
