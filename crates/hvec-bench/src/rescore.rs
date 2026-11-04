//! Two-stage retrieval: scan every vector with a small code, re-score the
//! best `depth` candidates with a larger one, keep the top k.
//!
//! The first stage is the codec held in memory for the scan (`binary`,
//! `pq48`); the second is the representation fetched only for candidates
//! (`int8`, `f32`). Ground truth is the exact f32 top-k, as in
//! [`crate::recall`].
//!
//! Alongside the sweep over depths, [`evaluate`] reports where the true
//! neighbours land in the first stage's ranking. With an exact second stage,
//! recall at depth `N` is simply the fraction of true neighbours ranked within
//! `N` by the first stage, so the depth needed for a target recall is a
//! percentile of those ranks. That is what lets the depth be compared with
//! the first stage's noise-to-spacing ratio.

use std::time::Instant;

use hvec_core::{Codec, CodecError, Encoded, Metric, distance};
use serde::{Deserialize, Serialize};

use crate::metrics::recall_at_k;
use crate::questions::relevance;
use crate::recall::{RecallQuery, centred_noise, median, ranked};

/// Recall targets for which the needed depth is reported.
pub const DEPTH_TARGETS: [f64; 4] = [0.90, 0.95, 0.99, 1.0];

/// Where the first stage puts the true neighbours, before any re-scoring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstStageProfile {
    pub first: String,
    pub metric: Metric,
    pub k: usize,
    pub vectors: usize,
    pub queries: usize,
    pub first_bytes: usize,
    /// As in [`crate::recall::RecallReport::rank_noise`].
    pub rank_noise: f64,
    /// As in [`crate::recall::RecallReport::neighbour_band`].
    pub neighbour_band: f64,
    /// Recall@k of the first stage alone (depth = k, no re-scoring).
    pub recall_at_k: f64,
    /// `(target, depth)`: the smallest candidate depth at which an exact
    /// second stage reaches the target mean recall@k.
    pub depth_for_recall: Vec<(f64, usize)>,
    /// Median over queries of the depth at which that query's whole exact
    /// top-k is among the candidates.
    pub median_full_depth: f64,
}

/// How many first-stage candidates a cell re-scores.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Policy {
    /// No second stage: the first stage's top k is the answer.
    None,
    /// The same number of candidates for every query.
    Fixed { depth: usize },
    /// Adaptive: every candidate whose first-stage score is within
    /// `sigmas × noise` of the first stage's k-th score, so crowded queries
    /// get more candidates and well-separated ones fewer. Bounded by k below
    /// and by the plan's `max_depth` above.
    Margin { sigmas: f64 },
}

/// What [`evaluate`] runs over one first stage.
#[derive(Debug, Clone)]
pub struct Plan<'a> {
    /// Re-scoring representations, each already fitted if it needs it.
    pub seconds: &'a [&'a dyn Codec],
    /// Fixed depths; values below k are raised to k.
    pub depths: &'a [usize],
    /// Margins for the adaptive policy, in units of `noise`.
    pub margins: &'a [f64],
    /// The first stage's score noise, calibrated without labels (for example
    /// `rank_noise` on self-queries); the adaptive policy's unit.
    pub noise: f64,
    /// Cap on any policy's per-query depth.
    pub max_depth: usize,
}

/// One (first stage, second stage, policy) cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescoreReport {
    pub first: String,
    /// `none` for the no-re-score cell.
    pub second: String,
    pub policy: Policy,
    pub metric: Metric,
    pub k: usize,
    /// The fixed depth, or the mean depth rounded for adaptive cells.
    pub depth: usize,
    /// Mean candidates re-scored per query.
    pub mean_depth: f64,
    /// 90th percentile of candidates re-scored per query.
    pub p90_depth: usize,
    pub vectors: usize,
    pub queries: usize,
    pub recall_at_k: f64,
    /// 10th percentile of per-query recall@k: the worst queries' experience.
    pub p10_recall_at_k: f64,
    pub top1_agreement: f64,
    pub labelled_queries: usize,
    pub ndcg_at_k: Option<f64>,
    pub doc_recall_at_k: Option<f64>,
    pub first_bytes: usize,
    pub second_bytes: usize,
    /// Mean bytes touched per query: the whole first-stage scan plus the
    /// re-scored candidates. Compare with `vectors × 4 × dim` for f32.
    pub bytes_read_per_query: f64,
    /// Time spent re-scoring candidates, over all queries.
    pub rescore_ms: u64,
}

struct Cell {
    second: Option<usize>,
    policy: Policy,
    recalls: Vec<f64>,
    depths: Vec<usize>,
    top1: usize,
    ndcg_sum: f64,
    docrec_sum: f64,
    ms: u64,
}

fn percentile<T: Copy + PartialOrd>(v: &[T], p: f64) -> Option<T> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((p * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1;
    Some(s[i])
}

/// Run the first stage once and every (second stage, policy) cell of `plan`
/// over it, plus the no-re-score cell.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    ids: &[i64],
    vectors: &[Vec<f32>],
    queries: &[RecallQuery],
    first: &dyn Codec,
    metric: Metric,
    k: usize,
    plan: &Plan<'_>,
    sources: Option<&[String]>,
) -> Result<(FirstStageProfile, Vec<RescoreReport>), CodecError> {
    assert_eq!(ids.len(), vectors.len(), "ids and vectors must align");
    if let Some(s) = sources {
        assert_eq!(s.len(), ids.len(), "ids and sources must align");
    }
    let cap = plan.max_depth.max(k).min(vectors.len());
    let mut depths: Vec<usize> = plan.depths.iter().map(|d| (*d).max(k).min(cap)).collect();
    depths.sort_unstable();
    depths.dedup();

    let enc1 = vectors.iter().map(|v| first.encode(v)).collect::<Result<Vec<_>, _>>()?;
    let enc2: Vec<Vec<Encoded>> = plan
        .seconds
        .iter()
        .map(|c| vectors.iter().map(|v| c.encode(v)).collect::<Result<Vec<_>, _>>())
        .collect::<Result<_, _>>()?;
    // A score that ranks last, for a query's excluded self.
    let worst = if metric.higher_is_better() {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };

    let mut cells: Vec<Cell> = Vec::new();
    let mut push = |second: Option<usize>, policy: Policy| {
        cells.push(Cell {
            second,
            policy,
            recalls: Vec::with_capacity(queries.len()),
            depths: Vec::with_capacity(queries.len()),
            top1: 0,
            ndcg_sum: 0.0,
            docrec_sum: 0.0,
            ms: 0,
        });
    };
    push(None, Policy::None);
    for s_i in 0..plan.seconds.len() {
        for &depth in &depths {
            push(Some(s_i), Policy::Fixed { depth });
        }
        for &sigmas in plan.margins {
            push(Some(s_i), Policy::Margin { sigmas });
        }
    }
    let mut labelled = 0usize;

    let mut pooled_ranks: Vec<usize> = Vec::with_capacity(queries.len() * k);
    let mut full_depths: Vec<f64> = Vec::with_capacity(queries.len());
    let mut bands: Vec<f64> = Vec::with_capacity(queries.len());
    let mut noise_sum = 0f64;
    let mut first_recall_sum = 0f64;

    for q in queries {
        // Scores indexed like `ids`; the excluded id scores worst everywhere.
        let score_all = |f: &dyn Fn(usize) -> Result<f32, CodecError>| -> Result<Vec<(i64, f32)>, CodecError> {
            (0..ids.len())
                .map(|i| Ok((ids[i], if Some(ids[i]) == q.exclude { worst } else { f(i)? })))
                .collect()
        };
        let exact = score_all(&|i| Ok(distance::score(&q.vector, &vectors[i], metric)))?;
        let prepared1 = first.prepare_query(&q.vector)?;
        let approx = score_all(&|i| first.score_prepared(&prepared1, &enc1[i], metric))?;
        let n_live = ids.len() - usize::from(q.exclude.is_some_and(|x| ids.contains(&x)));

        let exact_order = ranked(&exact, metric);
        let approx_order = ranked(&approx, metric);
        let truth: Vec<i64> = exact_order.iter().take(k).map(|&i| ids[i]).collect();

        // Spacing and noise, exactly as `recall::evaluate` measures them.
        if exact_order.len() >= 2 * k {
            let s = |r: usize| f64::from(exact[exact_order[r - 1]].1);
            bands.push((s(k) - s(2 * k)).abs());
        }
        noise_sum += centred_noise(&exact, &approx, &exact_order[..exact_order.len().min(2 * k)]);

        // First-stage rank (1-based) of every true neighbour.
        let mut rank_of = vec![0usize; ids.len()];
        for (r, &i) in approx_order.iter().enumerate() {
            rank_of[i] = r + 1;
        }
        let ranks: Vec<usize> = exact_order.iter().take(k).map(|&i| rank_of[i]).collect();
        first_recall_sum += ranks.iter().filter(|&&r| r <= k).count() as f64 / k as f64;
        full_depths.push(ranks.iter().copied().max().unwrap_or(0) as f64);
        pooled_ranks.extend(ranks);

        // The adaptive policy's depth: candidates within the margin of the
        // first stage's own k-th score. Uses first-stage scores only.
        let kth = approx[approx_order[k.min(n_live).max(1) - 1]].1;
        let within = |sigmas: f64| -> usize {
            let margin = (sigmas * plan.noise) as f32;
            let n = if metric.higher_is_better() {
                approx_order[..cap.min(n_live)].partition_point(|&i| approx[i].1 >= kth - margin)
            } else {
                approx_order[..cap.min(n_live)].partition_point(|&i| approx[i].1 <= kth + margin)
            };
            n.max(k).min(cap.min(n_live))
        };

        let is_labelled = sources.is_some() && !q.relevant.is_empty();
        if is_labelled {
            labelled += 1;
        }
        let prepared2: Vec<Vec<f32>> = plan
            .seconds
            .iter()
            .map(|c| c.prepare_query(&q.vector))
            .collect::<Result<_, _>>()?;
        for cell in &mut cells {
            let t = Instant::now();
            let (kept, depth): (Vec<usize>, usize) = match (cell.second, cell.policy) {
                (None, _) | (_, Policy::None) => (approx_order.iter().take(k).copied().collect(), 0),
                (Some(s_i), policy) => {
                    let depth = match policy {
                        Policy::Fixed { depth } => depth.min(n_live),
                        Policy::Margin { sigmas } => within(sigmas),
                        Policy::None => unreachable!(),
                    };
                    let second = plan.seconds[s_i];
                    let cands: Vec<(i64, f32)> = approx_order[..depth]
                        .iter()
                        .map(|&i| {
                            second
                                .score_prepared(&prepared2[s_i], &enc2[s_i][i], metric)
                                .map(|s| (i as i64, s))
                        })
                        .collect::<Result<_, _>>()?;
                    let kept = ranked(&cands, metric)
                        .into_iter()
                        .take(k)
                        .map(|c| cands[c].0 as usize)
                        .collect();
                    (kept, depth)
                }
            };
            cell.ms += t.elapsed().as_millis() as u64;
            cell.depths.push(depth);
            let got: Vec<i64> = kept.iter().map(|&i| ids[i]).collect();
            cell.recalls.push(recall_at_k(&truth, &got, k));
            if !truth.is_empty() && truth.first() == got.first() {
                cell.top1 += 1;
            }
            if let (true, Some(src)) = (is_labelled, sources) {
                let retrieved: Vec<String> = kept.iter().map(|&i| src[i].clone()).collect();
                if let Some(r) = relevance(&q.relevant, &retrieved) {
                    cell.ndcg_sum += r.ndcg;
                    cell.docrec_sum += r.doc_recall;
                }
            }
        }
    }

    let nq = queries.len().max(1) as f64;
    pooled_ranks.sort_unstable();
    let depth_for_recall = DEPTH_TARGETS
        .iter()
        .map(|&t| {
            if pooled_ranks.is_empty() {
                return (t, 0);
            }
            let need = ((t * pooled_ranks.len() as f64).ceil() as usize).clamp(1, pooled_ranks.len());
            (t, pooled_ranks[need - 1])
        })
        .collect();
    let profile = FirstStageProfile {
        first: first.name().to_owned(),
        metric,
        k,
        vectors: vectors.len(),
        queries: queries.len(),
        first_bytes: first.encoded_size(),
        rank_noise: noise_sum / nq,
        neighbour_band: median(bands),
        recall_at_k: first_recall_sum / nq,
        depth_for_recall,
        median_full_depth: median(full_depths),
    };
    let label_mean = |sum: f64| (labelled > 0).then(|| sum / labelled as f64);
    let reports = cells
        .into_iter()
        .map(|c| {
            let second_bytes = c.second.map_or(0, |s| plan.seconds[s].encoded_size());
            let mean_depth = c.depths.iter().sum::<usize>() as f64 / nq;
            RescoreReport {
                first: first.name().to_owned(),
                second: c
                    .second
                    .map_or_else(|| "none".to_owned(), |s| plan.seconds[s].name().to_owned()),
                policy: c.policy,
                metric,
                k,
                depth: match c.policy {
                    Policy::Fixed { depth } => depth,
                    _ => mean_depth.round() as usize,
                },
                mean_depth,
                p90_depth: percentile(&c.depths, 0.9).unwrap_or(0),
                vectors: vectors.len(),
                queries: queries.len(),
                recall_at_k: c.recalls.iter().sum::<f64>() / nq,
                p10_recall_at_k: percentile(&c.recalls, 0.1).unwrap_or(0.0),
                top1_agreement: c.top1 as f64 / nq,
                labelled_queries: labelled,
                ndcg_at_k: label_mean(c.ndcg_sum),
                doc_recall_at_k: label_mean(c.docrec_sum),
                first_bytes: first.encoded_size(),
                second_bytes,
                bytes_read_per_query: (vectors.len() * first.encoded_size()) as f64 + mean_depth * second_bytes as f64,
                rescore_ms: c.ms,
            }
        })
        .collect();
    Ok((profile, reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall::{self, self_queries};
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

    fn clustered() -> (Vec<i64>, Vec<Vec<f32>>) {
        let mut rng = XorShift(7);
        let dim = 64;
        let centres: Vec<Vec<f32>> = (0..8).map(|_| (0..dim).map(|_| rng.next_f32()).collect()).collect();
        let (mut ids, mut vecs) = (Vec::new(), Vec::new());
        for (c, centre) in centres.iter().enumerate() {
            for m in 0..40 {
                let mut v: Vec<f32> = centre.iter().map(|x| x + 0.3 * rng.next_f32()).collect();
                distance::normalize(&mut v);
                ids.push((c * 100 + m) as i64);
                vecs.push(v);
            }
        }
        (ids, vecs)
    }

    fn run(
        seconds: &[&dyn Codec],
        depths: &[usize],
        margins: &[f64],
        noise: f64,
        max_depth: usize,
    ) -> (FirstStageProfile, Vec<RescoreReport>) {
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 30);
        let plan = Plan {
            seconds,
            depths,
            margins,
            noise,
            max_depth,
        };
        evaluate(
            &ids,
            &vecs,
            &queries,
            &BinaryCodec::new(64),
            Metric::Cosine,
            10,
            &plan,
            None,
        )
        .unwrap()
    }

    fn fixed(reports: &[RescoreReport]) -> Vec<&RescoreReport> {
        reports
            .iter()
            .filter(|r| matches!(r.policy, Policy::Fixed { .. }))
            .collect()
    }

    #[test]
    fn deeper_f32_rescoring_never_hurts_and_reaches_exact() {
        let f32c = F32Codec::new(64);
        let (profile, reports) = run(&[&f32c], &[10, 20, 40, 80, 320], &[], 0.0, 1000);
        let none = &reports[0];
        assert_eq!(none.policy, Policy::None);
        assert_eq!(none.second, "none");
        assert!((none.recall_at_k - profile.recall_at_k).abs() < 1e-12);
        let f = fixed(&reports);
        let recalls: Vec<f64> = f.iter().map(|r| r.recall_at_k).collect();
        assert!(recalls.windows(2).all(|w| w[1] >= w[0]), "{recalls:?}");
        // Every vector re-scored exactly (the excluded self drops out): exact.
        assert_eq!(f.last().unwrap().recall_at_k, 1.0);
        assert_eq!(f.last().unwrap().p10_recall_at_k, 1.0);
        // Re-scoring the codec's own top k reorders but cannot change membership.
        assert!((f[0].recall_at_k - profile.recall_at_k).abs() < 1e-12);
        assert!(f[0].top1_agreement >= none.top1_agreement);
        let d: Vec<usize> = profile.depth_for_recall.iter().map(|(_, d)| *d).collect();
        assert!(d.windows(2).all(|w| w[1] >= w[0]) && d[0] >= 10, "{d:?}");
    }

    #[test]
    fn depth_for_recall_matches_the_sweep_for_an_exact_second_stage() {
        let f32c = F32Codec::new(64);
        let (profile, _) = run(&[&f32c], &[10], &[], 0.0, 1000);
        let (t, d) = profile.depth_for_recall[1];
        let (_, at) = run(&[&f32c], &[d], &[], 0.0, 1000);
        assert!(fixed(&at)[0].recall_at_k >= t - 1e-9, "at depth {d}");
        if d > 10 {
            let (_, below) = run(&[&f32c], &[d - 1], &[], 0.0, 1000);
            assert!(fixed(&below)[0].recall_at_k < t, "at depth {}", d - 1);
        }
    }

    #[test]
    fn margin_policy_varies_depth_per_query_and_grows_with_the_margin() {
        let f32c = F32Codec::new(64);
        let (profile, reports) = run(&[&f32c], &[], &[0.0, 1.0, 3.0, 1e6], profile_noise(), 200);
        let m: Vec<&RescoreReport> = reports
            .iter()
            .filter(|r| matches!(r.policy, Policy::Margin { .. }))
            .collect();
        assert_eq!(m.len(), 4);
        // A zero margin keeps only the first stage's top k (plus exact ties).
        assert!(m[0].mean_depth >= 10.0 && m[0].mean_depth < 11.0, "{}", m[0].mean_depth);
        let depths: Vec<f64> = m.iter().map(|r| r.mean_depth).collect();
        assert!(depths.windows(2).all(|w| w[1] >= w[0]), "{depths:?}");
        // An unbounded margin hits the cap for every query.
        assert_eq!(m[3].mean_depth, 200.0);
        assert!(m[2].p90_depth as f64 >= m[2].mean_depth);
        // Depth differs between queries at a moderate margin.
        assert!((m[1].p90_depth as f64) > m[1].mean_depth.floor(), "{:?}", m[1]);
        assert!(m[2].recall_at_k >= profile.recall_at_k);
    }

    fn profile_noise() -> f64 {
        let (ids, vecs) = clustered();
        let q = self_queries(&ids, &vecs, 30);
        recall::evaluate(&ids, &vecs, &q, &BinaryCodec::new(64), Metric::Cosine, 10, None)
            .unwrap()
            .rank_noise
    }

    #[test]
    fn profile_spacing_and_noise_match_recall_evaluate() {
        let i8c = Int8Codec::new(64);
        let (profile, reports) = run(&[&i8c], &[10, 50], &[], 0.0, 1000);
        let (ids, vecs) = clustered();
        let queries = self_queries(&ids, &vecs, 30);
        let r = recall::evaluate(&ids, &vecs, &queries, &BinaryCodec::new(64), Metric::Cosine, 10, None).unwrap();
        assert!((profile.rank_noise - r.rank_noise).abs() < 1e-9);
        assert_eq!(profile.neighbour_band, r.neighbour_band);
        assert!((profile.recall_at_k - r.recall_at_k).abs() < 1e-12);
        let f = fixed(&reports);
        assert_eq!(f[1].bytes_read_per_query, (320 * 8 + 50 * i8c.encoded_size()) as f64);
        assert_eq!(reports[0].bytes_read_per_query, (320 * 8) as f64);
        // Depth is capped at the collection size.
        let (_, all) = run(&[&i8c], &[1000], &[], 0.0, 1000);
        assert_eq!(fixed(&all)[0].depth, 320);
    }
}
