---
title: "Routing: choosing the compression per query or per collection"
date: 2026-11-04 22:00:00 +0000
result: "Perfect per-query routing would gain 5.5–7 points of recall@10 over a fixed re-score depth, but the scan's own scores reveal almost none of it: no router built on them captures more than 6%. Per collection, the noise-to-spacing curve picks the right codec but not the right depth."
references: [li2024quantmoe, thakur2021beir, xiao2023bge, reimers2019sbert]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-routing-oracle](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-routing-oracle).
Follow-up to [the hybrid × adaptive matrix]({{ "/experiments/hybrid-adaptive-matrix/" | relative_url }}).

<p align="center"><img src="{{ "/assets/animations/adaptive-depth.svg" | relative_url }}" width="820" alt="Animated schematic of adaptive re-scoring depth: a crowded query and a well-separated query share one noise margin but get very different numbers of candidates."></p>

**Hypothesis.** Routing each query to the compression that suits it, the way a mixture-of-experts
gate routes tokens [1], has real headroom, because queries need very different re-score depths.
But a router sees only the scan's own noisy scores, so it will capture little of that headroom.

**Setup.** `bench rescore --trace` records, for 38,124 queries over six scan codecs and ten
collections [2, 3, 4], where the scan ranks each true neighbour and the band visible in the
scan's own scores. With an exact second stage that is enough to replay any depth policy offline:
fixed depth, the margin policy, a ratio router (depth from σ ÷ the visible band), and a hindsight
oracle solved exactly for each budget.

**Result.** Confirmed for per-query routing; half confirmed per collection.

| mean depth | fixed | margin | ratio router | oracle |
|---:|---:|---:|---:|---:|
| 20 | 0.832 | −0.000 | −0.020 | +0.071 |
| 30 | 0.875 | +0.003 | −0.008 | +0.069 |
| 50 | 0.900 | +0.004 | −0.001 | +0.067 |
| 100 | 0.939 | +0.003 | +0.000 | +0.055 |

The depth a query needs varies 6.3× between its 10th and 90th percentile, but the visible band
tracks the true band only at Spearman 0.52–0.70. Per collection, predicting the cheapest
(codec, depth) from the noise-to-spacing curve picks the right scan size in 10 of 10 collections,
yet reaches anywhere from 0.86 to 1.00 when aiming for 95%.

**Reading.** The oracle's headroom is mostly chance: which neighbour the code's noise happened to
push down, which the same noisy scores cannot reveal. Keep the cheap margin policy for its
worst-query gain; choose the codec per collection from the curve; measure, don't predict, the
depth.

{% include references.html %}
