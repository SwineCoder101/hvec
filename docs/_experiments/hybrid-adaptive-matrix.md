---
title: "The hybrid × adaptive matrix"
date: 2026-11-04 21:00:00 +0000
result: "PCA plus variance-weighted bit allocation beats product quantization at equal bytes on average (0.768 vs 0.747 recall@10 at 48 bytes), all of it on MiniLM. Adaptive re-score depth gains half a point at equal work, up to 4 points for the worst queries."
references: [li2024quantmoe, huang1963block, max1960quantizing, jegou2011pq, thakur2021beir, xiao2023bge, reimers2019sbert]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-hybrid-adaptive-matrix](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-hybrid-adaptive-matrix).
Follow-up to [two-stage retrieval]({{ "/experiments/two-stage-rescoring/" | relative_url }}).

<p align="center"><img src="{{ "/assets/animations/adaptive-bits.svg" | relative_url }}" width="820" alt="Animated bar chart of a 48-byte budget spread over 384 principal components of SciFact's bge-small embeddings: uniform binary, variance-based allocation, and query-weighted allocation."></p>

**Two kinds of adaptive compression.** *Adaptive bits* spend a scan code's bytes unevenly, the
idea behind structure-aware mixed-precision quantization of mixture-of-experts models [1].
`mixed<B>` gives each dimension 0 to 8 bits by reverse water-filling on its variance [2], with a
Gaussian-optimal quantizer step [3]. `-pca` rotates into principal components first; `-w` weights
each component by the query's energy in it. *Adaptive depth* re-scores every candidate within λσ
of the scan's own 10th score, with σ calibrated on self-queries.

**Setup.** Ten collections (five BEIR corpora [5] × bge-small [6] and MiniLM [7]); eleven scan
codecs at 24, 48 and 96 bytes, uniform (`binary`, `binary-centred`, `pq<m>` [4]) and adaptive
(`mixed`); no re-score, fixed depths 20/50/100, or margins 1σ/2σ/3σ; int8 or f32 second stage.

**Result.**

| bytes | best uniform scan, recall@10 | `mixed-pca-w`, recall@10 | bge-small | MiniLM |
|---:|---|---:|---|---|
| 24 | `pq24` 0.606 | **0.635** | 0.587 → 0.592 | 0.624 → 0.677 |
| 48 | `pq48` 0.747 | **0.768** | 0.724 → 0.723 | 0.769 → 0.813 |
| 96 | `pq96` 0.870 | **0.886** | 0.855 → 0.855 | 0.885 → 0.916 |

Without the rotation, `mixed48` (0.696) is no better than `binary-centred` (0.698). Measured
exactly at equal mean depth, adaptive depth adds 0.3–0.6 points of recall@10 over a fixed depth
(ahead in 156 of 180 comparisons) and up to 4 points of worst-decile recall, at a 90th-percentile
depth 1.3–2.1× the mean. *Correction:* the first version of the write-up said 1.5 points, from
interpolating between too few fixed depths.

**Reading.** PCA plus bit allocation never trails PQ by more than a point at equal bytes and
leads by 3–5 points on MiniLM. On bge-small, where 60% of a vector's energy is a shared mean
direction, the two tie; PQ's codebooks probably absorb that component more cheaply. The cost:
`mixed48-pca-w` is the slowest scan measured, 2.2× f32 per vector ([zero-copy scan]({{ "/experiments/zero-copy-scan/" | relative_url }})).
The full matrix and a benefits-and-drawbacks table for every combination are in the write-up.

{% include references.html %}
