---
title: "Two-stage retrieval: how deep must the re-score go?"
date: 2026-11-04 20:00:00 +0000
result: "The depth an f32 re-score needs follows the same noise-to-spacing ratio that predicts codec loss (Spearman ≈ 0.93). At 48 bytes, 50 f32 candidates recover 0.95 (binary) to 0.99 (pq48) of the exact top 10 and nearly all of the nDCG loss."
references: [thakur2021beir, xiao2023bge, reimers2019sbert, jegou2011pq, charikar2002simhash]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-two-stage-rescoring](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-two-stage-rescoring).
Follow-up to [neighbour spacing]({{ "/experiments/neighbour-spacing/" | relative_url }}).

<p align="center"><img src="{{ "/assets/animations/hybrid-rescoring.svg" | relative_url }}" width="820" alt="Animated schematic of two-stage retrieval: a small code shortlists candidates, an exact f32 re-score lifts true neighbours into the top 10, and one neighbour ranked outside the shortlist is missed."></p>

**Hypothesis.** A codec with noise σ displaces a true neighbour past about (σ ÷ band) × k
competitors, so the candidate depth an exact re-score needs grows with the noise-to-spacing
ratio, roughly in proportion to it.

**Setup.** The ten collections of the spacing experiment: five BEIR-format corpora [1] × bge-small
[2] and MiniLM [3]. Scan codecs `binary`, `binary-centred` [5], `pq96`, `pq48`, `pq24` [4];
second stage `int8` or `f32`, read for candidates only; depths 10 to 1,000. With an exact second
stage, the depth needed for a target recall is a quantile of where the scan ranks the true
neighbours, so it is measured exactly.

**Result.** Confirmed, with one correction. The depth tracks the ratio with ρ ≈ 0.93 at every
target. It is proportional at 90% recall but grows faster than proportional for higher targets,
because the last neighbours sit in the tail of the code's error:

| target recall@10 | depth / k | Spearman ρ | held-out corpus error |
|---|---|---:|---:|
| 90% | 3.07 × r^1.06 | 0.930 | ×1.25 |
| 95% | 4.76 × r^1.29 | 0.934 | ×1.31 |
| 99% | 11.22 × r^1.74 | 0.925 | ×1.41 |

| first stage | depth 10 | 20 | 50 | 100 | nDCG loss at depth 10 → 50 |
|---|---:|---:|---:|---:|---|
| binary, f32 re-score | 0.669 | 0.841 | 0.948 | 0.981 | 0.0138 → 0.0003 |
| pq48, f32 re-score | 0.747 | 0.913 | 0.986 | 0.998 | 0.0078 → 0.0002 |
| pq48, int8 re-score | 0.747 | 0.910 | 0.979 | 0.989 | 0.0078 → 0.0002 |
| pq96, f32 re-score | 0.870 | 0.986 | 1.000 | 1.000 | 0.0005 → −0.0001 |

**Reading.** Re-scoring 50 candidates costs 75 KB per query on top of the scan, 3–4% of the
bytes an f32 scan reads. An int8 second stage matches f32 up to a depth of about 30, then stops
at int8's own 0.990. Caveats: brute force only; the f32 copy must still be stored; one fixed
depth for every query.

{% include references.html %}
