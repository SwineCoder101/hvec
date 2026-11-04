---
title: "Zero-copy scan: which codecs are actually faster?"
date: 2026-11-04 22:20:00 +0000
result: "Copying was most of the original search's cost: a contiguous in-place scan is 5.1× faster. Fewer bytes do not make a scan faster, except with popcount: binary-hamming plus an f32 re-score reaches 95% recall@10 on FiQA about 19× faster than exact f32."
references: [charikar2002simhash, jegou2011pq, thakur2021beir, maia2018fiqa, xiao2023bge]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-zero-copy-scan](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-zero-copy-scan).

**Hypothesis.** Most of the original search's time is copying, not scoring: every query read
every row, blob and text out of SQLite and allocated a vector per row. And once copying is gone,
asymmetric codecs (an f32 query against compressed codes) will still be no faster than f32,
because their scoring, not their bytes, sets the time.

**Setup.** `hvec bench scan` on SciFact (8,036 vectors) and FiQA [4] (71,274), both from BEIR [3], bge-small [5],
384 dims; codes scanned in place in one contiguous buffer, borrowed by the codec as `&[u8]`; 1 to
10 threads; a sequential-read bandwidth ceiling for comparison. Apple M4 Pro, idle.
`binary-hamming` stores `binary`'s sign bits [1] but binarises the query too and scores by XOR
and popcount.

**Result.** Both confirmed.

| f32 search path, FiQA | ms / query |
|---|---:|
| SQLite rows copied (the original) | 74.7 |
| SQLite rows, blobs borrowed | 54.5 |
| … with SQLite's memory map | 27.4 |
| contiguous buffer, in place | **14.7** |

| codec, one thread, FiQA | bytes | ns / vector | share of bandwidth |
|---|---:|---:|---:|
| f32 | 1,536 | 207 | 9.5% |
| int8 | 392 | 261 | 1.9% |
| binary | 48 | 246 | 0.3% |
| pq48 [2] | 48 | 203 | 0.3% |
| mixed48-pca-w | 48 | 462 | 0.1% |
| binary-hamming | 48 | **10** | 6.1% |

A `binary-hamming` scan alone keeps only 0.40 of FiQA's top 10, but re-scoring 369 candidates in
f32 reaches 95%, for about 0.79 ms against 14.8 ms for exact f32 on one thread. That is roughly
19× faster, with nDCG within 0.0003 of f32 at 500 candidates.

**Reading.** Compression buys memory everywhere, and time only with symmetric scoring. Caveats:
brute force; the hybrid's re-score cost is computed from measured per-vector costs, assuming the
f32 vectors are in memory; no explicit SIMD; one machine. A memory-mapped code file is left out
because the workspace forbids `unsafe`.

{% include references.html %}
