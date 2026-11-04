# Zero-copy scan: where a search spends its time, and which codecs are actually faster

Every earlier experiment measured what compression costs in recall. This one measures what it
buys in time, which the README had so far declined to claim: "binary is no faster than int8
here". The question has two parts. How much of a search's time is moving bytes rather than
scoring them, and does reading fewer bytes make a scan faster?

**Hypothesis.**
1. Most of the original search's time is copying, not scoring. Every query read every row,
   blob and text out of SQLite and allocated a vector per row. Scanning a contiguous buffer in
   place will be several times faster.
2. Once copying is gone, the asymmetric codecs (an f32 query against compressed codes) will
   still be no faster than f32. Their scoring is limited by computation, not by bytes. Only a
   symmetric code, the query binarised too and compared with XOR and popcount, will scan faster.

**Result.** Both confirmed.
- **Copying is most of the original cost.** The cached buffer searches 5.1× faster than the
  original row scan on FiQA (14.7 against 74.7 ms) and 5.2× on SciFact. SQLite's own memory map
  alone gives 2.7–2.9×.
- **Fewer bytes do not mean a faster scan, except for popcount.** `int8`, `binary`, `pq48`,
  `pq24` and `mixed48-pca-w` all cost 200 to 490 ns per vector on one thread, against 207 ns for
  f32. They use under 2% of the machine's memory bandwidth. `binary-hamming` costs 10 ns, 20×
  less than f32.
- **The fast scan needs a re-score.** `binary-hamming` alone keeps only 0.40 of FiQA's top 10. A
  hamming scan plus an f32 re-score reaches 95% recall with 369 candidates, for about 0.79 ms
  against 14.8 ms for exact f32 on one thread. That is the first speed claim hvec can make: about
  19× faster at 95% recall@10, with nDCG within 0.0003 of f32 at 500 candidates.

## Setup

| | |
|---|---|
| Collections | `scifact-bge-small` (8,036 vectors) and `fiqa-bge-small` (71,274), 384 dims, from [neighbour spacing](../2026-11-neighbour-spacing/) |
| Queries | 50 stored vectors per collection, each excluded from its own ranking when recall is counted; median of 3 timed passes after one warm-up pass |
| Storage paths (f32) | SQLite rows copied (the original `Store::search`: every row and its text per query); SQLite rows with borrowed blobs, SQLite's memory map off and on; the cached contiguous buffer (`Store::search` now) |
| Codecs, scanned in place | `f32`, `int8`, `binary`, `binary-centred`, `binary-hamming`, `pq48`, `pq24`, `mixed48-pca-w`, at 1, 2, 4, 8 and 10 threads |
| Bandwidth ceiling | A sequential sum of a 1 GiB `u64` buffer at the same thread counts |
| Machine | Apple M4 Pro, 10 performance + 4 efficiency cores, 24 GiB, nothing else running; release build |
| Code | hvec commit `8f87908` |
| Command | `./run.sh` (`results/<collection>.json`, `results/summary.txt`, `results/machine.txt`); the re-score depths for `binary-hamming` are in `results/rescore-<collection>.json` |

## Results

### Storage path: the same exact f32 search (ms per query)

| path | SciFact (8,036) | FiQA (71,274) | speed-up on FiQA |
|---|---:|---:|---:|
| SQLite rows, copied (the original search) | 8.75 | 74.70 | 1.0× |
| SQLite rows, blob borrowed in place | 6.41 | 54.50 | 1.4× |
| SQLite rows, borrowed, SQLite memory map on | 3.01 | 27.40 | 2.7× |
| cached contiguous buffer, text for the hits only | **1.68** | **14.68** | **5.1×** |

Building the buffer once costs 47 ms for FiQA's 110 MB from SQLite (borrowed blobs, one copy),
or 8.6 ms from a flat code file read in one call.

### Codec scan, in place, one thread (FiQA)

| codec | bytes / vector | ms / query | ns / vector | GB/s | share of bandwidth | recall@10 alone |
|---|---:|---:|---:|---:|---:|---:|
| `f32` | 1,536 | 14.76 | 207 | 7.42 | 9.5% | 1.000 |
| `int8` | 392 | 18.64 | 261 | 1.50 | 1.9% | 0.978 |
| `binary` | 48 | 17.56 | 246 | 0.19 | 0.3% | 0.504 |
| `binary-centred` | 48 | 20.48 | 287 | 0.17 | 0.2% | 0.562 |
| `pq48` | 48 | 14.44 | 203 | 0.24 | 0.3% | 0.644 |
| `pq24` | 24 | 14.06 | 197 | 0.12 | 0.2% | 0.422 |
| `mixed48-pca-w` | 48 | 32.94 | 462 | 0.10 | 0.1% | 0.708 |
| `binary-hamming` | 48 | **0.72** | **10** | 4.76 | 6.1% | 0.372 |

Memory bandwidth measured 78 GB/s on one thread and 238 GB/s on ten. Recall here is over
self-queries; the judged-query figures in the other experiments are the ones to compare
codecs by.

### Thread scaling (FiQA, ms per query)

| codec | 1 | 2 | 4 | 8 | 10 | speed-up at 10 |
|---|---:|---:|---:|---:|---:|---:|
| `f32` | 14.76 | 7.56 | 4.00 | 2.09 | 1.71 | 8.7× (64 GB/s, 27% of bandwidth) |
| `pq48` | 14.44 | 7.36 | 3.78 | 2.00 | 1.62 | 8.9× |
| `mixed48-pca-w` | 32.94 | 17.34 | 8.49 | 4.60 | 3.84 | 8.6× |
| `binary-hamming` | 0.72 | 0.42 | 0.25 | 0.18 | 0.17 | 4.3× |

On SciFact, `binary-hamming` stops scaling at two threads. A whole scan takes 0.09 ms there,
and starting the threads costs about as much as the work.

### The fast hybrid: `binary-hamming` scan, f32 re-score (one thread)

Re-score cost from the measured f32 cost per vector (207 ns). Fetching the candidates is
treated as reading from memory; it is not timed separately.

| collection | candidates for 90 / 95 / 99% recall@10 | time at 95% | exact f32 scan | speed-up at 95% |
|---|---|---:|---:|---:|
| SciFact | 115 / 215 / 627 | 0.13 ms | 1.65 ms | 12× |
| FiQA | 186 / 369 / 1,254 | 0.79 ms | 14.76 ms | 19× |

For comparison, the same 95% target needs 105 candidates behind a `binary` scan and 44 behind
`pq48` on FiQA. Those hybrids read far fewer bytes, but their scans already take as long as
exact f32.

## Discussion

**The original search was mostly bookkeeping.** Reading every chunk's text on every query, and
allocating a vector per row, cost more than the scoring. The fix needs no unsafe code and no new
index: keep the codes in one buffer, score them in place, and read text for the hits only. SQLite's
memory map is a cheap partial version of the same thing.

**A smaller code only helps when the scoring is as cheap as the read.** At 48 bytes per vector a
scan reads 3.4 MB for FiQA, which this machine streams in well under a millisecond. But
asymmetric scoring does a float operation per dimension whatever the code size: `binary` walks
384 bits and adds or subtracts a float for each. So it costs what f32 costs. Only popcount does
work in proportion to the bytes. Compression buys memory everywhere, and time only with symmetric
scoring.

**Adaptive bits have a speed cost.** `mixed48-pca-w`, the best 48-byte scan for recall in the
[matrix](../2026-11-hybrid-adaptive-matrix/), is the slowest codec here: 2.2× f32 per vector,
because every dimension's code has to be unpacked from a bit stream of varying widths. A version
with byte-aligned groups, or a lookup table per group of dimensions as PQ uses, would close most
of that gap.

**The memory-mapped file is the one zero-copy path left out.** The workspace forbids `unsafe`, and
mapping a file needs one `unsafe` call. Reading a flat file is already fast (8.6 ms for 110 MB),
so a map would mostly save memory and start-up time on collections far larger than these.

**Caveats.**
- **Brute force only.** With an index (HNSW, IVF) the scan is not over every vector, and the
  ranking of these paths changes.
- **The hybrid times are computed, not measured end to end.** The re-score cost assumes the f32
  candidates are in memory. From disk, hundreds of random 1.5 KB reads per query would dominate.
- **Scoring is portable Rust with no explicit SIMD.** A tuned f32 kernel would move every row of
  the codec table, f32 most of all.
- **Recall in the codec table is over self-queries**, a harder and less realistic test than the
  judged queries the other experiments use.
- **One machine.** Other machines will give different absolute numbers; the ratios between paths
  and codecs are the result.
