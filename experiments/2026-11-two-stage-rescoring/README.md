# Two-stage retrieval: how deep must the re-score go?

The neighbour-spacing experiment found that a codec's loss of the exact top 10 is governed by its
noise ÷ the corpus's neighbour band. The standard way to win the loss back is *hybrid*
retrieval: scan every vector with a small code, then re-score the best N candidates with a
bigger representation and keep the top 10. This experiment measures how large N must be, and
whether the same ratio predicts it.

**Hypothesis.** A codec with noise σ displaces a true neighbour past about (σ ÷ band) × k
competitors, so the candidate depth an exact re-score needs for a target recall grows with the
ratio, roughly in proportion to it. Re-scoring a few times k candidates will recover nearly all
of the loss at a small fraction of an f32 scan's bytes.

**Result.** Confirmed, with one correction. Across 50 (collection, first stage) points, the
depth needed tracks the ratio with Spearman ρ ≈ 0.93 at every target. It is proportional at
90% recall (depth ≈ 3.1·k·r^1.06) but grows faster than proportional for higher targets
(r^1.29 at 95%, r^1.74 at 99%): the last neighbours are in the error distribution's tail. Fitted
on four corpora, the curve predicts the fifth's depth within a factor of 1.25 (90%) to 1.41
(99%) on average. At 48 bytes per vector, re-scoring 50 candidates in f32 recovers 0.948 of the
top 10 for `binary` and 0.986 for `pq48`, and removes essentially all of the nDCG loss, for
3.2 to 3.8% of the bytes an f32 scan reads. An int8 second stage is just as good up to depth
about 30, then stops at int8's own recall of 0.990.

## Setup

| | |
|---|---|
| Collections | The ten from [neighbour spacing](../2026-11-neighbour-spacing/): SciFact, NFCorpus, ArguAna, CoSQA, FiQA × bge-small-en-v1.5, all-MiniLM-L6-v2 (384 dims, 200-word chunks) |
| Queries | Each set's judged test queries |
| First stage (scanned) | `binary`, `binary-centred` (48 B), `pq96`, `pq48`, `pq24` |
| Second stage (candidates only) | `int8` (392 B), `f32` (1,536 B) |
| Depths | 10, 20, 30, 50, 100, 200, 500, 1000 |
| k | 10 |
| Depth for a target recall | With an exact second stage, recall at depth N is the fraction of true top-10 neighbours the first stage ranks within N. The depth for target t is the t-quantile of those ranks, pooled over queries. It is exact, not interpolated from the sweep |
| Search | brute force, no index |
| Hardware | Apple M4 Pro, release build |
| Command | `./run.sh` (reuses the spacing experiment's collections; builds them if missing) |

New command: `hvec bench rescore --collection <c> --codecs <first> --rescore <second> --depths …`.

## Results

### Depth needed, as a multiple of k

f32 second stage. The full table, all 50 points, is in `results/summary.txt`.

| collection | first stage | noise/band | scan alone | 90% | 95% | 99% |
|---|---|---:|---:|---:|---:|---:|
| ArguAna, MiniLM | pq96 | 0.20 | 0.924 | 1.0 | 1.1 | 1.4 |
| SciFact, bge-small | pq48 | 0.83 | 0.711 | 2.1 | 3.1 | 5.9 |
| SciFact, bge-small | binary | 1.39 | 0.611 | 4.0 | 7.1 | 19.2 |
| FiQA, bge-small | binary | 1.65 | 0.537 | 6.0 | 10.5 | 32.7 |
| NFCorpus, bge-small | binary | 1.65 | 0.556 | 6.4 | 11.6 | 42.4 |
| FiQA, bge-small | pq24 | 1.35 | 0.445 | 9.8 | 16.5 | 45.3 |

| target recall | fit, depth / k | Spearman ρ | held-out corpus error (geometric mean factor) |
|---|---|---:|---:|
| 90% | 3.07 × r^1.06 | 0.930 | 1.25 |
| 95% | 4.76 × r^1.29 | 0.934 | 1.31 |
| 99% | 11.22 × r^1.74 | 0.925 | 1.41 |

### Recall@10 and nDCG loss by depth (means over the ten collections)

| first stage | second | depth 10 | 20 | 30 | 50 | 100 | 1000 | nDCG loss at depth 10 → 50 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| binary (48 B) | f32 | 0.669 | 0.841 | 0.900 | 0.948 | 0.981 | 1.000 | 0.0138 → 0.0003 |
| binary (48 B) | int8 | 0.669 | 0.839 | 0.898 | 0.944 | 0.974 | 0.991 | 0.0138 → 0.0004 |
| binary-centred (48 B) | f32 | 0.698 | 0.869 | 0.924 | 0.966 | 0.991 | 1.000 | 0.0125 → 0.0002 |
| pq48 (48 B) | f32 | 0.747 | 0.913 | 0.959 | 0.986 | 0.998 | 1.000 | 0.0078 → 0.0002 |
| pq48 (48 B) | int8 | 0.747 | 0.910 | 0.954 | 0.979 | 0.989 | 0.991 | 0.0078 → 0.0002 |
| pq24 (24 B) | f32 | 0.606 | 0.779 | 0.855 | 0.918 | 0.968 | 1.000 | 0.0309 → 0.0046 |
| pq96 (96 B) | f32 | 0.870 | 0.986 | 0.997 | 1.000 | 1.000 | 1.000 | 0.0005 → −0.0001 |

Depth 10 re-scores the scan's own top 10. It cannot change which 10 are returned, but it fixes
their order, so nDCG already recovers part of the way: binary's scan alone loses 0.024 nDCG
on average, and that halves to 0.014.

**Cost.** A 48-byte scan of every vector reads 3.1% of the bytes an f32 scan reads. Re-scoring
100 f32 candidates adds 150 KB per query. On FiQA's 71,274 chunks that brings the total to
3.3% of an f32 scan; on SciFact's 8,036 chunks, 4.4%. The scan dominates, so the re-score is
nearly free on large collections and only matters on small ones.

## Discussion

**The ratio predicts how deep to go, not just how much is lost.** The same number that
predicted the loss in the spacing experiment predicts the re-score depth. One f32 pass gives
the band, the codec's noise is roughly fixed per (codec, model), and the curve gives the depth
to within about a third. In practice: at 48 bytes on bge-small, plan for 50 to 100 candidates
with `binary` and 30 to 50 with `pq48`. MiniLM needs about half of that.

**Why the exponent grows with the target.** At 90% the depth is proportional to the ratio,
which is the simple picture: noise σ moves a neighbour past σ ÷ band × k competitors. Reaching
99% means catching the neighbours the codec displaced furthest, which are in the tail of its
error. They also fall into the sparser region below rank 2k, where the band is a worse guide
to density.

**int8 is the right second stage only up to a point.** An int8 re-score matches f32 up to a
depth of about 30. After that, int8's own ranking error (0.990 of the top 10) becomes the
ceiling. If 99% or more is the target, re-score in f32. The bytes are the same at these depths.

**Caveats.**
- **Brute-force scan.** With an index (HNSW, IVF) the first stage is no longer a full scan, and
  the cost side changes completely. The depth needed should not change.
- **f32 must still be stored somewhere.** The disk cost of the second stage is the full 1,536
  bytes per vector. Only the scan's 48 bytes need to be in memory.
- **The same queries and collections as the spacing experiment**, including CoSQA's duplicates.
- **Depth is fixed for every query.** Adaptive per-query depth is tested in the
  [hybrid × adaptive matrix](../2026-11-hybrid-adaptive-matrix/).
