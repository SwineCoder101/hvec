# Routing: how much can choosing the compression per query or per collection gain?

The adaptive-depth policy in the [hybrid × adaptive matrix](../2026-11-hybrid-adaptive-matrix/)
is a simple router: each query gets as many re-scored candidates as its scan scores suggest it
needs. The natural generalisation is to route each query, or each collection, to whichever form
of compression serves it best, the way a mixture-of-experts gate routes tokens to experts. Before
building a better router, this experiment measures how much there is to gain, and how much of it
a router can actually see.

**Hypothesis.** Per-query routing has real headroom, because the depth a query needs varies a lot
between queries. But a router only sees the scan's own scores, which are noisy, so a router
built on them will capture a small share of that headroom. Per-collection routing, choosing a
codec and a depth once from the noise-to-spacing curve, will get close to the best choice.

**Result.** The first two are confirmed. The third holds for the codec choice and fails for the
depth.
- **The headroom is real:** a hindsight oracle beats a fixed depth by 5.5 to 7 points of
  recall@10 at the same mean depth. The depth a query needs for full recall varies 6.3× between
  the 10th and 90th percentile.
- **Almost none of it is visible.** The band a router can read from the scan's own scores tracks
  the true band with Spearman 0.52 to 0.70. The margin policy captures 0 to 6% of the headroom.
  A router that sets depth from noise ÷ band_hat does *worse* than a fixed depth at every budget
  up to 50 candidates.
- **Per-collection routing picks the right codec, but not the right depth.** Predicting each
  collection's cheapest scan codec and depth from the curve, fitted on the other corpora, picks
  the same scan size as the best choice in 10 of 10 collections at 90% and 95% recall. Its depth,
  though, is coarse. Aiming for 95%, the recall it actually reaches ranges from 0.86 to 1.00, and
  aiming for 90%, from 0.71 to 0.99. So the hypothesis holds for the codec choice and fails for
  the depth. A deployed version would have to measure the depth on a sample, not predict it.

## Setup

| | |
|---|---|
| Collections | The ten from [neighbour spacing](../2026-11-neighbour-spacing/): five corpora × bge-small, MiniLM |
| Queries | Each set's judged test queries: 38,124 query traces over 60 (collection, scan codec) pairs |
| Scan codecs | `binary`, `binary-centred`, `pq48`, `pq24`, `mixed48-pca-w`, `mixed24-pca-w` |
| Second stage | f32 (exact), so recall at depth N is the share of the true top 10 the scan ranks within N |
| Traces | `bench rescore --trace`: per query, the scan's rank of each true neighbour, the exact band and the band read from the scan's own scores (`band_hat`), and the margin policy's depth from 0.25σ to 6σ |
| Policies | fixed depth; margin (λσ below the scan's 10th score, σ calibrated on self-queries); ratio router, depth = a·k·(σ ÷ band_hat)^β with β = 1 and 1.3, swept over a; oracle, the best per-query depths for each mean-depth budget, solved exactly (Lagrangian over each query's steps) |
| Comparison | Every policy traces a curve of mean depth against mean recall@10; curves are compared at mean depths of 15, 20, 30, 50 and 100 |
| Per-collection routing | For targets of 90/95/99%: fit depth/k against noise ÷ band on four corpora, predict each scan codec's depth on the fifth, pick the cheapest (scan bytes × vectors + depth × 1,536 B), and check its real recall |
| Code | hvec commit `cd1864a` (`--trace`) |
| Command | `./run.sh` (records the traces), then `python3 -I analyse.py results/traces 10` |

## Results

### Per-query routing at equal mean depth

Means over the 60 traces with every policy defined at that budget. Δ is against a fixed depth.

| mean depth | fixed | margin | ratio router (best β) | oracle | headroom | margin captures | worst decile, fixed → margin |
|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | 0.796 | −0.002 | −0.025 | +0.067 | 0.067 | −4% | 0.596 → 0.615 |
| 20 | 0.832 | −0.000 | −0.020 | +0.071 | 0.071 | 0% | 0.638 → 0.661 |
| 30 | 0.875 | +0.003 | −0.008 | +0.069 | 0.069 | 4% | 0.690 → 0.730 |
| 50 | 0.900 | +0.004 | −0.001 | +0.067 | 0.067 | 5% | 0.738 → 0.757 |
| 100 | 0.939 | +0.003 | +0.000 | +0.055 | 0.055 | 6% | 0.786 → 0.831 |

The number of traces falls with the budget (from 45 to 7), because the margin grid does not
reach large mean depths for the better codecs. Compared exactly at each margin's own mean
depth instead (all 60 traces), the margin policy gains +0.006 at 1σ, +0.006 at 2σ and +0.003 at
3σ. It is ahead in 156 of 180 comparisons, and worst-decile recall rises by 4.1, 2.0 and 0.7
points.

### What a router can see

| scan codec | Spearman, band_hat vs exact band | Spearman, 1 / band_hat vs depth needed |
|---|---:|---:|
| `binary` | 0.61 | 0.56 |
| `binary-centred` | 0.63 | 0.55 |
| `pq48` | 0.70 | 0.58 |
| `pq24` | 0.55 | 0.53 |
| `mixed48-pca-w` | 0.68 | 0.52 |
| `mixed24-pca-w` | 0.52 | 0.48 |

### Per-collection routing

| target | same scan size as the best | reaches the target (within 0.005) | recall actually reached | cost vs the best choice |
|---|---:|---:|---|---:|
| 90% | 10 / 10 | 5 / 10 | 0.713 to 0.989 | ×1.00 |
| 95% | 10 / 10 | 4 / 10 | 0.861 to 0.997 | ×1.01 |
| 99% | 7 / 10 | 4 / 10 | 0.950 to 1.000 | ×1.06 |

The cost ratio looks perfect partly because the predictions that undershoot also read less.

Under this cost model (bytes read: the whole scan plus the re-scored f32 candidates) the best
choice is almost always the smallest scan with a deeper re-score. At 95% it is `mixed24-pca-w`
in 9 of 10 collections, at depths from 17 (ArguAna, MiniLM) to 111 (FiQA, bge-small), reading
1.7 to 2.5% of the bytes of an f32 scan.

## Discussion

**Where the oracle's headroom comes from.** The oracle knows, for each query, how far the scan
displaced its worst-placed true neighbour. Mostly that is chance: one neighbour's score happened
to be pushed down by the code's noise. Nothing in the scan's own output predicts it well, because
the same noise that displaced the neighbour is all the scan reports. The band a query's scores
show is a property of the query, and it predicts only part of the variation.

**Why the ratio router loses to fixed depth.** It sets depth in proportion to 1 / band_hat, so its
errors in band_hat become large errors in depth. Some queries get far too many candidates, which
uses up the budget. The margin policy is more robust: it counts candidates near the 10th score
directly, rather than extrapolating from one gap.

**What this means for routing.**
- **Per query:** a cheap margin policy is worth keeping for its worst-query gain, but a smarter
  per-query router built on scan scores alone is not worth building.
- **Per collection:** the noise-to-spacing curve picks the right codec, but its depth is off by
  up to 9 points of recall at a 95% target. The practical version of "route to the best
  compression" chooses the codec from the curve, then measures the depth directly. With an exact
  second stage, measuring takes one pass of self-queries with traces.
- **Next:** storage-side multiplexing, a layered code read to different depths per candidate,
  remains open; it changes what hybrid costs on disk, which this experiment does not measure.

**Caveats.**
- **The cost model counts bytes, not time.** The [zero-copy scan](../2026-11-zero-copy-scan/)
  experiment shows that every asymmetric scan costs about the same per vector regardless of its
  size. In time, the smallest scan does not win; only the symmetric `binary-hamming` scan is
  faster.
- **The traces cover six scan codecs, with an f32 second stage only.**
- **Few collections.** Ten collections, two of each corpus, make a small sample for the
  per-collection test.
