# The hybrid × adaptive matrix

Two ways to make compression adaptive, crossed with the hybrid (two-stage) retrieval from the
[re-scoring experiment](../2026-11-two-stage-rescoring/):

- **Adaptive bits.** Spend the scan code's bytes unevenly. `mixed<B>` gives each dimension 0 to 8
  bits by greedy reverse water-filling on its variance. `-pca` first rotates into the corpus's
  principal components, so the variance has somewhere to concentrate. `-w` weights each
  component by how much energy queries put into it, so the allocation minimises score error
  rather than reconstruction error. This is the embedding analogue of structure-aware,
  mixed-precision quantization of mixture-of-experts models ([QuantMoE-Bench][quantmoe]), where
  the blocks and experts that matter more get more bits.
- **Adaptive depth.** Re-score every candidate whose scan score is within λσ of the scan's own
  10th score, where σ is the scan code's noise, calibrated once per collection on self-queries.
  Crowded queries get deep re-scoring, well-separated ones shallow.

**Hypothesis.**
1. With a PCA rotation, adaptive bits beat uniform sign bits at equal bytes and approach
   product quantization. Without the rotation they do little, because raw embedding dimensions
   have fairly even variance.
2. At the same mean number of re-scored candidates, adaptive depth beats a fixed depth,
   mostly by rescuing the worst queries.

**Result.** Both confirmed, and the first more strongly than expected.
- **Adaptive bits beat PQ on average.** `mixed48-pca-w` keeps 0.768 of the exact top 10 with no
  re-scoring, against 0.747 for `pq48` and 0.669 for `binary`. It wins at 24 and 96 bytes too.
  The gain is all on MiniLM (+4.4 points); on bge-small the two tie (0.723 vs 0.724), and `pq48`
  keeps a slightly smaller nDCG loss (0.0160 vs 0.0177).
- **The rotation is what makes it work.** Without it, `mixed48` (0.696) is no better than
  `binary-centred` (0.698).
- **Adaptive depth beats fixed depth at equal mean depth in 522 of 526 cells.** On average it
  adds 1.5 points of recall@10 and 3.7 points of worst-decile recall, with no change in nDCG.
  The price is variable work: the 90th-percentile query re-scores 1.3 to 2.1× the mean.
- **The two combine.** `mixed48-pca-w` with a 1σ margin re-scores 19 candidates on average and
  keeps 0.929 of the top 10. `binary` with a fixed 20 keeps 0.841 for the same bytes.

## Setup

| | |
|---|---|
| Collections | The ten from [neighbour spacing](../2026-11-neighbour-spacing/): SciFact, NFCorpus, ArguAna, CoSQA, FiQA × bge-small-en-v1.5, all-MiniLM-L6-v2 (384 dims, 200-word chunks, 6,260 to 71,274 per collection) |
| Queries | Each set's judged test queries (300 to 1,406) |
| Scan codecs | Uniform: `binary`, `binary-centred` (48 B), `pq24`, `pq48`, `pq96`. Adaptive: `mixed48` (raw dimensions), `mixed24-pca`, `mixed48-pca`, `mixed24-pca-w`, `mixed48-pca-w`, `mixed96-pca-w` |
| Second stage | `int8` (392 B) or `f32` (1,536 B), read for candidates only |
| Policies | none; fixed depth 20, 50, 100; margin 1σ, 2σ, 3σ (capped at 1,000) |
| Calibration | σ = the scan code's rank noise on 200 self-queries per collection. The test queries' exact scores never set their own depth |
| Fitting | Trained codecs fitted on every vector of the collection. PQ: 256 centroids per subspace, 20 Lloyd iterations. `mixed`: PCA by Jacobi on up to 20,000 vectors |
| Metrics | recall@10 against exact f32; 10th-percentile query recall@10 (worst decile); nDCG@10 loss against the f32 ceiling, from the relevance judgments; mean and p90 depth |
| Search | brute force, no index |
| Hardware | Apple M4 Pro, release build |
| Code | hvec commit `6113ab7` (`bench rescore` with policies, `mixed<B>`) |
| Command | `./run.sh` (reuses the spacing experiment's collections; builds them if missing) |

## Results

All numbers are means over the ten collections. The full matrix, 143 cells per collection, is in
`results/summary.txt`; the raw reports are `results/<corpus>-<model>-k10.json`.

### The matrix: recall@10 by scan codec and re-scoring policy (f32 second stage)

| scan codec | bytes | none | fixed 20 | fixed 50 | fixed 100 | margin 1σ (mean depth) | margin 2σ (mean depth) | nDCG loss, none → fixed 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `binary` | 48 | 0.669 | 0.841 | 0.948 | 0.981 | 0.869 (23) | 0.963 (52) | 0.0240 → 0.0003 |
| `binary-centred` | 48 | 0.698 | 0.869 | 0.966 | 0.991 | 0.911 (26) | 0.984 (65) | 0.0236 → 0.0002 |
| `mixed48` | 48 | 0.696 | 0.868 | 0.966 | 0.991 | 0.909 (26) | 0.983 (65) | 0.0242 → 0.0000 |
| `pq48` | 48 | 0.747 | 0.912 | 0.986 | 0.998 | 0.924 (21) | 0.985 (41) | 0.0160 → 0.0002 |
| `mixed48-pca` | 48 | 0.755 | 0.920 | 0.989 | 0.998 | 0.919 (19) | 0.982 (35) | 0.0167 → 0.0000 |
| `mixed48-pca-w` | 48 | **0.768** | **0.931** | **0.991** | **0.999** | **0.929 (19)** | 0.983 (36) | 0.0177 → 0.0000 |
| `pq24` | 24 | 0.606 | 0.779 | 0.918 | 0.968 | 0.887 (42) | 0.979 (144) | 0.0586 → 0.0046 |
| `mixed24-pca` | 24 | 0.631 | 0.807 | 0.937 | 0.977 | 0.880 (31) | 0.971 (89) | 0.0497 → 0.0037 |
| `mixed24-pca-w` | 24 | **0.635** | **0.815** | **0.937** | **0.977** | 0.888 (33) | 0.973 (100) | 0.0523 → 0.0047 |
| `pq96` | 96 | 0.870 | 0.986 | 1.000 | 1.000 | 0.958 (14) | 0.990 (18) | 0.0031 → −0.0001 |
| `mixed96-pca-w` | 96 | **0.886** | **0.990** | 1.000 | 1.000 | **0.963 (14)** | **0.991 (18)** | 0.0022 → 0.0000 |

An int8 second stage matches f32 to within a point up to a depth of about 50, then levels off
at int8's own recall. At depth 100: `binary` 0.974 against 0.981, `pq48` 0.989 against 0.998, `mixed48-pca-w`
0.990 against 0.999.

### Adaptive bits alone, by byte budget (no re-scoring)

| bytes | codec | recall@10 | bge-small | MiniLM | worst decile | nDCG loss |
|---:|---|---:|---:|---:|---:|---:|
| 24 | `mixed24-pca-w` | **0.635** | 0.592 | **0.677** | 0.37 | 0.052 |
| 24 | `mixed24-pca` | 0.631 | **0.596** | 0.667 | 0.37 | **0.050** |
| 24 | `pq24` | 0.606 | 0.587 | 0.624 | 0.33 | 0.059 |
| 48 | `mixed48-pca-w` | **0.768** | 0.723 | **0.813** | **0.55** | 0.018 |
| 48 | `mixed48-pca` | 0.755 | 0.717 | 0.793 | 0.54 | 0.017 |
| 48 | `pq48` | 0.747 | **0.724** | 0.769 | 0.52 | **0.016** |
| 48 | `binary-centred` | 0.698 | 0.667 | 0.728 | 0.43 | 0.024 |
| 48 | `mixed48` (no rotation) | 0.696 | 0.663 | 0.729 | 0.43 | 0.024 |
| 48 | `binary` | 0.669 | 0.622 | 0.715 | 0.41 | 0.024 |
| 96 | `mixed96-pca-w` | **0.886** | 0.855 | **0.916** | **0.74** | **0.002** |
| 96 | `pq96` | 0.870 | **0.855** | 0.885 | 0.73 | 0.003 |

### Adaptive depth against fixed depth at the same mean depth

Each margin cell is compared with the fixed-depth curve of the same collection and codec,
interpolated in log depth at the margin's mean depth. Cells whose mean depth falls outside the
measured fixed range (20 to 100) are left out.

| margin | cells compared | Δ recall@10 | Δ worst decile | Δ nDCG | p90 depth / mean depth |
|---|---:|---:|---:|---:|---:|
| 1σ | 218 | +0.020 | +0.048 | +0.001 | 1.29–1.60 |
| 2σ | 192 | +0.014 | +0.033 | +0.000 | 1.39–1.94 |
| 3σ | 116 | +0.008 | +0.025 | +0.000 | 1.49–2.09 |
| all | 526 | **+0.015** | **+0.037** | 0.000 | |

Adaptive is ahead in 522 of the 526 cells. The gain is largest at small depths and for the
better scan codecs (`pq96` and `mixed96-pca-w` gain 3.1 to 3.3 points at a 1σ margin), and it
shrinks as the depth grows and every policy approaches full recall.

### Benefits and drawbacks of each combination

Against the plain 48-byte `binary` scan with no re-scoring (recall@10 0.669, worst decile 0.41,
nDCG loss 0.024). Memory is per vector: what is scanned in RAM, plus what is fetched for
candidates.

| combination | memory per vector | recall@10 | worst decile | nDCG loss | benefits | drawbacks |
|---|---|---:|---:|---:|---|---|
| `binary`, no re-score | 48 B | 0.669 | 0.41 | 0.024 | Stateless, nothing to fit | Loses a third of the top 10; its asymmetric score is no faster than f32 |
| `pq48`, no re-score | 48 B (+ codebooks) | 0.747 | 0.52 | 0.016 | Best nDCG at 48 B with no re-score | k-means fit per collection; codebooks to store; refit when the corpus drifts |
| `mixed48-pca-w`, no re-score | 48 B (+ a 384 × 384 rotation) | 0.768 | 0.55 | 0.018 | Best recall at 24, 48 and 96 B; PCA fit, no iterative clustering | Ties PQ on bge-small and loses slightly on nDCG; one matrix–vector product per query; refit on drift |
| hybrid, fixed depth, int8 | 48 B RAM + 392 B disk | 0.944–0.983 at depth 50 | 0.84–0.97 | ≈ 0 | Second store 4× smaller than f32 | Levels off at int8's own 0.99; random reads per query |
| hybrid, fixed depth, f32 | 48 B RAM + 1,536 B disk | 0.948–0.991 at depth 50 | 0.86–0.98 | ≈ 0 | Reaches exact; the depth is predictable from noise ÷ band | Keeps a full f32 copy; N random reads per query |
| hybrid, adaptive depth | as above | +0.015 over fixed at equal mean depth | +0.037 | ≈ 0 | Fewer bad queries for the same average work | Variable work per query (p90 1.3–2.1× the mean); σ calibrated per collection; one more parameter |
| adaptive bits + adaptive depth (`mixed48-pca-w`, 1σ) | 48 B RAM + 1,536 B disk | 0.929 at mean depth 19 | | ≈ 0 | The best recall per candidate re-scored of any 48-byte scan | Every drawback above together: two fits, a rotation and variable depth |

## Discussion

**PCA plus bit allocation is a serious alternative to PQ.** It never trails `pq<m>` by more than
a point of recall at equal bytes, and it leads by 3 to 5 points on MiniLM. It fits with one
eigendecomposition instead of k-means, and the query-weighted allocation (`-w`) is worth about
another point at 48 bytes. Its nDCG at 48 bytes is slightly worse than PQ's despite the better
recall, which says the neighbours it recovers are not always the judged ones. That is the same
pattern the labelled SciFact experiment found for centring.

**Why MiniLM gains and bge-small does not.** Both models have a PCA spectrum to exploit, but on
bge-small (measured on SciFact) 60% of an average vector's energy is the shared mean direction.
PQ's codebooks probably absorb that common component for free, while a quantizer working per
principal component has to spend bits on it. The `-w` allocation does spend bits on it, which is the bump of bits on the
last components in the [README animation](../../docs/assets/animations/adaptive-bits.svg).
Centring the rotation, or giving the mean its own exact term, is the obvious next variant.

**Adaptive depth is a small, reliable win, not a large one.** 1.5 points at equal work, larger
for the worst queries, and almost never negative. The [routing experiment](../2026-11-routing-oracle/)
asks how much more a better router could get. A hindsight oracle shows about 6 points of
headroom, but the band a router can see from the scan's own scores tracks the true band only
moderately, so most of that headroom is out of reach.

**Caveats.**
- **Recall is against exact f32, not the answer.** nDCG uses binary relevance. Whether any of
  this reaches a chat model's answer is Experiment 1.
- **No fit or encode times.** The cost of fitting the PCA and the PQ codebooks is not measured
  here; the [zero-copy scan](../2026-11-zero-copy-scan/) experiment times the scans.
- **Parameters not counted in bytes per vector.** The rotation is 590 KB per collection, and
  PQ's codebooks are of similar size, as before.
- **Margin cells with few comparisons.** 3σ cells often exceed the measured fixed range, so
  they're compared on few collections.
- **The same collections as the spacing experiment**, including CoSQA's duplicates and
  bge-small without its query instruction.

[quantmoe]: https://github.com/UNITES-Lab/moe-quantization
