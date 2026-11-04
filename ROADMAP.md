# Roadmap

Phases, in order. Each phase ends with something runnable and, from phase 3 on, a published
experiment. Dates are deliberately absent; the order is the commitment.

## Phase 1 · Pipeline (done)

Workspace, `Codec` trait with the f32 baseline, Anthropic / OpenAI-compatible / local
connectors, SQLite store scored through the codec, run log, interactive shell, mock-server
integration tests, CI.

## Phase 2 · Codecs and retrieval-only measurement (done)

- [x] `int8` scalar codec (per-vector scale and offset) and `binary` sign codec, both with
  compressed-domain `score` and tests against the f32 reference on random vectors.
- [x] `hvec bench recall --collection <name> --codecs int8,binary`: re-encode the collection's
  vectors with each codec, rank self-queries or a query file, and report recall@k, top-1
  agreement and score error against f32. No chat model involved, so it is free and fast.
- [x] `hvec ingest --codec` and per-codec collections as the unit of comparison.
- [x] Run it on a real embedding model and corpus and publish the first table:
  [SciFact, int8 and binary](experiments/2026-10-scifact-int8-binary/).
- [x] Trained codecs: `fit` and `params` on the `Codec` trait, parameters stored per collection,
  fitted on first ingest and by `bench recall`. First instance: `binary-centred`, sign bits of
  the mean-centred vector. Measured in
  [SciFact, mean-centred binary](experiments/2026-10-scifact-centred-binary/).

Exit: a table of recall@10 and score error per codec for one embedding model, in the run log.

## Phase 3 · Question sets and the matrix (tooling done; Experiment 1 waiting on API access)

- [x] JSONL question sets with gold answers (`questions/<name>.jsonl`): `question`, `answers`,
  optional `source` hint. Loader with validation.
- [x] `hvec bench run --questions <set> --collections a,b,c --chat p,q`: one run per cell, with
  exact and contains scoring and source-hit recorded alongside timings and tokens. Errors are
  recorded and skipped; `--fail-fast` to abort.
- [x] `hvec report`: group `bench` rows by question set, collection, embedding model, codec,
  payload and chat model; accuracy, retrieval hit rate, nDCG, tokens and latency per cell;
  `--batch`, `--set`, `--json`.
- [x] A BEIR-style importer for question sets with relevance judgments:
  `hvec questions import-beir`, the `relevant` and `query` fields, nDCG@k and document recall@k
  in both `bench recall --questions` and `bench run`. The SciFact claim labels give 188 of the
  300 test claims a supports/refutes gold answer.
- [x] Measure the existing codecs against the judgments rather than against f32:
  [SciFact, codecs against the relevance judgments](experiments/2026-10-scifact-labelled/).
  Binary loses 39 points of f32 agreement but 4 points of nDCG@10.
- [ ] **Experiment 1** (needs API access to two chat models; the script is ready in
  [experiments/2026-10-scifact-rag-two-models](experiments/2026-10-scifact-rag-two-models/)).

Exit: **Experiment 1**, published on the site. int8 and binary without rescoring versus f32, one
small local embedding model, two chat models of very different size, a few hundred questions.
The claim under test: the chat-model axis matters.

## Phase 4 · Judged metrics (tooling done; re-scoring waits on Experiment 1)

- [x] LLM-as-judge scoring with a cheaper model: `hvec judge run --batch <id> --judge <profile>`
  grades correctness against gold and faithfulness to the retrieved passages. Judge prompt
  versioned like the RAG prompt; verdicts are `judge` rows that `report` joins into cells.
- [x] Agreement check between judge and exact / contains: `hvec judge agreement`, counting the
  paraphrases the rule missed and the substrings it wrongly accepted.
- [ ] Experiment 1 re-scored with the judge; a short post on where exact match and the judge
  disagree.

Exit: Experiment 1 re-scored with the judge; a short post on where exact match and the judge
disagree.

## Phase 5 · More codecs, and collapsing the layers (done)

- [x] Product quantization with trained codebooks (`pq<m>`), and Matryoshka truncation as a codec
  (`mrl<k>`), composable with any other codec (`mrl<k>+int8`, `mrl<k>+pq<m>`). Embedding
  profiles take task prefixes for models that need them.
- [x] **Experiment 2**: native low-precision embeddings versus post-hoc quantization of an f32
  model at the same byte budget, against the SciFact judgments:
  [SciFact, native truncation against post-hoc quantization](experiments/2026-10-scifact-native-vs-posthoc/).

## Phase 5b · The payload axis (tooling done; Experiment 3 waiting on API access)

- [x] A `payload` stage between retrieval and the prompt: `none`, a naive truncation baseline
  (`truncate:<words>`), and an adapter for an external compressor such as
  [Headroom](https://github.com/headroomlabs-ai/headroom) running as a local service
  (`http:<url>`). Recorded per run like the codec, with the context size before and after.
- [ ] **Experiment 3**: key compression alone, payload compression alone, and both together, on
  the same questions and chat models. The claim under test: the two layers' effects on answers
  are independent and additive, or they are not. Script ready in
  [experiments/2026-10-scifact-key-and-payload](experiments/2026-10-scifact-key-and-payload/).

## Phase 6 · Agent memory and the encryption track (tooling done; research note published)

- [x] A memory-loop benchmark: `hvec bench memory --script <jsonl>` replays an agent that writes
  tool results and turns into the store and retrieves them on later steps, measured per step on
  latency, bytes and answer quality, once per codec. A sample script is in
  `questions/memory-smoke.jsonl`; a real run needs a chat model.
- [x] Exploration of similarity on encrypted vectors (CKKS) as a codec with a very different cost
  model: [a research note](docs/_posts/2026-10-12-encrypted-similarity-cost-model.md) with the
  numbers. No code; the note says why not yet.

## Phase 7 · Learned rotation before quantization

The binary write-ups blamed the lost recall on geometry, not on bit depth: BGE vectors share a
large common component and their variance is spread unevenly over the axes, so one sign bit per
dimension wastes most of its resolution. Centring removed the common component and recovered
only four or five points. A rotation applied after embedding and before the codec spreads the
variance evenly over the axes, which is what a sign bit or a PQ subspace needs. This is the
cheap alternative to training a codec-robust embedding model: the model stays fixed, and the
rotation is a `d × d` matrix fitted on the corpus.

- [ ] A `rot` stage that composes like `mrl<k>` does: `rot+binary`, `rot+binary-centred`,
  `rot+pq48`, `mrl256+rot+binary`. It is a trained codec in the existing sense: `fit` learns the
  rotation on the first ingest, `params` stores the matrix (or the seed) with the collection,
  and `score` rotates the f32 query once per search before the asymmetric scan, so the stored
  side still scores on the bytes. The hook for that now exists (`Codec::prepare_query` and
  `score_prepared`, added in phase 8), and `mixed<B>-pca` already carries a PCA fit that
  `rot:itq` can start from.
- [ ] Two fits behind the same name. `rot` with a seed is a random orthogonal rotation
  (RaBitQ-style, a Haar-random matrix from the QR of a Gaussian), which needs no corpus pass and
  gives a baseline for how much the learned one buys. `rot:itq` is Iterative Quantization: PCA
  to the target dimension, then alternate between the sign bits and the orthogonal Procrustes
  solution until the quantization error stops falling.
- [ ] **Experiment 4**: `binary`, `binary-centred`, `rot+binary` and `rot:itq+binary` at 48
  bytes, and `pq48` against `rot:itq+pq48`, on SciFact against the judgments with the f32
  ceiling in the same table. The claim under test: a rotation closes most of the gap between
  post-hoc binary and the f32 ceiling without touching the embedding model. The number to beat
  is binary's 4.4 points of nDCG@10. If it does not, the write-up says how far short, and
  whether training the embedder for the codec (Matryoshka-style, with the sign function in the
  loss) is the only way left.
- [ ] Symmetric scoring for the rotated binary codec: binarise the rotated query too and score
  by Hamming distance with popcount. This is the first codec path where a speed claim is
  possible, so the write-up has to include scan time at equal bytes against the asymmetric path.

Exit: Experiment 4 published, and a stated answer to whether rotation makes binary usable on
this embedding model.

## Phase 8 · Spacing, hybrid and adaptive quantization, routing, zero-copy (done; follow-ups open)

Every codec number before this phase came from one corpus, so none of them said what a codec
would cost elsewhere. This phase finds the quantity that sets the cost, then uses it twice: to
decide how deep a hybrid re-score must go, and to decide where an adaptive code spends its bits.

- [x] Neighbour spacing in `bench recall`: `neighbour_band` (exact score gap from rank k to 2k),
  `boundary_gap` (k to k + 1) and `rank_noise` (the codec's score-error spread over the exact
  top 2k, per-query bias removed).
- [x] **Neighbour-spacing experiment** (`experiments/2026-11-neighbour-spacing`): SciFact,
  NFCorpus, ArguAna, FiQA and CoSQA (code, via CoIR) × bge-small and MiniLM × nine codecs.
  Noise ÷ band ranks recall@10 loss with ρ = 0.98 over 90 points (noise alone 0.64) and predicts
  a held-out corpus within 0.024. Content type matters only through spacing, and the embedding
  model moves spacing as much as the corpus does. Published with a site page and a post.
- [x] `hvec bench rescore`: scan with one codec, re-score candidates with another (int8 or f32),
  and report recall, worst-decile recall, nDCG, depth and bytes read per query. It also reports
  the exact depth an f32 re-score needs for 90/95/99/100% recall.
- [x] **Two-stage re-scoring experiment** (`experiments/2026-11-two-stage-rescoring`): the
  depth needed tracks the same ratio (ρ ≈ 0.93; depth ≈ 4.8·k·r^1.29 for 95% recall). At 48
  bytes, 50 f32 candidates recover 0.95 (`binary`) to 0.99 (`pq48`) of the top 10 and nearly all
  of the nDCG loss, for 3–4% of an f32 scan's bytes. An int8 second stage stops at int8's own
  0.990.
- [x] Adaptive depth: a `--margins` policy that re-scores every candidate within λσ of the
  scan's own 10th score, with σ calibrated on self-queries only.
- [x] Adaptive bits: `mixed<B>[-pca][-w]`, a fixed byte budget with bits allocated per
  dimension by greedy reverse water-filling and a Gaussian-optimal quantizer step. `-pca`
  rotates into principal components first; `-w` weights each component by query energy, which
  targets score error rather than reconstruction error. On SciFact at 48 bytes with no
  re-score: `binary` 0.611, `mixed48` 0.653, `mixed48-pca` 0.694, `mixed48-pca-w` 0.703, `pq48`
  0.711.
- [x] `Codec::prepare_query` / `score_prepared`, so transformed-space codecs rotate the query
  once per search; used by the store's search, `bench recall` and `bench rescore`.
- [x] Animated README figures for hybrid re-scoring, adaptive depth and adaptive bits
  (`docs/assets/animations/`, regenerated by `make_animations.py`).
- [x] **Hybrid × adaptive matrix** (`experiments/2026-11-hybrid-adaptive-matrix`): 11 scan codecs
  × none/fixed/margin × int8/f32 over ten collections. `mixed48-pca-w` beats `pq48` at equal
  bytes on average (0.768 vs 0.747 recall@10; the gain is all on MiniLM, a tie on bge-small), and
  leads at 24 and 96 bytes too. Adaptive depth gains 0.3–0.6 points of recall at equal mean
  depth, and up to 4 points for the worst decile. (The first version said 1.5 points; that came
  from interpolating between sparse fixed depths, and is corrected in the write-up.)
- [x] Per-query traces (`bench rescore --trace`) and the **routing experiment**
  (`experiments/2026-11-routing-oracle`). A hindsight oracle has 5.5–7 points of headroom over
  fixed depth. The band visible in the scan's own scores tracks the true band only at Spearman
  0.52–0.70, and no router built on it captures more than 6%. Per-collection routing picks the
  right codec size in 10/10 collections, but its predicted depth misses a 95% target by up to
  9 points.
- [x] **Zero-copy** (`experiments/2026-11-zero-copy-scan`). Borrowed codes in the `Codec` trait,
  `CodeBuffer` (a contiguous, cached per-collection buffer scanned in place, optionally over
  threads), in-place f32 and single-pass cosine kernels, and `binary-hamming` (popcount). The
  cached buffer is 5.1× faster than the original row scan. Every asymmetric codec costs about
  what f32 costs per vector. `binary-hamming` is 20× cheaper, and with an f32 re-score it reaches
  95% recall@10 on FiQA about 19× faster than exact f32: the first speed claim hvec makes.
- [ ] Choose the codec automatically at ingest from noise ÷ band, then measure the re-score
  depth for a target recall on a sample of self-queries. The routing experiment showed that the
  curve chooses the codec well but predicts the depth badly.
- [ ] Layered (residual) codes: one stored code read to different depths per candidate, so
  hybrid retrieval needs no second full-precision copy on disk.
- [ ] A faster `mixed<B>` scan (byte-aligned groups, or a lookup table per group as PQ uses):
  today it is the best 48-byte code for recall and the slowest to scan, 2.2× f32 per vector.
- [ ] Time the hybrid end to end, with the f32 candidates fetched from disk, not memory.
- [ ] SIMD kernels for f32 and int8, so the asymmetric codecs are compared at their best.
- [ ] Decide whether to allow one reviewed `unsafe` call for memory-mapped code files
  (`unsafe_code` is forbidden workspace-wide today). A flat file already loads 110 MB in 9 ms, so
  the case is memory footprint and start-up on much larger collections.
- [ ] Per-vector adaptive bits, the analogue of MoE PTQ's outlier scorer: more bits for vectors
  in crowded neighbourhoods. Needs a variable-size encoding or two tiers.
- [ ] Per-cluster PCA (Sub-MoE's grouped subspaces), if `mixed<B>-pca` gets within reach of
  `pq<m>` at equal bytes.
- [ ] Spacing follow-ups the write-up flagged: a deduplicated code corpus (CoSQA is 70%
  duplicates), FiQA subsampled to 8,000 chunks to separate corpus size from spacing, a third
  embedding model, and a tail statistic for the aggressive codecs the curve misses.

## Ideas from MoE compression

Weight compression for Mixture-of-Experts models is working on a problem shaped like this one:
many similar objects (experts, vectors) sharing structure, where some matter more than others.
Three lines of work, and what each would mean as an hvec codec or experiment:

- **MoE post-training quantization** (QuantMoE-Bench, UNITES-Lab; arXiv:2406.08155,
  [code](https://github.com/UNITES-Lab/moe-quantization)). Fine-grained, structure-aware mixed
  precision: blocks, experts and linear layers get different bit widths, chosen by importance
  scorers (expert usage frequency, a linear-weight outlier score) tuned to MoE's sparse
  activation. *In hvec:* `mixed<B>` allocates a byte budget's bits per dimension by variance,
  and `-w` weights that variance by how much energy queries put into the dimension, the analogue
  of weighting an expert by how often it is routed to. The per-vector analogue of an outlier
  scorer, giving more bits to vectors in crowded neighbourhoods, is not built yet.
- **MoE-SVD** (citation not yet verified; we have only the description so far). Training-free:
  SVD decomposes each expert's weights into low-rank factors, the V matrices are shared across
  experts, and the U matrices are trimmed to their top-k columns. *In hvec:* PCA is the SVD of
  the centred corpus. `mixed<B>-pca` with zero bits on the trailing components is a low-rank
  codec, and its single rotation per collection is the shared V. Top-k U trimming corresponds to
  truncating components, which is `mrl<k>` applied after the rotation instead of on raw
  dimensions.
- **Sub-MoE, subspace expert merging** (arXiv:2506.23266,
  [code](https://github.com/siruihan2024/Sub-MoE)). Experts are clustered by cosine similarity,
  each group gets a joint SVD, and the V matrices are merged by expert frequency to resolve
  parameter conflicts. *In hvec:* one PCA per k-means cluster instead of one per collection
  (closest to locally optimised PQ), with each cluster's bit allocation weighted by how often
  queries land in it. This is the natural next codec if `mixed<B>-pca` beats `pq<m>` at equal
  bytes.

## Next

- Run Experiments 1 and 3 and the judge re-scoring as soon as two chat models are reachable;
  the scripts need only `ANTHROPIC_API_KEY` (or an Ollama server) and a built `hvec`.
- Phase 8 follow-ups, starting with choosing the codec at ingest and layered codes.
- Codec follow-ups the write-ups flagged: a per-vector scale for `binary-centred`. The
  re-scoring stage is done (phase 8).
- Graded relevance: NFCorpus ships graded judgments, but hvec's nDCG treats relevance as
  binary. Use the grades, then re-score the five-corpus results.

## Always

- Negative results are published.
- Every published number has a commit hash, a config and a command next to it.
- Production users should use their database's built-in quantization and an eval harness.
  hvec exists to understand codecs, not to pick a database setting.
