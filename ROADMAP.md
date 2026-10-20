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
  side still scores on the bytes.
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

## Next

- Run Experiments 1 and 3 and the judge re-scoring as soon as two chat models are reachable;
  the scripts need only `ANTHROPIC_API_KEY` (or an Ollama server) and a built `hvec`.
- Codec follow-ups the write-ups flagged: a per-vector scale for `binary-centred`, and a
  rescoring stage (fetch 5-10× candidates by codec, re-rank with int8 or f32).
- A second corpus with graded, denser judgments than SciFact's.

## Always

- Negative results are published.
- Every published number has a commit hash, a config and a command next to it.
- Production users should use their database's built-in quantization and an eval harness.
  hvec exists to understand codecs, not to pick a database setting.
