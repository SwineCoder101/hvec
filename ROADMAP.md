# Roadmap

Phases, in order. Each phase ends with something runnable and, from phase 3 on, a published
experiment. Dates are deliberately absent; the order is the commitment.

## Phase 1 · Pipeline (done)

Workspace, `Codec` trait with the f32 baseline, Anthropic / OpenAI-compatible / local
connectors, SQLite store scored through the codec, run log, interactive shell, mock-server
integration tests, CI.

## Phase 2 · Codecs and retrieval-only measurement

- `int8` scalar codec (per-vector scale and offset) and `binary` sign codec, both with
  compressed-domain `score` and tests against the f32 reference on random vectors.
- `hvec bench recall --collection <name> --codecs int8,binary`: re-encode the collection's
  vectors with each codec, sample stored chunks as queries, and report recall@k and mean score
  error against f32. No chat model involved, so it is free and fast.
- `hvec ingest --codec` already exists; collections per codec become the unit of comparison.

Exit: a table of recall@10 and score error per codec for one embedding model, in the run log.

## Phase 3 · Question sets and the matrix

- JSONL question sets with gold answers (`questions/<name>.jsonl`): `question`, `answer`,
  optional `source` hints. A small loader and a BEIR-style importer.
- `hvec bench run --questions <set> --collections a,b,c --chat p,q`: one run per cell, with
  exact-match and contains-answer scoring recorded alongside the existing timings and tokens.
- `hvec report`: group the run log by embedding model, codec and chat model; print answer
  accuracy, recall, tokens and latency per cell.

Exit: **Experiment 1**, published on the site. int8 and binary without rescoring versus f32, one
small local embedding model, two chat models of very different size, a few hundred questions.
The claim under test: the chat-model axis matters.

## Phase 4 · Judged metrics

- LLM-as-judge scoring with a cheaper model: correctness against gold, and faithfulness to the
  retrieved passages. Judge prompt versioned like the RAG prompt.
- Agreement check between judge and exact match on the phase 3 data before trusting it.

Exit: Experiment 1 re-scored with the judge; a short post on where exact match and the judge
disagree.

## Phase 5 · More codecs, and collapsing the layers

- Product quantization with trained codebooks, and Matryoshka truncation as a codec.
- **Experiment 2**: native low-precision embeddings versus post-hoc quantization of an f32 model
  at the same byte budget. If native wins, the right codec is a different embedding model.

## Phase 5b · The payload axis

- A `payload` stage between retrieval and the prompt: `none`, a naive truncation baseline, and
  an adapter for an external compressor such as [Headroom](https://github.com/headroomlabs-ai/headroom)
  running as a local proxy. Recorded per run like the codec.
- **Experiment 3**: key compression alone, payload compression alone, and both together, on the
  same questions and chat models. The claim under test: the two layers' effects on answers are
  independent and additive, or they are not.

## Phase 6 · Agent memory and the encryption track

- A memory-loop benchmark: an agent that writes tool results and turns back into the store and
  retrieves them on later steps, measured per step on latency, bytes and answer quality.
- Exploration of similarity on encrypted vectors (CKKS) as a codec with a very different cost
  model. Research track; no promise of a usable result.

## Always

- Negative results are published.
- Every published number has a commit hash, a config and a command next to it.
- Production users should use their database's built-in quantization and an eval harness.
  hvec exists to understand codecs, not to pick a database setting.
