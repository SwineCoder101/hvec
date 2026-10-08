---
layout: page
title: Glossary
permalink: /glossary/
---

The jargon used across hvec, its code and this site, with a note on how each term shows up in
the tool. Papers are linked by key into the [references]({{ "/references/" | relative_url }}).

## Vectors and similarity

**Embedding**
: A fixed-length list of numbers produced by a model from a piece of text (or an image, or audio)
  so that similar inputs land near each other. In hvec every chunk of an ingested document becomes
  one embedding, computed by the collection's embedding model. See [reimers2019sbert]({{ "/references/#reimers2019sbert" | relative_url }}).

**Vector**, **dimension**
: An embedding is a vector; its dimension is how many numbers it has. `bge-small-en-v1.5` produces
  384 dimensions, `text-embedding-3-small` produces 1536. Dimension is recorded on every collection
  because compression behaviour depends on it.

**Embedding model** versus **chat model**
: Two different models in every RAG pipeline. The embedding model turns text into vectors for
  retrieval. The chat model (Claude, GPT, Llama) reads the retrieved text and writes the answer. hvec
  keeps them on separate axes of the benchmark: compression depends on the first, tolerance to
  degraded retrieval depends on the second.

**Metric**
: The function that scores how close two vectors are. hvec supports three: **dot product** (inner
  product, higher is closer), **cosine similarity** (dot product of the unit-length vectors, so only
  the angle matters) and **L2** (squared Euclidean distance, lower is closer). Set in config as
  `metric` and stored per collection.

**Normalisation**
: Scaling a vector to unit length. For normalised vectors dot product and cosine are the same thing,
  and binary quantization works much better. Most sentence-embedding models already emit
  normalised vectors.

**Nearest-neighbour search**, **top-k**
: Finding the `k` stored vectors closest to a query under the metric. `k` is the `--k` flag and the
  `/k` command. **Exact** search scores every stored vector; **approximate** (ANN) search uses an
  index to skip most of them.

**Index**
: A data structure (HNSW graph, inverted file, tree) that makes ANN search fast by not scoring every
  vector. hvec deliberately has none: it scans everything so that any loss it measures comes from the
  codec and not from the index. See [malkov2020hnsw]({{ "/references/#malkov2020hnsw" | relative_url }}).

**Brute-force scan**
: Scoring every stored vector against the query. Exact, simple and what hvec's store does.

## Compression

**Codec**
: Short for coder-decoder: a pair of functions that compress a vector to bytes and reconstruct it.
  In hvec a codec is the `Codec` trait: `encode`, `decode`, and crucially `score`, which computes
  the metric directly on the compressed bytes. Every collection is stored through one codec and
  records its name (`f32`, later `int8`, `binary`, `pq`).

**Embedding as compression**
: An embedding model is itself a lossy codec: it keeps one operation, semantic similarity, and
  discards the rest, including easy readability. It does not save bytes. A 200-word chunk is about
  1 KB of text; its 1536-dimension f32 embedding is 6 KB. What it compresses is the cost of
  comparing two passages, from "read both" to one dot product. Embeddings are more reversible than
  they look ([morris2023vec2text]({{ "/references/#morris2023vec2text" | relative_url }})). hvec's codecs are a second layer applied to
  this first one, preserving the same operation while giving the bytes back.

**Key versus payload**
: In retrieval the **key** is what you search by (the vector) and the **payload** is what you get
  back (the chunk text). hvec compresses the key only. The agent never reads a vector, compressed
  or not; it reads the payload text that the retrieval step returns, uncompressed, into its context
  window. Compressing the payload itself (summarising or pruning passages before the model reads
  them, as in [jiang2023llmlingua]({{ "/references/#jiang2023llmlingua" | relative_url }})) is a separate layer that changes what the agent
  sees rather than what it finds. It is out of scope for hvec today.

**Homomorphic compression**
: A compression scheme is homomorphic with respect to an operation when you can apply the
  operation to the compressed form and get the same result, or a controlled approximation, as on
  the original. For vector search the operation is the metric. Product quantization with asymmetric
  distance is the canonical example. The term is borrowed from homomorphic encryption, where the
  operation is applied to ciphertext.

**Baseline**
: The uncompressed f32 representation. It is exact, so every other codec is measured as a
  deviation from it. hvec's `f32` codec exists for this reason and is the only codec in milestone 1.

**Compressed-domain**, **asymmetric distance computation (ADC)**
: Scoring where the query stays uncompressed and the stored vector stays compressed. Introduced for
  PQ in [jegou2011pq]({{ "/references/#jegou2011pq" | relative_url }}). The `score` method on the codec trait is ADC.
  **Symmetric** distance compresses both sides and is cheaper but lossier.

**Quantization**
: Mapping continuous values onto a finite set of levels. Every lossy vector codec is a form of it.
  The classic treatment is [gray1984vq]({{ "/references/#gray1984vq" | relative_url }}).

**Scalar quantization**, **int8**
: Quantizing each dimension independently, usually to 8 bits with a per-vector or per-dimension
  scale. About 4× smaller than f32 with small error. The first codec hvec will add.

**Binary quantization**, **1-bit**
: Keeping only the sign of each dimension. 32× smaller than f32. Hamming distance between the bit
  strings estimates the angle between normalised vectors, per
  [charikar2002simhash]({{ "/references/#charikar2002simhash" | relative_url }}). Large error on its own, often used as a first pass
  before re-ranking. [gao2024rabitq]({{ "/references/#gao2024rabitq" | relative_url }}) gives a version with error bounds.

**Product quantization (PQ)**
: Split the vector into `m` sub-vectors, cluster each subspace into 256 centroids with k-means, and
  store one byte per sub-vector. Distances come from `m` table lookups. 8× to 64× compression with
  tunable error. See [jegou2011pq]({{ "/references/#jegou2011pq" | relative_url }}) and its optimised form [ge2014opq]({{ "/references/#ge2014opq" | relative_url }}).

**Codebook**, **centroid**
: A codebook is the table of representative vectors a quantizer maps onto; each entry is a centroid.
  PQ has one codebook per subspace. Codebooks are trained with k-means ([lloyd1982kmeans]({{ "/references/#lloyd1982kmeans" | relative_url }})).

**Sketch**, **random projection**
: Multiplying by a random matrix to reduce dimension while preserving distances within a provable
  bound ([johnson1984jl]({{ "/references/#johnson1984jl" | relative_url }})). A homomorphic codec for L2 with theory attached.

**Matryoshka embeddings**
: Embeddings trained so that the first `n` dimensions work on their own. Truncation becomes a
  nearly free codec. See [kusupati2022mrl]({{ "/references/#kusupati2022mrl" | relative_url }}).

**Compression ratio**
: Bytes of the uncompressed f32 vector divided by bytes of the encoded one. The `Codec` trait
  reports it; f32 is 1.0, int8 about 4, binary 32.

**Homomorphic encryption (HE, FHE)**, **CKKS**, **ciphertext**
: Encryption schemes where arithmetic on ciphertext corresponds to arithmetic on the plaintext.
  Fully homomorphic encryption supports arbitrary computation ([gentry2009fhe]({{ "/references/#gentry2009fhe" | relative_url }}));
  CKKS supports approximate real-number arithmetic, which fits inner products on embeddings
  ([cheon2017ckks]({{ "/references/#cheon2017ckks" | relative_url }})). An exploration track in hvec, not part of milestone 1.

## Retrieval-augmented generation

**RAG**
: Retrieval-augmented generation. Retrieve relevant passages for a question, put them in the
  prompt, and let a chat model answer from them. Named in [lewis2020rag]({{ "/references/#lewis2020rag" | relative_url }}).
  hvec's `query` and shell turns are a RAG pipeline with the codec in the retrieval step.

**Chunk**, **chunking**
: A passage of a document small enough to embed and retrieve on its own. hvec splits text into
  overlapping windows of words (`--chunk-words`, `--overlap-words`). The chunker is part of the
  pipeline under test, so it is deterministic and recorded.

**Collection**
: A named set of chunks embedded with one embedding model and stored through one codec. Created by
  `ingest` or `/new`. A collection refuses to mix embedding models.

**Context**, **passages**
: The retrieved chunks placed in the prompt. `--show-context` and `/context` print them.

**Prompt version**
: An identifier for the fixed RAG prompt template. Changing the prompt changes every result, so the
  version is recorded on each run.

**Session**
: A persisted conversation. Each shell launch creates one; `-s name` resumes it. Earlier turns are
  replayed as plain history while each new question retrieves fresh context.

**Profile**
: A named model configuration in `config.toml`: provider, model id, endpoint, key variable. Chat
  profiles and embedding profiles are separate. `--chat`, `--embedder` and `/model` select them.

**Token**
: The unit a chat model reads and writes, roughly three quarters of a word. Input and output token
  counts are recorded per turn because they are the cost.

## Measurement

**Run**, **run log**
: One recorded pipeline execution: setup (collection, embedding model, dimension, codec, metric,
  chat model) plus outcome (timings, tokens, retrieved chunk ids and scores, question, answer).
  Stored in SQLite; inspect with `runs list` and `runs show`.

**KPI**
: Key performance indicator. The metrics hvec collects to rank codecs: retrieval-side (recall,
  distance error, bytes per vector, latency) and generation-side (answer correctness, faithfulness,
  tokens, latency).

**Recall@k**
: Of the `k` nearest neighbours under exact f32 search, the fraction that a codec's top-`k` also
  contains. 1.0 means the compressed ranking agrees perfectly with the baseline. The standard
  retrieval metric, used by [aumuller2020annb]({{ "/references/#aumuller2020annb" | relative_url }}).

**Distance error**
: The difference between a codec's estimated score and the exact f32 score for the same pair.
  hvec reports the mean absolute error.

**Answer correctness**
: Whether the chat model's answer matches a known gold answer, by exact match or by a judge. The
  generation-side headline KPI.

**Faithfulness**
: Whether the answer is supported by the retrieved passages rather than invented. Defined as an
  LLM-scored metric in [es2023ragas]({{ "/references/#es2023ragas" | relative_url }}).

**LLM-as-judge**
: Using a language model to grade another model's answers. Scales better than humans and has known
  biases; see [zheng2023judge]({{ "/references/#zheng2023judge" | relative_url }}).

**Benchmark matrix**
: The grid hvec runs: embedding model × codec × chat model × dataset. One run per cell.

**Experiment**
: A directory under `experiments/` with a hypothesis, a reproducible command, raw results and a
  write-up. Negative results are kept. Finished write-ups are published on this site.

**Gold answers**, **dataset**
: A question set with known correct answers, used to score answer correctness. BEIR
  ([thakur2021beir]({{ "/references/#thakur2021beir" | relative_url }})) is a likely source.

## Tooling

**ONNX**, **ONNX Runtime**, **fastembed**
: ONNX is a portable model format; ONNX Runtime executes it; fastembed is the Rust crate that wraps
  both to run embedding models locally. hvec's `local` embedding provider uses it so benchmarks are
  free and offline.

**OpenAI-compatible**
: Any server that speaks OpenAI's `/chat/completions` and `/embeddings` HTTP protocol: OpenAI
  itself, Ollama, vLLM, LM Studio. One hvec provider covers all of them.

**SQLite**
: The single-file database hvec stores everything in: chunks and their encoded vectors, sessions,
  and the run log. One file means a run, its context and its conversation can be joined later.

Missing a term? Open an issue or edit
[`docs/glossary.md`](https://github.com/SwineCoder101/hvec/blob/main/docs/glossary.md).
