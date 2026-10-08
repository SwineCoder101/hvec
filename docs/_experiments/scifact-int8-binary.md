---
title: "SciFact: int8 and binary codecs against f32, two embedding models"
date: 2026-10-08 20:30:00 +0000
result: "int8 keeps recall@10 at 0.99 at 3.9× smaller; binary keeps 0.61 (bge-small) and 0.71 (MiniLM) at 32× smaller, and widening k does not recover it."
references: [jegou2011pq, charikar2002simhash, gong2013itq, xiao2023bge, thakur2021beir, gao2024rabitq]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-10-scifact-int8-binary](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-10-scifact-int8-binary).

**Hypothesis.** Scalar int8 quantization will be close to lossless for retrieval on real
scientific-abstract embeddings, and binary (sign) quantization will lose a large fraction of
neighbours, by an amount that depends on the embedding model.

**Result.** Confirmed. 8,036 chunks from BEIR SciFact [5], 300 real test queries, cosine,
brute force, no rescoring:

| embedding model | codec | recall@10 | top-1 | mean score error | bytes | ratio |
|---|---|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 [4] | int8 | 0.987 | 0.997 | 0.0005 | 392 | 3.9× |
| bge-small-en-v1.5 | binary | 0.611 | 0.700 | 0.307 | 48 | 32× |
| all-MiniLM-L6-v2 | int8 | 0.995 | 1.000 | 0.0003 | 392 | 3.9× |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.034 | 48 | 32× |

Recall@100 for binary is 0.604 and 0.708: the lost neighbours are reordered far away, not just
past the cutoff.

**Reading.** int8 is close to free on both models. Binary behaves very differently per model in
score error (BGE vectors are not centred, so many sign bits agree across the whole corpus and
discriminate nothing) but only moderately differently in recall. Sign quantization loses rank
information even when scores track well, which is the gap that learned rotations [3] and
error-bounded binary schemes [6] exist to close. The asymmetric sign score is the SimHash
estimator [2]; product quantization [1] is the next codec in line.

**Caveats.** One corpus, two small models, no relevance labels, no chat model. Nothing here
says whether 0.61 recall changes an answer. That is the next experiment.

{% include references.html %}
