---
title: "SciFact: codecs against the relevance judgments"
date: 2026-10-12 17:30:00 +0000
result: "Binary loses 39 points of agreement with f32 on bge-small but only 4.4 points of nDCG@10 against the human labels; int8 is indistinguishable from f32; centring's gain barely registers."
references: [thakur2021beir, xiao2023bge, reimers2019sbert, charikar2002simhash, jegou2011pq]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-10-scifact-labelled](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-10-scifact-labelled).
Follow-up to [SciFact: mean-centred binary]({{ "/experiments/scifact-centred-binary/" | relative_url }}).

**Hypothesis.** Agreement with the f32 ranking overstates a binary codec's retrieval loss,
because the neighbours it reorders are near-ties, and near-ties are rarely the judged-relevant
abstract. Scored against BEIR's relevance judgments [1], nDCG@10 will fall far less than the
30 to 40 points of top-10 agreement that binary loses.

**Result.** Confirmed. New tooling imports a BEIR dataset as a question set with judgments
(`hvec questions import-beir`) and `bench recall --questions` reports nDCG@k and document
recall@k next to agreement with f32. On the same 8,036 SciFact chunks and 300 claims:

| embedding model | codec | recall@10 vs f32 | nDCG@10 | doc recall@10 | bytes |
|---|---|---:|---:|---:|---:|
| bge-small-en-v1.5 [2] | f32 | 1.000 | **0.698** | **0.818** | 1536 |
| bge-small-en-v1.5 | int8 | 0.987 | 0.698 | 0.818 | 392 |
| bge-small-en-v1.5 | binary | 0.611 | 0.654 | 0.768 | 48 |
| bge-small-en-v1.5 | binary-centred | 0.656 | 0.658 | 0.770 | 48 |
| all-MiniLM-L6-v2 [3] | f32 | 1.000 | **0.627** | **0.778** | 1536 |
| all-MiniLM-L6-v2 | int8 | 0.995 | 0.627 | 0.778 | 392 |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.605 | 0.733 | 48 |
| all-MiniLM-L6-v2 | binary-centred | 0.719 | 0.598 | 0.739 | 48 |

**Reading.** The sign code [4] loses four of every ten f32 neighbours and six percent of the
labelled retrieval quality. int8 loses nothing the labels can see. The centring step that
recovered 4.5 points of f32 agreement recovers 0.4 nDCG points on bge-small and costs 0.7 on
MiniLM: it is a better estimator of cosine, not a better retriever. And bge-small compressed
32× still retrieves better than MiniLM at full precision, so the model axis outweighs the codec
axis; that is the premise of the planned comparison of native low-precision embeddings with
post-hoc quantization at equal bytes.

**Caveats.** One corpus with sparse binary judgments (1.13 per claim); two small 384-dim
models; absolute nDCG a few points under published figures because the corpus is chunked into
200-word windows for the RAG pipeline. No chat model yet: the question set now carries gold
verdicts for 188 claims so that the next experiment can ask whether a 4-point nDCG loss moves
the answer.

{% include references.html %}
