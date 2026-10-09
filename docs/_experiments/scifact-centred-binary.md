---
title: "SciFact: mean-centred binary against plain binary"
date: 2026-10-09 13:00:00 +0000
result: "Centring cuts bge-small's binary score error six-fold (0.307 to 0.048) but lifts recall@10 only from 0.611 to 0.656; the binary loss is a resolution problem, not a centring problem."
references: [mu2018allbutthetop, charikar2002simhash, gong2013itq, gao2024rabitq, xiao2023bge, thakur2021beir]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-10-scifact-centred-binary](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-10-scifact-centred-binary).
Follow-up to [SciFact: int8 and binary]({{ "/experiments/scifact-int8-binary/" | relative_url }}).

**Hypothesis.** Subtracting the corpus mean before taking sign bits will recover most of the
binary loss on bge-small, whose vectors share a large common component, and change little on
MiniLM, whose vectors are already close to centred.

**Result.** Half right. The new `binary-centred` codec stores `sign(x − μ)` at the same 48 bytes
per vector as `binary`, with the corpus mean μ and one scale stored once per collection. On
8,036 SciFact chunks [6], 300 real queries, cosine, brute force, no rescoring:

| embedding model | codec | recall@10 | top-1 | mean score error | bytes | ratio |
|---|---|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 [5] | binary | 0.611 | 0.700 | 0.307 | 48 | 32× |
| bge-small-en-v1.5 | binary-centred | **0.656** | **0.727** | **0.048** | 48 | 32× |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.034 | 48 | 32× |
| all-MiniLM-L6-v2 | binary-centred | 0.719 | 0.717 | 0.036 | 48 | 32× |

**Reading.** Sixty percent of a bge-small vector's energy lies along the corpus mean (‖μ‖ = 0.78
for unit vectors; 0.34 for MiniLM). That shared component biases the plain sign estimator [2],
and modelling it removes the bias: the score error falls into the range MiniLM was already in,
as the word-vector literature on anisotropy would predict [1]. But a constant bias does not
reorder anything. What reorders neighbours is noise, and a 384-bit sign code estimates cosine
with a noise of 0.016 to 0.035 at the top-10 level, against gaps between consecutive neighbours
of 0.003 to 0.008. Centring halves that noise on bge-small, worth four or five recall points, and
that is all it can be worth. On MiniLM it slightly raises the noise, because the decoded vector
`μ + α·s` no longer has unit norm, and top-1 agreement drops a point.

**Next.** The lever that moves recall is per-bit noise, which points at a rotation before the
sign bits (learned as in ITQ [3], or random as in RaBitQ [4]) and then at rescoring a wider
candidate set with int8 or f32. And the question the series is building towards, with two chat
models: does 0.66 recall change the answer?

{% include references.html %}
