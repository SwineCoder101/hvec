# SciFact: mean-centred binary against plain binary, two embedding models

Follow-up to [SciFact: int8 and binary](../2026-10-scifact-int8-binary/), which found that
binary (sign) quantization keeps 0.61 of the exact top-10 on `bge-small` and 0.71 on MiniLM, and
blamed part of the bge-small loss on BGE vectors sharing a large common component.

**Hypothesis.** Subtracting the corpus mean before taking sign bits will recover most of the
binary loss on bge-small, whose vectors share a large common component, and change little on
MiniLM, whose vectors are already close to centred.

**Result.** Half right. Centring fixes the *scores*: bge-small's mean score error falls six-fold,
from 0.307 to 0.048, into the range MiniLM was already in. It barely moves the *ranking*:
recall@10 rises from 0.611 to 0.656 on bge-small and from 0.705 to 0.719 on MiniLM, and MiniLM's
top-1 agreement drops a point. The binary loss is a resolution problem, not a centring problem:
neighbours in the top 10 differ in cosine by about 0.003 to 0.008, and a 384-bit sign code
estimates cosine with a noise of 0.016 to 0.035, centred or not.

## Setup

Identical to the first experiment except for the codec list and the code version.

| | |
|---|---|
| Corpus | BEIR SciFact, 5,183 scientific abstracts (title + text): 8,036 chunks of 200 words, no overlap |
| Queries | The 300 SciFact test claims, embedded with the collection's model; also 300 self-queries |
| Embedding models | `bge-small-en-v1.5` (Xenova ONNX, 384 dims) and `all-MiniLM-L6-v2` (Qdrant ONNX, 384 dims), fastembed on CPU |
| Metric | cosine |
| k | 10, and 100 |
| Codecs | `f32` baseline, `int8`, `binary`, `binary-centred` |
| Fitting | `binary-centred` is fitted on all 8,036 corpus vectors, which is what `ingest --codec binary-centred` does. The mean is unsupervised, so there is no held-out split |
| Search | brute force, no index, no rescoring |
| Hardware | Apple M4 Pro, 14 cores, release build |
| Code | hvec commit `1d4a4e4` |
| Command | `./run.sh` in this directory (downloads data and models on first run; reuses the first experiment's converter) |

### What `binary-centred` is

Per vector it stores `sign(x − μ)`, one bit per dimension, 48 bytes at 384 dims: the same size
as `binary`. Once per collection it stores `μ`, the corpus mean, and one scale `α`, the mean
`|x_i − μ_i|` over the corpus. The decoded vector is `μ + α·s` with `s ∈ {−1, +1}^d`, and the
score is the metric between the query and that decoded vector, computed in one pass over the
query, `μ` and the bits without materialising it.

### Corpus statistics

Computed from the stored f32 vectors.

| | bge-small-en-v1.5 | all-MiniLM-L6-v2 |
|---|---:|---:|
| ‖μ‖ (vectors are unit length) | 0.777 | 0.340 |
| share of energy along the mean, ‖μ‖² / mean ‖x‖² | 0.604 | 0.115 |
| mean cosine between two random chunks | 0.603 | 0.113 |
| mean ‖x − μ‖ | 0.627 | 0.939 |
| fitted scale α | 0.0255 | 0.0381 |
| dimensions with the same sign in more than 95% of the corpus | 20 of 384 | 2 of 384 |
| the same, after centring | 0 | 0 |
| median cosine gap between consecutive exact neighbours in the top 10 | 0.0034 | 0.0082 |
| median cosine of the 1st / 10th exact neighbour | 0.867 / 0.782 | 0.713 / 0.528 |

Sixty percent of a bge-small vector is the same vector as every other bge-small vector. For
MiniLM it is eleven percent.

## Results

300 real queries, k = 10. Score error is |codec score − exact cosine| over every (query, chunk)
pair. Scan time is for 300 queries against all 8,036 chunks.

| embedding model | codec | recall@10 | top-1 | mean score error | max score error | bytes | ratio | scan ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | f32 | 1.000 | 1.000 | 0 | 0 | 1536 | 1.0× | 1203 |
| bge-small-en-v1.5 | int8 | 0.987 | 0.997 | 0.00049 | 0.0034 | 392 | 3.9× | 538 |
| bge-small-en-v1.5 | binary | 0.611 | 0.700 | 0.307 | 0.458 | 48 | 32× | 310 |
| bge-small-en-v1.5 | **binary-centred** | **0.656** | **0.727** | **0.048** | **0.285** | 48 | 32× | 600 |
| all-MiniLM-L6-v2 | f32 | 1.000 | 1.000 | 0 | 0 | 1536 | 1.0× | 1200 |
| all-MiniLM-L6-v2 | int8 | 0.995 | 1.000 | 0.00027 | 0.0019 | 392 | 3.9× | 517 |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.034 | 0.213 | 48 | 32× | 301 |
| all-MiniLM-L6-v2 | **binary-centred** | **0.719** | **0.717** | **0.036** | **0.237** | 48 | 32× | 600 |

Other query sets, `binary` → `binary-centred`:

| | bge-small recall / top-1 | MiniLM recall / top-1 |
|---|---|---|
| real queries, k = 100 | 0.604 → 0.653 / 0.700 → 0.727 | 0.708 → 0.727 / 0.730 → 0.717 |
| self-queries, k = 10 | 0.637 → 0.687 / 0.690 → 0.750 | 0.720 → 0.742 / 0.743 → 0.780 |

Estimator noise, measured directly: the standard deviation of (codec score − exact cosine) over
the 300 self-queries' exact top-10 pairs, which is where ranking is decided.

| | bge-small | MiniLM |
|---|---:|---:|
| `binary` | 0.025 | 0.032 |
| `binary-centred` | 0.016 | 0.035 |
| median gap between consecutive neighbours | 0.003 | 0.008 |

Raw output for every row is in `results/`, one JSON file per (model, query source, k).

## Discussion

**Centring does exactly what the theory says it does to the scores.** On bge-small the plain
sign estimator is biased by the shared component: it treats every decoded vector as a unit vector
of independent signs, when sixty percent of the real vector is a constant that all chunks share.
Modelling that constant explicitly, as `μ + α·s`, removes the bias and the mean error drops from
0.307 to 0.048. On MiniLM there was little bias to remove and the error is unchanged. The
anisotropy of embedding spaces and the fix of subtracting the mean are well known from word
vectors [mu2018allbutthetop]; the surprise here is only how large ‖μ‖ is for a modern retrieval
model.

**Centring does very little to the ranking, because the ranking was never limited by the
bias.** A constant offset in every score does not change the order. What changes the order is
noise, and the noise of a 384-bit sign code at the top-10 level is 0.016 to 0.035 in cosine,
against neighbour gaps of 0.003 to 0.008. Centring halves the noise on bge-small, which is worth
four or five recall points, and that is all it can be worth. Only 20 of 384 dimensions had
near-constant signs before centring; the other 364 bits were already carrying information, just
not enough of it. The first experiment's reading that "BGE vectors are not centred, so many sign
bits agree and discriminate nothing" was true of the scores and wrong about the recall.

**On MiniLM, centring costs a little.** Top-1 agreement goes from 0.730 to 0.717 and the noise
goes up from 0.032 to 0.035. Plain `binary` decodes to exactly unit-length vectors, so its cosine
estimate has no per-vector norm term. `binary-centred` decodes to `μ + α·s`, whose norm varies
with `μ·s`, and when ‖μ‖ is small that variation is noise rather than signal. A per-vector scale
(4 more bytes, 52 instead of 48) or a norm-free ranking for corpora known to be unit length would
remove it; both are one-line variants worth measuring.

**Scan time doubled, for a boring reason.** `binary-centred` does two more multiply-adds per
dimension than `binary` (`q·μ` and `μ·s`). The first is constant per query and the second could
be stored per vector at encode time. The implementation is deliberately the naive one; the
600 ms is not a property of the codec.

**What is wrong with this experiment.** Everything that was wrong with the first one: one
corpus, two small 384-dimension models, agreement with f32 rather than relevance labels, no chat
model. In addition, the mean is fitted on the same vectors it is used to rank. For an
unsupervised mean over 8,036 vectors this changes nothing measurable, and it is what `ingest`
does, but a split would be the right thing for any codec with more parameters.

**Next.** The lever that moves recall is not the offset but the per-bit noise, so the next codec
is a rotation before the sign bits: a learned one as in ITQ [gong2013itq] or a random one as in
RaBitQ [gao2024rabitq], which spreads the variance evenly across bits so no bit is wasted on a
low-variance dimension. After that, rescoring: fetch 5–10× candidates by bits and re-rank them
with int8 or f32, which is how 1-bit codes are used in production and which this series has
deliberately left out so far. And the question this whole series is building towards, with
`bench run` and two chat models: does 0.66 recall change the answer?
