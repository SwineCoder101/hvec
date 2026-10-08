# SciFact: int8 and binary codecs against f32, two embedding models

**Hypothesis.** Scalar int8 quantization will be close to lossless for retrieval on real
scientific-abstract embeddings, and binary (sign) quantization will lose a large fraction of
neighbours. The size of the binary loss will depend on the embedding model, not only on the
codec, because sign bits discard more information when the model's vectors are not centred.

**Result.** Confirmed on both counts. int8 keeps recall@10 at 0.987 (bge-small) and 0.995
(MiniLM) at 3.9× smaller, with mean score error below 0.0005. Binary keeps 0.611 (bge-small) and
0.705 (MiniLM) at 32× smaller. The two models differ by an order of magnitude in binary score
error (0.31 versus 0.03) but by only ten points of recall, and widening k to 100 does not recover
the lost neighbours.

## Setup

| | |
|---|---|
| Corpus | BEIR SciFact, 5,183 scientific abstracts (title + text), from the BEIR release |
| Chunking | 200 words, no overlap, one or two chunks per abstract: 8,036 chunks |
| Queries | The 300 SciFact test claims (those with relevance judgments), embedded with the same model as the corpus; also 300 self-queries (stored vectors, each excluding itself) |
| Embedding models | `bge-small-en-v1.5` (Xenova ONNX, 384 dims) and `all-MiniLM-L6-v2` (Qdrant ONNX, 384 dims), both via fastembed on CPU |
| Metric | cosine |
| k | 10 (and 100 for one check) |
| Codecs | `f32` baseline, `int8` (per-vector affine, 392 bytes), `binary` (sign bits, 48 bytes) |
| Search | brute force over all 8,036 chunks, no index, no rescoring |
| Hardware | Apple M4 Pro, 14 cores, release build |
| Code | hvec commit `2b048f2` |
| Command | `./run.sh` in this directory (downloads the data and models on first run) |

Relevance judgments are not used here; this experiment measures agreement with the exact f32
ranking, not retrieval quality against human labels. That is the phase 3 question.

## Results

300 real queries, k = 10. Score error is |codec score − exact cosine| over every (query, chunk) pair.

| embedding model | codec | recall@10 | top-1 agreement | mean score error | max score error | bytes/vector | ratio | scan ms (300 × 8,036) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | f32 | 1.000 | 1.000 | 0 | 0 | 1536 | 1.0× | 1201 |
| bge-small-en-v1.5 | int8 | 0.987 | 0.997 | 0.00049 | 0.0034 | 392 | 3.9× | 600 |
| bge-small-en-v1.5 | binary | 0.611 | 0.700 | 0.307 | 0.458 | 48 | 32× | 589 |
| all-MiniLM-L6-v2 | f32 | 1.000 | 1.000 | 0 | 0 | 1536 | 1.0× | 1200 |
| all-MiniLM-L6-v2 | int8 | 0.995 | 1.000 | 0.00027 | 0.0019 | 392 | 3.9× | 600 |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.034 | 0.213 | 48 | 32× | 593 |

Self-queries (300 stored vectors, each excluding itself) give the same picture:
int8 0.983 / 0.998 and binary 0.637 / 0.720 for bge-small / MiniLM.

k = 100 on the real queries: int8 0.988 / 0.997, binary 0.604 / 0.708. The binary loss is not a
tail effect that a larger candidate set repairs.

Raw output for every row is in `results/`, one JSON file per (model, query source, k).

## Discussion

**int8 is close to free.** On both models the per-vector affine quantizer loses about one
neighbour in a hundred at k=10 and never moves the top-1 on MiniLM. The score error is three
orders of magnitude below the typical gap between neighbours. If a later experiment shows int8
changing answers, the cause will be somewhere other than the codec.

**Binary loses a third of the neighbours, and the model matters more for scores than for
ranking.** The bge-small binary score error (0.31 mean) is ten times the MiniLM error (0.034).
BGE vectors carry a strong shared component, so many dimensions have the same sign across the
whole corpus and contribute nothing to discrimination; MiniLM vectors are closer to centred.
Yet recall differs by only ten points. Sign bits lose rank information even when the scores
track well, because within the top-10 neighbours cosine gaps are often smaller than the
resolution one bit per dimension provides.

**Widening k does not help.** Recall@100 is the same as recall@10 for binary. The neighbours
that fall out are not just beyond the cutoff; they are reordered far away. A rescoring step
would need to fetch many more than 10× candidates to recover them, which is the usual
production fix and exactly what this experiment deliberately omits.

**Binary is not faster here.** The scan time of binary equals int8 because hvec scores binary
asymmetrically against a float query, one dimension at a time. A symmetric Hamming path with
popcount would be far faster but changes the error model; that is a separate codec.

**What is wrong with this experiment.** One corpus, one domain, two small 384-dimension
models. Larger or anisotropic models (e.g. 1536-dimension API embeddings) may behave
differently. Scores are measured against f32, not against relevance labels, so a codec that
reorders two equally relevant passages is penalised as if it had lost one. And no chat model is
involved, so nothing here says whether the answers change.

**Next.** Centre the vectors (subtract the corpus mean) before binarising, which is the standard
fix for the BGE effect, and add it as a codec option. Then run this corpus through `bench run`
with a question set and two chat models to see whether 0.61 recall reaches the answer.
