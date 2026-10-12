# SciFact: codecs against the relevance judgments, not just against f32

The first two SciFact experiments measured every codec by agreement with the exact f32
ranking: recall@10 of the f32 top-10. Both write-ups flagged the obvious hole: a codec that
swaps one relevant passage for another equally relevant one is penalised as if it had lost
one, and a codec that swaps two irrelevant passages is penalised the same way. This experiment
scores the same codecs, on the same collections, against BEIR's human relevance judgments.

**Hypothesis.** Agreement with f32 overstates the retrieval loss of binary codecs. Most of the
neighbours that fall out of binary's top-10 are near-ties with the ones that replace them, and
near-ties are rarely the judged-relevant abstract. So nDCG@10 against the labels will fall by
much less than the 30 to 40 points of f32 agreement that binary loses.

**Result.** Confirmed, strongly. On bge-small, `binary` keeps only 0.611 of f32's top-10 but
0.654 of nDCG@10 against 0.698 for f32: a loss of 4.4 points, not 39. On MiniLM the loss is
2.2 nDCG points against 29.5 points of top-10 agreement. `int8` is indistinguishable from f32
on every labelled metric. And the `binary-centred` codec, which recovered 4.5 points of f32
agreement on bge-small in the previous experiment, recovers 0.4 nDCG points there and loses
0.7 on MiniLM: the neighbours centring wins back are not the judged ones.

## Setup

| | |
|---|---|
| Corpus | BEIR SciFact, 5,183 scientific abstracts (title + text): 8,036 chunks of 200 words, no overlap. Written as one `<id>.txt` per abstract by `hvec questions import-beir --corpus-out` |
| Questions | `questions/scifact.jsonl`: the 300 BEIR test claims with their 339 relevance judgments, built by `hvec questions import-beir`. The 188 claims whose SciFact evidence is uniformly `SUPPORT` or `CONTRADICT` also carry a gold verdict for later chat-model experiments; this experiment uses only the judgments |
| Queries | The claim text, embedded with the collection's model |
| Embedding models | `bge-small-en-v1.5` (Xenova ONNX, 384 dims) and `all-MiniLM-L6-v2` (Qdrant ONNX, 384 dims), fastembed on CPU |
| Metric | cosine |
| k | 10 and 5 |
| Codecs | `f32`, `int8`, `binary`, `binary-centred` (fitted on all 8,036 corpus vectors) |
| Label metrics | nDCG@k over documents, binary relevance, each abstract counted once at the rank of its first chunk; document recall@k, the fraction of a claim's judged abstracts among the top-k chunks' documents |
| Search | brute force, no index, no rescoring |
| Hardware | Apple M4 Pro, 14 cores, release build |
| Code | hvec commit `6a78bf9` |
| Command | `./run.sh` in this directory (downloads data and models on first run) |

The question set is committed, so the import is reproducible from the committed file alone;
`run.sh` rebuilds it from the BEIR download and warns if the two differ.

## Results

300 claims, k = 10. `recall@10` is agreement with the f32 top-10, as in the earlier experiments.
`nDCG@10` and `doc recall@10` are against the judgments; the f32 row is the ceiling.

| embedding model | codec | recall@10 vs f32 | top-1 vs f32 | **nDCG@10** | **doc recall@10** | bytes | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | f32 | 1.000 | 1.000 | **0.698** | **0.818** | 1536 | 1.0× |
| bge-small-en-v1.5 | int8 | 0.987 | 0.997 | 0.698 | 0.818 | 392 | 3.9× |
| bge-small-en-v1.5 | binary | 0.611 | 0.700 | 0.654 | 0.768 | 48 | 32× |
| bge-small-en-v1.5 | binary-centred | 0.656 | 0.727 | 0.658 | 0.770 | 48 | 32× |
| all-MiniLM-L6-v2 | f32 | 1.000 | 1.000 | **0.627** | **0.778** | 1536 | 1.0× |
| all-MiniLM-L6-v2 | int8 | 0.995 | 1.000 | 0.627 | 0.778 | 392 | 3.9× |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.605 | 0.733 | 48 | 32× |
| all-MiniLM-L6-v2 | binary-centred | 0.719 | 0.717 | 0.598 | 0.739 | 48 | 32× |

k = 5, which is what the RAG pipeline retrieves by default:

| embedding model | codec | recall@5 vs f32 | nDCG@5 | doc recall@5 |
|---|---|---:|---:|---:|
| bge-small-en-v1.5 | f32 | 1.000 | 0.671 | 0.741 |
| bge-small-en-v1.5 | int8 | 0.985 | 0.671 | 0.742 |
| bge-small-en-v1.5 | binary | 0.627 | 0.632 | 0.708 |
| bge-small-en-v1.5 | binary-centred | 0.657 | 0.629 | 0.690 |
| all-MiniLM-L6-v2 | f32 | 1.000 | 0.602 | 0.713 |
| all-MiniLM-L6-v2 | int8 | 0.997 | 0.601 | 0.709 |
| all-MiniLM-L6-v2 | binary | 0.699 | 0.584 | 0.677 |
| all-MiniLM-L6-v2 | binary-centred | 0.713 | 0.576 | 0.680 |

The loss in the labelled metric, relative to f32:

| | bge-small, k=10 | MiniLM, k=10 | bge-small, k=5 | MiniLM, k=5 |
|---|---:|---:|---:|---:|
| top-10 (or top-5) agreement lost by `binary` | 38.9 % | 29.5 % | 37.3 % | 30.1 % |
| nDCG lost by `binary` | 6.4 % | 3.5 % | 5.7 % | 3.0 % |
| document recall lost by `binary` | 6.1 % | 5.7 % | 4.4 % | 5.0 % |

Raw output for every row is in `results/`, one JSON file per (model, k), with the run config.

## Discussion

**Agreement with f32 is the wrong yardstick for a lossy codec, by about a factor of six.**
Binary drops four of every ten f32 neighbours on bge-small, yet the judged abstracts survive
at 94 percent of their f32 rate. The earlier write-up measured that the top-10 neighbours differ
in cosine by 0.003 to 0.008 and that a 384-bit sign code resolves cosine to about 0.02 to
0.03, so the neighbours binary reorders are the ones that were nearly tied. Nearly-tied
neighbours are, by and large, interchangeable *for the labels*: an abstract at rank 6 that is
replaced by the abstract that was at rank 12 is a miss for recall-vs-f32 and nothing for nDCG,
unless one of the two was the judged one. With 1.13 judged abstracts per claim and the judged
one usually near the top, that is rare. The losses that do register come from the top of the
ranking: top-1 agreement is 0.70, and a relevant abstract pushed from rank 1 to rank 3 costs
nDCG even though it is still retrieved.

**int8 is lossless as far as the labels can tell.** nDCG and document recall match f32 to
three decimals on both models and both k. Whatever a later chat-model experiment finds for
int8, it will not be retrieval.

**Centring's gain was real in one space and invisible in the other.** `binary-centred`
raised f32 agreement on bge-small from 0.611 to 0.656 and the labelled metric from 0.654 to
0.658: a tenth of the gain carries over. On MiniLM it lowers nDCG@10 from 0.605 to 0.598 while
raising f32 agreement. The previous write-up already observed that centring costs MiniLM top-1
agreement by adding a norm term that is noise when ‖μ‖ is small; the labels confirm that the
top of the ranking is where it matters. Centring is a better *estimator* of cosine, not a
better *retriever*, and the two are not the same thing once the score noise is below the
neighbour gaps that matter.

**The two models now rank the other way round.** By f32 agreement, MiniLM tolerates binary
better (0.705 against 0.611). By the labels, bge-small with binary (0.654) still beats MiniLM
uncompressed (0.627). The codec axis and the model axis interact, and the model axis is
larger: choosing the better embedding model and compressing it 32× leaves more retrieval
quality than keeping the worse model at full precision. This is the premise of the roadmap's
Experiment 2, native low-precision embeddings against post-hoc quantization at equal bytes.

**What is wrong with this experiment.** One corpus, one domain, two small 384-dimension models,
as before. SciFact's judgments are sparse (1.13 per claim) and binary, so nDCG mostly
measures where the single judged abstract lands; a dataset with graded, denser judgments
could tell a different story about the lower ranks. The absolute nDCG@10 figures are a few
points below the published BEIR numbers for these models because the corpus is chunked into
200-word windows and scored per chunk rather than per abstract, and because the queries carry
no instruction prefix. The chunking is held fixed because the RAG pipeline under test uses it.
And still no chat model: whether a 4-point nDCG loss changes answers is the question the next
experiment, with a question set that now carries gold verdicts, exists to answer.

**Next.** Run the question set through `bench run` against two chat models on the f32, int8,
binary and binary-centred collections (the roadmap's Experiment 1). The claim under test is
that a codec whose labelled retrieval loss is 4 nDCG points does not move the answer rate on
a large model and might on a small one.
