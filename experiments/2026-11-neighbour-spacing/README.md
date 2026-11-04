# Five corpora, two models: neighbour spacing predicts codec loss

The SciFact experiments kept running into the same explanation for binary's loss. Neighbours in
the top 10 differ in cosine by a few thousandths, and a 384-bit sign code estimates cosine with
a noise of a few hundredths, so it reorders them. This experiment checks whether that holds as
a quantitative rule across kinds of text, and answers the question behind it: does the *type*
of content change what a codec costs?

**Hypothesis.** A codec's loss of agreement with the exact f32 top-k is governed by one number:
the ratio of the codec's score noise to the score spacing of the corpus's neighbours. Corpora
with tighter spacing lose more under the same codec. The ratio will predict the loss across
corpora and models much better than the noise alone, which is what a codec's error figure
usually reports.

**Result.** Confirmed. Over 90 (corpus, model, codec) points, noise ÷ band ranks recall@10
loss with Spearman ρ = 0.980; noise alone manages 0.644. A single two-parameter curve fitted
on four corpora predicts the fifth's recall@10 with a mean absolute error of 0.024, against 0.090
for the same curve on noise alone. Content type matters, but only through spacing. Across the
five corpora the band varies about 2.2× within a model: FiQA's financial Q&A is the tightest,
ArguAna's arguments the loosest, and Python code sits in the middle. Changing the embedding
model on the *same* corpus moves the band by as much (MiniLM's is 1.7 to 2.5× bge-small's).
MiniLM spreads neighbours further apart than it amplifies codec noise, which is why it has
tolerated binary better in every experiment so far.

## Setup

| | |
|---|---|
| Corpora | Five BEIR-format retrieval sets [1]: **SciFact** (scientific abstracts, 300 claims), **NFCorpus** [2] (medical articles, 323 queries), **ArguAna** [3] (arguments; the query is itself an argument and the target its counterargument, 1,406 queries), **FiQA** [4] (financial forum Q&A, 648 queries), **CoSQA** [5] from CoIR [6] (Python functions, web-search queries, 500 queries) |
| Chunking | 200 words, no overlap: 8,036 / 6,260 / 11,232 / 71,274 / 20,648 chunks |
| Queries | Each set's test split with its relevance judgments, via `hvec questions import-beir`; the query text is embedded with the collection's model |
| Embedding models | `bge-small-en-v1.5` (Xenova ONNX, 384 dims) and `all-MiniLM-L6-v2` (Qdrant ONNX, 384 dims), fastembed on CPU, no query instruction |
| Metric | cosine |
| k | 10 |
| Codecs | `f32` baseline, `int8`, `binary`, `binary-centred`, `pq96`, `pq48`, `pq24`, `mrl256`, `mrl192`, `mrl96` |
| Fitting | Trained codecs (`binary-centred`, `pq<m>`) fitted on every vector of the collection, as `ingest --codec` does |
| Search | brute force, no index, no rescoring |
| Hardware | Apple M4 Pro, 14 cores, release build |
| Code | hvec commit after `cc9a5eb` (the commit that adds this experiment) |
| Command | `./run.sh` in this directory. It downloads ~27 MB of data and needs `uv` to read CoSQA's parquet files. About 45 minutes, mostly embedding FiQA |

### The three new numbers

`bench recall` now reports, next to recall and nDCG:

- **`boundary_gap`**: the median over queries of the exact score gap between rank k and rank
  k + 1, the margin between the last true neighbour and the first outsider.
- **`neighbour_band`**: the median over queries of the exact score gap between rank k and rank
  2k, the width of the score window that holds the next k competitors. This is the spacing
  measure used below. It needs only f32 and is the same for every codec on a collection.
- **`rank_noise`**: the mean over queries of the standard deviation of the codec's score error
  over the exact top-2k, with each query's mean error subtracted first. A constant shift
  changes no ranking, so only this spread can reorder neighbours. It is not the same as `mean
  err`. Binary on bge-small has a mean error of 0.31, nearly all of it bias, but a rank noise of
  0.019.

## Results

### Spacing by corpus and model

Exact f32, median over queries. Binary and pq48 are both 48 bytes, 32× smaller than f32.

| corpus | content | chunks | band, bge-small | band, MiniLM | binary recall@10, bge / MiniLM | pq48 recall@10, bge / MiniLM | f32 nDCG@10, bge / MiniLM |
|---|---|---:|---:|---:|---:|---:|---:|
| FiQA | financial Q&A | 71,274 | **0.0108** | 0.0271 | 0.537 / 0.666 | 0.642 / 0.716 | 0.387 / 0.367 |
| NFCorpus | medical | 6,260 | 0.0121 | 0.0308 | 0.556 / 0.669 | 0.684 / 0.742 | 0.314 / 0.282 |
| SciFact | scientific abstracts | 8,036 | 0.0135 | 0.0307 | 0.611 / 0.705 | 0.711 / 0.764 | 0.698 / 0.627 |
| CoSQA | Python functions | 20,648 | 0.0148 | **0.0248** | 0.660 / 0.719 | 0.773 / 0.775 | 0.269 / 0.279 |
| ArguAna | arguments | 11,232 | **0.0233** | **0.0529** | 0.748 / 0.815 | 0.811 / 0.849 | 0.421 / 0.361 |

On bge-small, binary's recall follows the band's order exactly. FiQA is tightest and loses
most; ArguAna is loosest and loses least. On MiniLM the order holds except for CoSQA, which
MiniLM packs tightest of all but which loses less than FiQA, NFCorpus or SciFact; its codec
noise is also the lowest (see the caveats). The two models agree on the order of the corpora
apart from CoSQA, and SciFact and NFCorpus, which tie on MiniLM.

### Does the ratio predict the loss?

All 90 lossy points (5 corpora × 2 models × 9 codecs). The fitted curve is
`ln(loss / recall) = −0.727 + 1.045 · ln(noise / band)`, so the odds of losing a neighbour are
very nearly proportional to the ratio: **odds ≈ 0.48 × noise ÷ band**. A ratio of 1 means about
0.67 of the top 10 survives.

| predictor | Spearman ρ with 1 − recall@10 | Spearman ρ with nDCG@10 loss | held-out recall@10 MAE (fit on four corpora, predict the fifth) |
|---|---:|---:|---:|
| rank noise ÷ neighbour band | **0.980** | **0.848** | **0.024** |
| rank noise alone | 0.644 | 0.680 | 0.090 |

Mean absolute error of the held-out prediction, by corpus:

| held-out corpus | MAE, ratio | worst point | MAE, noise alone |
|---|---:|---|---:|
| SciFact | 0.016 | 0.048 | 0.079 |
| CoSQA | 0.015 | 0.031 | 0.046 |
| NFCorpus | 0.024 | 0.096 | 0.090 |
| ArguAna | 0.031 | 0.111 | 0.136 |
| FiQA | 0.035 | 0.166 (bge-small, pq24) | 0.101 |

**Predicting before running the codec.** The useful version of this needs only f32 on the new
corpus. Measure its band, borrow the codec's noise for that model from the other four corpora,
and read the recall off the curve. Leave-one-corpus-out, that gives an MAE of 0.033, against
0.062 for simply averaging the codec's recall on the other four corpora. It is better on four
corpora and worse on one. On CoSQA it gives 0.062 against 0.039, because codec noise there is
lower than elsewhere (binary on bge-small 0.014, against 0.018 to 0.020 on the prose corpora).
Rank noise per (codec, model) varies by 1.4 to 1.75× across corpora, so borrowing it is an
approximation.

The full per-codec table is in `results/summary.txt`; the raw reports, with score errors,
top-1 agreement, document recall and timings, are in `results/<corpus>-<model>-k10.json`.

## Discussion

**The denominator does the work.** Noise alone is a property of the codec and the model, and it
gets the ordering within one collection right. It cannot say that the same codec will lose 46
points on FiQA and 25 on ArguAna, because the codec's noise is about the same on both (binary
on bge-small: 0.0177 and 0.0183). The band can. A codec's error figure, quoted on its own as
it usually is, does not tell you what it will cost on your corpus. Its ratio to the corpus's
spacing does.

**Content type matters through spacing, and less than the model.** The original question was
whether changing the kind of text changes the result. It does, by up to 21 points of binary
recall between FiQA and ArguAna on bge-small. But code, the content most unlike prose here,
is not an outlier: CoSQA sits between SciFact and ArguAna on bge-small. Changing the model on
one corpus shifts the band as much as the whole range of corpora does. ArguAna is loose because
its queries are full arguments whose best match is one specific counterargument, so the top of
the ranking is well separated. FiQA is tight because it is the largest corpus, full of short,
similar forum answers. Size alone does not explain the band, though. NFCorpus is the smallest
corpus and the second tightest.

**Why MiniLM tolerates binary better.** The int8-and-binary experiment found that MiniLM keeps
0.705 of the top 10 under binary where bge-small keeps 0.611, and put it down to centring. The
centred-binary experiment showed centring was not the reason. This experiment gives the
answer. On every corpus MiniLM's band is 1.7 to 2.5× wider than bge-small's, while its codec
noise is only 1.3 to 1.5× larger. Its neighbours sit further apart in cosine than the extra
quantization noise can close.

**What the curve misses.** The five worst predictions are all `pq24` or `mrl96`, the most
aggressive codecs. On FiQA and NFCorpus, `pq24` loses 9 to 16 points more than predicted. On
ArguAna, `pq24` and `mrl96` lose 8 to 10 points *less* than predicted. Rank noise is
measured over the exact top-2k only. A codec that occasionally scores a distant vector very
high, pulling it in from far outside the window, does damage that this noise does not see.
Measuring noise over a wider window, or using a tail statistic, is the obvious refinement.
ArguAna's better-than-predicted points suggest the opposite effect: when the top of the ranking
is well separated, the band understates how much noise it takes to displace the top 10.

**Labels.** The ratio predicts nDCG loss less well (ρ = 0.848) than recall loss, as expected.
nDCG depends on whether the reordered neighbours are the judged ones, which spacing cannot
know. On these corpora binary costs 0.7 to 5.4 nDCG points where it costs 19 to 46 points of
agreement, the same gap the labelled SciFact experiment found.

**What is wrong with this experiment.**
- **CoSQA's corpus is mostly duplicates.** Only 6,267 of its 20,604 functions are distinct, and
  86% sit in groups of identical code (the largest has 20 copies). Exact ties are why its
  boundary gap is zero. A codec scores identical vectors identically, so duplicates move
  together and do not count as losses. The effective corpus is about 6,300 functions, and that
  probably explains CoSQA's lower codec noise. A deduplicated code set is the next code
  corpus to try.
- **Only two models, both 384-dimensional English text models.** The model effect is two points,
  not a trend.
- **Spacing and size are not separated.** FiQA is both the largest corpus and the tightest.
  Subsampling FiQA to 8,000 chunks would show how much of its band comes from size.
- **The fit's form was chosen after seeing the results.** The curve is a logit-linear relation
  between loss and log ratio. The held-out numbers test its parameters, not its form, and the
  Spearman figures do not depend on it.
- **Brute force only, no rescoring.** With an index or an f32 rescoring pass the relationship
  will change, and rescoring is exactly what the band would tell you how deep to do.

**Next.**
- **Rescoring depth from the band.** If a codec's ratio is r, how many candidates must be rescored
  in f32 to recover the exact top 10? A curve of that is directly useful.
- **A deduplicated code corpus.**
- **Identifier-heavy content.** The hashes-and-amounts corpus discussed as the next experiment
  should show whether spacing can be measured at all where f32 itself does not retrieve.

## References

1. Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets and Benchmarks 2021.
2. Boteva et al., *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, ECIR 2016.
3. Wachsmuth, Syed and Stein, *Retrieval of the Best Counterargument without Prior Topic Knowledge*, ACL 2018.
4. Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*, WWW Companion 2018.
5. Huang et al., *CoSQA: 20,000+ Web Queries for Code Search and Question Answering*, ACL 2021.
6. Li et al., *CoIR: A Comprehensive Benchmark for Code Information Retrieval Models*, arXiv:2407.02883, 2024.
