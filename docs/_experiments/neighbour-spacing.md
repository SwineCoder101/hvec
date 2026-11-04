---
title: "Five corpora, two models: neighbour spacing predicts codec loss"
date: 2026-11-04 18:00:00 +0000
result: "Codec noise divided by the corpus's neighbour spacing ranks recall@10 loss with ρ = 0.98 over 90 points, against 0.64 for noise alone. Content type matters only through that spacing, and changing the embedding model moves it as much as changing the corpus does."
references: [thakur2021beir, boteva2016nfcorpus, wachsmuth2018arguana, maia2018fiqa, huang2021cosqa, li2024coir, xiao2023bge, reimers2019sbert]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-neighbour-spacing](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-neighbour-spacing).
Follow-up to [SciFact: mean-centred binary]({{ "/experiments/scifact-centred-binary/" | relative_url }}),
which found that binary's loss is a resolution problem: top-10 neighbours sit closer together in
cosine than a sign code can tell apart.

**Hypothesis.** A codec's loss of agreement with the exact f32 top-k is governed by the ratio of
its score noise to the spacing of the corpus's neighbours. The ratio will predict the loss across
corpora and models much better than the noise alone.

**Setup.** Five BEIR-format sets [1] of different kinds: SciFact (scientific abstracts),
NFCorpus [2] (medical), ArguAna [3] (arguments), FiQA [4] (financial Q&A) and CoSQA [5] from
CoIR [6] (Python functions). They range from 6,260 to 71,274 chunks of 200 words. Two embedding
models, bge-small-en-v1.5 [7] and all-MiniLM-L6-v2 [8], both 384 dimensions. Nine codecs:
`int8`, `binary`, `binary-centred`, `pq96/48/24`, `mrl256/192/96`. k = 10, brute force, no
rescoring. `bench recall` gains three numbers:

- **`neighbour_band`**: the median exact score gap between rank 10 and rank 20. It needs only f32.
- **`rank_noise`**: the spread of the codec's score error over the exact top 20, with each
  query's bias removed.
- **`boundary_gap`**: the gap between rank 10 and rank 11.

**Result.** Confirmed.

| corpus | content | band, bge-small | band, MiniLM | binary recall@10, bge / MiniLM | pq48 recall@10, bge / MiniLM |
|---|---|---:|---:|---:|---:|
| FiQA | financial Q&A | **0.0108** | 0.0271 | 0.537 / 0.666 | 0.642 / 0.716 |
| NFCorpus | medical | 0.0121 | 0.0308 | 0.556 / 0.669 | 0.684 / 0.742 |
| SciFact | scientific abstracts | 0.0135 | 0.0307 | 0.611 / 0.705 | 0.711 / 0.764 |
| CoSQA | Python functions | 0.0148 | **0.0248** | 0.660 / 0.719 | 0.773 / 0.775 |
| ArguAna | arguments | **0.0233** | **0.0529** | 0.748 / 0.815 | 0.811 / 0.849 |

| predictor of recall@10 loss, 90 points | Spearman ρ | held-out MAE (fit on four corpora, predict the fifth) |
|---|---:|---:|
| rank noise ÷ neighbour band | **0.980** | **0.024** |
| rank noise alone | 0.644 | 0.090 |

The fitted curve is nearly proportional: **the odds of losing a neighbour ≈ 0.48 × noise ÷
band**. At a ratio of 1, about two thirds of the top 10 survives. Predicting a new corpus from
its f32 band alone, with the codec's noise borrowed from the other four corpora, gives an MAE of
0.033 against 0.062 for averaging the other corpora's recall. It does better on four corpora and
worse on CoSQA.

**Reading.** The same codec has about the same noise everywhere. Binary on bge-small has 0.0177
on FiQA and 0.0183 on ArguAna, yet loses 46 points of the top 10 on FiQA and 25 on ArguAna. The
band explains the difference. Content type matters only through spacing, and code is not an
outlier: on bge-small, CoSQA sits between SciFact and ArguAna. The embedding model moves the
band as much as the whole range of corpora does. MiniLM's band is 1.7 to 2.5× wider than
bge-small's while its codec noise is only 1.3 to 1.5× larger, which is why MiniLM has tolerated
binary better in every experiment so far. Against the labels the ratio is a weaker guide
(ρ = 0.848 with nDCG loss), and binary costs only 0.7 to 5.4 nDCG points.

**Caveats.**
- **CoSQA is mostly duplicates.** Only 6,267 of its 20,604 functions are distinct, which zeroes
  its boundary gap and probably lowers its codec noise.
- **The most aggressive codecs (`pq24`, `mrl96`) miss the curve by up to 16 points.** They
  miss in both directions.
- **Two models only.**
- **FiQA is both the largest and the tightest corpus**, so size and spacing are not separated.
- **The curve's form was chosen after seeing the data.**
- **No rescoring.**

{% include references.html %}
