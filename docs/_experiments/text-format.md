---
title: "Text format: is there a best layout for a record?"
date: 2026-11-05 21:00:00 +0000
result: "With the facts held fixed, layout moves f32 nDCG@10 by up to 2×, and templated prose is last in every context. The best format depends on the embedding model (JSON on bge-small, a table row on MiniLM), not the query type. Its advantage survives compression: a binary-hamming scan plus 100 f32 re-scores keeps 0.96–0.99 of it at 1/19 of the time."
references: [lebret2016wikibio, xiao2023bge, reimers2019sbert, charikar2002simhash, jegou2011pq]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-11-text-format](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-11-text-format).
Follow-up to [neighbour spacing]({{ "/experiments/neighbour-spacing/" | relative_url }}) and
[zero-copy scan]({{ "/experiments/zero-copy-scan/" | relative_url }}).

Every earlier experiment held the text fixed and changed the codec. This one holds the facts
fixed and changes only their layout. The question: for a given embedding model and kind of
query, which format and fast path give the best nDCG@10 within a time budget?

**Format can only change speed indirectly.** The scan never sees text. With one record per chunk,
every format has the same vector count and the same scan cost. Format can only move time through
neighbour spacing, which sets how many f32 re-scores a compressed scan needs. It also changes how
many tokens a retrieved record costs the chat model.

**Hypothesis.**
1. The best format depends on the query type. Field-labelled layouts (key-value, JSON, table)
   win attribute lookups at f32. But they repeat the same keys in every record, which packs
   neighbours closer, so they lose at a tight time budget.
2. Format acts on compressed retrieval only through f32 nDCG and noise ÷ band. The depth curve
   fitted on some formats predicts the depth on a held-out one.

**Setup.** 20,000 WikiBio [1] biographies, each written seven ways.
- **Five controlled formats** with identical facts, the cleaned infobox: templated prose, bullets,
  key-value, JSON, and a two-row table.
- **Two references:** the article's own opening paragraph, and a key-value header above it.

There are 500 lookup queries ("who was born on 25 april 1985 and played as guard / offensive
tackle?") and 500 descriptive ones ("people who died in columbus , ohio"), the same for every
format and judged from the infobox. Each record is one chunk, verbatim, using the new
`ingest --whole-files`. The models are bge-small [2] and MiniLM [3], 384 dims. The fast path is
a `binary-hamming` scan [4] (`pq48` [5] for comparison) with an f32 re-score, timed from measured
per-vector costs on one thread. Apple M4 Pro.

**Result.** Hypothesis 1 is refuted on both counts. Hypothesis 2 is confirmed.

Exact f32 nDCG@10:

| format | tokens | bge lookup | bge descriptive | MiniLM lookup | MiniLM descriptive |
|---|---:|---:|---:|---:|---:|
| templated prose (baseline) | 114 | 0.370 | 0.174 | 0.179 | 0.130 |
| bullets | 75 | 0.402 | 0.180 | 0.336 | 0.161 |
| key-value | 74 | 0.435 | 0.191 | 0.276 | 0.143 |
| JSON | 114 | **0.441** | **0.224** | 0.240 | 0.142 |
| table row | 119 | 0.415 | 0.211 | **0.353** | **0.181** |
| Wikipedia paragraph (reference) | 74 | 0.181 | 0.113 | 0.098 | 0.091 |
| key-value header + paragraph (reference) | 147 | 0.370 | 0.158 | 0.208 | 0.136 |

Every structured layout beats prose in every context. The model, not the query type, decides
which one wins: JSON on bge-small, the table row on MiniLM, where JSON is only fourth.

The predicted speed penalty is too small to matter. Within a context, format changes the
neighbour band by at most 1.3×, while the model changes it by about 2×. A `binary-hamming`
scan plus 100 f32 re-scores takes 0.22 ms against about 4.1 ms for exact f32 over 20,000
vectors.

| context | best format | f32 nDCG | hybrid nDCG (share kept) | prose f32 nDCG | prose hybrid (share kept) |
|---|---|---:|---:|---:|---:|
| bge-small, lookup | JSON | 0.441 | 0.423 (0.959) | 0.370 | 0.335 (0.906) |
| bge-small, descriptive | JSON | 0.224 | 0.219 (0.977) | 0.174 | 0.169 (0.976) |
| MiniLM, lookup | table | 0.353 | 0.349 (0.987) | 0.179 | 0.170 (0.948) |
| MiniLM, descriptive | table | 0.181 | 0.179 (0.992) | 0.130 | 0.126 (0.976) |

The fast path on the best format beats exact f32 on prose in every context, at 1/19 of the time.

With one format left out at a time, noise ÷ band ranks the re-score depth at Spearman 0.985. It
predicts the held-out format's depth for 95% recall within a factor of 1.16, tighter than across
corpora (1.25–1.41). Together with f32 nDCG, it predicts a hybrid's nDCG on the held-out format
within 0.003–0.007.

**Reading.** Choose the format for the model. Measure f32 nDCG on a sample in two or three
layouts; the compressed cost follows from spacing. Key-value and bullets are about 35% cheaper in
tokens than JSON and table, and within 0.02 of the best on lookups. Putting a key-value header
above a prose document doubles its nDCG.

Caveats:
- With 500 queries per type, gaps under about 0.02 are ties.
- The queries are templated, and the judgments match infobox values exactly, so absolute nDCG is
  low.
- The domain is biographies only.
- Only retrieval is measured, not answers.
- Hybrid times are computed from per-vector costs, with the candidates in memory.

{% include references.html %}
