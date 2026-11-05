# Text format: is there a best layout for a record, given the model and the time budget?

Every earlier experiment held the text fixed and changed the codec. Neighbour spacing found
that the kind of content changes what a codec costs, but only through the score spacing of a
corpus's neighbours. This experiment holds the content fixed and changes only its layout: the
same facts written as prose, bullets, key-value pairs, JSON or a table row.

**The question.** For a given context (an embedding model and a kind of query), which format and
fast path (first-stage codec, re-score depth) give the best nDCG@10 within a fixed time per query?

**Format can only change speed indirectly.** The scan never sees text. With one record per chunk,
every format has the same number of vectors and so the same scan cost per codec. Format moves
time in two ways:
1. **Spacing.** A format that spreads neighbours further apart lowers a codec's noise ÷ band, so a
   `binary-hamming` scan needs fewer f32 re-scores for the same recall
   ([two-stage re-scoring](../2026-11-two-stage-rescoring/)).
2. **Tokens.** What a retrieved record costs the chat model to read. Measured as size per record;
   no model is called.

**Hypothesis.**
1. The best format depends on the query type. Templated, field-labelled layouts (key-value,
   JSON, table) match attribute lookups best at f32. They also repeat the same keys in every
   record, which should pack neighbours closer together. Their compressed scans would then need
   deeper re-scores, so a format can win at f32 and lose at a tight time budget.
2. Format acts on compressed retrieval only through two numbers: f32 nDCG (how well the query
   matches the text) and noise ÷ band (how well the ranking survives the codec). If so, the
   noise ÷ band curve fitted on some formats predicts the re-score depth on a held-out format
   about as well as it did across corpora (within a factor of 1.25–1.41).

**Result.** Hypothesis 1 is refuted on both counts. Hypothesis 2 is confirmed.
- **Format matters more than any codec here.** With the facts held fixed, it moves exact f32
  nDCG@10 by up to 2×: on MiniLM lookups, templated prose scores 0.179 and a table row 0.353.
  Templated prose is the worst of the five controlled formats in all four contexts.
- **The best format depends on the embedding model, not the query type.** On bge-small, JSON is
  best for both query types (0.441 lookup, 0.224 descriptive), with key-value and table close
  behind on lookups. On MiniLM, the table row is best for both (0.353, 0.181), and JSON is only
  fourth (0.240 on lookups).
- **There is no accuracy-for-speed trade-off.** Within a context, format changes the neighbour
  band by at most 1.3×, against 1.7–2× between the two models. A `binary-hamming` scan plus 100 f32
  re-scores (0.22 ms against 4.1 ms for exact f32, 19× less) keeps 0.96–0.99 of the winning
  format's f32 nDCG, and 0.91–0.98 of prose's. So the fast path on the best format beats
  exact f32 on prose in every context, at 1/19 of the time.
- **Noise ÷ band predicts depth on a held-out format** within a factor of 1.16 at 95% recall
  (Spearman 0.985). That is tighter than across corpora. Together with f32 nDCG, it predicts a
  hybrid's nDCG on a held-out format within 0.003–0.007. In practice this means you can pick a
  format by measuring f32 nDCG on a sample, and the codec cost follows from spacing.

## Setup

| | |
|---|---|
| Corpus | WikiBio [1] test split (`thoughtworks/wiki_bio` on Hugging Face, parquet, 72,831 biographies): infobox plus the article's opening paragraph. 20,000 records sampled with seed 2026 |
| Cleaning | Image, caption, website and update-stamp fields dropped. Values over 25 words or empty or "unknown" dropped. At most 16 fields per record (median 12). PTB bracket tokens (`-lrb-`) restored. Text stays lowercased, as released |
| Formats | Seven, below. One record per chunk, verbatim, line breaks kept (`ingest --whole-files`) |
| Queries | 500 `lookup` and 500 `descriptive` queries from the infobox, the same for every format |
| Embedding models | `bge-small-en-v1.5` and `all-MiniLM-L6-v2`, 384 dims, fastembed on CPU, no query instruction |
| Measurements | `bench recall` (f32 nDCG@10, `neighbour_band`, `rank_noise` for `int8`, `binary`, `binary-hamming`, `pq48`). `bench rescore` (`binary-hamming` and `pq48` first stages, f32 second stage, depths 10–1000 and adaptive margins). `bench scan` (one thread, cost per vector) |
| Time per query | Scan + mean re-score depth × the f32 cost per vector, on one thread. The candidates are assumed to be in memory, as in [zero-copy scan](../2026-11-zero-copy-scan/) |
| Machine | Apple M4 Pro, 26 GB, release build |
| Code | hvec commit `cb5ffd1` |
| Command | `./run.sh` (about an hour, mostly embedding). Results in `results/`: `recall/`, `rescore/`, `scan/`, `stats.json`, `examples.md`, `summary.txt` |

### Formats

`results/examples.md` shows one record in all seven. The first five hold exactly the same facts
(the cleaned infobox). Only the layout differs.

| format | layout | approx. tokens / record |
|---|---|---:|
| `paragraph` | one templated sentence per field: "ruth born's birth place is bay city , michigan." | 114 |
| `bullets` | `- birth place: bay city , michigan`, one field per line | 75 |
| `kv` | `name: ruth born; birth date: 08 august 1925; ...` on one line | 74 |
| `json` | `{"name": "ruth born", "birth_date": "08 august 1925", ...}` | 114 |
| `table` | a two-row Markdown table, header then values | 119 |
| `wiki` | the article's own opening paragraph, up to 150 words. Different facts: the natural-prose reference, not a controlled comparison | 74 |
| `header-prose` | `kv` as a header, a blank line, then `wiki`. More facts than either | 147 |

Token counts are a word-and-punctuation split from `stats.json`, not a model's tokenizer.

### Queries

- **`lookup`**: two of a record's attributes, for example "who was born on 25 april 1985 and
  played as guard / offensive tackle?". Relevant to every record with both values, and kept only
  when that is at most three records.
- **`descriptive`**: one attribute value shared by 5 to 50 records, for example "people who died
  in columbus , ohio". Relevant to all of them (5,860 judgments in all).

The queries use their own phrasing ("was born in", "played as"), not the field labels the
documents use ("birth place", "position"). That way no infobox format gets the query's words for
free.

## Results

### Exact f32 nDCG@10 by format

| format | tokens | bge lookup | bge descriptive | MiniLM lookup | MiniLM descriptive |
|---|---:|---:|---:|---:|---:|
| `paragraph` (baseline prose) | 114 | 0.370 | 0.174 | 0.179 | 0.130 |
| `bullets` | 75 | 0.402 | 0.180 | 0.336 | 0.161 |
| `kv` | 74 | 0.435 | 0.191 | 0.276 | 0.143 |
| `json` | 114 | **0.441** | **0.224** | 0.240 | 0.142 |
| `table` | 119 | 0.415 | 0.211 | **0.353** | **0.181** |
| `wiki` (reference) | 74 | 0.181 | 0.113 | 0.098 | 0.091 |
| `header-prose` (reference) | 147 | 0.370 | 0.158 | 0.208 | 0.136 |

Every structured layout beats templated prose in every context, by 4% (bullets, bge
descriptive) to 97% (table, MiniLM lookup). The model decides which one wins. JSON is best on
bge-small and middling on MiniLM. The table row is best on MiniLM and within 0.03 of the top on
bge-small.

### Spacing and re-score depth

| | bge-small | MiniLM |
|---|---|---|
| neighbour band, range across formats | 0.0082–0.0116 | 0.0157–0.0213 |
| `binary-hamming` noise ÷ band | 4.47–5.39 | 2.12–2.80 |
| `pq48` noise ÷ band | 1.08–1.41 | 0.87–1.18 |
| f32 re-scores for 95% recall@10 behind `binary-hamming` | 476–966 | 225–441 |

Within a context (one model and one query type), format changes the band by at most 1.3×. The
model changes it by about 2×. The
spacing penalty predicted for structured layouts appears once: JSON on bge-small has the tightest
band (0.0082 on lookups) and the highest `binary-hamming` noise ÷ band (5.39). It is too small
to matter, though. JSON still needs fewer re-scores for 95% recall than prose (760 against 966).

### The fast path: `binary-hamming` scan, 100 f32 re-scores (0.22 ms against about 4.1 ms)

| context | best format | its f32 nDCG | its hybrid nDCG (share kept) | `paragraph` f32 nDCG | `paragraph` hybrid (share kept) |
|---|---|---:|---:|---:|---:|
| bge-small, lookup | `json` | 0.441 | 0.423 (0.959) | 0.370 | 0.335 (0.906) |
| bge-small, descriptive | `json` | 0.224 | 0.219 (0.977) | 0.174 | 0.169 (0.976) |
| MiniLM, lookup | `table` | 0.353 | 0.349 (0.987) | 0.179 | 0.170 (0.948) |
| MiniLM, descriptive | `table` | 0.181 | 0.179 (0.992) | 0.130 | 0.126 (0.976) |

Scan cost does not depend on format. Across all 14 collections an exact f32 scan takes
3.87–4.25 ms and a `binary-hamming` scan 0.19–0.22 ms (20,000 vectors, one thread). At 100
candidates, agreement with the exact top 10 is only 0.64–0.87, but most of the neighbours lost
are not relevant, so nDCG barely moves. In every context, the best format and operating point
within 10% of an exact scan's time (0.41 ms) is the same as within 100%: JSON on bge-small and
the table row on MiniLM.

### Cross-format checks, leaving one format out at a time

56 (format, model, query type, first stage) points.

| target recall@10 | Spearman ρ (noise ÷ band, depth) | fitted depth | held-out format, mean factor error |
|---|---:|---|---:|
| 90% | 0.985 | 3.08·k·r^1.70 | 1.14 |
| 95% | 0.985 | 4.74·k·r^1.83 | 1.16 |
| 99% | 0.968 | 10.59·k·r^1.98 | 1.24 |

The share of f32 nDCG a hybrid keeps, predicted from log noise ÷ band and fitted without the
held-out format, has these errors:

| N | error | f32 nDCG × the mean share kept |
|---:|---:|---:|
| 10 | 0.0067 | 0.0181 |
| 30 | 0.0047 | 0.0117 |
| 100 | 0.0026 | 0.0048 |

## Discussion

**The answer to the question is mostly "pick the format for the model".** The fast path costs
every format about the same, so the format that wins at f32 also wins at a tenth of the time. You
can choose a format by embedding a sample in two or three layouts and measuring f32 nDCG. The
compressed cost then follows from noise ÷ band, which held on formats it was not fitted on.

**Prose is the weakest layout for attribute queries, even with identical facts.** The templated
sentences repeat the person's name in every sentence ("ruth born's ... is ..."), which may dilute
the fields. That makes `paragraph` a pessimistic stand-in for prose. The real Wikipedia paragraph
(`wiki`) is lower still, but it lacks many of the facts asked about.

**A metadata header is a cheap fix for prose documents.** Putting the key-value fields above the
Wikipedia paragraph doubles its nDCG (bge-small lookups 0.181 → 0.370, MiniLM 0.098 → 0.208). The
record alone still scores higher, and the header costs the most tokens (147).

**Tokens: key-value and bullets are about 35% cheaper than JSON and table** (74–75 against
114–119 per record). On bge-small lookups, key-value is within 0.006 of JSON. On MiniLM lookups,
bullets are within 0.017 of the table. For descriptive queries the cheaper format gives up more
(0.033 and 0.020).

**Why JSON on bge-small and a table on MiniLM?** Not tested. Finding out would take an ablation,
for example JSON without its quotes and braces, or a table without its pipes.

**The depth curve is steeper here than across corpora** (r^1.83 at 95% against r^1.29). Most of
the ratio's range in this experiment comes from the model and the codec, not the format. So the
check shows that the curve carries across formats, not that its exponent is universal.

**Caveats.**
- **No confidence intervals.** With 500 queries per type, treat gaps under about 0.02 nDCG as
  ties: JSON against key-value on bge-small lookups, table against bullets on MiniLM lookups.
  The per-query scores are not saved, so there is no bootstrap.
- **Absolute nDCG is low** (0.09–0.44). Three causes: the queries are templated, the
  judgments match infobox values exactly (so "new york city" and "new york city , new york ,
  u.s." are separate categories), and no query instruction is used for bge-small. All seven
  formats share these, but the ratios between formats may not hold for realistic queries.
- **`wiki` and `header-prose` are not controlled comparisons.** Their facts differ from the
  infobox's.
- **One domain.** Biographies are short, field-heavy records. Long documents, code, or tables
  with many rows may rank the formats differently.
- **Retrieval only.** Whether a format helps a chat model answer needs a chat model and a judge,
  as Experiments 1 and 3 do.
- **Time is computed from per-vector costs**, not timed end to end, and assumes the f32
  candidates are in memory.

**Next.**
- Embed one format and return another (for example, match on `kv` and hand the model `wiki`).
  Retrieval and reading may want different layouts. `ingest` stores the text it embeds, so this
  needs a separate payload text per chunk.
- A second structured source, and realistic queries, to see whether "the model decides" holds.
- Save per-query scores in `bench recall` so format comparisons can carry confidence intervals.

## References

1. Lebret, Grangier and Auli, "Neural Text Generation from Structured Data with Application to
   the Biography Domain", EMNLP 2016. [arXiv:1603.07771](https://arxiv.org/abs/1603.07771)
