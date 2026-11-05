# Text format: is there a best layout for a record, given the query and the time budget? (not yet run)

**Status: scripted, not run.** `./run.sh` has been checked end to end at a small scale (1,500
records, 60 queries per type, one model). The full run is about an hour on the M4 Pro, mostly
embedding. It needs no API key and no chat model.

Every earlier experiment held the text fixed and changed the codec. Neighbour spacing found
that the kind of content changes what a codec costs, but only through the score spacing of
a corpus's neighbours. This experiment holds the content fixed and changes only its layout:
the same facts written as prose, bullets, key-value pairs, JSON or a table row.

**The question.** For a given context (an embedding model and a kind of query), which format
and fast path (first-stage codec, re-score depth) give the best nDCG@10 within a fixed time
per query?

**Format can only change speed indirectly.** The scan never sees text. With one record per
chunk, every format has the same number of vectors and so the same scan cost per codec.
Format moves time in two ways:
1. **Spacing.** A format that spreads neighbours further apart lowers a codec's noise ÷ band,
   so a `binary-hamming` scan needs fewer f32 re-scores for the same recall
   ([two-stage re-scoring](../2026-11-two-stage-rescoring/): depth ∝ (noise ÷ band)^1.29 at 95%).
2. **Tokens.** What a retrieved record costs the chat model to read. Measured as size per
   record; no model is called.

**Hypothesis.**
1. The best format depends on the query type. Templated, field-labelled layouts (key-value,
   JSON, table) match attribute lookups best at f32. They also repeat the same keys in every
   record, which should pack neighbours closer together. Their compressed scans would then
   need deeper re-scores, so a format can win at f32 and lose at a tight time budget.
2. Format acts on compressed retrieval only through two numbers: f32 nDCG (how well the
   query matches the text) and noise ÷ band (how well the ranking survives the codec). If so,
   the noise ÷ band curve fitted on some formats predicts the re-score depth on a held-out
   format about as well as it did across corpora (within a factor of 1.25–1.41). Choosing a
   format would then need only those two numbers from a sample, not a full grid.

**Result.** Not yet run.

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
| Machine | Apple M4 Pro, release build |
| Command | `./run.sh` (results in `results/`; `RECORDS`, `QUERIES`, `FORMATS`, `MODELS` override the defaults) |

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
  played as guard / offensive tackle?". Relevant to every record with both values, and kept
  only when that is at most three records.
- **`descriptive`**: one attribute value shared by 5 to 50 records, for example "people who
  died in columbus , ohio". Relevant to all of them.

The queries use their own phrasing ("was born in", "played as"), not the field labels the
documents use ("birth place", "position"). That way no infobox format gets the query's words for
free.

## Results

Not yet run. `summarise.py` prints, for each (model, query type):
- **A per-format table:** tokens, f32 nDCG@10, neighbour band, noise ÷ band and the depth
  needed for 95% recall@10 behind each first stage.
- **The best formats at 10, 25, 50 and 100% of an exact f32 scan's time**, each with the
  operating point that reaches it.

Then the two leave-one-format-out checks from hypothesis 2.

## Discussion

**Caveats, known before the run.**
- **`wiki` is not a controlled comparison.** The paragraph often lacks the infobox facts that
  the queries ask about, so its scores mix format with content. `paragraph` is the controlled
  prose format. Its templated sentences are stilted, so any prose disadvantage it shows may
  be from the template rather than from prose itself.
- **Judgments are exact-match on infobox values**, so they miss variants. "new york city"
  and "new york city , new york , u.s." are separate categories, and a record that matches
  in meaning but not in string counts as irrelevant. This affects every format equally, but
  it caps nDCG.
- **The queries are templated.** Real queries are messier and may favour prose more than
  these do.
- **One domain.** Biographies are short, field-heavy records. Long documents, code or
  tables with many rows may rank the formats differently. A second structured source is the
  natural next step.
- **Retrieval only.** Whether a format helps the chat model answer, as opposed to retrieve,
  needs a chat model and a judge. Neither is available locally yet, as for Experiments 1
  and 3.
- **Time is computed from per-vector costs**, not timed end to end, as in the zero-copy
  experiment.

**Next.**
- Embed one format and return another (for example, match on `kv` and hand the model
  `wiki`). Retrieval and reading may want different layouts. `ingest` stores the text it
  embeds, so this needs a separate payload text per chunk.

## References

1. Lebret, Grangier and Auli, "Neural Text Generation from Structured Data with Application to
   the Biography Domain", EMNLP 2016. [arXiv:1603.07771](https://arxiv.org/abs/1603.07771)
