# Question sets

JSON Lines, one question per line, used by `hvec bench run`:

```json
{"id": "q1", "question": "What is the refund window?", "answers": ["30 days", "thirty days"], "source": "refunds.md"}
```

- `id`: unique within the file.
- `question`: sent to the pipeline verbatim.
- `answers`: every acceptable surface form. Scoring normalises case, punctuation, whitespace and
  leading articles, then checks equality (`exact`) and substring (`contains`).
- `source` (optional): a substring of the chunk source path that retrieval should hit. Lets the
  retrieval step be scored without a chat model.
- `relevant` (optional): document ids judged relevant, BEIR style. A chunk belongs to a document
  when the file stem of its source path equals the id (`corpus/14717.txt` is document `14717`).
  Questions with judgments are scored with nDCG@k and document recall@k as well as a hit flag,
  and `bench recall --questions` uses them to measure codecs against the labels.
- `query` (optional): the text embedded for retrieval when it should differ from the question
  the chat model sees. The SciFact set uses the bare claim as the query and a verdict question
  as the prompt.

Lines starting with `#` are comments. Keep sets small and sharp; a few hundred questions is
plenty for a first experiment. Name the file after the corpus it belongs to.

## Sets in this directory

- `hvec-docs.jsonl`: a dozen questions about this repository's own documentation. A smoke test.
- `scifact.jsonl`: the 300 BEIR SciFact test claims with their 339 relevance judgments, built by
  `hvec questions import-beir`; 188 carry a supports/refutes gold answer from the original
  SciFact release. The corpus is written by the same command with `--corpus-out`.
- `memory-smoke.jsonl`: an agent memory script for `hvec bench memory`, with `write` and `ask`
  steps instead of questions.

## Importing a BEIR dataset

```sh
hvec questions import-beir path/to/dataset --out questions/<name>.jsonl --corpus-out corpus/<name>/ [--split test]
hvec questions check questions/<name>.jsonl
```
