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

Lines starting with `#` are comments. Keep sets small and sharp; a few hundred questions is
plenty for a first experiment. Name the file after the corpus it belongs to.
