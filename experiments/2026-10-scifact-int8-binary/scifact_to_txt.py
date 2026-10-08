"""Convert BEIR SciFact to one .txt per document plus a query file. Stdlib only."""
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
docs = out / "corpus"; docs.mkdir(parents=True, exist_ok=True)
n = 0
with open(src / "corpus.jsonl") as f:
    for line in f:
        d = json.loads(line)
        (docs / f"{d['_id']}.txt").write_text((d.get("title", "") + "\n\n" + d.get("text", "")).strip() + "\n")
        n += 1
test_ids = set()
with open(src / "qrels" / "test.tsv") as f:
    next(f)
    for line in f:
        test_ids.add(line.split("\t")[0])
qs = []
with open(src / "queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        if q["_id"] in test_ids:
            qs.append(q["text"].replace("\n", " ").strip())
(out / "queries.txt").write_text("\n".join(qs) + "\n")
print(f"docs: {n}, test queries: {len(qs)}")
