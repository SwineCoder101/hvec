"""Convert a CoIR dataset (Hugging Face parquet) into BEIR's directory layout.

    uv run --no-project --with pyarrow python -I coir_to_beir.py <parquet-dir> <out-dir>

<parquet-dir> holds corpus.parquet, queries.parquet and test.parquet as published under
huggingface.co/datasets/CoIR-Retrieval/<name>. <out-dir> gets corpus.jsonl, queries.jsonl and
qrels/test.tsv, which `hvec questions import-beir` reads.
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main(src: Path, out: Path) -> None:
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    for name in ("corpus", "queries"):
        rows = pq.read_table(src / f"{name}.parquet", columns=["_id", "title", "text"]).to_pylist()
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"_id": r["_id"], "title": r["title"] or "", "text": r["text"] or ""}) + "\n")
    qrels = pq.read_table(src / "test.parquet").to_pylist()
    with open(out / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in qrels:
            f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
