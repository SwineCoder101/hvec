"""Write one WikiBio sample as several BEIR datasets, the same records in a different layout each.

    python3 -I wikibio_to_formats.py <test.parquet> <out-dir> [records] [queries-per-type] [seed]

Every format but `wiki` and `header-prose` holds exactly the same facts: the cleaned infobox
fields of each record. `wiki` is the record's own Wikipedia paragraph, which holds different
facts (it is the natural-prose reference), and `header-prose` is the infobox as a key-value
header above that paragraph.

The queries and judgments are the same for every format, from the infobox:
- `lookup`: two of a record's attributes ("who was born in X and played as Y?"), relevant
  to every record with both values, kept only when that is at most 3 records.
- `descriptive`: one attribute value shared by 5 to 50 records ("people who worked as X"),
  relevant to all of them.

Writes `<out>/<format>/{corpus.jsonl,queries.jsonl,qrels/lookup.tsv,qrels/descriptive.tsv}`,
`<out>/stats.json` (size per record per format) and `<out>/examples.md` (one record in every
format). Needs pyarrow.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FORMATS = ["paragraph", "bullets", "kv", "json", "table", "wiki", "header-prose"]

# Fields with no retrievable content (images, layout, update stamps).
DROP = {
    "article_title", "image", "image_size", "imagesize", "caption", "alt", "website", "signature",
    "pcupdate", "ntupdate", "image_caption", "logo", "landscape", "module", "footnotes",
}
MAX_FIELDS = 16
MAX_VALUE_WORDS = 25
MAX_WIKI_WORDS = 150
JUNK_VALUES = {"", "unknown", "none", "n/a", "--", "-", "?"}

# Query phrasing. Deliberately not the field labels the documents use, so no format gets the
# query's own words for free.
PHRASES = {
    "birth_date": "was born on {}",
    "birth_place": "was born in {}",
    "death_date": "died on {}",
    "death_place": "died in {}",
    "position": "played as {}",
    "occupation": "worked as {}",
    "nationality": "had {} nationality",
    "party": "belonged to the {} party",
    "alma_mater": "studied at {}",
    "genre": "worked in {}",
    "religion": "followed {}",
    "spouse": "was married to {}",
    "clubs": "played for {}",
    "residence": "lived in {}",
}
DESCRIPTIVE_FIELDS = ["occupation", "position", "nationality", "party", "genre", "religion",
                      "birth_place", "death_place", "alma_mater", "residence"]

PTB = {"-lrb-": "(", "-rrb-": ")", "-lsb-": "[", "-rsb-": "]", "-lcb-": "{", "-rcb-": "}",
       "``": '"', "''": '"'}
TOKEN = re.compile(r"\w+|[^\w\s]")


def clean(s: str) -> str:
    for k, v in PTB.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def record(row):
    t = row["input_text"]["table"]
    fields = {}
    for h, v in zip(t["column_header"], t["content"]):
        h = h.strip()
        v = clean(v)
        if h in DROP or h.endswith("image") or h.endswith("caption") or h in fields:
            continue
        if v.lower() in JUNK_VALUES or len(v.split()) > MAX_VALUE_WORDS:
            continue
        fields[h] = v
    name = fields.pop("name", None)
    if not name and "article_title" in t["column_header"]:
        title = clean(t["content"][t["column_header"].index("article_title")])
        name = re.sub(r"\s*\(.*?\)\s*$", "", title)
    if not name:
        return None
    fields = dict(list(fields.items())[: MAX_FIELDS - 1])
    if len(fields) < 3:
        return None
    words, wiki = 0, []
    for sent in row["target_text"].split("\n"):
        sent = clean(sent)
        if not sent:
            continue
        if wiki and words + len(sent.split()) > MAX_WIKI_WORDS:
            break
        wiki.append(sent)
        words += len(sent.split())
    return {"name": name, "fields": fields, "wiki": " ".join(wiki)}


def label(h: str) -> str:
    return h.replace("_", " ")


def render(r, fmt: str) -> str:
    name, f = r["name"], r["fields"]
    items = [("name", name)] + list(f.items())
    kv = "; ".join(f"{label(h)}: {v}" for h, v in items)
    if fmt == "paragraph":
        return " ".join([f"{name}."] + [f"{name}'s {label(h)} is {v}." for h, v in f.items()])
    if fmt == "bullets":
        return "\n".join(f"- {label(h)}: {v}" for h, v in items)
    if fmt == "kv":
        return kv
    if fmt == "json":
        return json.dumps(dict(items), ensure_ascii=False)
    if fmt == "table":
        head = "| " + " | ".join(label(h) for h, _ in items) + " |"
        rule = "|" + "---|" * len(items)
        row = "| " + " | ".join(v.replace("|", "/") for _, v in items) + " |"
        return "\n".join([head, rule, row])
    if fmt == "wiki":
        return r["wiki"]
    if fmt == "header-prose":
        return kv + "\n\n" + r["wiki"]
    raise ValueError(fmt)


def make_queries(recs, ids, n, rng):
    by_value = defaultdict(set)
    for i, r in zip(ids, recs):
        for h, v in r["fields"].items():
            if h in PHRASES:
                by_value[(h, v)].add(i)

    lookup, seen = [], set()
    order = list(range(len(recs)))
    rng.shuffle(order)
    for j in order:
        if len(lookup) >= n:
            break
        cand = [h for h in recs[j]["fields"] if h in PHRASES]
        if len(cand) < 2:
            continue
        a, b = sorted(rng.sample(cand, 2), key=list(PHRASES).index)
        va, vb = recs[j]["fields"][a], recs[j]["fields"][b]
        rel = by_value[(a, va)] & by_value[(b, vb)]
        if len(rel) > 3 or (a, va, b, vb) in seen:
            continue
        seen.add((a, va, b, vb))
        if (a, b) in (("birth_date", "birth_place"), ("death_date", "death_place")):
            text = f"who {PHRASES[a].format(va)} in {vb}?"
        else:
            text = f"who {PHRASES[a].format(va)} and {PHRASES[b].format(vb)}?"
        lookup.append((text, sorted(rel)))

    cats = [(h, v) for (h, v), s in by_value.items() if h in DESCRIPTIVE_FIELDS and 5 <= len(s) <= 50]
    cats.sort()
    rng.shuffle(cats)
    descriptive = [(f"people who {PHRASES[h].format(v)}", sorted(by_value[(h, v)])) for h, v in cats[:n]]
    return lookup, descriptive


def main(src: Path, out: Path, n_records: int, n_queries: int, seed: int) -> None:
    rng = random.Random(seed)
    rows = pq.read_table(src).to_pylist()
    order = list(range(len(rows)))
    rng.shuffle(order)
    recs, ids = [], []
    for i in order:
        r = record(rows[i])
        if r:
            recs.append(r)
            ids.append(f"wb{i:05d}")
        if len(recs) >= n_records:
            break
    lookup, descriptive = make_queries(recs, ids, n_queries, rng)

    queries = [{"_id": f"L{q:04d}", "text": t} for q, (t, _) in enumerate(lookup)]
    queries += [{"_id": f"D{q:04d}", "text": t} for q, (t, _) in enumerate(descriptive)]
    qrels = {
        "lookup": [(f"L{q:04d}", d) for q, (_, rel) in enumerate(lookup) for d in rel],
        "descriptive": [(f"D{q:04d}", d) for q, (_, rel) in enumerate(descriptive) for d in rel],
    }

    stats = {"records": len(recs), "lookup_queries": len(lookup), "descriptive_queries": len(descriptive),
             "seed": seed, "formats": {}}
    for fmt in FORMATS:
        d = out / fmt
        (d / "qrels").mkdir(parents=True, exist_ok=True)
        texts = [render(r, fmt) for r in recs]
        with open(d / "corpus.jsonl", "w") as fh:
            for i, t in zip(ids, texts):
                fh.write(json.dumps({"_id": i, "title": "", "text": t}, ensure_ascii=False) + "\n")
        with open(d / "queries.jsonl", "w") as fh:
            for q in queries:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        for split, rows_ in qrels.items():
            with open(d / "qrels" / f"{split}.tsv", "w") as fh:
                fh.write("query-id\tcorpus-id\tscore\n")
                fh.writelines(f"{q}\t{c}\t1\n" for q, c in rows_)
        toks = sorted(len(TOKEN.findall(t)) for t in texts)
        stats["formats"][fmt] = {
            "mean_bytes": sum(len(t.encode()) for t in texts) / len(texts),
            "mean_words": sum(len(t.split()) for t in texts) / len(texts),
            "mean_tokens": sum(toks) / len(toks),
            "p99_tokens": toks[int(0.99 * (len(toks) - 1))],
            "over_400_tokens": sum(t > 400 for t in toks),
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    ex = recs[0]
    md = [f"# One record ({ids[0]}) in every format\n"]
    for fmt in FORMATS:
        md += [f"## {fmt}\n", "```", render(ex, fmt), "```\n"]
    md += ["## Queries\n"] + [f"- lookup: {t}" for t, _ in lookup[:3]] + [f"- descriptive: {t}" for t, _ in descriptive[:3]]
    (out / "examples.md").write_text("\n".join(md) + "\n")
    print(f"{len(recs)} records, {len(lookup)} lookup and {len(descriptive)} descriptive queries, "
          f"{len(FORMATS)} formats in {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), int(a[2]) if len(a) > 2 else 20000,
         int(a[3]) if len(a) > 3 else 500, int(a[4]) if len(a) > 4 else 2026)
