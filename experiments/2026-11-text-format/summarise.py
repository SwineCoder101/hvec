"""Summarise the text-format run: accuracy against time per query, per format and context.

    python3 -I summarise.py <results-dir> <k>

Reads recall/<format>-<model>-<qtype>.json (bench recall), rescore/<...>.json (bench rescore),
scan/<format>-<model>.json (bench scan) and stats.json (record sizes). A context is an
(embedding model, query type) pair. For each context it prints:

1. Per format: tokens per record, exact f32 nDCG@k, the neighbour band, each first stage's noise
   over band and the depth an f32 re-score needs for 95% recall@k behind it.
2. The frontier: every operating point (exact f32; a first-stage scan alone; a scan plus an f32
   re-score of N candidates, fixed or adaptive), timed as scan + N x the f32 cost per vector on
   one thread, and the best format at budgets of 10, 25, 50 and 100% of an exact f32 scan.

Then the two cross-format checks:
- depth: does noise / band predict the re-score depth on a format it was not fitted on?
- accuracy: does (f32 nDCG, noise / band) predict a hybrid's nDCG on a held-out format?
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

QTYPES = ("lookup", "descriptive")
BUDGETS = (0.10, 0.25, 0.50, 1.00)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def fit(xs, ys):
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def split_name(stem, formats):
    """`<format>-<model>[-<qtype>]` -> (format, model, qtype or None). Formats may contain '-'."""
    qtype = None
    for q in QTYPES:
        if stem.endswith("-" + q):
            qtype, stem = q, stem[: -len(q) - 1]
    fmt = max((f for f in formats if stem.startswith(f + "-")), key=len)
    return fmt, stem[len(fmt) + 1:], qtype


def policy_name(p):
    return {"none": "scan only", "fixed": f"N={p.get('depth')}", "margin": f"{p.get('sigmas')}σ"}[p["kind"]]


def main(results: Path, k: int) -> None:
    stats = json.loads((results / "stats.json").read_text())
    formats = list(stats["formats"])

    cost = {}  # (format, model) -> {codec: ms per query, "f32_ns": ns per vector, "vectors": n}
    for p in sorted((results / "scan").glob("*.json")):
        r = json.loads(p.read_text())
        fmt, model, _ = split_name(p.stem, formats)
        c = {x["codec"]: x["ms_per_query"] for x in r["codecs"] if x["threads"] == 1}
        c["f32_ns"] = c["f32"] * 1e6 / r["vectors"]
        c["vectors"] = r["vectors"]
        cost[(fmt, model)] = c

    rows = defaultdict(dict)  # (model, qtype) -> format -> row
    profiles = []  # one per (format, model, qtype, first stage), for the cross-format checks
    for p in sorted((results / "recall").glob("*.json")):
        fmt, model, qtype = split_name(p.stem, formats)
        if (fmt, model) not in cost:
            continue
        c = cost[(fmt, model)]
        recall = {r["codec"]: r for r in json.loads(p.read_text())}
        resc = json.loads((results / "rescore" / p.name).read_text())
        f32 = recall["f32"]
        row = {
            "tokens": stats["formats"][fmt]["mean_tokens"],
            "ndcg": f32["ndcg_at_k"],
            "band": f32["neighbour_band"],
            "ratio": {n: r["rank_noise"] / f32["neighbour_band"] for n, r in recall.items() if n != "f32"},
            "depth95": {},
            "points": [("exact f32", c["f32"], f32["ndcg_at_k"])],
        }
        for pr in resc["profiles"]:
            d = dict((t, n) for t, n in pr["depth_for_recall"])
            row["depth95"][pr["first"]] = d.get(0.95)
            profiles.append({"format": fmt, "model": model, "qtype": qtype, "first": pr["first"],
                             "ratio": pr["rank_noise"] / pr["neighbour_band"], "depth": d, "f32_ndcg": f32["ndcg_at_k"],
                             "cells": {}})
        by_first = {pp["first"]: pp for pp in profiles[-len(resc["profiles"]):]} if resc["profiles"] else {}
        for cell in resc["cells"]:
            if cell["first"] not in c:
                continue
            ms = c[cell["first"]] + cell["mean_depth"] * c["f32_ns"] / 1e6
            row["points"].append((f"{cell['first']} {policy_name(cell['policy'])}", ms, cell["ndcg_at_k"]))
            if cell["policy"]["kind"] == "fixed" and cell["first"] in by_first:
                by_first[cell["first"]]["cells"][cell["policy"]["depth"]] = cell["ndcg_at_k"]
        row["f32_ms"] = c["f32"]
        rows[(model, qtype)][fmt] = row

    firsts = sorted({p["first"] for p in profiles})
    for (model, qtype), fr in sorted(rows.items()):
        print(f"== {model}, {qtype} queries")
        head = f"{'format':<13} {'tokens':>6} {'f32 nDCG':>8} {'band':>7}"
        for f in firsts:
            head += f" {f + ' n/b':>18} {'N@95%':>6}"
        print(head)
        for fmt in formats:
            if fmt not in fr:
                continue
            r = fr[fmt]
            line = f"{fmt:<13} {r['tokens']:>6.0f} {r['ndcg']:>8.4f} {r['band']:>7.4f}"
            for f in firsts:
                line += f" {r['ratio'].get(f, float('nan')):>18.3f} {r['depth95'].get(f) or '-':>6}"
            print(line)
        f32_ms = sum(r["f32_ms"] for r in fr.values()) / len(fr)
        print(f"best nDCG@{k} within a time budget (exact f32 scan ≈ {f32_ms:.2f} ms/query, one thread)")
        for b in BUDGETS:
            limit = b * f32_ms
            best = []
            for fmt, r in fr.items():
                pts = [p for p in r["points"] if p[1] <= limit]
                if pts:
                    name, ms, nd = max(pts, key=lambda p: p[2])
                    best.append((nd, fmt, name, ms))
            best.sort(reverse=True)
            cells = "; ".join(f"{fmt} {nd:.4f} ({name}, {ms:.2f} ms)" for nd, fmt, name, ms in best[:3])
            print(f"  {100 * b:>4.0f}% ({limit:.2f} ms): {cells or 'nothing fits'}")
        print()

    if len({p["format"] for p in profiles}) < 3:
        return
    print("Cross-format checks, leaving one format out at a time")
    for t in (0.9, 0.95, 0.99):
        pts = [p for p in profiles if p["depth"].get(t)]
        xs = [math.log(p["ratio"]) for p in pts]
        ys = [math.log(max(p["depth"][t], k) / k) for p in pts]
        errs = []
        for fmt in formats:
            tr = [(x, y) for x, y, p in zip(xs, ys, pts) if p["format"] != fmt]
            te = [(x, y) for x, y, p in zip(xs, ys, pts) if p["format"] == fmt]
            if len(tr) < 3 or not te:
                continue
            a, b = fit(*zip(*tr))
            errs += [abs(a + b * x - y) for x, y in te]
        a, b = fit(xs, ys)
        print(f"  depth for {t:.0%} recall: Spearman ρ(noise/band, depth) = {pearson(ranks(xs), ranks(ys)):.3f}, "
              f"depth ≈ {math.exp(a):.2f}·k·r^{b:.2f}, held-out format within a factor of "
              f"{math.exp(sum(errs) / len(errs)):.2f} on average ({len(pts)} points)")
    depths = sorted({d for p in profiles for d in p["cells"]})
    for d in depths:
        pts = [p for p in profiles if d in p["cells"] and p["f32_ndcg"]]
        if len(pts) < 4:
            continue
        # The share of exact f32's nDCG a hybrid keeps, against log noise/band.
        xs = [math.log(p["ratio"]) for p in pts]
        ys = [p["cells"][d] / p["f32_ndcg"] for p in pts]
        errs, naive = [], []
        for fmt in formats:
            tr = [(x, y) for x, y, p in zip(xs, ys, pts) if p["format"] != fmt]
            te = [(x, y, p) for x, y, p in zip(xs, ys, pts) if p["format"] == fmt]
            if len(tr) < 3 or not te:
                continue
            a, b = fit(*zip(*tr))
            mean = sum(y for _, y in tr) / len(tr)
            for x, y, p in te:
                errs.append(abs((a + b * x - y) * p["f32_ndcg"]))
                naive.append(abs((mean - y) * p["f32_ndcg"]))
        if errs:
            print(f"  hybrid nDCG at N={d}: predicted from f32 nDCG and noise/band within {sum(errs) / len(errs):.4f} "
                  f"on a held-out format (f32 nDCG times the mean retained share: {sum(naive) / len(naive):.4f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
