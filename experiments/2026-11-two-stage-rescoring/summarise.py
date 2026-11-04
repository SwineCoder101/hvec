"""Summarise `bench rescore` JSON: the depth each first stage needs, against its noise-to-band
ratio, and the recall and nDCG of each (first, second, depth) cell.

    python3 -I summarise.py <results-dir> <k>
"""

import json
import math
import sys
from pathlib import Path


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
    return cov / ((sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5)


def fit(xs, ys):
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main(results: Path, k: int) -> None:
    profiles, cells = [], []
    for path in sorted(results.glob(f"*-k{k}.json")):
        col = path.name[: -len(f"-k{k}.json")]
        data = json.loads(path.read_text())
        for p in data["profiles"]:
            p["collection"] = col
            p["ratio"] = p["rank_noise"] / p["neighbour_band"]
            p["depth"] = {t: d for t, d in p["depth_for_recall"]}
            profiles.append(p)
        for c in data["cells"]:
            c["collection"] = col
            cells.append(c)

    print("Depth an exact (f32) re-score needs, as a multiple of k")
    print(f"{'collection':<20} {'first':<15} {'bytes':>5} {'noise/band':>10} {'recall@k':>8} "
          f"{'90%':>6} {'95%':>6} {'99%':>6} {'median full':>11}")
    for p in profiles:
        d = p["depth"]
        print(f"{p['collection']:<20} {p['first']:<15} {p['first_bytes']:>5} {p['ratio']:>10.3f} "
              f"{p['recall_at_k']:>8.4f} {d[0.9] / k:>6.1f} {d[0.95] / k:>6.1f} {d[0.99] / k:>6.1f} "
              f"{p['median_full_depth'] / k:>11.1f}")

    print()
    print(f"{len(profiles)} (collection, first stage) points")
    for t in (0.9, 0.95, 0.99):
        xs = [math.log(p["ratio"]) for p in profiles]
        ys = [math.log(p["depth"][t] / k) for p in profiles]
        a, b = fit(xs, ys)
        rho = pearson(ranks(xs), ranks(ys))
        # Leave one corpus out.
        errs = []
        for corpus in sorted({p["collection"].split("-")[0] for p in profiles}):
            tr = [p for p in profiles if not p["collection"].startswith(corpus + "-")]
            te = [p for p in profiles if p["collection"].startswith(corpus + "-")]
            a2, b2 = fit([math.log(p["ratio"]) for p in tr], [math.log(p["depth"][t] / k) for p in tr])
            for p in te:
                pred = math.exp(a2 + b2 * math.log(p["ratio"]))
                errs.append(abs(math.log(pred / (p["depth"][t] / k))))
        geo = math.exp(sum(errs) / len(errs))
        print(f"recall {t:.0%}: depth/k = {math.exp(a):.2f} × (noise/band)^{b:.2f};  spearman {rho:.3f};  "
              f"held-out corpus error: a factor of {geo:.2f} on average")

    print()
    print("Recall@k and nDCG@k by cell (mean over collections), and the best first stage per byte")
    keys = sorted({(c["first"], c["second"], c["depth"]) for c in cells}, key=lambda x: (x[0], x[1], x[2]))
    print(f"{'first':<15} {'second':<6} {'depth':>6} {'recall@k':>9} {'ndcg loss vs f32':>16}")
    # The exact f32 nDCG per collection comes from the spacing experiment's results.
    f32_ndcg = {}
    spacing = results.parent.parent / "2026-11-neighbour-spacing" / "results"
    for path in spacing.glob(f"*-k{k}.json"):
        f32_ndcg[path.name[: -len(f"-k{k}.json")]] = json.loads(path.read_text())[0]["ndcg_at_k"]
    for key in keys:
        sel = [c for c in cells if (c["first"], c["second"], c["depth"]) == key]
        rec = sum(c["recall_at_k"] for c in sel) / len(sel)
        losses = [f32_ndcg[c["collection"]] - c["ndcg_at_k"] for c in sel if c["collection"] in f32_ndcg]
        loss = f"{sum(losses) / len(losses):>16.4f}" if losses else f"{'-':>16}"
        print(f"{key[0]:<15} {key[1]:<6} {key[2]:>6} {rec:>9.4f} {loss}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
