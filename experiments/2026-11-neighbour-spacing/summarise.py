"""Summarise the bench recall JSON files: spacing per collection, then every codec's loss
against its noise-to-spacing ratio, and the rank correlation between the two.

    python3 -I summarise.py <results-dir> <k>
"""

import json
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
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def main(results: Path, k: str) -> None:
    rows = []
    print(f"{'collection':<20} {'vectors':>8} {'queries':>8} {'gap k:k+1':>10} {'band k:2k':>10} {'f32 ndcg':>9}")
    for path in sorted(results.glob(f"*-k{k}.json")):
        col = path.name[: -len(f"-k{k}.json")]
        reports = json.loads(path.read_text())
        f32 = reports[0]
        print(f"{col:<20} {f32['vectors']:>8} {f32['queries']:>8} {f32['boundary_gap']:>10.5f} "
              f"{f32['neighbour_band']:>10.5f} {f32['ndcg_at_k'] or 0:>9.4f}")
        for r in reports[1:]:
            rows.append((col, r, f32))
    print()
    print(f"{'collection':<20} {'codec':<15} {'bytes':>6} {'noise':>8} {'noise/band':>10} "
          f"{'recall@k':>9} {'ndcg loss':>9}")
    for col, r, f32 in rows:
        ndcg_loss = (f32["ndcg_at_k"] or 0) - (r["ndcg_at_k"] or 0)
        print(f"{col:<20} {r['codec']:<15} {r['bytes_per_vector']:>6} {r['rank_noise']:>8.5f} "
              f"{r['rank_noise'] / r['neighbour_band']:>10.3f} {r['recall_at_k']:>9.4f} {ndcg_loss:>9.4f}")
    lossy = [(c, r, f) for c, r, f in rows if r["rank_noise"] > 0]
    ratio = [r["rank_noise"] / r["neighbour_band"] for _, r, _ in lossy]
    noise = [r["rank_noise"] for _, r, _ in lossy]
    loss = [1 - r["recall_at_k"] for _, r, _ in lossy]
    ndcg = [(f["ndcg_at_k"] or 0) - (r["ndcg_at_k"] or 0) for _, r, f in lossy]
    print()
    print(f"{len(lossy)} (collection, codec) points")
    print(f"spearman(noise/band, 1 - recall@k) = {spearman(ratio, loss):.3f}")
    print(f"spearman(noise alone, 1 - recall@k) = {spearman(noise, loss):.3f}")
    print(f"spearman(noise/band, ndcg loss)     = {spearman(ratio, ndcg):.3f}")
    print(f"spearman(noise alone, ndcg loss)    = {spearman(noise, ndcg):.3f}")
    # Within each codec: does the corpus with the tighter band lose more?
    print()
    print("within each codec, across collections: spearman(band, 1 - recall@k)")
    for codec in sorted({r["codec"] for _, r, _ in lossy}):
        pts = [(r["neighbour_band"], 1 - r["recall_at_k"]) for _, r, _ in lossy if r["codec"] == codec]
        if len(pts) >= 3:
            print(f"  {codec:<15} {spearman([p[0] for p in pts], [p[1] for p in pts]):>7.3f}  (n={len(pts)})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
