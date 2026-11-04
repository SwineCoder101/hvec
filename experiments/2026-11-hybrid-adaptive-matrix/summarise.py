"""Summarise the hybrid × adaptive matrix from `bench rescore` JSON.

    python3 -I summarise.py <results-dir> <k> <spacing-results-dir>

The spacing experiment's results give each collection's exact f32 nDCG, the ceiling that nDCG
loss is measured from. Prints:
  1. the matrix: every (scan codec, second stage, policy) averaged over collections;
  2. adaptive storage alone: mixed<B> against uniform codecs at equal bytes, no re-score;
  3. adaptive depth against fixed depth at the same mean depth (fixed interpolated in log depth);
  4. benefits and drawbacks of each family of combination against the plain 48-byte binary scan.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def policy_name(p):
    if p["kind"] == "none":
        return "none"
    if p["kind"] == "fixed":
        return f"fixed {p['depth']}"
    return f"margin {p['sigmas']:g}σ"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load(results: Path, k: int, spacing: Path):
    ceiling = {}
    for path in spacing.glob(f"*-k{k}.json"):
        ceiling[path.name[: -len(f"-k{k}.json")]] = json.loads(path.read_text())[0]["ndcg_at_k"]
    cells = []
    for path in sorted(results.glob(f"*-k{k}.json")):
        col = path.name[: -len(f"-k{k}.json")]
        data = json.loads(path.read_text())
        for c in data["cells"]:
            c["collection"] = col
            c["model"] = col.split("-", 1)[1]
            c["policy_name"] = policy_name(c["policy"])
            c["ndcg_loss"] = ceiling[col] - c["ndcg_at_k"] if col in ceiling and c["ndcg_at_k"] is not None else None
            c["f32_scan"] = c["vectors"] * 4 * 384
            cells.append(c)
    return cells


def interp_fixed(fixed_pts, depth):
    """Linear interpolation of (depth, value) in log depth; None outside the measured range."""
    pts = sorted(fixed_pts)
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= depth <= d1:
            t = (math.log(depth) - math.log(d0)) / (math.log(d1) - math.log(d0)) if d1 > d0 else 0.0
            return v0 + t * (v1 - v0)
    return None


def main(results: Path, k: int, spacing: Path) -> None:
    cells = load(results, k, spacing)
    cols = sorted({c["collection"] for c in cells})
    print(f"{len(cols)} collections: {', '.join(cols)}\n")

    groups = defaultdict(list)
    for c in cells:
        groups[(c["first"], c["second"], c["policy_name"])].append(c)

    def row(key):
        g = groups[key]
        return dict(
            first=key[0], second=key[1], policy=key[2], ram=g[0]["first_bytes"], disk=g[0]["second_bytes"],
            depth=mean(c["mean_depth"] for c in g), p90=mean(c["p90_depth"] for c in g),
            recall=mean(c["recall_at_k"] for c in g), p10=mean(c["p10_recall_at_k"] for c in g),
            top1=mean(c["top1_agreement"] for c in g),
            ndcg_loss=mean(c["ndcg_loss"] for c in g if c["ndcg_loss"] is not None),
            read=mean(c["bytes_read_per_query"] / c["f32_scan"] for c in g), n=len(g),
        )

    order = sorted(groups, key=lambda x: (x[0], x[1] != "none", x[1], x[2]))
    rows = [row(key) for key in order]
    print("1. The matrix (means over collections). ram = scanned bytes/vector, disk = second-stage bytes/vector,")
    print("   read = bytes touched per query as a fraction of a full f32 scan")
    hdr = f"{'scan codec':<15} {'second':<6} {'policy':<12} {'ram':>4} {'disk':>5} {'depth':>6} {'p90':>6} " \
          f"{'recall@10':>9} {'p10':>6} {'top1':>6} {'ndcg loss':>9} {'read':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['first']:<15} {r['second']:<6} {r['policy']:<12} {r['ram']:>4} {r['disk']:>5} {r['depth']:>6.1f} "
              f"{r['p90']:>6.0f} {r['recall']:>9.4f} {r['p10']:>6.3f} {r['top1']:>6.3f} {r['ndcg_loss']:>9.4f} "
              f"{r['read']:>7.4f}")

    print("\n2. Adaptive storage alone (no re-score), by byte budget; per model, then both")
    budgets = defaultdict(list)
    for r in rows:
        if r["policy"] == "none":
            budgets[r["ram"]].append(r)
    for b in sorted(budgets):
        for r in sorted(budgets[b], key=lambda r: -r["recall"]):
            per_model = []
            for m in ("bge-small", "minilm"):
                g = [c for c in groups[(r["first"], "none", "none")] if c["model"] == m]
                per_model.append(mean(c["recall_at_k"] for c in g))
            print(f"  {b:>3} B  {r['first']:<15} recall@10 {r['recall']:.4f} (bge {per_model[0]:.4f}, "
                  f"MiniLM {per_model[1]:.4f})  p10 {r['p10']:.3f}  ndcg loss {r['ndcg_loss']:.4f}")

    print("\n3. Adaptive depth against fixed depth at the same mean depth (per collection, then averaged)")
    print(f"   {'scan codec':<15} {'second':<6} {'margin':<6} {'mean depth':>10} {'Δrecall':>8} {'Δp10':>7} "
          f"{'Δndcg':>8} {'p90/mean':>8} {'cells':>5}")
    by_first_second = defaultdict(list)
    for c in cells:
        by_first_second[(c["first"], c["second"], c["collection"])].append(c)
    deltas = defaultdict(list)
    for (first, second, col), g in by_first_second.items():
        if second == "none":
            continue
        fixed = [c for c in g if c["policy"]["kind"] == "fixed"]
        none = groups[(first, "none", "none")]
        base = [c for c in none if c["collection"] == col][0]
        # A fixed depth of k is the no-re-score membership with an exact reorder; anchor the curve there.
        anchor_rec = base["recall_at_k"]
        pts_rec = [(c["mean_depth"], c["recall_at_k"]) for c in fixed] + [(k, anchor_rec)]
        pts_p10 = [(c["mean_depth"], c["p10_recall_at_k"]) for c in fixed] + [(k, base["p10_recall_at_k"])]
        pts_ndcg = [(c["mean_depth"], c["ndcg_at_k"]) for c in fixed if c["ndcg_at_k"] is not None]
        for c in g:
            if c["policy"]["kind"] != "margin":
                continue
            fr, fp = interp_fixed(pts_rec, c["mean_depth"]), interp_fixed(pts_p10, c["mean_depth"])
            fn = interp_fixed(pts_ndcg, c["mean_depth"]) if pts_ndcg else None
            if fr is None:
                continue
            deltas[(first, second, c["policy"]["sigmas"])].append((
                c["mean_depth"], c["recall_at_k"] - fr, c["p10_recall_at_k"] - fp,
                (c["ndcg_at_k"] - fn) if fn is not None else None,
                c["p90_depth"] / c["mean_depth"]))
    for key in sorted(deltas):
        d = deltas[key]
        print(f"   {key[0]:<15} {key[1]:<6} {key[2]:<6g} {mean(x[0] for x in d):>10.1f} "
              f"{mean(x[1] for x in d):>+8.4f} {mean(x[2] for x in d):>+7.3f} "
              f"{mean(x[3] for x in d if x[3] is not None):>+8.4f} {mean(x[4] for x in d):>8.2f} {len(d):>5}")
    allr = [x[1] for d in deltas.values() for x in d]
    allp = [x[2] for d in deltas.values() for x in d]
    if allr:
        wins = sum(1 for x in allr if x > 0)
        print(f"   overall: Δrecall {mean(allr):+.4f}, Δp10 {mean(allp):+.3f}; adaptive ahead in {wins} of {len(allr)} cells")

    print("\n4. Benefits and drawbacks against the plain 48-byte binary scan (binary, none)")
    base = row(("binary", "none", "none"))
    fam = [
        ("uniform scan, no re-score", lambda r: r["policy"] == "none" and not r["first"].startswith("mixed")),
        ("adaptive bits, no re-score", lambda r: r["policy"] == "none" and r["first"].startswith("mixed")),
        ("hybrid fixed depth, int8", lambda r: r["policy"].startswith("fixed") and r["second"] == "int8"),
        ("hybrid fixed depth, f32", lambda r: r["policy"].startswith("fixed") and r["second"] == "f32"),
        ("hybrid adaptive depth, int8", lambda r: r["policy"].startswith("margin") and r["second"] == "int8"),
        ("hybrid adaptive depth, f32", lambda r: r["policy"].startswith("margin") and r["second"] == "f32"),
    ]
    print(f"   baseline: recall {base['recall']:.4f}, p10 {base['p10']:.3f}, ndcg loss {base['ndcg_loss']:.4f}, "
          f"48 B/vector, read {base['read']:.4f} of an f32 scan")
    for name, pred in fam:
        sel = [r for r in rows if pred(r)]
        best = max(sel, key=lambda r: r["recall"])
        print(f"   {name:<28} best: {best['first']} {best['second']} {best['policy']}: recall {best['recall']:.4f} "
              f"({best['recall'] - base['recall']:+.4f}), p10 {best['p10']:.3f}, ndcg loss {best['ndcg_loss']:.4f}, "
              f"ram {best['ram']} B + disk {best['disk']} B, read {best['read']:.4f}, mean depth {best['depth']:.0f} "
              f"(p90 {best['p90']:.0f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
