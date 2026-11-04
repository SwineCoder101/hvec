"""Replay depth-routing policies from `bench rescore --trace` output.

    python3 -I analyse.py <traces-dir> <k>

With an exact (f32) second stage, a query's recall@k at depth N is the share of its true top-k
neighbours that the scan ranks within N, so every policy below is scored exactly from the traces:

  fixed     the same depth N for every query
  margin    every candidate within λσ of the scan's own k-th score (σ calibrated on self-queries)
  ratio     depth = a·k·(σ / band_hat)^β, band_hat read from the scan's own scores (free at query time)
  oracle    the best per-query depths for a mean-depth budget, in hindsight (Lagrangian, exact)

Each policy traces a curve of mean depth against mean recall@k. They are compared at equal mean
depth. Headroom = oracle − fixed; captured = (policy − fixed) / headroom. The second part asks the
per-collection question: predict each collection's cheapest (scan codec, fixed depth) for a recall
target from noise ÷ band, with the curve fitted on the other corpora, and check it.
"""

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CAP = 1000
BUDGETS = [15, 20, 30, 50, 100]
TARGETS = [0.90, 0.95, 0.99]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = mean(ra), mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


class Trace:
    def __init__(self, line):
        d = json.loads(line)
        self.collection, self.first, self.k = d["collection"], d["first"], d["k"]
        self.first_bytes, self.vectors, self.noise = d["first_bytes"], d["vectors"], d["calibrated_noise"]
        self.margins = d["trace_margins"]
        self.ranks, self.band, self.band_hat = d["ranks"], d["band"], d["band_hat"]
        self.margin_depths = d["margin_depths"]
        self.n = len(self.ranks)

    def recall(self, q, depth):
        return sum(1 for r in self.ranks[q] if r <= depth) / self.k

    def evaluate(self, depths):
        rec = [self.recall(q, d) for q, d in enumerate(depths)]
        return dict(depth=mean(depths), recall=mean(rec), p10=pct(rec, 0.10), p90_depth=pct(depths, 0.90))

    # Policy curves: lists of evaluate() results, ordered by mean depth.
    def fixed_curve(self):
        return [self.evaluate([d] * self.n) for d in sorted({self.k, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500, CAP})]

    def margin_curve(self):
        return [self.evaluate([md[i] for md in self.margin_depths]) for i in range(len(self.margins))]

    def ratio_curve(self, beta):
        out = []
        for a in [0.25 * 1.25 ** i for i in range(30)]:
            depths = []
            for bh in self.band_hat:
                r = self.noise / bh if bh > 0 else 50.0
                depths.append(int(min(CAP, max(self.k, round(a * self.k * r ** beta)))))
            out.append(self.evaluate(depths))
        return sorted(out, key=lambda e: e["depth"])

    def oracle_curve(self):
        # Per query, the candidate depths are k and each true neighbour's rank (≤ CAP); recall is a
        # step function of depth, so the optimum for a budget is on these points. For each
        # multiplier λ every query maximises recall − λ·depth independently.
        options = []
        for q in range(self.n):
            pts = sorted({self.k} | {r for r in self.ranks[q] if self.k < r <= CAP})
            options.append([(d, self.recall(q, d)) for d in pts])
        out = []
        for lam in [0.0] + [10 ** (e / 8) for e in range(-48, 9)]:
            depths = []
            for opts in options:
                best = max(opts, key=lambda o: (o[1] - lam * o[0], -o[0]))
                depths.append(best[0])
            out.append(self.evaluate(depths))
        return sorted(out, key=lambda e: e["depth"])


def at_budget(curve, budget):
    """Linear interpolation of recall and p10 along a curve at a mean depth."""
    for a, b in zip(curve, curve[1:]):
        if a["depth"] <= budget <= b["depth"]:
            t = 0.0 if b["depth"] == a["depth"] else (budget - a["depth"]) / (b["depth"] - a["depth"])
            return {key: a[key] + t * (b[key] - a[key]) for key in ("recall", "p10", "p90_depth")}
    return None


def main(traces_dir: Path, k: int) -> None:
    def lines(p):
        text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
        return [l for l in text.splitlines() if l.strip()]

    files = sorted(traces_dir.glob("*.jsonl")) + sorted(traces_dir.glob("*.jsonl.gz"))
    traces = [Trace(l) for p in files for l in lines(p)]
    print(f"{len(traces)} (collection, scan codec) traces, {sum(t.n for t in traces)} query traces in all\n")

    # Does the scan's own band track the exact one, query by query?
    print("1. Can a router see the band? Spearman, per trace, over queries")
    rows = defaultdict(list)
    for t in traces:
        full = [max(r) for r in t.ranks]
        rows[t.first].append((spearman(t.band_hat, t.band), spearman([1 / b if b > 0 else 1e9 for b in t.band_hat], full)))
    for first, v in sorted(rows.items()):
        print(f"   {first:<15} band_hat vs exact band {mean(x[0] for x in v):.3f}   1/band_hat vs depth for full recall {mean(x[1] for x in v):.3f}")
    print("   (how much per-query depth needed varies:",
          f"median p90/p10 ratio of full-recall depth {mean(pct([max(r) for r in t.ranks], .9) / max(1, pct([max(r) for r in t.ranks], .1)) for t in traces):.1f}×)")

    print("\n2. Per-query routing at equal mean depth (means over traces). Δ against fixed depth")
    print(f"   {'budget':>6} {'fixed':>7} {'margin':>8} {'ratio β=1':>10} {'ratio β=1.3':>11} {'oracle':>8} "
          f"{'headroom':>8} {'margin cap.':>11} {'ratio cap.':>10} {'p10 fixed→margin':>17}")
    table = defaultdict(lambda: defaultdict(list))
    for t in traces:
        curves = dict(fixed=t.fixed_curve(), margin=t.margin_curve(), r1=t.ratio_curve(1.0),
                      r13=t.ratio_curve(1.29), oracle=t.oracle_curve())
        for b in BUDGETS:
            vals = {name: at_budget(c, b) for name, c in curves.items()}
            if any(v is None for v in vals.values()):
                continue
            for name, v in vals.items():
                table[b][name].append(v)
            table[b]["first"].append(t.first)
    for b in BUDGETS:
        row = table[b]
        if not row["fixed"]:
            continue
        r = {name: mean(v["recall"] for v in row[name]) for name in ("fixed", "margin", "r1", "r13", "oracle")}
        head = r["oracle"] - r["fixed"]
        cap = lambda name: (r[name] - r["fixed"]) / head if head > 1e-9 else float("nan")
        best_ratio = max(("r1", "r13"), key=lambda n: r[n])
        p10f, p10m = mean(v["p10"] for v in row["fixed"]), mean(v["p10"] for v in row["margin"])
        print(f"   {b:>6} {r['fixed']:>7.4f} {r['margin'] - r['fixed']:>+8.4f} {r['r1'] - r['fixed']:>+10.4f} "
              f"{r['r13'] - r['fixed']:>+11.4f} {r['oracle'] - r['fixed']:>+8.4f} {head:>8.4f} {cap('margin'):>11.0%} "
              f"{cap(best_ratio):>10.0%} {p10f:>8.3f}→{p10m:.3f}   (n={len(row['fixed'])})")

    print("\n   by scan codec, at a mean depth of 30: fixed recall, then Δ for margin, best ratio, oracle")
    b = 30
    by = defaultdict(lambda: defaultdict(list))
    for i, first in enumerate(table[b]["first"]):
        for name in ("fixed", "margin", "r1", "r13", "oracle"):
            by[first][name].append(table[b][name][i]["recall"])
    for first in sorted(by):
        r = {n: mean(v) for n, v in by[first].items()}
        print(f"   {first:<15} fixed {r['fixed']:.4f}  margin {r['margin'] - r['fixed']:+.4f}  "
              f"ratio {max(r['r1'], r['r13']) - r['fixed']:+.4f}  oracle {r['oracle'] - r['fixed']:+.4f}")
    print("   p90 depth at a mean of 30: fixed 30, margin "
          f"{mean(v['p90_depth'] for v in table[b]['margin']):.0f}, oracle {mean(v['p90_depth'] for v in table[b]['oracle']):.0f}")

    # Per-collection routing: choose the scan codec and a fixed depth for a target recall.
    print("\n3. Per-collection routing: the cheapest (scan codec, fixed depth) reaching a recall target")
    print("   cost = bytes read per query = vectors × scan bytes + depth × 1536; RAM = scan bytes per vector")
    by_col = defaultdict(list)
    for t in traces:
        by_col[t.collection].append(t)
    if len({c.split("-")[0] for c in by_col}) < 2:
        print("   (needs at least two corpora: the depth curve is fitted on the others)")
        return

    def needed(t, target):
        # Smallest fixed depth whose mean recall reaches the target (exact, from the ranks).
        ranks = sorted(r for rs in t.ranks for r in rs)
        need = math.ceil(target * len(ranks))
        return max(t.k, ranks[need - 1]) if need > 0 else t.k

    for target in TARGETS:
        pts = []
        for col, ts in by_col.items():
            for t in ts:
                ratio = t.noise / sorted(t.band)[len(t.band) // 2]
                pts.append((col, t, ratio, needed(t, target)))
        corpora = sorted({c.split("-")[0] for c in by_col})
        hits, regret, ram_match = 0, [], 0
        for col in sorted(by_col):
            corpus = col.split("-")[0]
            train = [p for p in pts if p[0].split("-")[0] != corpus]
            xs = [math.log(p[2]) for p in train]
            ys = [math.log(p[3] / k) for p in train]
            mx, my = mean(xs), mean(ys)
            beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            alpha = my - beta * mx
            mine = [p for p in pts if p[0] == col]
            cost = lambda t, d: t.vectors * t.first_bytes + d * 1536
            best = min(mine, key=lambda p: (cost(p[1], p[3]), p[1].first_bytes))
            pred = []
            for p in mine:
                d = int(min(CAP, max(k, round(k * math.exp(alpha + beta * math.log(p[2]))))))
                pred.append((p, d))
            chosen, d = min(pred, key=lambda x: (cost(x[0][1], x[1]), x[0][1].first_bytes))
            got = chosen[1].evaluate([d] * chosen[1].n)["recall"]
            ok = got >= target - 0.005
            hits += ok
            regret.append(cost(chosen[1], d) / cost(best[1], best[3]))
            ram_match += chosen[1].first_bytes == best[1].first_bytes
        print(f"   target {target:.0%}: predicted choice reaches the target (within 0.005) in {hits}/{len(by_col)} collections; "
              f"cost vs the best choice ×{mean(regret):.2f} on average; same scan size as the best in {ram_match}/{len(by_col)}")
    print("\n   best choice per collection at 95%:")
    for col in sorted(by_col):
        opts = [(t.vectors * t.first_bytes + needed(t, 0.95) * 1536, t.first, needed(t, 0.95), t.first_bytes) for t in by_col[col]]
        opts.sort()
        c, first, d, b = opts[0]
        cheapest_ram = min(opts, key=lambda o: (o[3], o[0]))
        print(f"   {col:<20} {first:<15} depth {d:>4}  read {c / (by_col[col][0].vectors * 1536):.4f} of f32   "
              f"(smallest scan: {cheapest_ram[1]} at depth {cheapest_ram[2]})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
