"""Summarise `bench scan --json` reports.

    python3 -I summarise.py <results-dir>

For each collection: the storage paths with their speed-up over the original row scan, the cost
of building the buffer, and each codec's scan (ms/query, ns per vector, GB/s, share of the measured
memory bandwidth, recall@10). Then, per collection, the hybrid estimate: a scan plus N f32
re-scores, using the measured per-vector costs.
"""

import json
import sys
from pathlib import Path


def main(results: Path) -> None:
    for path in sorted(results.glob("*.json")):
        r = json.loads(path.read_text())
        n = r["vectors"]
        print(f"== {r['collection']}: {n} vectors × {r['dimension']} dims, {r['queries']} queries, k={r['k']}")
        base = next((p["ms_per_query"] for p in r["paths"] if p["path"].startswith("sqlite rows, copied")), None)
        print(f"   {'storage path (f32)':<34} {'ms/query':>9} {'speed-up':>9}")
        for p in r["paths"]:
            up = f"{base / p['ms_per_query']:.1f}×" if base else "-"
            print(f"   {p['path']:<34} {p['ms_per_query']:>9.2f} {up:>9}")
        print(f"   {'building the buffer':<34} {'ms':>9} {'MB':>9}")
        for l in r["loads"]:
            print(f"   {l['source']:<34} {l['ms']:>9.1f} {l['bytes'] / 1e6:>9.1f}")
        bw = {b["threads"]: b["gb_per_s"] for b in r["bandwidth"]}
        print(f"   memory bandwidth: " + ", ".join(f"{t} thread(s) {g:.0f} GB/s" for t, g in sorted(bw.items())))
        print(f"   {'codec':<16} {'B/vec':>6} {'thr':>4} {'ms/query':>9} {'ns/vector':>10} {'GB/s':>7} {'% bw':>6} {'recall@10':>10}")
        for c in r["codecs"]:
            ns = c["ms_per_query"] * 1e6 / n
            share = 100 * c["gb_per_s"] / bw[c["threads"]] if c["threads"] in bw else float("nan")
            print(f"   {c['codec']:<16} {c['bytes_per_vector']:>6} {c['threads']:>4} {c['ms_per_query']:>9.3f} {ns:>10.2f} "
                  f"{c['gb_per_s']:>7.2f} {share:>5.1f}% {c['recall_at_k']:>10.4f}")
        one = {c["codec"]: c for c in r["codecs"] if c["threads"] == 1}
        if "f32" in one:
            f32_ns = one["f32"]["ms_per_query"] * 1e6 / n
            print(f"   hybrid estimate, 1 thread: scan + N f32 re-scores at {f32_ns:.1f} ns each (random access ignored)")
            for scan in ("binary-hamming", "binary", "pq48"):
                if scan not in one:
                    continue
                s_ms = one[scan]["ms_per_query"]
                cells = ", ".join(f"N={d}: {s_ms + d * f32_ns / 1e6:.3f} ms" for d in (50, 100, 200))
                print(f"     {scan:<16} scan {s_ms:.3f} ms → {cells}  (exact f32 scan {one['f32']['ms_per_query']:.3f} ms)")
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
