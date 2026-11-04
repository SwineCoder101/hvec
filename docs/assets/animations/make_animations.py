"""Generate the README's animated SVGs for hybrid and adaptive quantization.

    python3 -I docs/assets/animations/make_animations.py

Writes hybrid-rescoring.svg, adaptive-depth.svg and adaptive-bits.svg next to this file. The
first two are schematic (synthetic scores, fixed seed). adaptive-bits.svg uses real data:
SciFact's bge-small-en-v1.5 vectors, their PCA spectrum, and the 48-byte bit allocations that
`mixed48-pca` and `mixed48-pca-w` make, stored in scifact-bge-small-alloc48.json.

The animations are CSS keyframes, so GitHub renders them through <img>. Under
prefers-reduced-motion they stop on their final, fully labelled frame. Colours are the dataviz
reference palette (blue and orange slots, validated all-pairs in both modes), with light and dark
steps selected by prefers-color-scheme.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

BASE_CSS = f"""
  svg {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
         --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --other:#898781;
         --band:rgba(42,120,214,0.10); --band2:rgba(42,120,214,0.20); }}
  @media (prefers-color-scheme: dark) {{
    svg {{ --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
           --s1:#3987e5; --s2:#d95926; --band:rgba(57,135,229,0.16); --band2:rgba(57,135,229,0.28); }}
  }}
  text {{ font-family: {FONT}; fill: var(--ink); }}
  .bg {{ fill: var(--surface); stroke: var(--grid); }}
  .h {{ font-size: 15px; font-weight: 600; }}
  .cap {{ font-size: 13px; fill: var(--ink2); }}
  .small {{ font-size: 11px; fill: var(--muted); }}
  .lbl {{ font-size: 12px; fill: var(--ink2); }}
  .num {{ font-size: 12px; fill: var(--ink); font-variant-numeric: tabular-nums; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .true {{ fill: var(--s1); stroke: var(--surface); stroke-width: 2; }}
  .other {{ fill: var(--other); fill-opacity: .55; stroke: var(--surface); stroke-width: 2; }}
  .lost {{ fill: var(--s2); stroke: var(--surface); stroke-width: 2; }}
  .cut {{ stroke: var(--ink); stroke-width: 1.5; stroke-dasharray: 4 3; }}
  .fixed {{ stroke: var(--s2); stroke-width: 2; stroke-dasharray: 5 3; }}
  .band {{ fill: var(--band); }}
  .band2 {{ fill: var(--band2); }}
"""
REDUCED = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"


def svg(w, h, title, desc, css, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">
<title id="t">{title}</title>
<desc id="d">{desc}</desc>
<style>{BASE_CSS}{css}
  {REDUCED}
</style>
<rect class="bg" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="8"/>
{body}
</svg>
"""


def phase(name, dur, stops):
    """A keyframe rule from [(percent, css)] stops."""
    frames = " ".join(f"{p}% {{ {css} }}" for p, css in stops)
    return f"@keyframes {name} {{ {frames} }}\n"


def captions(prefix, dur, items, y, x=24, cls="cap"):
    """Cross-fading captions; the last is the reduced-motion resting frame."""
    css, body = "", ""
    n = len(items)
    for i, (start, end, text) in enumerate(items):
        last = i == n - 1
        name = f"{prefix}{i}"
        a, b = start, end
        stops = [(0, "opacity:0"), (max(a - 0.01, 0), "opacity:0"), (min(a + 2, b), "opacity:1"),
                 (max(b - 2, a), "opacity:1"), (b, "opacity:0"), (100, "opacity:0")]
        if a == 0:
            stops = [(0, "opacity:1")] + stops[3:]
        css += phase(name, dur, stops)
        css += f".{name} {{ opacity:{1 if last else 0}; animation: {name} {dur}s infinite; }}\n"
        body += f'<text class="{cls} {name}" x="{x}" y="{y}">{text}</text>\n'
    return css, body


# ---------------------------------------------------------------------------
# 1. Hybrid re-scoring
# ---------------------------------------------------------------------------

def hybrid():
    rng = random.Random(11)
    n, k, depth = 44, 10, 30
    # Find a draw that shows the story: the code's top 10 keeps 6 true neighbours, the top 30
    # keeps 9, one is ranked beyond the shortlist.
    while True:
        exact = sorted((rng.gauss(0, 1) for _ in range(n)), reverse=True)
        approx = [e + rng.gauss(0, 0.55) for e in exact]
        order = sorted(range(n), key=lambda i: -approx[i])
        rank = {i: r for r, i in enumerate(order)}
        true = set(range(k))
        in10 = sum(1 for i in true if rank[i] < k)
        in30 = sum(1 for i in true if rank[i] < depth)
        if in10 == 6 and in30 == 9:
            break
    lost = next(i for i in true if rank[i] >= depth)
    short = set(order[:depth])
    rescored_order = sorted(short, key=lambda i: -exact[i])
    final10 = set(rescored_order[:k])
    assert len(final10 & true) == 9

    W, H = 820, 370
    x0, x1 = 50, 790
    lo = min(min(exact), min(approx)) - 0.1
    hi = max(max(exact), max(approx)) + 0.1
    X = lambda s: x0 + (s - lo) / (hi - lo) * (x1 - x0)
    lanes = [150, 166, 182, 198, 214, 230]
    dur = 16
    css = ""
    body = '<text class="h" x="24" y="32">Hybrid quantization: scan with a small code, re-score a shortlist exactly</text>\n'

    # The code's 10th and the shortlist edge, both on approx scores; the final 10th on exact.
    a_sorted = sorted(approx, reverse=True)
    cut10 = (a_sorted[k - 1] + a_sorted[k]) / 2
    cut30 = (a_sorted[depth - 1] + a_sorted[depth]) / 2
    e_short = sorted((exact[i] for i in short), reverse=True)
    final_cut = (e_short[k - 1] + e_short[k]) / 2

    # Shortlist band (phase 2 on) and top-10 zone.
    css += phase("sl", dur, [(0, "opacity:0"), (28, "opacity:0"), (32, "opacity:1"), (94, "opacity:1"), (100, "opacity:0")])
    css += ".sl { opacity:1; animation: sl %ds infinite; }\n" % dur
    body += f'<rect class="band sl" x="{X(cut30):.1f}" y="128" width="{x1 - X(cut30):.1f}" height="118" rx="4"/>\n'
    body += f'<text class="small sl" x="{X(cut30) + 6:.1f}" y="142">shortlist: the code\'s top {depth} (f32 fetched for these only)</text>\n'

    # Top-10 cut: appears at the code's 10th, then moves to the exact 10th after re-scoring.
    shift = X(final_cut) - X(cut10)
    css += phase("cut", dur, [(0, "opacity:0;transform:translateX(0)"), (12, "opacity:0;transform:translateX(0)"),
                              (16, "opacity:1;transform:translateX(0)"), (50, "opacity:1;transform:translateX(0)"),
                              (62, f"opacity:1;transform:translateX({shift:.1f}px)"),
                              (94, f"opacity:1;transform:translateX({shift:.1f}px)"),
                              (100, f"opacity:0;transform:translateX({shift:.1f}px)")])
    css += f".cutg {{ transform: translateX({shift:.1f}px); animation: cut {dur}s infinite; }}\n"
    body += (f'<g class="cutg"><line class="cut" x1="{X(cut10):.1f}" y1="148" x2="{X(cut10):.1f}" y2="252"/>'
             f'<text class="lbl" x="{X(cut10) + 6:.1f}" y="262">top 10 →</text></g>\n')

    # Axis.
    body += f'<line class="axis" x1="{x0}" y1="252" x2="{x1}" y2="252"/>\n'
    body += f'<text class="small" x="{x1}" y="276" text-anchor="end">higher score →</text>\n'
    body += f'<text class="small" x="{x0}" y="276">each dot is one stored vector, placed by its score against the query</text>\n'

    # Dots. Shortlisted ones slide from the code's score to the exact score.
    for i in range(n):
        y = lanes[i % len(lanes)]
        cls = "true" if i in true else "other"
        dx = X(exact[i]) - X(approx[i]) if i in short else 0.0
        name = f"d{i}"
        stops = [(0, "opacity:0;transform:translateX(0)"), (4, "opacity:1;transform:translateX(0)"),
                 (50, "opacity:1;transform:translateX(0)"), (62, f"opacity:1;transform:translateX({dx:.1f}px)"),
                 (94, f"opacity:1;transform:translateX({dx:.1f}px)"), (100, f"opacity:0;transform:translateX({dx:.1f}px)")]
        if i not in short and i != lost:
            stops = [(0, "opacity:0"), (4, "opacity:1"), (50, "opacity:1"), (60, "opacity:.35"),
                     (94, "opacity:.35"), (100, "opacity:0")]
            rest = "opacity:.35"
        else:
            rest = f"transform:translateX({dx:.1f}px)"
        css += phase(name, dur, stops)
        css += f".{name} {{ {rest}; animation: {name} {dur}s infinite; }}\n"
        if i == lost:
            css += phase("lostc", dur, [(0, "fill:var(--s1)"), (64, "fill:var(--s1)"), (68, "fill:var(--s2)"),
                                        (100, "fill:var(--s2)")])
            body += f'<g class="{name}"><circle class="lost" style="animation: lostc {dur}s infinite" cx="{X(approx[i]):.1f}" cy="{y}" r="6"/></g>\n'
            lost_xy = (X(approx[i]), y)
        else:
            body += f'<g class="{name}"><circle class="{cls}" cx="{X(approx[i]):.1f}" cy="{y}" r="6"/></g>\n'

    # Label for the missed neighbour, shown in the last phase.
    css += phase("ml", dur, [(0, "opacity:0"), (66, "opacity:0"), (70, "opacity:1"), (94, "opacity:1"), (100, "opacity:0")])
    css += f".ml {{ opacity:1; animation: ml {dur}s infinite; }}\n"
    lx, ly = lost_xy
    body += (f'<g class="ml"><line class="axis" x1="{lx:.1f}" y1="{ly - 8}" x2="{lx:.1f}" y2="104"/>'
             f'<text class="lbl" x="{lx - 4:.1f}" y="98" text-anchor="start">missed: the code ranked it #{rank[lost] + 1}, outside the shortlist</text></g>\n')

    # Phase counters.
    c, b = captions("hc", dur, [
        (0, 14, "1 · Scan. Every vector is scored from its 48-byte code, in memory."),
        (14, 30, f"2 · Cut at the code's top 10: it keeps {in10} of the 10 true neighbours."),
        (30, 50, f"3 · Instead, shortlist the code's top {depth} and fetch their f32 vectors."),
        (50, 66, "4 · Re-score the shortlist exactly. True neighbours rise into the top 10."),
        (66, 100, f"Result: {len(final10 & true)} of 10 true neighbours. A deeper shortlist catches the last one."),
    ], y=64)
    css += c
    body += b

    # Legend (two series plus the missed state; text in ink, colour on the mark).
    lg = 312
    body += (f'<circle class="true" cx="32" cy="{lg}" r="6"/><text class="lbl" x="44" y="{lg + 4}">true top-10 neighbour (exact f32)</text>'
             f'<circle class="other" cx="268" cy="{lg}" r="6"/><text class="lbl" x="280" y="{lg + 4}">other vector</text>'
             f'<circle class="lost" cx="384" cy="{lg}" r="6"/><text class="lbl" x="396" y="{lg + 4}">true neighbour the hybrid missed</text>\n')
    body += (f'<text class="small" x="24" y="{lg + 26}">Schematic: {n} synthetic vectors. Measured over ten collections, re-scoring 50 f32 candidates recovers</text>'
             f'<text class="small" x="24" y="{lg + 40}">0.95 (binary) to 0.99 (pq48) of the top 10, for 3–4% of the bytes an f32 scan reads.</text>\n')

    desc = (f"Animated schematic of two-stage retrieval. {n} vectors are scored from small codes; the code's top 10 holds "
            f"{in10} of the 10 true neighbours. The code's top {depth} are re-scored with exact f32 vectors, which lifts "
            f"{len(final10 & true)} true neighbours into the top 10; one, ranked {rank[lost] + 1} by the code, falls outside "
            "the shortlist and is missed.")
    return svg(W, H, "Hybrid quantization: two-stage re-scoring", desc, css, body)


# ---------------------------------------------------------------------------
# 2. Adaptive depth
# ---------------------------------------------------------------------------

def adaptive_depth():
    rng = random.Random(10)
    k, fixed, margin = 10, 30, 0.045
    # Codec scores for two queries: one with a crowded top, one well separated.
    crowded = sorted((0.80 - abs(rng.gauss(0, 0.04)) for _ in range(70)), reverse=True)
    separated = sorted((0.80 - abs(rng.gauss(0, 0.18)) for _ in range(70)), reverse=True)

    W, H = 820, 446
    x0, x1 = 50, 790
    lo, hi = 0.33, 0.81
    X = lambda s: x0 + (s - lo) / (hi - lo) * (x1 - x0)
    dur = 12
    css, body = "", ""
    body += '<text class="h" x="24" y="32">Adaptive depth: re-score every candidate within λσ of the 10th score</text>\n'
    c, b = captions("ac", dur, [
        (0, 30, "The code's score noise σ is calibrated once per collection, on self-queries."),
        (30, 62, "The margin λσ is the same for every query; how many candidates fall inside it is not."),
        (62, 100, "A fixed shortlist (dashed) over-fetches for one query and under-fetches for the other."),
    ], y=60)
    css += c
    body += b

    css += phase("grow", dur, [(0, "transform:scaleX(0)"), (30, "transform:scaleX(0)"), (52, "transform:scaleX(1)"),
                               (94, "transform:scaleX(1)"), (100, "transform:scaleX(0)")])
    css += phase("show", dur, [(0, "opacity:0"), (52, "opacity:0"), (56, "opacity:1"), (94, "opacity:1"), (100, "opacity:0")])
    css += phase("fix", dur, [(0, "opacity:0"), (62, "opacity:0"), (66, "opacity:1"), (94, "opacity:1"), (100, "opacity:0")])
    css += f".grow {{ transform-box: fill-box; transform-origin: right; animation: grow {dur}s infinite; }}\n"
    css += f".show {{ opacity:1; animation: show {dur}s infinite; }}\n"
    css += f".fix {{ opacity:1; animation: fix {dur}s infinite; }}\n"

    def panel(scores, top, name):
        out = ""
        kth = scores[k - 1]
        inside = sum(1 for s in scores if s >= kth - margin)
        base = top + 112
        out += f'<text class="lbl" x="24" y="{top}">{name}</text>\n'
        out += f'<line class="axis" x1="{x0}" y1="{base}" x2="{x1}" y2="{base}"/>\n'
        out += (f'<rect class="band grow" x="{X(kth - margin):.1f}" y="{top + 12}" width="{X(kth) - X(kth - margin):.1f}" '
                f'height="{base - top - 12}" rx="3"/>\n')
        # Dots in lanes; candidates inside the margin turn blue once the band has grown.
        for i, s in enumerate(scores):
            y = top + 24 + (i % 6) * 15
            if s >= kth - margin and i >= k:
                n2 = f"in{name[:3]}{i}"
                css_dot = phase(n2, dur, [(0, "fill:var(--other);fill-opacity:.55"), (52, "fill:var(--other);fill-opacity:.55"),
                                          (56, "fill:var(--s1);fill-opacity:1"), (94, "fill:var(--s1);fill-opacity:1"),
                                          (100, "fill:var(--other);fill-opacity:.55")])
                nonlocal_css.append(css_dot)
                out += f'<circle class="other" style="fill:var(--s1);fill-opacity:1;animation:{n2} {dur}s infinite" cx="{X(s):.1f}" cy="{y}" r="5"/>\n'
            elif i < k:
                out += f'<circle class="true" cx="{X(s):.1f}" cy="{y}" r="5"/>\n'
            else:
                out += f'<circle class="other" cx="{X(s):.1f}" cy="{y}" r="5"/>\n'
        out += f'<line class="cut" x1="{X(kth):.1f}" y1="{top + 12}" x2="{X(kth):.1f}" y2="{base}"/>\n'
        out += f'<text class="small" x="{X(kth) + 4:.1f}" y="{base + 14}">10th</text>\n'
        fx = X(scores[fixed - 1])
        out += f'<line class="fixed fix" x1="{fx:.1f}" y1="{top + 12}" x2="{fx:.1f}" y2="{base}"/>\n'
        out += f'<text class="small fix" x="{fx - 4:.1f}" y="{base + 14}" text-anchor="end">fixed N = {fixed}</text>\n'
        out += f'<text class="num show" x="{x1}" y="{top}" text-anchor="end">re-scores {inside} candidates</text>\n'
        return out, inside

    nonlocal_css = []
    p1, n1 = panel(crowded, 96, "Crowded query: many candidates score close to the 10th")
    p2, n2 = panel(separated, 256, "Well-separated query: the 10th stands clear of the rest")
    css += "".join(nonlocal_css)
    body += p1 + p2
    lg = 412
    body += (f'<circle class="true" cx="32" cy="{lg}" r="5"/><text class="lbl" x="44" y="{lg + 4}">the code\'s top 10</text>'
             f'<circle class="other" style="fill:var(--s1);fill-opacity:1" cx="190" cy="{lg}" r="5"/><text class="lbl" x="202" y="{lg + 4}">re-scored by margin</text>'
             f'<circle class="other" cx="364" cy="{lg}" r="5"/><text class="lbl" x="376" y="{lg + 4}">not re-scored</text>\n')
    body += (f'<text class="small" x="24" y="{lg + 24}">Schematic: synthetic code scores, one σ for both queries. '
             'Measured results: experiments/2026-11-hybrid-adaptive-matrix.</text>\n')
    desc = (f"Animated schematic of adaptive re-scoring depth. Two queries share one noise-based margin below their 10th "
            f"code score. The crowded query has {n1} candidates inside the margin, the well-separated one {n2}; a fixed "
            f"shortlist of {fixed} would under-fetch for the first and over-fetch for the second.")
    return svg(W, H, "Adaptive depth: a per-query margin instead of a fixed shortlist", desc, css, body)


# ---------------------------------------------------------------------------
# 3. Adaptive bits (real data)
# ---------------------------------------------------------------------------

def adaptive_bits():
    data = json.loads((HERE / "scifact-bge-small-alloc48.json").read_text())
    var, plain, wt = data["var"], data["plain"], data["weighted"]
    d, group = len(var), 8
    groups = d // group
    states = [
        ("uniform", [1] * d, "Uniform: 1 bit on every dimension (binary, 48 B)", "0.611"),
        ("plain", plain, "mixed48-pca: rotate to principal components, spend bits where the variance is", "0.694"),
        ("weighted", wt, "mixed48-pca-w: weight variance by how much energy queries put there", "0.703"),
    ]
    sums = {name: [sum(bits[g * group:(g + 1) * group]) for g in range(groups)] for name, bits, _, _ in states}
    zero = {name: sum(1 for b in bits if b == 0) for name, bits, _, _ in states}

    W, H = 820, 486
    x0, x1 = 64, 790
    bw = (x1 - x0) / groups
    top, base = 180, 370
    ymax = 40
    Y = lambda v: base - v / ymax * (base - top)
    dur = 15
    css, body = "", ""
    body += '<text class="h" x="24" y="32">Adaptive bits: the same 48 bytes, spent where they matter</text>\n'

    # Variance spectrum, its own small chart (one axis), log scale.
    import math
    vt, vb = 70, 138
    # Clip at 0.001: one near-dead component would otherwise stretch the axis to 1e-5.
    lmin = -3
    lv = [max(math.log10(max(v, 1e-12)), lmin) for v in var]
    VY = lambda l: vb - (l - lmin) / (0 - lmin) * (vb - vt)
    VX = lambda i: x0 + (i + 0.5) / d * (x1 - x0)
    pts = " ".join(f"{VX(i):.1f},{VY(lv[i]):.1f}" for i in range(d))
    for t in range(lmin, 1):
        body += f'<line class="grid" x1="{x0}" y1="{VY(t):.1f}" x2="{x1}" y2="{VY(t):.1f}"/>'
        body += f'<text class="small" x="{x0 - 6}" y="{VY(t) + 4:.1f}" text-anchor="end">{10 ** t:g}</text>\n'
    body += f'<polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="2"/>\n'
    body += f'<text class="lbl" x="{x0}" y="{vt - 10}">variance per principal component, relative to the first (log scale, clipped at 0.001)</text>\n'

    # Bars: total bits per group of 8 components, morphing between states.
    for g in range(0, ymax + 1, 8):
        body += f'<line class="grid" x1="{x0}" y1="{Y(g):.1f}" x2="{x1}" y2="{Y(g):.1f}"/>'
        body += f'<text class="small" x="{x0 - 6}" y="{Y(g) + 4:.1f}" text-anchor="end">{g}</text>\n'
    body += f'<text class="lbl" x="{x0}" y="{top - 12}">bits per group of 8 components (48 bytes = 384 bits in all)</text>\n'
    seq = ["uniform", "plain", "weighted"]
    # 0–28% uniform, 33–61% plain, 66–94% weighted, back to uniform.
    for gi in range(groups):
        s = [sums[n][gi] / ymax for n in seq]
        name = f"b{gi}"
        stops = [(0, f"transform:scaleY({s[0]:.4f})"), (28, f"transform:scaleY({s[0]:.4f})"),
                 (33, f"transform:scaleY({s[1]:.4f})"), (61, f"transform:scaleY({s[1]:.4f})"),
                 (66, f"transform:scaleY({s[2]:.4f})"), (94, f"transform:scaleY({s[2]:.4f})"),
                 (100, f"transform:scaleY({s[0]:.4f})")]
        css += phase(name, dur, stops)
        css += f".{name} {{ transform-box: fill-box; transform-origin: bottom; transform: scaleY({s[2]:.4f}); animation: {name} {dur}s infinite; }}\n"
        body += f'<rect class="{name}" x="{x0 + gi * bw + 1:.1f}" y="{top}" width="{bw - 2:.1f}" height="{base - top}" fill="var(--s1)"/>\n'
    body += f'<line class="axis" x1="{x0}" y1="{base}" x2="{x1}" y2="{base}"/>\n'
    body += f'<text class="small" x="{x0}" y="{base + 16}">component 1</text>'
    body += f'<text class="small" x="{x1}" y="{base + 16}" text-anchor="end">component 384</text>\n'

    items = []
    spans = [(0, 30), (31, 63), (64, 100)]
    for (name, _, label, rec), (a, b) in zip(states, spans):
        items.append((a, b, f"{label}"))
    c1, b1 = captions("bc", dur, items, y=410, cls="cap")
    stats = []
    for (name, _, _, rec), (a, b) in zip(states, spans):
        stats.append((a, b, f"recall@10 without re-scoring: {rec} · components with 0 bits: {zero[name]} of {d}"))
    c2, b2 = captions("bs", dur, stats, y=432, cls="num")
    css += c1 + c2
    body += b1 + b2
    # The query-weighted state puts a few bits back on the last components: low variance, but
    # they carry the corpus's shared mean direction, which every query projects onto.
    css += phase("tail", dur, [(0, "opacity:0"), (66, "opacity:0"), (70, "opacity:1"), (94, "opacity:1"), (100, "opacity:0")])
    css += f".tail {{ opacity:1; animation: tail {dur}s infinite; }}\n"
    body += (f'<text class="small tail" x="{x1}" y="{Y(12):.1f}" text-anchor="end">bits return to the last 44 components: little variance,</text>'
             f'<text class="small tail" x="{x1}" y="{Y(12) + 14:.1f}" text-anchor="end">but 28% of the corpus mean, which every query points along</text>\n')
    body += ('<text class="small" x="24" y="458">Real data: SciFact, bge-small-en-v1.5, 8,036 chunks, 300 claims. pq48 at the same 48 bytes reaches 0.711.</text>'
             '<text class="small" x="24" y="472">Mixed precision by importance is the idea behind structure-aware MoE quantization (QuantMoE-Bench).</text>\n')
    desc = ("Animated bar chart of how a 48-byte budget is spread over 384 principal components of SciFact's bge-small "
            f"embeddings. Uniform binary gives every component 1 bit (recall@10 0.611). Variance-based allocation "
            f"concentrates bits on the leading components and gives {zero['plain']} components none (0.694); "
            f"query-weighted allocation goes further, {zero['weighted']} components with none (0.703).")
    return svg(W, H, "Adaptive bits: mixed-precision allocation at 48 bytes", desc, css, body)


if __name__ == "__main__":
    for name, fn in [("hybrid-rescoring.svg", hybrid), ("adaptive-depth.svg", adaptive_depth), ("adaptive-bits.svg", adaptive_bits)]:
        (HERE / name).write_text(fn())
        print("wrote", HERE / name)
