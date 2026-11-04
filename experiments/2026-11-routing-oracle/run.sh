#!/usr/bin/env bash
# Routing: how much is there to gain by choosing the compression per query or per collection,
# and how much of it a free router captures. Records per-query traces (where the scan ranks each
# true neighbour, the band the scan itself sees, the margin policy's depth on a 0.25σ grid) for
# six scan codecs on the ten collections, then replays fixed, margin, ratio-router and oracle
# policies offline. Retrieval only. Reuses ../2026-11-neighbour-spacing's collections (and
# builds them if missing). Needs: a built `hvec` (cargo build --release).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
SPACING="$HERE/../2026-11-neighbour-spacing"
SPACING_WORK="${SPACING_WORK:-$SPACING/work}"
K="${K:-10}"
CORPORA="${CORPORA:-scifact nfcorpus arguana cosqa fiqa}"
MODELS="${MODELS:-bge-small minilm}"
FIRST="${FIRST:-binary,binary-centred,pq48,pq24,mixed48-pca-w,mixed24-pca-w}"
mkdir -p "$HERE/results/traces"

export HVEC_CONFIG="$SPACING_WORK/config.toml"
if [ ! -f "$HVEC_CONFIG" ]; then
  WORK="$SPACING_WORK" CORPORA="$CORPORA" MODELS="$MODELS" "$SPACING/run.sh"
fi

for c in $CORPORA; do
  for m in $MODELS; do
    col="$c-$m"
    # The cells themselves are not needed: the traces replay any depth policy exactly.
    "$HVEC" bench rescore --collection "$col" -k "$K" --codecs "$FIRST" --rescore f32 \
      --depths "$K" --margins 1 --max-depth 1000 --questions "$SPACING_WORK/$c.jsonl" \
      --trace "$HERE/results/traces/$col.jsonl" --no-record --json > /dev/null
    echo "== $col done"
  done
done
gzip -9f "$HERE"/results/traces/*.jsonl
python3 -I "$HERE/analyse.py" "$HERE/results/traces" "$K" | tee "$HERE/results/summary.txt"
