#!/usr/bin/env bash
# The hybrid × adaptive matrix. Rows: what is held in memory and scanned, uniform codecs
# (binary, binary-centred, pq<m>) against adaptive bit allocation (mixed<B>, mixed<B>-pca,
# mixed<B>-pca-w) at 24, 48 and 96 bytes. Columns: the re-scoring policy, none, fixed depth
# (hybrid) or a per-query margin (hybrid + adaptive), with an int8 or f32 second stage.
# Retrieval only, no chat model. Reuses the ten collections from ../2026-11-neighbour-spacing
# (and builds them if missing). Needs: a built `hvec` (cargo build --release).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
SPACING="$HERE/../2026-11-neighbour-spacing"
SPACING_WORK="${SPACING_WORK:-$SPACING/work}"
K="${K:-10}"
CORPORA="${CORPORA:-scifact nfcorpus arguana cosqa fiqa}"
MODELS="${MODELS:-bge-small minilm}"
FIRST="${FIRST:-binary,binary-centred,pq48,mixed48,mixed48-pca,mixed48-pca-w,pq24,mixed24-pca,mixed24-pca-w,pq96,mixed96-pca-w}"
SECOND="${SECOND:-int8,f32}"
DEPTHS="${DEPTHS:-20,50,100}"
MARGINS="${MARGINS:-1,2,3}"
MAX_DEPTH="${MAX_DEPTH:-1000}"
mkdir -p "$HERE/results"

export HVEC_CONFIG="$SPACING_WORK/config.toml"
if [ ! -f "$HVEC_CONFIG" ]; then
  WORK="$SPACING_WORK" CORPORA="$CORPORA" MODELS="$MODELS" "$SPACING/run.sh"
fi

for c in $CORPORA; do
  for m in $MODELS; do
    col="$c-$m"
    "$HVEC" bench rescore --collection "$col" -k "$K" --codecs "$FIRST" --rescore "$SECOND" \
      --depths "$DEPTHS" --margins "$MARGINS" --max-depth "$MAX_DEPTH" \
      --questions "$SPACING_WORK/$c.jsonl" --json > "$HERE/results/$col-k$K.json"
    echo "== $col done"
  done
done
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
python3 -I "$HERE/summarise.py" "$HERE/results" "$K" "$SPACING/results" | tee "$HERE/results/summary.txt"
