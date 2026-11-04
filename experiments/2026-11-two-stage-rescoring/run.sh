#!/usr/bin/env bash
# Two-stage retrieval: scan with a 24-96 byte code, re-score the best N candidates with int8 or
# f32, keep the top 10. Measures how deep the re-score must go for a target recall, and whether
# the first stage's noise-to-spacing ratio predicts that depth. Retrieval only, no chat model.
# Reuses the ten collections built by ../2026-11-neighbour-spacing (and runs it if they are
# missing). Needs: a built `hvec` (cargo build --release).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
SPACING="$HERE/../2026-11-neighbour-spacing"
SPACING_WORK="${SPACING_WORK:-$SPACING/work}"
K="${K:-10}"
CORPORA="${CORPORA:-scifact nfcorpus arguana cosqa fiqa}"
MODELS="${MODELS:-bge-small minilm}"
FIRST="${FIRST:-binary,binary-centred,pq96,pq48,pq24}"
SECOND="${SECOND:-int8,f32}"
DEPTHS="${DEPTHS:-10,20,30,50,100,200,500,1000}"
mkdir -p "$HERE/results"

export HVEC_CONFIG="$SPACING_WORK/config.toml"
if [ ! -f "$HVEC_CONFIG" ]; then
  WORK="$SPACING_WORK" CORPORA="$CORPORA" MODELS="$MODELS" "$SPACING/run.sh"
fi

for c in $CORPORA; do
  for m in $MODELS; do
    col="$c-$m"
    "$HVEC" bench rescore --collection "$col" -k "$K" --codecs "$FIRST" --rescore "$SECOND" \
      --depths "$DEPTHS" --questions "$SPACING_WORK/$c.jsonl" --json > "$HERE/results/$col-k$K.json"
    echo "== $col done"
  done
done
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
python3 -I "$HERE/summarise.py" "$HERE/results" "$K" | tee "$HERE/results/summary.txt"
