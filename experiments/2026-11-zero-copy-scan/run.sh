#!/usr/bin/env bash
# Zero-copy scan: where a brute-force search spends its time, and what each codec's scan costs
# against the machine's memory bandwidth. Times the storage paths (SQLite rows copied, SQLite
# rows borrowed with and without SQLite's memory map, the cached contiguous buffer), the cost of
# building the buffer, and every codec scanned in place at several thread counts, with recall@10
# against exact f32. Run on a quiet machine: nothing else should be using the CPU.
# Reuses ../2026-11-neighbour-spacing's collections. Needs: a built `hvec` (cargo build --release).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
SPACING="$HERE/../2026-11-neighbour-spacing"
export HVEC_CONFIG="${HVEC_CONFIG:-$SPACING/work/config.toml}"
COLLECTIONS="${COLLECTIONS:-scifact-bge-small fiqa-bge-small}"
CODECS="${CODECS:-f32,int8,binary,binary-centred,binary-hamming,pq48,pq24,mixed48-pca-w}"
THREADS="${THREADS:-1,2,4,8,10}"
SAMPLE="${SAMPLE:-50}"
REPEAT="${REPEAT:-3}"
mkdir -p "$HERE/results"
[ -f "$HVEC_CONFIG" ] || { echo "build the collections first: $SPACING/run.sh"; exit 1; }

{
  echo "hvec $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo '+dirty')"
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null | paste -sd' ' - | sed 's/^/performance and efficiency cores: /' || true
  sysctl -n hw.memsize 2>/dev/null | awk '{printf "memory: %.0f GB\n", $1/1e9}' || true
} > "$HERE/results/machine.txt"

for col in $COLLECTIONS; do
  "$HVEC" bench scan --collection "$col" --codecs "$CODECS" --threads "$THREADS" --sample "$SAMPLE" \
    --repeat "$REPEAT" --json > "$HERE/results/$col.json"
  echo "== $col done"
done
python3 -I "$HERE/summarise.py" "$HERE/results" | tee "$HERE/results/summary.txt"
cat "$HERE/results/machine.txt"
