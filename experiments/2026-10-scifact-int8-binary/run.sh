#!/usr/bin/env bash
# Reproduce the SciFact int8/binary recall experiment.
# Needs: a built `hvec` (cargo build --release), curl, unzip, python3.
# Downloads ~3 MB of data and two small embedding models on first run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HVEC="${HVEC:-$HERE/../../target/release/hvec}"
WORK="${WORK:-$HERE/work}"
K="${K:-10}"
mkdir -p "$WORK/dl" "$WORK/txt" "$HERE/results"

if [ ! -f "$WORK/dl/scifact/corpus.jsonl" ]; then
  curl -sL -o "$WORK/dl/scifact.zip" https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  (cd "$WORK/dl" && unzip -oq scifact.zip)
fi
python3 -I "$HERE/scifact_to_txt.py" "$WORK/dl/scifact" "$WORK/txt"

export HVEC_CONFIG="$WORK/config.toml"
cat > "$HVEC_CONFIG" <<CFG
db_path = "$WORK/hvec.db"
default_chat = "anthropic"
default_embedder = "bge-small"
metric = "cosine"

[chat.anthropic]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"

[embed.bge-small]
provider = "local"
model = "BGESmallENV15"

[embed.minilm]
provider = "local"
model = "AllMiniLML6V2"
CFG

for m in bge-small minilm; do
  "$HVEC" collections delete "scifact-$m" >/dev/null 2>&1 || true
  "$HVEC" ingest "$WORK/txt/corpus" --collection "scifact-$m" --embedder "$m" --ext txt --chunk-words 200 --overlap-words 0 --batch 64
  "$HVEC" bench recall --collection "scifact-$m" -k "$K" --queries "$WORK/txt/queries.txt" --json > "$HERE/results/$m-queries.json"
  "$HVEC" bench recall --collection "scifact-$m" -k "$K" --sample 300 --json > "$HERE/results/$m-self.json"
  echo "== $m (300 real queries) =="
  "$HVEC" bench recall --collection "scifact-$m" -k "$K" --queries "$WORK/txt/queries.txt" --no-record
done
echo "results in $HERE/results"
