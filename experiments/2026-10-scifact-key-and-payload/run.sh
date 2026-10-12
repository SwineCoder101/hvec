#!/usr/bin/env bash
# Experiment 3: key compression alone, payload compression alone, and both together, on the
# same questions and chat model. The claim under test: the two layers' effects on answers are
# independent and additive, or they are not.
#
# Needs: a built `hvec`, curl, unzip, a chat model (ANTHROPIC_API_KEY by default; set CHAT to an
# Ollama profile for a local one) and, for the external-compressor cell, a compressor serving
# POST {question, passages[]} -> {passages[]} at PAYLOAD_URL (see README.md for a Headroom shim).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
QUESTIONS="$ROOT/questions/scifact.jsonl"
EMBEDDER="${EMBEDDER:-bge-small}"
CHAT="${CHAT:-sonnet}"
JUDGE_CHAT="${JUDGE_CHAT:-sonnet}"
K="${K:-5}"
TRUNCATE_WORDS="${TRUNCATE_WORDS:-60}"
PAYLOAD_URL="${PAYLOAD_URL:-}"
LIMIT="${LIMIT:-}"
mkdir -p "$WORK/dl" "$HERE/results"

if [ ! -f "$WORK/dl/scifact/corpus.jsonl" ]; then
  curl -sL -o "$WORK/dl/scifact.zip" https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  (cd "$WORK/dl" && unzip -oq scifact.zip)
fi
[ -d "$WORK/corpus" ] || "$HVEC" questions import-beir "$WORK/dl/scifact" --out "$WORK/unused.jsonl" --corpus-out "$WORK/corpus"

export HVEC_CONFIG="$WORK/config.toml"
cat > "$HVEC_CONFIG" <<CFG
db_path = "$WORK/hvec.db"
default_chat = "$CHAT"
default_embedder = "$EMBEDDER"
metric = "cosine"

[chat.sonnet]
provider = "anthropic"
model = "claude-sonnet-5-5"
api_key_env = "ANTHROPIC_API_KEY"
max_tokens = 512

[chat.ollama]
provider = "openai"
model = "llama3.2:3b"
base_url = "http://localhost:11434/v1"
max_tokens = 512

[embed.bge-small]
provider = "local"
model = "BGESmallENV15"
CFG

for codec in f32 binary; do
  col="scifact-$EMBEDDER-$codec"
  if ! "$HVEC" collections list | grep -q "^$col "; then
    "$HVEC" ingest "$WORK/corpus" --collection "$col" --embedder "$EMBEDDER" --codec "$codec" \
      --ext txt --chunk-words 200 --overlap-words 0 --batch 64
  fi
done
grep -v '^#' "$QUESTIONS" | grep '"answers":\["' > "$WORK/scifact-gold.jsonl"
LIMIT_FLAG=""; [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"

PAYLOADS="none truncate:$TRUNCATE_WORDS"
[ -n "$PAYLOAD_URL" ] && PAYLOADS="$PAYLOADS http:$PAYLOAD_URL"
BATCHES=""
for payload in $PAYLOADS; do
  # shellcheck disable=SC2086
  "$HVEC" bench run --questions "$WORK/scifact-gold.jsonl" \
    --collections "scifact-$EMBEDDER-f32,scifact-$EMBEDDER-binary" --chat "$CHAT" -k "$K" \
    --payload "$payload" --label exp3 $LIMIT_FLAG 2>&1 | tee "$WORK/bench-$payload.log"
  b="$(grep -o '^batch [0-9a-f]*' "$WORK/bench-$payload.log" | head -1 | cut -d' ' -f2)"
  "$HVEC" judge run --batch "$b" --judge "$JUDGE_CHAT"
  BATCHES="$BATCHES $b"
done
"$HVEC" report --set scifact-gold --json > "$HERE/results/cells.json"
"$HVEC" report --set scifact-gold
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
echo "batches:$BATCHES; results in $HERE/results"
