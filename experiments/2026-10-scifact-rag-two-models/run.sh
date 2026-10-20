#!/usr/bin/env bash
# Experiment 1: does codec loss reach the answer? The SciFact verdict questions (188 claims with
# supports/refutes gold) through the RAG pipeline on f32, int8, binary and binary-centred
# collections, against two chat models of very different size, then judged by a third.
#
# Needs: a built `hvec`, curl, unzip, and chat models. By default two Claude profiles over
# ANTHROPIC_API_KEY. For local models, start `ollama serve`, pull the models, and run with
#   SMALL_CHAT=ollama-small LARGE_CHAT=ollama-large JUDGE_CHAT=ollama-judge ./run.sh
# with OLLAMA_SMALL / OLLAMA_LARGE / OLLAMA_JUDGE naming the Ollama models (defaults below).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
# Secrets live in the repo-root .env (git-ignored); exported variables take precedence.
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
QUESTIONS="$ROOT/questions/scifact.jsonl"
EMBEDDER="${EMBEDDER:-bge-small}"
SMALL_CHAT="${SMALL_CHAT:-haiku}"
LARGE_CHAT="${LARGE_CHAT:-opus}"
JUDGE_CHAT="${JUDGE_CHAT:-sonnet}"
OLLAMA_SMALL="${OLLAMA_SMALL:-llama3.2:3b}"
OLLAMA_LARGE="${OLLAMA_LARGE:-llama3.1:8b}"
OLLAMA_JUDGE="${OLLAMA_JUDGE:-$OLLAMA_LARGE}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
K="${K:-5}"
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
default_chat = "$LARGE_CHAT"
default_embedder = "$EMBEDDER"
metric = "cosine"

[chat.opus]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"
max_tokens = 512

[chat.sonnet]
provider = "anthropic"
model = "claude-sonnet-5-5"
api_key_env = "ANTHROPIC_API_KEY"
max_tokens = 512

[chat.haiku]
provider = "anthropic"
model = "claude-haiku-5-5"
api_key_env = "ANTHROPIC_API_KEY"
max_tokens = 512

[chat.ollama-small]
provider = "openai"
model = "$OLLAMA_SMALL"
base_url = "$OLLAMA_URL"
max_tokens = 512

[chat.ollama-large]
provider = "openai"
model = "$OLLAMA_LARGE"
base_url = "$OLLAMA_URL"
max_tokens = 512

[chat.ollama-judge]
provider = "openai"
model = "$OLLAMA_JUDGE"
base_url = "$OLLAMA_URL"
max_tokens = 512

[embed.bge-small]
provider = "local"
model = "BGESmallENV15"
CFG

for codec in f32 int8 binary binary-centred; do
  col="scifact-$EMBEDDER-$codec"
  if ! "$HVEC" collections list | grep -q "^$col "; then
    "$HVEC" ingest "$WORK/corpus" --collection "$col" --embedder "$EMBEDDER" --codec "$codec" \
      --ext txt --chunk-words 200 --overlap-words 0 --batch 64
  fi
done

# Only the claims with a gold verdict are informative for accuracy; the rest still score retrieval.
grep -v '^#' "$QUESTIONS" | grep '"answers":\["' > "$WORK/scifact-gold.jsonl"
LIMIT_FLAG=""; [ -n "$LIMIT" ] && LIMIT_FLAG="--limit $LIMIT"
# shellcheck disable=SC2086
"$HVEC" bench run --questions "$WORK/scifact-gold.jsonl" \
  --collections "scifact-$EMBEDDER-f32,scifact-$EMBEDDER-int8,scifact-$EMBEDDER-binary,scifact-$EMBEDDER-binary-centred" \
  --chat "$SMALL_CHAT,$LARGE_CHAT" -k "$K" --label exp1 $LIMIT_FLAG 2>&1 | tee "$WORK/bench-run.log"
BATCH="$(grep -o '^batch [0-9a-f]*' "$WORK/bench-run.log" | head -1 | cut -d' ' -f2)"
"$HVEC" judge run --batch "$BATCH" --judge "$JUDGE_CHAT" 2>&1 | tee "$WORK/judge.log"
"$HVEC" report --batch "$BATCH" --json > "$HERE/results/cells.json"
"$HVEC" judge agreement --batch "$BATCH" --json > "$HERE/results/agreement.json"
"$HVEC" report --batch "$BATCH"
"$HVEC" judge agreement --batch "$BATCH"
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
echo "batch $BATCH; results in $HERE/results"
