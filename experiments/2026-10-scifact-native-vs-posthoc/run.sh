#!/usr/bin/env bash
# Experiment 2: native low-precision embeddings (a Matryoshka-trained model truncated) against
# post-hoc quantization of a conventional model, at matched byte budgets, scored against the
# SciFact relevance judgments. Needs: a built `hvec` (cargo build --release), curl, unzip.
# Downloads ~3 MB of data and two embedding models (~600 MB for nomic) on first run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
# Secrets live in the repo-root .env (git-ignored); exported variables take precedence.
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
QUESTIONS="$ROOT/questions/scifact.jsonl"
K="${K:-10}"
mkdir -p "$WORK/dl" "$HERE/results"

if [ ! -f "$WORK/dl/scifact/corpus.jsonl" ]; then
  curl -sL -o "$WORK/dl/scifact.zip" https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  (cd "$WORK/dl" && unzip -oq scifact.zip)
fi
[ -d "$WORK/corpus" ] || "$HVEC" questions import-beir "$WORK/dl/scifact" --out "$WORK/unused.jsonl" --corpus-out "$WORK/corpus"

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

# Conventional 384-dim model, no task prefixes (as in the earlier SciFact experiments).
[embed.bge-small]
provider = "local"
model = "BGESmallENV15"

# Matryoshka-trained 768-dim model; trained with these task prefixes.
[embed.nomic]
provider = "local"
model = "NomicEmbedTextV15"
query_prefix = "search_query: "
document_prefix = "search_document: "
CFG

# Post-hoc levers on the conventional model; truncation here is a plain projection.
BGE_CODECS="int8,binary,binary-centred,pq48,pq24,mrl256,mrl192,mrl96,mrl64,mrl192+int8,mrl96+int8,mrl192+binary,mrl96+binary"
# Native lever (truncation the model was trained for) and the same post-hoc levers on top.
NOMIC_CODECS="int8,binary,binary-centred,pq96,pq48,mrl512,mrl256,mrl128,mrl64,mrl256+int8,mrl128+int8,mrl64+int8,mrl384+binary,mrl256+binary,mrl192+binary,mrl128+binary,mrl256+pq32"

for m in bge-small nomic; do
  if ! "$HVEC" collections list | grep -q "^scifact-$m "; then
    "$HVEC" ingest "$WORK/corpus" --collection "scifact-$m" --embedder "$m" --ext txt --chunk-words 200 --overlap-words 0 --batch 64
  fi
  if [ "$m" = nomic ]; then codecs="$NOMIC_CODECS"; else codecs="$BGE_CODECS"; fi
  "$HVEC" bench recall --collection "scifact-$m" -k "$K" --codecs "$codecs" --questions "$QUESTIONS" --json \
    > "$HERE/results/$m-k$K.json"
  echo "== $m (300 judged claims, k=$K) =="
  "$HVEC" bench recall --collection "scifact-$m" -k "$K" --codecs "$codecs" --questions "$QUESTIONS" --no-record
done
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
echo "results in $HERE/results"
