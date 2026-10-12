#!/usr/bin/env bash
# Reproduce the SciFact labelled-retrieval experiment: every codec measured against the
# BEIR relevance judgments (nDCG@k, document recall@k) as well as against the f32 ranking.
# Needs: a built `hvec` (cargo build --release), curl, unzip. Downloads ~3 MB of data and two
# small embedding models on first run. No chat model is involved.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
CODECS="${CODECS:-int8,binary,binary-centred}"
QUESTIONS="$ROOT/questions/scifact.jsonl"
mkdir -p "$WORK/dl" "$HERE/results"

if [ ! -f "$WORK/dl/scifact/corpus.jsonl" ]; then
  curl -sL -o "$WORK/dl/scifact.zip" https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
  (cd "$WORK/dl" && unzip -oq scifact.zip)
fi
if [ ! -f "$WORK/dl/release/data/claims_dev.jsonl" ]; then
  curl -sL -o "$WORK/dl/scifact-release.tar.gz" https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
  mkdir -p "$WORK/dl/release" && tar -xzf "$WORK/dl/scifact-release.tar.gz" -C "$WORK/dl/release"
fi
# The committed question set is rebuilt here so the command is on record; the output is identical.
"$HVEC" questions import-beir "$WORK/dl/scifact" --out "$WORK/scifact.jsonl" --corpus-out "$WORK/corpus" \
  --scifact-claims "$WORK/dl/release/data/claims_dev.jsonl"
if ! diff -q <(grep -v '^#' "$WORK/scifact.jsonl") <(grep -v '^#' "$QUESTIONS") >/dev/null; then
  echo "warning: rebuilt question set differs from $QUESTIONS; using the committed one" >&2
fi

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
  if ! "$HVEC" collections list | grep -q "^scifact-$m "; then
    "$HVEC" ingest "$WORK/corpus" --collection "scifact-$m" --embedder "$m" --ext txt --chunk-words 200 --overlap-words 0 --batch 64
  fi
  for k in 5 10; do
    "$HVEC" bench recall --collection "scifact-$m" -k "$k" --codecs "$CODECS" --questions "$QUESTIONS" --json \
      > "$HERE/results/$m-k$k.json"
  done
  echo "== $m (300 judged claims, k=10) =="
  "$HVEC" bench recall --collection "scifact-$m" -k 10 --codecs "$CODECS" --questions "$QUESTIONS" --no-record
done
echo "results in $HERE/results"
