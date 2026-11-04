#!/usr/bin/env bash
# Does neighbour spacing predict codec loss? The same codec grid on five corpora of different
# kinds (scientific abstracts, medical, argument, financial Q&A, Python code) and two embedding
# models, retrieval only, with each corpus's exact neighbour spacing recorded next to each
# codec's rank noise. Needs: a built `hvec` (cargo build --release), curl, unzip, and `uv`
# (only to read the code dataset's parquet files). Downloads ~27 MB of data on first run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
K="${K:-10}"
CORPORA="${CORPORA:-scifact nfcorpus arguana cosqa fiqa}"
MODELS="${MODELS:-bge-small minilm}"
CODECS="${CODECS:-int8,binary,binary-centred,pq96,pq48,pq24,mrl256,mrl192,mrl96}"
mkdir -p "$WORK/dl" "$HERE/results"

fetch() {  # fetch <corpus>: leaves a BEIR directory at $WORK/dl/<corpus>
  local c="$1"
  [ -f "$WORK/dl/$c/corpus.jsonl" ] && return
  if [ "$c" = cosqa ]; then
    local base=https://huggingface.co/datasets/CoIR-Retrieval/cosqa/resolve/main
    mkdir -p "$WORK/dl/cosqa-parquet"
    curl -sL -o "$WORK/dl/cosqa-parquet/corpus.parquet" "$base/corpus/corpus-00000-of-00001.parquet"
    curl -sL -o "$WORK/dl/cosqa-parquet/queries.parquet" "$base/queries/queries-00000-of-00001.parquet"
    curl -sL -o "$WORK/dl/cosqa-parquet/test.parquet" "$base/data/test-00000-of-00001.parquet"
    (cd "$WORK" && uv run -q --no-project --with pyarrow python -I "$HERE/coir_to_beir.py" \
      "$WORK/dl/cosqa-parquet" "$WORK/dl/cosqa")
  else
    curl -sL -o "$WORK/dl/$c.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$c.zip"
    (cd "$WORK/dl" && unzip -oq "$c.zip")
  fi
}

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

for c in $CORPORA; do
  fetch "$c"
  [ -d "$WORK/corpus-$c" ] || "$HVEC" questions import-beir "$WORK/dl/$c" --out "$WORK/$c.jsonl" --corpus-out "$WORK/corpus-$c"
  for m in $MODELS; do
    col="$c-$m"
    if ! "$HVEC" collections list | grep -q "^$col "; then
      "$HVEC" ingest "$WORK/corpus-$c" --collection "$col" --embedder "$m" --ext txt \
        --chunk-words 200 --overlap-words 0 --batch 64
    fi
    "$HVEC" bench recall --collection "$col" -k "$K" --codecs "$CODECS" --questions "$WORK/$c.jsonl" \
      --json > "$HERE/results/$col-k$K.json"
    echo "== $col done"
  done
done
cp "$HVEC_CONFIG" "$HERE/results/config.toml"
python3 -I "$HERE/summarise.py" "$HERE/results" "$K" | tee "$HERE/results/summary.txt"
echo "results in $HERE/results"
