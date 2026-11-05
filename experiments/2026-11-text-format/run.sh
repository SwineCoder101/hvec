#!/usr/bin/env bash
# Text format: the same WikiBio records written as prose, bullets, key-value, JSON, a table row
# and a key-value header above prose, plus each record's own Wikipedia paragraph. For each format,
# query type (lookup, descriptive) and embedding model: exact f32 nDCG@10, the neighbour band and
# each codec's rank noise, the re-score depth and nDCG behind a binary-hamming or pq48 scan, and
# the per-vector scan cost, so summarise.py can draw accuracy against time per query.
# Retrieval only, no chat model, no API key. One record per chunk (`ingest --whole-files`), so
# every format has the same number of vectors.
# Needs: a built `hvec` (cargo build --release), curl, and `uv` (to read WikiBio's parquet file).
# Downloads 41 MB on first run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
HVEC="${HVEC:-$ROOT/target/release/hvec}"
WORK="${WORK:-$HERE/work}"
RESULTS="${RESULTS:-$HERE/results}"
K="${K:-10}"
RECORDS="${RECORDS:-20000}"
QUERIES="${QUERIES:-500}"
SEED="${SEED:-2026}"
FORMATS="${FORMATS:-paragraph bullets kv json table wiki header-prose}"
MODELS="${MODELS:-bge-small minilm}"
QTYPES="${QTYPES:-lookup descriptive}"
CODECS="${CODECS:-int8,binary,binary-hamming,pq48}"
FIRST="${FIRST:-binary-hamming,pq48}"
DEPTHS="${DEPTHS:-10,20,30,50,100,200,500,1000}"
mkdir -p "$WORK/dl" "$RESULTS/recall" "$RESULTS/rescore" "$RESULTS/scan"

PARQUET="$WORK/dl/wikibio-test.parquet"
[ -f "$PARQUET" ] || curl -sfL -o "$PARQUET" \
  https://huggingface.co/datasets/thoughtworks/wiki_bio/resolve/main/data/test-00000-of-00001.parquet
[ -f "$WORK/beir/stats.json" ] || (cd "$WORK" && uv run -q --no-project --with pyarrow python -I \
  "$HERE/wikibio_to_formats.py" "$PARQUET" "$WORK/beir" "$RECORDS" "$QUERIES" "$SEED")
cp "$WORK/beir/stats.json" "$WORK/beir/examples.md" "$RESULTS/"

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

{
  echo "hvec $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo '+dirty')"
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  sysctl -n hw.memsize 2>/dev/null | awk '{printf "memory: %.0f GB\n", $1/1e9}' || true
} > "$RESULTS/machine.txt"

for f in $FORMATS; do
  for q in $QTYPES; do  # the question sets are the same for every format; the corpus is not
    [ -f "$WORK/questions-$q.jsonl" ] && [ -d "$WORK/corpus-$f" ] && continue
    "$HVEC" questions import-beir "$WORK/beir/$f" --split "$q" --out "$WORK/questions-$q.jsonl" \
      --corpus-out "$WORK/corpus-$f"
  done
  for m in $MODELS; do
    col="$f-$m"
    if ! "$HVEC" collections list | grep -q "^$col "; then
      "$HVEC" ingest "$WORK/corpus-$f" --collection "$col" --embedder "$m" --ext txt --whole-files --batch 64
    fi
    for q in $QTYPES; do
      "$HVEC" bench recall --collection "$col" -k "$K" --codecs "$CODECS" \
        --questions "$WORK/questions-$q.jsonl" --json > "$RESULTS/recall/$col-$q.json"
      "$HVEC" bench rescore --collection "$col" -k "$K" --codecs "$FIRST" --rescore f32 \
        --depths "$DEPTHS" --questions "$WORK/questions-$q.jsonl" --json > "$RESULTS/rescore/$col-$q.json"
    done
    # Per-vector scan cost, one thread. It does not depend on the text, only on the codec and the
    # vector count, which is the same for every format; measured per collection as a check.
    "$HVEC" bench scan --collection "$col" --codecs "f32,${FIRST}" --threads 1 --skip-sqlite \
      --bandwidth-mib 256 --json > "$RESULTS/scan/$col.json"
    echo "== $col done"
  done
done
cp "$HVEC_CONFIG" "$RESULTS/config.toml"
python3 -I "$HERE/summarise.py" "$RESULTS" "$K" | tee "$RESULTS/summary.txt"
