---
title: "Reproduced on a second machine: SciFact codecs against the relevance judgments"
date: 2026-10-27 17:00:00 +0000
excerpt: "The labelled SciFact experiment re-run from a clean checkout on Linux. Every labelled metric matches to four decimals; the scan timings do not, and that is the interesting part."
references: [thakur2021beir, xiao2023bge, reimers2019sbert, charikar2002simhash]
---

Every number hvec publishes is supposed to come with a commit hash, a config and a command, so
that someone else can get the same number. This post is the first time someone did. The
[labelled SciFact experiment]({{ "/experiments/scifact-labelled/" | relative_url }}) was run
from a clean checkout on a different machine, a different operating system and a different CPU
architecture, with no changes to the code or the scripts.

**Result.** The retrieval numbers reproduce. nDCG@10 and document recall@10 against the BEIR
judgments [1] match the published values to four decimals for every codec on both embedding
models. Agreement with the f32 ranking matches to three decimals, with one neighbour in 300
queries moving on MiniLM int8. The scan timings do not reproduce, and on this machine the
binary codecs scan *slower* than f32, which the original write-up's hardware hid.

## What was run

| | Original | Reproduction |
|---|---|---|
| Machine | Apple M4 Pro, 14 cores, macOS | Intel Core i9-9980HK, 8 cores / 16 threads, 61 GB, Linux 7.0 |
| Code | hvec commit `6a78bf9` | hvec commit `cc9a5eb` |
| Toolchain | stable Rust | Rust 1.99.1, release build, `lto = "thin"` |
| Embedding models | `bge-small-en-v1.5` [2] and `all-MiniLM-L6-v2` [3], Xenova and Qdrant ONNX via fastembed on CPU | same models, downloaded fresh |
| Corpus | BEIR SciFact, 5,183 abstracts, 8,036 chunks of 200 words | same, downloaded fresh and re-ingested |
| Questions | `questions/scifact.jsonl`, 300 claims, 339 judgments | same committed file; the rebuild from the BEIR download matched it |
| Command | `./run.sh` in `experiments/2026-10-scifact-labelled` | the same script, pointed with `WORK=` at the database the int8/binary experiment had just built |

Ingest took 309 seconds for bge-small and 126 seconds for MiniLM, both on CPU, both f32
collections. Nothing was cached from the earlier runs: this machine had never built hvec or
downloaded a model.

## Retrieval: identical

300 claims, k = 10. `recall@10` is agreement with the f32 top-10. `nDCG@10` and `doc recall@10`
are against the judgments. Published values in brackets where they differ.

| embedding model | codec | recall@10 vs f32 | top-1 vs f32 | nDCG@10 | doc recall@10 | bytes |
|---|---|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | f32 | 1.000 | 1.000 | **0.698** | **0.818** | 1536 |
| bge-small-en-v1.5 | int8 | 0.987 | 0.997 | 0.698 | 0.818 | 392 |
| bge-small-en-v1.5 | binary | 0.611 | 0.700 | 0.654 | 0.768 | 48 |
| bge-small-en-v1.5 | binary-centred | 0.656 | 0.727 | 0.658 | 0.770 | 48 |
| all-MiniLM-L6-v2 | f32 | 1.000 | 1.000 | **0.627** | **0.778** | 1536 |
| all-MiniLM-L6-v2 | int8 | 0.995 | 0.997 (1.000) | 0.627 | 0.778 | 392 |
| all-MiniLM-L6-v2 | binary | 0.705 | 0.730 | 0.605 | 0.733 | 48 |
| all-MiniLM-L6-v2 | binary-centred | 0.719 | 0.717 | 0.598 | 0.739 | 48 |

k = 5 reproduces the same way: nDCG@5 and document recall@5 match to four decimals on every row.

The only differences in the raw JSON are in the sixth or seventh decimal of the score errors,
and one query on MiniLM where int8 put a different vector at rank 1 than f32 did. Both come from
the ONNX Runtime producing embeddings that differ in the last bits between Apple silicon and
x86. The embeddings are not bit-identical across machines; the retrieval metrics are, to the
precision anyone would report them at.

This is the result hvec's design was built for. The codecs are deterministic functions of the
f32 vectors, the trained ones are fitted on the whole corpus with no random seed, and the scan
is brute force with no index. There is nothing in the retrieval path that can vary between
machines except the embedding model's floating-point noise, and that noise is far below the
gaps between neighbours that the labels care about.

## Timing: not identical, and the sign flipped

Scan time over the 8,036 vectors for 300 queries, milliseconds, k = 10, bge-small:

| codec | M4 Pro (published) | i9-9980HK (this run) |
|---|---:|---:|
| f32 | 1203 | 2871 |
| int8 | 600 | 926 |
| binary | 433 | 3938 |
| binary-centred | 600 | 4610 |

The x86 machine is slower across the board, which is expected of a 2019 laptop CPU against a
2024 Apple one. The ratio between codecs is what changed. On the M4 Pro, binary scanned in a
third of f32's time. On the i9, binary takes a third *longer* than f32, and binary-centred
longer still, while int8 keeps its three-fold advantage.

The README already says hvec makes no speed claim for any codec, because every codec is scored
asymmetrically: the query stays in f32 and the stored bytes are expanded on the fly. For binary
that means unpacking each sign bit [4] to a float and multiplying, 384 times per vector. On
Apple silicon the compiler evidently turns that into something fast; on this x86 target it
does not, and the per-bit branch or shift costs more than reading four bytes of f32. int8 has
no such step: one byte becomes one multiply-add.

Two conclusions, neither new but both now backed by a second machine. The byte count of a codec
says nothing about its scan speed until the scoring kernel is designed for it, and the only
honest speed comparison for binary is the symmetric one, Hamming distance over packed bits with
popcount, which is on the roadmap for the rotated-binary codec. Until that exists, the timing
columns in hvec's tables are a cost of measurement, not a property of the codec.

## What this does and does not show

It shows that the labelled retrieval numbers for int8, binary and binary-centred on SciFact are
stable across hardware and operating systems, and that the published commit, config and command
are sufficient to get them. It shows that scan timings in hvec are machine-specific and should
be read as such.

It does not add a corpus, a model or a chat model. The open question from the original
write-up is unchanged: whether a four-point nDCG loss moves the answer a chat model gives. The
two-chat-model experiment is scripted and still waiting on model access.

The regenerated results files, with this machine's timings, replace the committed ones in
`experiments/2026-10-scifact-labelled/results/` and `experiments/2026-10-scifact-int8-binary/results/`.
The write-ups keep the original hardware line and cite this post for the reproduction.

{% include references.html %}
