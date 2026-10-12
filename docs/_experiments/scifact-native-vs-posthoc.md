---
title: "SciFact: native truncation against post-hoc quantization at equal bytes"
date: 2026-10-12 18:30:00 +0000
result: "A Matryoshka-trained 768-dim model truncated natively never beats a conventional 384-dim model quantized post hoc at the same bytes; product quantization is the best codec under 400 bytes on both (pq48 on bge-small: 0.678 nDCG@10 at 48 bytes, two points under f32)."
references: [kusupati2022mrl, jegou2011pq, charikar2002simhash, thakur2021beir, xiao2023bge, lloyd1982kmeans]
---

*Full write-up, raw results and a one-command reproduction are in the repository:*
[experiments/2026-10-scifact-native-vs-posthoc](https://github.com/SwineCoder101/hvec/tree/main/experiments/2026-10-scifact-native-vs-posthoc).
Follow-up to [SciFact: codecs against the relevance judgments]({{ "/experiments/scifact-labelled/" | relative_url }}).

**Hypothesis.** A model trained with Matryoshka loss [1] carries its own codec: keep the first
`k` dimensions. Truncated to 256 or 128 dimensions, `nomic-embed-text-v1.5` will retrieve at
least as well against the SciFact judgments [4] as `bge-small-en-v1.5` [5] quantized post hoc to
the same bytes. If native wins, the right codec is a different embedding model.

**Result.** It does not win. Two new codec families make the comparison: `pq<m>`, product
quantization with trained codebooks [2][6], and `mrl<k>`, truncation composable with any other
codec (`mrl256+int8`). nDCG@10 against the judgments, by byte budget:

| bytes | bge-small-en-v1.5 (384 d) | nDCG@10 | nomic-embed-text-v1.5 (768 d) | nDCG@10 |
|---:|---|---:|---|---:|
| 3072 | | | f32 | **0.696** |
| 1536 | f32 | **0.698** | | |
| 1024 | mrl256 | 0.684 | mrl256 | 0.671 |
| 392 | int8 | 0.698 | | |
| 264 | | | mrl256+int8 | 0.671 |
| 256 | mrl64 | 0.520 | mrl64 | 0.520 |
| 96 | | | pq96 · binary-centred · binary | 0.686 · 0.683 · 0.673 |
| 48 | pq48 · binary-centred · binary | **0.678** · 0.658 · 0.654 | pq48 · mrl384+binary | 0.634 · 0.632 |
| 32 | | | mrl256+pq32 · mrl256+binary | 0.638 · 0.586 |
| 24 | pq24 · mrl192+binary | 0.602 · 0.580 | mrl192+binary | 0.523 |

**Reading.** Matryoshka training works per dimension: dropping two thirds of nomic costs 2.5
points where dropping three quarters of bge-small costs 10.5. But the two models retrieve
equally well at full precision and nomic needs twice the bytes to do it, so per byte the native
lever never catches up: post-hoc truncation of bge-small wins at 1,024 bytes and the two tie at
256. Product quantization is the best codec under 400 bytes on both models; at 48 bytes on
bge-small it keeps 0.678 where the sign code [3] keeps 0.654, two points under f32 at 32×
smaller. Fewer bits of more dimensions beats more bits of fewer: binary on all 768 nomic
dimensions (96 bytes) beats `mrl128+int8` (136 bytes). Nomic does tolerate binary much better
than bge-small (2.3 points lost against 4.4), having twice the sign bits and more centred vectors.

**Caveats.** One corpus; only two models, one of each kind, so dimension and training are
confounded; bge-small ran without its query instruction; plain PQ without a rotation; truncation
of bge-small keeps the leading dimensions where a random projection would be the fairer
post-hoc control. No chat model yet.

{% include references.html %}
