# SciFact: native truncation against post-hoc quantization at equal bytes

The roadmap's Experiment 2. A model trained to be truncated (Matryoshka representation
learning) ships its own compression: keep the first `k` dimensions. The question is whether
that *native* lever beats *post-hoc* quantization of a conventional model at the same byte
budget. If it does, the right codec is a different embedding model, and much of the codec work
collapses into a model choice.

**Hypothesis.** `nomic-embed-text-v1.5`, trained with Matryoshka loss over 64 to 768
dimensions, truncated to 256 or 128 dimensions will retrieve at least as well against the
SciFact judgments as `bge-small-en-v1.5` quantized post hoc to the same number of bytes, and
truncating bge-small (which was not trained for it) will fall apart quickly.

**Result.** Half of the second part, none of the first. Truncating nomic is much gentler *per
dimension dropped* than truncating bge-small: at one sixth of the dimensions nomic loses 17.6
nDCG points and bge-small 17.8, but at one third nomic loses 2.5 where bge-small (at one
quarter) loses 10.5. Per *byte*, though, nomic starts from twice the dimensions and never
catches up: at 1,024 bytes post-hoc truncation of bge-small (0.684) beats native truncation of
nomic (0.671); at 256 bytes the two are identical (0.520). And product quantization beats every
truncation on both models below 400 bytes: `pq48` on bge-small reaches 0.678 at 48 bytes, two
points under f32 at 32× smaller, better than any nomic configuration below 1,024 bytes. The
right codec on this corpus is not a different model; it is PQ.

## Setup

| | |
|---|---|
| Corpus | BEIR SciFact, 5,183 abstracts, 8,036 chunks of 200 words, no overlap, as in the earlier experiments |
| Questions | `questions/scifact.jsonl`, the 300 test claims with 339 relevance judgments; the claim text is the query |
| Embedding models | `bge-small-en-v1.5` (384 dims, no task prefixes, as before) and `nomic-embed-text-v1.5` (768 dims, Matryoshka-trained, with its `search_query: ` / `search_document: ` prefixes), both via fastembed on CPU |
| Metric | cosine; for `mrl<k>` the cosine of the two prefixes, as Matryoshka models are used |
| k | 10 |
| Codecs on bge-small | `int8`, `binary`, `binary-centred`, `pq48`, `pq24`, `mrl256`, `mrl192`, `mrl96`, `mrl64`, `mrl192+int8`, `mrl96+int8`, `mrl192+binary`, `mrl96+binary` |
| Codecs on nomic | `int8`, `binary`, `binary-centred`, `pq96`, `pq48`, `mrl512`, `mrl256`, `mrl128`, `mrl64`, `mrl256+int8`, `mrl128+int8`, `mrl64+int8`, `mrl384+binary`, `mrl256+binary`, `mrl192+binary`, `mrl128+binary`, `mrl256+pq32` |
| Trained codecs | `binary-centred` and `pq<m>` fitted on all 8,036 corpus vectors (PQ: 256 k-means centroids per subspace, 20 Lloyd iterations) |
| Search | brute force, no index, no rescoring |
| Hardware | Apple M4 Pro, 14 cores, release build |
| Code | hvec commit `f314280` |
| Command | `./run.sh` in this directory (downloads data and both models on first run; nomic is about 530 MB) |

## Results

nDCG@10 against the judgments, by byte budget. Each row is one codec on one model; the two
columns line up configurations of similar size. f32 is the ceiling for each model.

| bytes / vector | bge-small-en-v1.5 (384 d) | nDCG@10 | nomic-embed-text-v1.5 (768 d) | nDCG@10 |
|---:|---|---:|---|---:|
| 3072 | | | f32 | **0.696** |
| 2048 | | | mrl512 | 0.691 |
| 1536 | f32 | **0.698** | | |
| 1024 | mrl256 | 0.684 | mrl256 | 0.671 |
| 776 | | | int8 | 0.695 |
| 768 | mrl192 | 0.659 | | |
| 512 | | | mrl128 | 0.632 |
| 392 | int8 | 0.698 | | |
| 384 | mrl96 | 0.593 | | |
| 264 | | | mrl256+int8 | 0.671 |
| 256 | mrl64 | 0.520 | mrl64 | 0.520 |
| 200 | mrl192+int8 | 0.660 | | |
| 136 | | | mrl128+int8 | 0.634 |
| 104 | mrl96+int8 | 0.594 | | |
| 96 | | | pq96 · binary-centred · binary | 0.686 · 0.683 · 0.673 |
| 72 | | | mrl64+int8 | 0.520 |
| 48 | pq48 · binary-centred · binary | **0.678** · 0.658 · 0.654 | pq48 · mrl384+binary | 0.634 · 0.632 |
| 32 | | | mrl256+pq32 · mrl256+binary | 0.638 · 0.586 |
| 24 | pq24 · mrl192+binary | 0.602 · 0.580 | mrl192+binary | 0.523 |
| 16 | | | mrl128+binary | 0.431 |
| 12 | mrl96+binary | 0.412 | | |

The full tables with recall-vs-f32, top-1 agreement, score error, document recall and timings
are in `results/bge-small-k10.json` and `results/nomic-k10.json`; the condensed versions:

| bge-small | recall@10 vs f32 | nDCG@10 | doc recall@10 | bytes |
|---|---:|---:|---:|---:|
| f32 | 1.000 | 0.698 | 0.818 | 1536 |
| int8 | 0.987 | 0.698 | 0.818 | 392 |
| binary | 0.611 | 0.654 | 0.768 | 48 |
| binary-centred | 0.656 | 0.658 | 0.770 | 48 |
| pq48 | 0.711 | 0.678 | 0.797 | 48 |
| pq24 | 0.577 | 0.602 | 0.725 | 24 |
| mrl256 | 0.745 | 0.684 | 0.796 | 1024 |
| mrl192 | 0.649 | 0.659 | 0.767 | 768 |
| mrl96 | 0.456 | 0.593 | 0.712 | 384 |
| mrl64 | 0.350 | 0.520 | 0.635 | 256 |
| mrl192+int8 | 0.649 | 0.660 | 0.774 | 200 |
| mrl96+int8 | 0.458 | 0.594 | 0.710 | 104 |
| mrl192+binary | 0.428 | 0.580 | 0.703 | 24 |
| mrl96+binary | 0.252 | 0.412 | 0.557 | 12 |

| nomic-embed-text-v1.5 | recall@10 vs f32 | nDCG@10 | doc recall@10 | bytes |
|---|---:|---:|---:|---:|
| f32 | 1.000 | 0.696 | 0.835 | 3072 |
| int8 | 0.993 | 0.695 | 0.832 | 776 |
| binary | 0.633 | 0.673 | 0.805 | 96 |
| binary-centred | 0.727 | 0.683 | 0.821 | 96 |
| pq96 | 0.775 | 0.686 | 0.813 | 96 |
| pq48 | 0.646 | 0.634 | 0.782 | 48 |
| mrl512 | 0.815 | 0.691 | 0.822 | 2048 |
| mrl256 | 0.680 | 0.671 | 0.793 | 1024 |
| mrl128 | 0.538 | 0.632 | 0.767 | 512 |
| mrl64 | 0.384 | 0.520 | 0.640 | 256 |
| mrl256+int8 | 0.679 | 0.671 | 0.793 | 264 |
| mrl128+int8 | 0.537 | 0.634 | 0.774 | 136 |
| mrl64+int8 | 0.383 | 0.520 | 0.640 | 72 |
| mrl384+binary | 0.479 | 0.632 | 0.750 | 48 |
| mrl256+binary | 0.392 | 0.586 | 0.702 | 32 |
| mrl192+binary | 0.332 | 0.523 | 0.663 | 24 |
| mrl128+binary | 0.257 | 0.431 | 0.556 | 16 |
| mrl256+pq32 | 0.566 | 0.638 | 0.770 | 32 |

## Discussion

**Matryoshka training works as advertised, per dimension.** Dropping two thirds of nomic's
dimensions costs 2.5 nDCG points; dropping three quarters of bge-small's costs 10.5. Keeping
one sixth costs both about 17.7 points, so the advantage runs out at the smallest sizes, which
is consistent with the Matryoshka loss weighting the larger prefixes. Truncating a model that
was not trained for it is, as expected, a plain lossy projection: bge-small's loss grows
roughly with the fraction of dimensions removed, with no privileged prefix.

**Per byte, native truncation never wins.** The two models retrieve equally well at full
precision (0.698 against 0.696), but nomic needs twice the bytes to do it. Every nomic
truncation is therefore competing against a bge-small configuration with the same bytes and a
smaller fraction removed, or against bge-small's int8 and PQ, which lose almost nothing. At
1,024 bytes the post-hoc truncation wins; at 264 bytes nomic's `mrl256+int8` (0.671) is beaten
by bge-small's `pq48` at 48 bytes (0.678) and `int8` at 392 bytes (0.698). The roadmap's
conditional, "if native wins, the right codec is a different embedding model", comes out false
on this corpus. It would come out differently for a Matryoshka model with the *same* base
dimension as the conventional one; the comparison here is between the two small models a
practitioner would actually be choosing between.

**Product quantization is the best codec below 400 bytes on both models.** At 48 bytes on
bge-small, `pq48` keeps 0.678 against `binary`'s 0.654 and `binary-centred`'s 0.658: two
points under f32 at 32× smaller, where the earlier experiments had binary at 4.4 points under.
At 96 bytes on nomic, `pq96` (0.686) edges `binary-centred` (0.683) and `binary` (0.673). PQ's
8-dimensional subspaces with 256 centroids resolve cosine to 0.03 mean error on bge-small,
against 0.31 for plain binary and 0.05 for centred binary, and unlike the sign codes they spend
their byte on the actual distribution of the data rather than on one fixed bit per dimension.
The cost is a fitted codebook (393 KB per collection at 384 dims, fitted in a few seconds on
8,036 vectors) and, in this implementation, a scan that walks the centroids rather than a
lookup table.

**Fewer bits of more dimensions beats more bits of fewer dimensions.** On nomic, `binary` on
all 768 dimensions (96 bytes, 0.673) beats `mrl128+int8` (136 bytes, 0.634) and
`mrl256+binary` (32 bytes, 0.586). Truncation then binarisation compounds two losses and the
Matryoshka prefix does not protect the sign bits. The best 32-byte code on nomic is
`mrl256+pq32` (0.638), again PQ, and the best 48-byte code is bge-small's `pq48`, not anything
on nomic.

**Nomic tolerates binary far better than bge-small.** Binary costs nomic 2.3 nDCG points and
bge-small 4.4; centring costs nomic 1.3. Two reasons visible in the numbers: nomic has twice
as many sign bits to spend, and its vectors are closer to centred (binary mean score error
0.12 against 0.31). The same 96 bytes buys more on nomic than on bge-small's 48 bytes plus 48
bytes of anything else.

**What is wrong with this experiment.** One corpus, one domain, and only two models, one of
each kind; a 384-dimension Matryoshka model or a 768-dimension conventional one would make the
per-byte comparison cleaner. The nomic prefixes are the published ones but bge-small ran
without its recommended query instruction, as in the earlier experiments, which flatters the
comparison slightly for nomic. PQ here is plain PQ with evenly spaced initial centroids and no
rotation; optimised PQ would widen its margin, not close it. `mrl<k>` on bge-small keeps the
leading dimensions, which is one of many equally arbitrary projections for a non-Matryoshka
model; a random projection to the same size would be the fairer post-hoc control. And, still,
no chat model.

**Next.** PQ is now the codec to carry into the chat-model experiments alongside binary, since
it occupies the same 48 bytes with half the loss. A rotation before quantization (OPQ for PQ,
ITQ for binary) is the natural follow-up on the codec side, and a rescoring stage the natural
one on the pipeline side.
