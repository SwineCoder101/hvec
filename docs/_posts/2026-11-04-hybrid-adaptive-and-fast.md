---
title: "Hybrid, adaptive, and finally fast"
date: 2026-11-04 22:30:00 +0000
references: [li2024quantmoe, charikar2002simhash, jegou2011pq]
---

[The last post]({{ "/blog/2026/11/04/does-the-kind-of-text-matter/" | relative_url }}) found the
number that sets what a codec costs: its score noise divided by how tightly the corpus packs a
query's neighbours. Four experiments followed from it. Together they answer the three questions
people actually ask about vector compression: how do I get the lost neighbours back, can the
compression adapt to the data, and is any of it faster?

## Getting the neighbours back: hybrid retrieval

Scan with a small code, shortlist its best N candidates, and re-score only those with the full
vectors.

<p align="center"><img src="{{ "/assets/animations/hybrid-rescoring.svg" | relative_url }}" width="820" alt="Animated schematic of two-stage retrieval."></p>

The same noise-to-spacing ratio that predicts the loss also predicts how deep the shortlist must
be, to within about a third. At 48 bytes per vector, 50 candidates recover 95–99% of the exact
top 10 and essentially all of the loss in nDCG
([two-stage retrieval]({{ "/experiments/two-stage-rescoring/" | relative_url }})).

## Adaptive compression

Adaptive compression can mean spending bits unevenly, or spending work unevenly.

**Uneven bits.** Mixed-precision quantization of mixture-of-experts models gives more bits to the
blocks and experts that matter more [1]. For embeddings, "matters more" means a principal
component with more variance, or with more of the query's energy:

<p align="center"><img src="{{ "/assets/animations/adaptive-bits.svg" | relative_url }}" width="820" alt="Animated bar chart of a 48-byte bit allocation over 384 principal components."></p>

Rotating into principal components and allocating bits this way beats product quantization [3] at
24, 48 and 96 bytes on average, by 1.6 to 2.9 points of recall@10. The gain is all on one of the two
embedding models; on the other they tie
([the hybrid × adaptive matrix]({{ "/experiments/hybrid-adaptive-matrix/" | relative_url }})).

**Uneven work.** Re-score more candidates for crowded queries and fewer for well-separated ones:

<p align="center"><img src="{{ "/assets/animations/adaptive-depth.svg" | relative_url }}" width="820" alt="Animated schematic of adaptive re-scoring depth."></p>

This works, but only a little: half a point of recall at the same average work, and up to four
points for the worst queries. I first reported 1.5 points; that came from comparing against too
few fixed depths, and the write-up now says so. A hindsight oracle shows 6 points of headroom,
but the scan's own scores reveal almost none of it, and no router built on them captured more than
6% ([routing]({{ "/experiments/routing-oracle/" | relative_url }})). Routing is worth doing per
collection, where the noise-to-spacing curve picks the right codec, but not per query.

## Is any of it faster?

Until now hvec claimed nothing about speed. Zero-copy plumbing changed that: the codes now sit in
one contiguous buffer and are scored in place, with no copy or allocation per vector. That alone
made search 5× faster. Then the uncomfortable result. With copying out of the way, every
asymmetric codec (an f32 query against compressed codes) costs about what f32 costs per vector,
because its scoring, not its bytes, sets the time. A 48-byte binary scan reads 32× fewer bytes
and takes longer than f32.

The exception is symmetric scoring: binarise the query too and compare by XOR and popcount [2]. That
costs 10 ns per vector against 207 for f32. On its own it keeps only 40% of the top 10, but with
an f32 re-score of a few hundred candidates it reaches 95% recall about 19× faster than an exact
f32 scan ([zero-copy scan]({{ "/experiments/zero-copy-scan/" | relative_url }})).

So compression buys memory everywhere, and time only when the scoring is as cheap as the read.

{% include references.html %}
