---
title: "Two layers of compression, and which one the agent never sees"
date: 2026-10-08 12:00:00 +0000
excerpt: "An embedding is already a lossy codec. hvec adds a second one. Here is what each layer compresses, how they interact, and why the agent reads text, not codes."
references: [morris2023vec2text, jegou2011pq, kusupati2022mrl, guo2020scann, lewis2020rag, jiang2023llmlingua]
---

A fair objection to hvec goes like this. An embedding is already a compressed representation of
a passage. The text is what a human reads; the vector is what the retrieval system reads. So
quantizing embeddings is compressing something that was compressed once already. What is the
second layer for, and is it even measuring the right thing?

The objection is right about the premise and it sharpens the project rather than undermining it.

## Layer one: the embedding compresses an operation, not bytes

Take a 200-word chunk. As UTF-8 it is roughly 1 KB. Its embedding from a 1536-dimension model is
1536 × 4 bytes, about 6 KB. In storage terms the embedding model has made the data six times
larger. Nobody would call that compression.

What the embedding compresses is the cost of a question: are these two passages about the same
thing? Answering that from text means reading and understanding both. Answering it from vectors
is one dot product. The embedding model keeps exactly the information needed for that one
operation and discards the rest, including, mostly, the ability to read the passage back. That is
precisely the definition of homomorphic compression used throughout this project: preserve one
operation on the compressed form, give up everything else.

"Mostly" is doing some work there. Inversion attacks have shown that short texts can be
reconstructed from their embeddings almost word for word [1]. An embedding is a literal
compressed representation of the passage, not an opaque fingerprint, and it should be handled
with the same care as the text.

## Layer two: quantization compresses the same operation and returns the bytes

hvec's codecs take the vector the embedding model produced and preserve the same operation,
inner product or cosine, while shrinking the representation 4× to 32×. Product quantization [2]
is the canonical example: the stored vector becomes a handful of byte codes and the distance is
computed by table lookups on those codes, never on a reconstructed vector.

So the stack is two homomorphic codecs in series, both preserving similarity. The interesting
part is that their error models are different in kind.

- The embedding's error is semantic. It is whatever the model thinks "similar" means, and it
  cannot be measured against the text in any clean way. There is no ground truth vector.
- The quantizer's error is geometric. The f32 vector exists, so the quantizer's score can be
  compared with the exact score for every pair. Recall against f32 and mean score error are
  well defined.

This is why the f32 baseline sits at the centre of hvec. It is the boundary between the two
layers. Everything below it is measured exactly; everything above it is measured only through the
answer.

## The layers interact, which is why the embedding model is a benchmark axis

Quantization can only damage information the embedding kept. If a model spreads meaning evenly
across its dimensions, a uniform int8 codec treats it fairly. If a model packs most of its
meaning into a few leading dimensions, as Matryoshka-trained models deliberately do [3], the same
codec spends precision on dimensions that carry little and the loss lands unevenly. Two models
with the same dimension can react very differently to the same codec, and the right codec for
one may be the wrong one for the other. ScaNN made a related point about measuring quantization
by its effect on ranking rather than on reconstruction [4]; the same logic says you must measure
it per embedding model.

That is the reason the benchmark matrix is embedding model × codec × chat model × dataset, and
not codec alone.

## The agent never reads a vector

Here is the part the objection gets slightly wrong, and it matters for what hvec claims to
measure. In a retrieval-augmented pipeline [5] the vector is the **key**: it is what you search
by. The chunk text is the **payload**: it is what comes back. The retrieval step scores the query
against stored keys and returns the payloads for the best matches. The agent reads those
payloads, as plain text, in its context window.

So no, the agent does not read from codecs instead of embeddings. It reads neither. Switching
from f32 vectors to int8 codes changes how the keys are stored and how the scores are computed.
It does not change the format of anything the model sees. What it can change is *which* passages
come back, and that is the only route by which compression can alter the answer.

This is a useful boundary to draw. hvec compresses the key. Compressing the payload, by
summarising or pruning passages before the model reads them [6], is a third layer that changes
what the agent sees rather than what it finds. It is a legitimate axis with its own trade-offs,
and it is out of scope for hvec today. Keeping the two apart makes the question hvec answers
precise: does compressing the retrieval key change what gets retrieved enough to change the
answer, and does that depend on which model is answering?

## An experiment that falls out of this

If both layers compress the same operation, you can collapse them. Pick an embedding model that
emits int8 or binary vectors natively and compare it against post-hoc quantization of an f32
model to the same byte budget, on the same questions, with the same chat model. If native
low-precision wins, the right codec is "none, change layer one". If post-hoc wins, quantization
earns its place as a separate layer. Either result is worth publishing, and it will be one of the
first experiments on this site.

Terms used here are defined in the [glossary]({{ "/glossary/" | relative_url }}), including
new entries for *embedding as compression* and *key versus payload*.

{% include references.html %}
