---
title: "Similarity on encrypted vectors: the cost model, before any code"
date: 2026-10-12 18:00:00 +0000
references: [cheon2017ckks, gentry2009fhe, jegou2011pq, charikar2002simhash]
---

The roadmap's last track is similarity search on *encrypted* embeddings: a codec whose
compressed-domain operation is exact, where the thing it buys is not bytes but privacy. This note
works out what such a codec costs before writing any of it, because the arithmetic decides
whether it belongs in hvec's `Codec` trait at all. The short answer is that it does not, yet, and
the reasons are instructive.

## The scheme

CKKS [1] is the homomorphic encryption scheme built for approximate arithmetic on real numbers,
which is what a dot product of embeddings is. A ciphertext holds a vector of up to `N/2` complex
"slots" (so `N/2` reals), where `N` is the ring dimension, and supports slot-wise addition and
multiplication plus rotations of the slot vector. A dot product of two encrypted 384-dim vectors
is one slot-wise multiply followed by `log2(384)` rotate-and-add steps to sum the slots. The
result is an encrypted scalar; whoever holds the secret key decrypts it.

That is a homomorphic codec in exactly hvec's sense: `encode` is encryption, `score` is the
encrypted dot product, and `decode` is decryption. The shape fits the trait. The numbers do not.

## Sizes

A CKKS ciphertext is two polynomials of degree `N` with coefficients modulo a product of primes
`q`. For a dot product followed by one rescale, with the precision an embedding needs, a common
parameter set is `N = 8192` and `log2(q) ≈ 200` bits, giving `2 × 8192 × 200 / 8 ≈ 410 KB` per
ciphertext [2]. One ciphertext packs `N/2 = 4096` reals, so it can hold about 10 embeddings of
384 dimensions, or one with 3,712 dimensions of padding.

| representation | bytes per 384-dim vector | relative to f32 |
|---|---:|---:|
| f32 | 1,536 | 1× |
| int8 (hvec) | 392 | 0.26× |
| binary (hvec) | 48 | 0.03× |
| CKKS, 10 vectors packed per ciphertext | ≈ 41,000 | 27× **larger** |
| CKKS, one vector per ciphertext | ≈ 410,000 | 270× larger |

The first thing the encryption track does is invert hvec's axis: every other codec makes the
store smaller, this one makes it 27 to 270 times bigger. "Compression" is the wrong word, and
the write-up should say so.

## Time

Published library benchmarks put a CKKS ciphertext multiply with relinearisation and a
rotation each at roughly a millisecond at `N = 8192` on one core, with the exact figures
depending on the library and the depth budget. A packed dot product against 10 stored vectors
is one multiply and nine rotations: on the order of 10 ms. A brute-force scan of the
8,036-chunk SciFact collection is then about 800 packed operations, or several seconds per
query. hvec's f32 baseline scans the same collection in 4 ms, int8 in 2 ms.

| scan of 8,036 chunks, one query | time | ratio to f32 |
|---|---:|---:|
| f32, decoded (hvec baseline) | 4 ms | 1× |
| int8 compressed-domain (hvec) | 2 ms | 0.5× |
| CKKS packed dot products, one core (estimate) | 5 to 10 s | 1,000 to 2,500× |

Three orders of magnitude slower and 27× bigger, for exact scores. There is a fourth cost: the
query must also be encrypted under the same key, and top-k selection cannot be done on
encrypted scores without further machinery (comparison under CKKS is itself an approximate,
multi-level circuit), so the server returns *all* encrypted scores and the client decrypts and
ranks them. The bandwidth per query is then 8,036 encrypted scalars, which with packing is 800
ciphertexts, about 330 MB. That is the figure that ends the discussion for anything interactive.

## What would make it interesting

The cost model is not hopeless, only mismatched with brute-force scan. Three changes move it:

- **Ask a different question.** The agent-memory loop writes often and reads rarely from a
  small store (hundreds of entries, not thousands). At 100 memories the scan is 10 packed
  operations and 10 ciphertexts back: tens of milliseconds and 4 MB. Encrypted memory for an
  agent whose host must not read its memories is a plausible setting.
- **Quantize first, then encrypt.** CKKS over sign bits or int8 codes needs far less precision,
  which means smaller `q`, smaller ciphertexts and shallower circuits. A SimHash-style [4]
  encrypted Hamming distance is a well-studied primitive. The codec composition hvec already has
  (`mrl<k>+<codec>`) is the right shape: `enc+binary`.
- **Index, don't scan.** Clustering (as in IVF or PQ [3]) on plaintext centroids with encrypted
  residuals confines the encrypted work to one cluster. That leaks which cluster a query lands
  in, which is a privacy decision, not an engineering one.

## Decision

No CKKS codec in hvec for now. The measurement harness is built around the claim that a codec's
cost is bytes and the question is whether the answers change; an encrypted codec's cost is time
and bandwidth and its answers do not change by construction, so almost nothing in `bench recall`
or `bench run` would measure anything about it. If the track resumes, it starts with the
agent-memory benchmark at a hundred entries and a quantize-then-encrypt codec, where the numbers
above say the costs land within reach, and with a `Codec` extension that lets `score` return an
opaque value a client decrypts, because the current trait's `f32` result is the one assumption
an encrypted codec cannot meet.

{% include references.html %}
