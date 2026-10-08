# hvec

**A Rust sandbox for homomorphic compression of embedding vectors, aimed at vector databases.**

> Status: early experimental. The repository is being scaffolded. Expect APIs, crate layout, and
> results to change without notice. Contributions and experiments are welcome from day one; see
> [CONTRIBUTING.md](CONTRIBUTING.md).

## What is homomorphic compression?

A compression scheme is *homomorphic* with respect to an operation when you can perform that
operation directly on the compressed representation and get the same (or a controlled
approximation of the same) result you would get on the original data. You never have to
decompress.

For vector search the operations that matter are:

- inner product and cosine similarity,
- Euclidean (L2) distance,
- top-k nearest-neighbour selection built on the two above.

If a codec preserves these operations, a vector database can store compressed vectors, scan them
compressed, and rank them compressed. That cuts memory, bandwidth, and cache pressure at the
hottest part of the query path.

Familiar techniques already sit on this spectrum, even if they are rarely named this way:

| Technique | Compressed-domain op | Trade-off |
|---|---|---|
| Scalar / int8 quantization | Approximate dot product | Small loss, ~4x smaller |
| Binary / 1-bit quantization | Hamming distance as a proxy | Large loss, ~32x smaller |
| Product quantization (PQ) with asymmetric distance | Table-lookup distance | Tunable loss, 8-64x smaller |
| Random projections / JL sketches | Approximate inner product | Provable error bounds |
| Homomorphic encryption (CKKS and friends) | Exact ops on ciphertext | Privacy, heavy compute |

hvec is a place to implement these side by side in Rust, measure them honestly, and explore the
gaps between them.

## Goals

- **Implement** a family of compressed-domain similarity codecs behind a shared trait.
- **Measure** recall, distance error, throughput, and memory on real embedding datasets with
  reproducible benchmarks.
- **Explore** the boundary between lossy compression and privacy-preserving computation. The
  encryption side (computing similarity on encrypted vectors) is in scope as an experiment track.
- **Stay small.** This is a research sandbox, not a database. Results should be easy to lift
  into real systems.

## Non-goals

- Building a full vector database, indexing service, or network API.
- Production hardening or stability guarantees.
- Competing with mature libraries on raw speed before the ideas are proven.

## Planned layout

The crate structure is not final. The intended shape is:

```
hvec/
├── crates/
│   ├── hvec-core/      # Codec trait, vector types, distance kernels
│   ├── hvec-codecs/    # Scalar, binary, PQ, sketch-based codecs
│   ├── hvec-he/        # Homomorphic-encryption experiments
│   └── hvec-bench/     # Datasets, metrics, benchmark harness
├── experiments/        # One directory per experiment with its write-up
├── CONTRIBUTING.md
└── README.md
```

## Getting started

You need a recent stable Rust toolchain. Install it with [rustup](https://rustup.rs) if you do not
have one.

```sh
git clone https://github.com/SwineCoder101/hvec.git
cd hvec
cargo build
cargo test
```

Until the workspace lands, these commands will do nothing useful. Follow the issues tab for the
scaffolding tracker.

## Experiments

Each experiment lives in its own directory under `experiments/` and ships with a short write-up:
the hypothesis, the dataset, the baseline, the metric, and the result. Negative results are kept.
See the experiment section of [CONTRIBUTING.md](CONTRIBUTING.md) for the template.

## Contributing

Bug reports, experiment proposals, benchmark additions, and documentation fixes are all welcome.
Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

A license has not been chosen yet. Until one is added, all rights are reserved by the author.
A permissive license (MIT or Apache-2.0) is the intended direction. Contributions submitted
before the license lands will be covered by it once it is added.
