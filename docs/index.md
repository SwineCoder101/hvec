---
layout: home
title: Blog
---

**hvec** is a Rust command-line tool for testing claims about compressing embedding vectors:
what a codec saves in bytes, what it costs in retrieval quality against relevance judgments, and
whether that loss changes the answers a chat model gives. It ingests documents, stores embeddings
through a codec, retrieves in the compressed domain, asks a model, and records every stage. The
benchmark is a matrix of embedding model × codec × payload stage × chat model × dataset.

- Code and README: [github.com/SwineCoder101/hvec](https://github.com/SwineCoder101/hvec)
- Experiment write-ups: [experiments]({{ "/experiments/" | relative_url }})
- How to contribute: [CONTRIBUTING.md](https://github.com/SwineCoder101/hvec/blob/main/CONTRIBUTING.md)

Posts below cover progress, design notes and results.
