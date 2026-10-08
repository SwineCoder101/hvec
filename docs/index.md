---
layout: home
title: Blog
---

**hvec** is a Rust command-line tool for finding out whether compressing the vectors in a
retrieval-augmented generation pipeline changes the answers a chat model gives, and by how much.
It ingests documents, stores embeddings through a codec, retrieves in the compressed domain, asks
a model, and records every stage. The benchmark is a matrix of embedding model × codec × chat
model × dataset.

- Code and README: [github.com/SwineCoder101/hvec](https://github.com/SwineCoder101/hvec)
- Experiment write-ups: [experiments]({{ "/experiments/" | relative_url }})
- How to contribute: [CONTRIBUTING.md](https://github.com/SwineCoder101/hvec/blob/main/CONTRIBUTING.md)

Posts below cover progress, design notes and results.
