---
title: "hvec: measuring vector compression where it matters"
date: 2026-10-08
excerpt: "Why hvec exists, what milestone 1 ships, and what comes next."
---

Vector databases quantize embeddings to save memory, and the papers that introduce those
quantizers report recall@k against an exact search. That is the right metric for a search engine.
It is an indirect metric for a retrieval-augmented generation (RAG) system, where the retrieved
passages are consumed by a language model that may or may not notice a slightly worse ranking.

hvec exists to measure the direct thing: run the same questions through the same pipeline with
different codecs and different chat models, and compare the answers, the latency, and the token
cost. The benchmark is a matrix of embedding model × codec × chat model × dataset. Compression
behaviour depends on the embedding model. Tolerance to degraded retrieval depends on the chat
model. Both axes are recorded on every run.

## What milestone 1 ships

- A Cargo workspace with a `Codec` trait whose `score` method works on the compressed form.
  Only the f32 baseline exists so far; it is the thing every other codec will be measured against.
- Connectors for Anthropic, any OpenAI-compatible server (OpenAI, Ollama, vLLM), and local
  ONNX embeddings, all behind named profiles in a TOML config.
- A SQLite store that scores through the codec by brute force. Deliberately no index: the point
  is to measure the codec, not the ANN structure.
- An interactive shell. Type `hvec`, point it at a folder with `/new docs ./docs`, ask questions,
  switch models with `/model`, and every turn lands in a run log with timings, tokens, retrieved
  chunk ids and the prompt version.

## What comes next

Int8 and binary codecs selectable per collection, then product quantization, then a `bench`
command that runs a question set across the matrix and a `report` that summarises the log. The
first published experiment will be int8 and binary against f32 on a few hundred QA pairs with a
small local embedding model and two chat models.

If that sounds useful, the repository is at
[github.com/SwineCoder101/hvec](https://github.com/SwineCoder101/hvec) and contributions are
welcome.
