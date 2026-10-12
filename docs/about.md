---
layout: page
title: About
permalink: /about/
---

hvec is an open-source research sandbox for testing claims about compressing embedding vectors.
It asks two questions, in order.

**What does compressing the vectors save, and what does it cost?** A vector store is memory,
and compressing its vectors is the main lever on what that memory costs: millions of vectors,
search on a device, scans bound by memory bandwidth, many indexes. The price is retrieval
quality, and hvec measures it against relevance judgments rather than only against agreement
with the f32 ranking, because the two disagree by a factor of six on the same data. A codec earns
a claim in one of three forms: less memory at comparable quality, faster search, or useful
operations directly on the compressed vectors. So far hvec supports the first (int8 at 4× with no
measurable loss; product quantization at 32× for two points of nDCG@10) and the third, and makes
no speed claim: in its brute-force scan binary is no faster than int8.

**Does the loss reach the answer?** Smaller retrieval infrastructure is worth having even when
the agent reads the same amount of text; the second question is whether the measured retrieval
loss changes what a chat model says, per model. The codec is measured end to end, through
retrieval and generation, with every run recorded. The compression literature stops at retrieval
metrics; application teams care about answers; hvec covers both and keeps them separate.

hvec is not a proxy or a token saver. Tools such as
[Headroom](https://github.com/headroomlabs-ai/headroom) compress the text an agent reads; hvec
compresses the vectors a store searches by, and measures what each layer does to the answers.
They are complementary, and hvec's payload stage lets the two be measured in one run.

The project is MIT licensed and maintained by [SwineCoder101](https://github.com/SwineCoder101).
Issues and pull requests are welcome at
[github.com/SwineCoder101/hvec](https://github.com/SwineCoder101/hvec).
