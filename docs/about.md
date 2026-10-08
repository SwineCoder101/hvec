---
layout: page
title: About
permalink: /about/
---

hvec is an open-source research sandbox. The question it exists to answer is not "what is
recall@10 for this quantizer" but "does this compression change what a given model says". The
codec is measured end to end, through retrieval and generation, with every run recorded.

Why this matters: vector databases are increasingly the memory of LLM applications, and
compressing those vectors is the main lever for cost. The compression literature reports
retrieval metrics. Application teams care about answers. hvec sits in the gap.

The project is MIT licensed and maintained by [SwineCoder101](https://github.com/SwineCoder101).
Issues and pull requests are welcome at
[github.com/SwineCoder101/hvec](https://github.com/SwineCoder101/hvec).
