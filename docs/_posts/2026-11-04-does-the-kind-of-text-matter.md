---
title: "Does the kind of text change what a codec costs?"
date: 2026-11-04 18:30:00 +0000
references: [thakur2021beir, li2024coir, xiao2023bge, reimers2019sbert, jegou2011pq]
---

Every hvec experiment so far has used one corpus, SciFact's scientific abstracts. So every
codec number on this site carries an unstated condition: *on this kind of text*. The obvious
question is whether it would come out differently on financial forum posts, medical articles,
arguments or source code. This post is about the experiment that answers it. The answer is yes,
and the reason it is yes is more useful than the answer itself.

## A codec never sees the text

A codec compresses an embedding vector, and the text is gone by then. So the kind of text can
only matter through the vectors the embedding model makes of it. The question is which
property of those vectors matters.

The SciFact experiments had already pointed at one. When a 384-bit sign code (`binary`) drops a
neighbour from the top 10, it is almost always a near-tie. The 10th and 11th results on SciFact
differ in cosine by about a thousandth, and the sign code's score error is a couple of
hundredths. The codec is not getting the similarities badly wrong; it is getting them slightly
wrong, in a place where slightly wrong is enough to reorder the list.

That suggests a rule. What a codec costs should depend on two numbers:

- **its noise**: how much it jitters scores among the top candidates (the *rank noise*, with any
  constant offset removed, because a constant offset reorders nothing);
- **the corpus's spacing**: how far apart the candidates it has to tell apart sit (the
  *neighbour band*, the gap from the 10th to the 20th exact score).

The noise is a property of the codec and the embedding model. The spacing is a property of the
corpus and the model, and it is the only way the kind of text enters.

## The test

`hvec bench recall` now reports both numbers. The experiment runs nine codecs (int8, two
sign-bit codecs, three product quantizers [5], three truncations) over five retrieval sets with
relevance judgments [1], each embedded with two small models [3, 4]:

| corpus | what it is | band, bge-small | binary keeps of the f32 top 10 |
|---|---|---:|---:|
| FiQA | financial forum Q&A | 0.0108 | 0.537 |
| NFCorpus | medical articles | 0.0121 | 0.556 |
| SciFact | scientific abstracts | 0.0135 | 0.611 |
| CoSQA | Python functions [2] | 0.0148 | 0.660 |
| ArguAna | arguments and counterarguments | 0.0233 | 0.748 |

The binary codec's noise is nearly the same on FiQA and ArguAna (0.0177 and 0.0183). It loses 46
points of the top 10 on one and 25 on the other. The band explains the difference.

Across all 90 combinations of corpus, model and codec, the ratio of noise to band ranks the
loss almost perfectly (Spearman ρ = 0.98). Noise alone, the number a codec's spec sheet would
give you, manages 0.64. The relationship is close to proportional: **the odds of losing a
neighbour are about half the noise-to-band ratio**. Fit the curve on four corpora and it
predicts the fifth's recall to within 0.024 on average.

## So does the kind of text matter?

It does, by up to 21 points of binary recall between the tightest corpus and the loosest. Two
things in the result are worth more than that headline, though.

**Code is not special.** Source code was the content most unlike prose in the set, and it sits in
the middle of the range. "Text type" turns out to be a proxy, and the thing it stands in for can
be measured directly, in a few seconds, with nothing but the f32 vectors you already have.

**The model matters as much as the corpus.** On the same corpus, swapping bge-small for MiniLM
widens the band 1.7 to 2.5×, as much as the whole spread across the five corpora. MiniLM's codec
noise grows by only 1.3 to 1.5×, so it tolerates binary better on every corpus. That had puzzled
the earlier experiments, and this is the explanation.

## What to do with it

Before choosing a codec for a new collection, measure its neighbour band in f32. You can then
estimate the codec's loss from a noise figure measured elsewhere. Done that way, without running
the codec on the new corpus, the estimate was off by 0.033 on average, half the error of assuming
the codec behaves as it did on other corpora. It worked on four of the five corpora. It failed
on CoSQA, whose corpus turned out to be 70% duplicated code; that lowers the codec noise and
fools the borrowed figure.

None of this says how much the loss matters to an answer. Against the human relevance labels the
same codecs lose 0.7 to 5.4 nDCG points where they lose 19 to 46 points of agreement with f32,
and whether that reaches a chat model's answer is still the job of the RAG experiment that is
waiting for models to run on. The next retrieval-only question follows directly from this one:
if the ratio says how many true neighbours a codec scrambles, it should also say how deep a
cheap f32 rescoring pass has to go to recover them.

The write-up, with the caveats, is
[on the experiments page]({{ "/experiments/neighbour-spacing/" | relative_url }}).

{% include references.html %}
