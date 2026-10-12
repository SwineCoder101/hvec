# SciFact through two chat models: does the codec reach the answer? (not yet run)

**Status: scripted, not run.** This experiment needs two chat models and a judge model. The
session that built the tooling had no API key and no local chat server, so the script is
committed and the numbers are not. Run `./run.sh` with `ANTHROPIC_API_KEY` set (or point
`SMALL_CHAT`, `LARGE_CHAT` and `JUDGE_CHAT` at an Ollama profile) and replace this file with the
write-up; the template is in `CONTRIBUTING.md`.

**Hypothesis.** The labelled-retrieval experiment found that `binary` loses 4.4 points of
nDCG@10 and `int8` loses nothing against the SciFact judgments. Carried through to a chat model
that must answer *supports* or *refutes* from the retrieved abstracts, int8 will not move the
answer rate on either model; binary will lower it by a few points on a small model and by less,
possibly nothing measurable, on a large one, because a large model recovers from a weaker but
still relevant passage set more often. The claim under test is that the chat-model axis matters
as much as the codec axis.

**Setup.** The 188 SciFact test claims with a uniform SciFact verdict, phrased as "Does the
evidence support or refute this claim? …" with the bare claim as the retrieval query
(`questions/scifact.jsonl`). Corpus and chunking as in the retrieval experiments; `bge-small`
embeddings; collections for `f32`, `int8`, `binary` and `binary-centred`; k = 5; two chat
models of very different size (Claude Haiku 5.5 and Claude Opus 5.5 by default); every answer
then graded by a third model (`hvec judge run`, Claude Sonnet 5.5 by default) for correctness
and faithfulness, with the judge's agreement with `contains` reported alongside. One
`bench run` batch: 188 × 4 × 2 = 1,504 model calls plus 1,504 judge calls.

**What to report.** Per cell: contains accuracy, judge-correct rate, judge-faithful rate,
retrieval hit rate and nDCG@5, mean input tokens, generation latency; the agreement table;
and the answer-rate gap between f32 and each codec, per model. Keep the negative result if the
gaps are within noise: 188 questions resolve differences of about 7 points at 95 percent
confidence, so "no measurable effect" is itself the finding at that size.
