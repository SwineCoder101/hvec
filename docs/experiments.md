---
layout: page
title: Experiments
permalink: /experiments/
---

Each experiment in the repository's [`experiments/`](https://github.com/SwineCoder101/hvec/tree/main/experiments)
directory ships with a write-up: hypothesis, setup, results table with the f32 baseline, and
discussion. Finished write-ups are published here. Negative and inconclusive results are kept.

{% assign exps = site.experiments | sort: "date" | reverse %}
{% if exps.size == 0 %}
No experiments have been published yet. The first will compare int8 and binary codecs against
the f32 baseline on a small QA set. See the
[experiment template](https://github.com/SwineCoder101/hvec/blob/main/CONTRIBUTING.md#writing-up-an-experiment)
if you want to run one.
{% else %}
<ul>
{% for e in exps %}
  <li>
    <a href="{{ e.url | relative_url }}">{{ e.title }}</a>
    {% if e.date %}<small>({{ e.date | date: "%Y-%m-%d" }})</small>{% endif %}
    {% if e.result %}<br><em>{{ e.result }}</em>{% endif %}
  </li>
{% endfor %}
</ul>
{% endif %}
