---
layout: page
title: References
permalink: /references/
---

The papers behind the ideas in hvec, grouped by topic, each with a note on why it matters here.
Blog posts and experiment write-ups cite entries from this list by key. To add one, append to
[`docs/_data/references.yml`](https://github.com/SwineCoder101/hvec/blob/main/docs/_data/references.yml)
with a resolvable DOI or arXiv link. Definitions of the terms are in the
[glossary]({{ "/glossary/" | relative_url }}).

{% assign groups = site.data.references | group_by: "topic" %}
{% for g in groups %}
## {{ g.name }}

{% assign items = g.items | sort: "year" %}
{% for r in items %}
<p id="{{ r.key }}" class="ref">
<strong>{{ r.title }}</strong><br>
{{ r.authors }}. <em>{{ r.venue }}</em>, {{ r.year }}. {% if r.url %}<a href="{{ r.url }}">{{ r.url | remove: "https://" }}</a>{% endif %}<br>
<small>{{ r.note }} Cite as <code>{{ r.key }}</code>.</small>
</p>
{% endfor %}
{% endfor %}
