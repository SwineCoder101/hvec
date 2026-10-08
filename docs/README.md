# Site source

This folder is published with GitHub Pages at https://swinecoder101.github.io/hvec/ using Jekyll
and the `minima` theme. No local build is needed; GitHub builds on push to `main`.

- Blog post: add `_posts/YYYY-MM-DD-slug.md` with `title` and `date` front matter. To cite, list
  keys from `_data/references.yml` under `references:` in order, write `[1]`, `[2]` in the text, and
  end the post with `{% include references.html %}`.
- References: append entries to `_data/references.yml` with a resolvable DOI or arXiv url.
- Glossary: edit `glossary.md`; link terms to references with `/references/#key`.
- Experiment write-up: add `_experiments/<slug>.md` with `title`, `date` and a one-line `result`,
  mirroring the `README.md` in `experiments/<yyyy-mm>-<slug>/` and linking to it.

To preview locally: `gem install bundler jekyll github-pages` then `cd docs && jekyll serve`.
