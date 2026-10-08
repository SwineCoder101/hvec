# Site source

This folder is published with GitHub Pages at https://swinecoder101.github.io/hvec/ using Jekyll
and the `minima` theme. No local build is needed; GitHub builds on push to `main`.

- Blog post: add `_posts/YYYY-MM-DD-slug.md` with `title` and `date` front matter.
- Experiment write-up: add `_experiments/<slug>.md` with `title`, `date` and a one-line `result`,
  mirroring the `README.md` in `experiments/<yyyy-mm>-<slug>/` and linking to it.

To preview locally: `gem install bundler jekyll github-pages` then `cd docs && jekyll serve`.
