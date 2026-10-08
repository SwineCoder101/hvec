# Contributing to hvec

Thanks for your interest. hvec is an experimental sandbox, so the bar for contributing is low and
the bar for honesty about results is high. This document explains how to get set up, how to
propose and run experiments, and what a pull request needs to be merged.

## Ways to contribute

- **Experiments.** Implement a codec or a variation, benchmark it, write up the result.
- **Benchmarks and datasets.** Add a dataset loader, a metric, or a fairer baseline.
- **Core code.** Improve the codec trait, the distance kernels, or the harness.
- **Documentation.** Fix a mistake, clarify an explanation, add a reference.
- **Issues.** Report a bug, question a result, or suggest a direction.

If you are unsure whether something fits, open an issue first. It is cheaper than a pull request
that goes nowhere.

## Development setup

1. Install a stable Rust toolchain with [rustup](https://rustup.rs).
2. Install the components the project uses:

   ```sh
   rustup component add rustfmt clippy
   ```

3. Clone and build:

   ```sh
   git clone https://github.com/SwineCoder101/hvec.git
   cd hvec
   cargo build
   cargo test
   ```

Before pushing, run the same checks CI will run:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

## Proposing an experiment

Open an issue titled `experiment: <short name>` and fill in:

- **Hypothesis.** One or two sentences. What do you expect to be true, and why?
- **Method.** Which codec or variation, which parameters.
- **Dataset.** Which embedding set, how many vectors, what dimensionality.
- **Baseline.** What it is being compared to. Uncompressed f32 is the default baseline.
- **Metrics.** Pick from recall@k, mean distance error, queries per second, bytes per vector.
- **Done when.** The concrete result that closes the issue, positive or negative.

Small, sharp experiments are preferred over broad ones. If the idea needs several steps, split it.

## Writing up an experiment

Each experiment lives in `experiments/<yyyy-mm>-<short-name>/` and contains:

```
experiments/2026-10-pq-adc-baseline/
├── README.md        # Write-up (template below)
├── run.sh           # One command that reproduces the numbers
└── results/         # Raw output, CSV or JSON, committed if small
```

The `README.md` template:

```markdown
# <Title>

**Hypothesis.** ...
**Result.** One sentence. Confirmed, refuted, or inconclusive.

## Setup
Dataset, size, dimension, hardware, commit hash, exact command.

## Results
A table. Include the baseline in every table.

## Discussion
What surprised you. What you would try next. What is wrong with this experiment.
```

Negative and inconclusive results are merged. They save the next person's time.

## Code guidelines

- **Keep the codec trait honest.** A codec must declare which operations it preserves and with
  what error model. Do not silently decompress to compute a distance.
- **Prefer plain Rust.** Reach for `unsafe` or SIMD intrinsics only after a scalar version exists
  and is tested against it.
- **Test against the reference.** Every compressed-domain distance needs a test comparing it to
  the uncompressed f32 result on random vectors with a stated tolerance.
- **Benchmark with criterion** or the project harness. Do not paste numbers from a one-off
  `println!` into a write-up.
- **No new dependencies without a reason** stated in the pull request.
- Run `cargo fmt` and fix every `clippy` warning. Formatting arguments are settled by rustfmt.

## Pull requests

1. Fork the repository and create a branch from `main`. Name it after the change, for example
   `codec/int8-symmetric` or `bench/sift1m-loader`.
2. Keep the pull request focused. One codec, one experiment, or one fix per PR.
3. Write a description that says what changed, why, and how you verified it. Link the issue.
4. Make sure the checks in the development setup section pass locally.
5. Request a review. Expect questions about measurements. That is the point of the project.

Commits should follow the [Conventional Commits](https://www.conventionalcommits.org) style:

```
feat(codecs): add int8 symmetric scalar quantizer
fix(bench): use correct k when computing recall
docs: clarify error model in codec trait
exp: pq adc baseline on glove-100
```

Use your own name and email as the commit author. Squash fixup commits before requesting review.

## Reporting bugs

Open an issue with the Rust version, the operating system, the command you ran, and the full
output. If the bug is a wrong number rather than a crash, include the expected value and how you
derived it.

## Security and privacy

The encryption track handles homomorphic encryption schemes. Nothing in this repository is
audited or fit for protecting real data. If you find a flaw in an implementation, open an issue.
There is no private disclosure channel because there is nothing in production to protect.

## Code of conduct

Be direct about ideas and kind to people. Critique the measurement, not the person who made it.
Harassment, personal attacks, and discrimination are not tolerated and will result in removal
from the project. Report problems to the maintainer through a GitHub issue or by email to the
address on the maintainer's GitHub profile.

## License

A license has not been added yet. By contributing you agree that your contribution will be
released under the permissive license (MIT or Apache-2.0) the project adopts, and that you have
the right to submit it.
