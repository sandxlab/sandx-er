# Contributing to sandx-er

Thanks for your interest. This document covers how to set up a development environment, run tests, and submit a pull request.

---

## Development setup

```bash
git clone https://github.com/sandxlab/sandx-er
cd sandx-er
pip install -e ".[dev]"
```

For embedding-based features:

```bash
pip install -e ".[dev,embed]"
```

## Running tests

```bash
pytest tests/ -q
```

With coverage:

```bash
pytest tests/ --cov=sandx_er --cov-report=term-missing -q
```

## Linting

```bash
ruff check src tests
```

We use `ruff` with `line-length = 100`. Fix lint errors before opening a PR — CI will reject anything that fails.

## Code style

- Type-annotate all public functions and methods.
- No comments explaining *what* code does. Only add a comment when the *why* is non-obvious (a hidden constraint, a workaround, a subtle invariant).
- No docstrings that restate the function name. Keep docstrings short and focused on contracts, not mechanics.
- Every output carries a confidence score. Don't add binary-decision shortcuts.

## Before opening a PR

1. Tests pass: `pytest tests/ -q`
2. Lint passes: `ruff check src tests`
3. New behaviour has test coverage.
4. Benchmark results are not regressed — run `python -m benchmarks.febrl4` and verify F1 is not lower than the README table.

## Pull request process

- Branch off `main`. Name your branch `feat/short-description` or `fix/short-description`.
- Keep PRs focused. One logical change per PR.
- PR description should explain *why*, not just *what*.
- At least one approving review is required before merge.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/sandxlab/sandx-er/issues).

- **Bug reports:** include Python version, sandx-er version (`pip show sandx-er`), minimal reproducing example, and the full traceback.
- **Feature requests:** describe the use case, not just the feature. What problem does it solve?

## Design principles

sandx-er is infrastructure, not a product. Changes that add external dependencies without a strong justification will not be merged. Changes that improve correctness, performance, or composability with the other SandX engines are welcome.

---

Apache 2.0 license. By contributing you agree your changes are released under the same license.
