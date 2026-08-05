# Contributing

## Setting up

```bash
git clone https://github.com/IGORSVOLOHOVS/repo-quality-template
cd repo-quality-template
python scripts/install_dependencies.py --dev
pre-commit install
```

`pre-commit install` is not optional — it is what keeps formatting and secret
scanning out of code review.

## The loop

```bash
pytest                                   # tests, with coverage
ruff check . && ruff format .            # lint and format
pytest benchmarks --benchmark-only       # only if you touched the domain layer
python scripts/collect_quality_metrics.py
```

CI runs all of these. Running them locally first is faster than waiting for a
red build.

## Rules that the build enforces

| Rule | Enforced by |
| --- | --- |
| Coverage at or above 85 percent | `fail_under` in `pyproject.toml` |
| Cyclomatic complexity at most 10 per function | `ruff` rule `C90` |
| No credential in any commit | `gitleaks`, in pre-commit and CI |
| Exactly three branches | `scripts/enforce_branch_policy.py` |
| Formatting | `ruff format` |

## Branches

Work on `dev`. Do not push any other branch — see `docs/branching.md`. Short
lived work belongs in your local clone; rebase onto `dev` and push `dev`.

## Commits

Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`,
`build:`, `ci:`, `chore:`, `security:`.

Write what changed and why, in English. A commit message that only says what the
diff already shows is not worth reading.

## Adding a feature

1. Put the logic in `core.py`, with no I/O.
2. Add tests for it, including the empty and invalid cases.
3. If it could be slow, add a benchmark.
4. Update the README if the behaviour is user-visible.
5. Add a `CHANGELOG.md` entry under `[Unreleased]`.

Step 1 matters: logic that reaches into `cli.py` or `app.py` cannot be reused by
the other shell and is much harder to test.
