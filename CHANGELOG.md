# Changelog

## v1.1.0 (2026-11-08)

### Commits since v1.0.0

- dc68b34 docs(docs): name revision 4.0.0 and its exceptions in the README
- acbd4c6 ci(ci): add the five lab workflows and the issue and pull request templates
- fb3c3dc chore(core): add the AI environment, its hooks and the lab-ai-env skill
- 6909fa2 docs(docs): add the README, licence, rules and manual
- ecb410c build(build): add recipe.ps1 and the test suites it runs
- e2aeaa7 refactor(core): settle the sources under the pinned ruff and mypy
- c3c4556 build(build): pin the toolchain, and decide line endings and ignores
- a38ebdd chore: stop tracking coverage.xml
- d039dfa chore: add the pre-push check that mirrors CI
- 9d0a5de fix(ci): unescape the quotes in the tag guard
- bfc2924 fix(ci): make Run workflow work instead of failing
- b3a748d docs: remove the course brief - it is the course's text, not this project's
- 32c3afc chore: add the pre-push check that mirrors CI
- a6816b1 chore: add the pre-push check that mirrors CI
- 703b079 feat: architecture, benchmarks and an ISO 25010 assessment of this project
- 69f9891 chore: add the pre-push check that mirrors CI
- 85f31b4 style: format the added scripts to this repository's rules
- 4383c17 fix: type the quality metrics script for mypy --strict

### Issues

- #1 Pin the toolchain, and decide line endings and ignores in zero-annottation-od (open)
- #2 Bring the sources of zero-annottation-od under the pinned tools (open)
- #3 Add recipe.ps1 and its test suites to zero-annottation-od (open)
- #4 Add the README, licence, rules and manual to zero-annottation-od (open)
- #5 Add the AI environment to zero-annottation-od (open)
- #6 Add the gates and templates to zero-annottation-od (open)
- #7 State the revision 4.0.0 compliance line in zero-annottation-od (open)

### Pull requests

- #14 build(build): pin the toolchain, and decide line endings and ignores (closed)
- #15 refactor(core): settle the sources under the pinned ruff and mypy (closed)
- #16 build(build): add recipe.ps1 and the test suites it runs (closed)
- #17 docs(docs): add the README, licence, rules and manual (closed)
- #18 chore(core): add the AI environment, its hooks and the lab-ai-env skill (closed)
- #19 ci(ci): add the five lab workflows and the issue and pull request templates (closed)
- #20 docs(docs): name revision 4.0.0 and its exceptions in the README (closed)

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-08-05

The pipeline itself is unchanged. This release closes the three gaps that stood
between the repository and the project quality standard.

### Added

- **Release artefacts.** A `v*.*.*` tag now builds a `zeod` executable, a zip
  and a SHA-256 for each, published as a GitHub Release. Someone without Python
  can now run the detector.
- **Dependency install script** - `scripts/install_dependencies.py`, one command
  for the whole toolchain, with a lock resolved in a clean virtual environment.
- **Secret scanning.** `gitleaks` now runs over full history in CI, not only
  over the working tree via pre-commit.
- **Branch policy enforcement** - `scripts/enforce_branch_policy.py`, wired into
  CI so that a fourth branch fails the build.
- `SECURITY.md`, `CONTRIBUTING.md`, `docs/branching.md`.
- Helper scripts for the release build, quality metrics and profiling.
- `pyinstaller` as a `release` extra, `radon` as a `dev` extra.

### Changed

- Branch layout is now exactly `release`, `dev` and `test`; CI triggers follow.

[Unreleased]: https://github.com/IGORSVOLOHOVS/zero-annottation-od/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/IGORSVOLOHOVS/zero-annottation-od/releases/tag/v1.0.0
