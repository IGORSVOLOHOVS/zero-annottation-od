# Changelog

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
