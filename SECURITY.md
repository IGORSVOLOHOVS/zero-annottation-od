# Security

## Reporting a vulnerability

Email **igorsvolohovs@gmail.com** with a description and, if possible, a way to
reproduce it. Please do not open a public issue for a security problem.

Expect an acknowledgement within a few days.

## What this project does to protect itself

| Control | Where it runs |
| --- | --- |
| Secret scanning before commit | `gitleaks` in `.pre-commit-config.yaml` |
| Secret scanning over full history | `gitleaks` job in `.github/workflows/ci.yml`, with `fetch-depth: 0` |
| Private key detection | `detect-private-key` pre-commit hook |
| Static security linting | `ruff` rule set `S` (bandit rules) |
| Pinned dependencies | `requirements-dev.lock` |
| Artefact integrity | `.sha256` published beside every release file |

## Handling credentials

Never write a credential into a source file. Read it from the environment:

```python
import os

token = os.environ["SERVICE_TOKEN"]
```

Commit a `.env.example` listing the variable **names** with empty values. The
real `.env` is in `.gitignore` and must never be committed.

## If a credential does get committed

A commit that removes it is not enough — it stays in history and remains valid.

1. **Revoke it first.** Rotate the key at the provider. This is the only step
   that actually stops the leak.
2. Replace the literal with an environment lookup and commit that.
3. Only then consider rewriting history with `git filter-repo`, and understand
   that it needs a force-push and breaks every existing clone.

Order matters: rewriting history without revoking leaves a working credential in
every clone and fork that already exists.
