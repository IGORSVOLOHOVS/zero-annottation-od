# Branching

Three branches exist. No others, ever.

| Branch | Holds | Protected |
| --- | --- | --- |
| `release` | What is published. Every commit is tagged `vX.Y.Z` and has a GitHub Release with a downloadable artefact | yes |
| `test` | Release candidate. Full CI plus manual checks before it moves to `release` | yes |
| `dev` | Day-to-day work. CI must be green | yes |

```
dev ──────► test ──────► release ──► tag vX.Y.Z ──► artefact
 ▲                                        │
 └──────────── hotfix merged back ────────┘
```

## Why only three

Long-lived feature branches are where work goes to be forgotten. The audit of
this account found repositories whose default branch was nearly empty while the
real code sat in a branch nobody had touched in a year — `Sandbox`,
`soal-coffee` and `WebSearchAI` among them. Three branches make that impossible:
if it is not in `dev`, it does not exist.

Short-lived work happens locally. Rebase onto `dev` and push `dev`; do not push
the local branch.

## Enforcement

`scripts/enforce_branch_policy.py` fails if a fourth branch exists, locally or
on the remote. It runs on every push and every Monday via
`.github/workflows/branch-policy.yml`.

```bash
python scripts/enforce_branch_policy.py --remote origin   # report
python scripts/enforce_branch_policy.py --delete-extra    # remove merged extras
```

`--delete-extra` refuses to delete a branch whose tip is not already contained
in `release`, `dev` or `test`. A branch holding unique commits is named and
kept, never silently dropped.

## Setting this up on an existing repository

1. **Look before deleting.** List every branch and what is only in it:
   ```bash
   git fetch --all
   for b in $(git branch -r --format='%(refname:short)'); do
     echo "$b: $(git log --oneline origin/dev.."$b" 2>/dev/null | wc -l) unique commits"
   done
   ```
2. Merge or cherry-pick anything worth keeping into `dev`.
3. Create the three branches; rename `main` or `master` to `release`.
4. Delete the rest: `git push origin --delete <branch>`.
5. Set `release` as the default branch and protect all three in repository
   settings: require CI to pass, require pull requests, forbid force-push.

Step 1 is not optional. Deleting a remote branch is not undoable from the GitHub
UI.

## Releasing

```bash
git checkout release && git merge --ff-only test
git tag -a v1.2.0 -m "v1.2.0"
git push origin release --tags
```

The tag triggers `.github/workflows/release.yml`, which runs the tests, builds
the executable and the zip, writes a checksum for each, and publishes a GitHub
Release. Update `CHANGELOG.md` before tagging — the release notes are taken from
it.
