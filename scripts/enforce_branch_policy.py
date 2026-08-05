"""Point 14: exactly three branches - release, dev, test. Nothing else.

    python scripts/enforce_branch_policy.py                 # report
    python scripts/enforce_branch_policy.py --remote origin # check the remote too
    python scripts/enforce_branch_policy.py --delete-extra  # actually remove them

Reporting is the default and deleting is opt-in on purpose: a stray branch is
often the only copy of something. --delete-extra refuses to touch a branch whose
tip is not already contained in release, dev or test, so nothing unique is lost
without being named first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

ALLOWED = ("release", "dev", "test")


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def local_branches() -> list[str]:
    return [
        b.strip() for b in git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines() if b.strip()
    ]


def remote_branches(remote: str) -> list[str]:
    out = git("ls-remote", "--heads", remote)
    return [line.split("refs/heads/", 1)[1].strip() for line in out.splitlines() if "refs/heads/" in line]


def is_merged_into_allowed(branch: str, existing: set[str]) -> bool:
    """True when this branch's tip is already an ancestor of an allowed branch."""
    for target in ALLOWED:
        if target not in existing or target == branch:
            continue
        proc = subprocess.run(["git", "merge-base", "--is-ancestor", branch, target], capture_output=True)
        if proc.returncode == 0:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remote", help="also check this remote, e.g. origin")
    parser.add_argument("--delete-extra", action="store_true", help="delete extra branches that are fully merged")
    args = parser.parse_args()

    local = local_branches()
    extra_local = [b for b in local if b not in ALLOWED]

    # A CI checkout creates exactly one local branch, so "these three exist" can
    # only be judged against the remote. When --remote is given it is the
    # authority for what exists; the local list is then informational.
    remote = remote_branches(args.remote) if args.remote else []
    authoritative = remote if args.remote else local
    missing = [b for b in ALLOWED if b not in authoritative]

    print("local branches:" if local else "local branches: none (detached HEAD)")
    for b in sorted(local):
        print(f"  {'OK   ' if b in ALLOWED else 'EXTRA'} {b}")

    extra_remote: list[str] = []
    if args.remote:
        extra_remote = [b for b in remote if b not in ALLOWED]
        print(f"\n{args.remote} branches:")
        for b in sorted(remote):
            print(f"  {'OK   ' if b in ALLOWED else 'EXTRA'} {b}")

    if missing:
        where = args.remote if args.remote else "locally"
        print(f"\nmissing required branches on {where}: {', '.join(missing)}")

    if args.delete_extra and extra_local:
        existing = set(local)
        print("\ndeleting extra local branches:")
        for b in extra_local:
            if is_merged_into_allowed(b, existing):
                git("branch", "-d", b)
                print(f"  deleted {b} (already merged)")
            else:
                print(f"  KEPT {b} - has commits not present in release/dev/test; merge or export it first")

    problems = extra_local or extra_remote or missing
    if problems:
        print("\nbranch policy NOT satisfied")
        if extra_remote:
            print(f"  extra on {args.remote}: {', '.join(extra_remote)}")
        return 1

    print("\nbranch policy satisfied: release, dev, test and nothing else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
