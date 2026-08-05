"""Run what CI runs, here, before anything is pushed.

Every check below costs a second or two on this machine. Sending them to a
hosted runner instead means waiting minutes for a result that was available
immediately, burning runner time, and - because a failed workflow emails whoever
watches the repository - interrupting someone with a mistake that never needed to
leave the laptop.

The checks are not a fixed list. They are read out of this repository's own
`.github/workflows`, so this script fails exactly when CI would fail and stays
quiet otherwise. A repository whose CI does not lint is not linted here either;
inventing a stricter gate than CI would block pushes for a failure that was never
going to happen.

    python scripts/check_before_push.py          # report
    python scripts/check_before_push.py --install-hook

`--install-hook` writes .git/hooks/pre-push, so `git push` refuses to send code
that CI would reject. Bypass a single push with `git push --no-verify` when the
failure is genuinely unrelated.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

HOOK = """#!/bin/sh
# Installed by scripts/check_before_push.py.
# Runs the same checks CI runs, so a red pipeline is caught here instead of on
# a hosted runner. Bypass once with: git push --no-verify
exec "{python}" "{script}" --quiet
"""

# Tools worth mirroring locally. Anything else in CI (docker build, deploy,
# secret scanning) either cannot run here or costs more than it saves.
TOOLS = ("ruff", "mypy", "pytest")

# `uv run pytest`, `poetry run mypy`, `python -m ruff check` all mean the same
# thing once the runner prefix is removed.
PREFIXES = (
    ["uv", "run"],
    ["poetry", "run"],
    ["python", "-m"],
    ["python3", "-m"],
    ["py", "-m"],
)
YAML_KEY = re.compile(r"^-?\s*(name|uses|with|env|if|id|shell|working-directory):")

# pytest's "no tests were collected" - a repository with an empty tests/
# directory is not a failing repository.
PYTEST_EMPTY = 5


def have(module: str) -> bool:
    return (
        subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, check=False).returncode
        == 0
    )


def _command_on(line: str) -> list[str] | None:
    """The tool invocation on one workflow line, if there is one."""
    text = line.strip()
    if not text or text.startswith("#") or YAML_KEY.match(text):
        return None
    text = re.sub(r"^-?\s*run:\s*\|?-?\s*", "", text)
    text = text.lstrip("- ").split("#", 1)[0]
    # Only the first command of a chain; the rest are separate concerns.
    text = re.split(r"&&|\|\||;", text)[0]
    tokens = text.split()
    for prefix in PREFIXES:
        if tokens[: len(prefix)] == prefix:
            tokens = tokens[len(prefix) :]
    return tokens if tokens and tokens[0] in TOOLS else None


def ci_commands() -> list[list[str]]:
    """Every lint, type-check and test command this repository's CI runs."""
    found: list[list[str]] = []
    if not WORKFLOWS.is_dir():
        return found
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        for line in workflow.read_text(encoding="utf-8", errors="replace").splitlines():
            command = _command_on(line)
            if command and command not in found:
                found.append(command)
    return found


def localise(command: list[str]) -> list[str] | None:
    """Adjust a CI command for this machine, or drop it if it cannot run.

    Coverage flags are a reporting concern of the runner: keeping them here
    would fail the push on a missing plugin rather than on a real defect.
    """
    if not have(command[0]):
        return None
    if command[0] == "pytest" and not have("pytest_cov"):
        command = [t for t in command if not t.startswith("--cov")]
    return command


def not_our_fault(output: str) -> str:
    """Why a test run failed for a reason a push cannot fix.

    The runner installs the package before testing; this machine usually has
    not. Blocking a push on that is a false alarm, and a hook that cries wolf
    gets bypassed with --no-verify until it stops being read at all.
    """
    # "1 error during collection" but "2 errors during collection" - matching
    # the singular alone silently missed every repository with more than one
    # failing test module, which is most of them.
    if "ModuleNotFoundError" in output and "during collection" in output:
        return "the package is not installed here; CI installs it before testing"
    if "Required test coverage" in output and "Total coverage: 0.00%" in output:
        return "coverage measured nothing because the package is not installed here"
    return ""


def run(command: list[str], *, quiet: bool) -> bool | None:
    """True passed, False failed, None could not be judged on this machine."""
    proc = subprocess.run(
        [sys.executable, "-m", *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # the return code is the result, not an error
    )
    label = " ".join(command)
    if proc.returncode == 0 or (command[0] == "pytest" and proc.returncode == PYTEST_EMPTY):
        print(f"  OK   {label}")
        return True

    output = proc.stdout + proc.stderr
    if command[0] == "pytest" and (reason := not_our_fault(output)):
        print(f"  skip {label} - {reason}")
        return None

    print(f"  FAIL {label}")
    tail = output.strip().splitlines()
    for line in tail[-6:] if quiet else tail[-12:]:
        print(f"       {line}")
    return False


def install_hook() -> int:
    hooks = ROOT / ".git" / "hooks"
    if not hooks.is_dir():
        print("not a git repository (no .git/hooks)")
        return 1
    target = hooks / "pre-push"
    target.write_text(
        HOOK.format(
            python=sys.executable.replace("\\", "/"),
            script=str(Path(__file__).resolve()).replace("\\", "/"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    target.chmod(0o755)
    print(f"installed {target}")
    print("`git push` will now run these checks first; --no-verify skips them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-hook", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="shorter output, for hook use")
    args = parser.parse_args()

    if args.install_hook:
        return install_hook()

    commands = ci_commands()
    if not commands:
        print("CI here runs no lint or test steps - nothing to check")
        return 0

    print("running the checks CI runs:")
    results: list[bool] = []
    for command in commands:
        local = localise(command)
        if local is None:
            print(f"  skip {' '.join(command)} (not installed here)")
            continue
        verdict = run(local, quiet=args.quiet)
        if verdict is not None:
            results.append(verdict)

    if not results:
        print("\nnone of CI's tools are installed here - nothing checked")
        return 0
    if all(results):
        print("\nall clear - safe to push")
        return 0
    print("\nCI would fail on this. Fix it here rather than on a runner.")
    return 1


if __name__ == "__main__":
    if shutil.which("git") is None:
        print("git not found")
    raise SystemExit(main())
