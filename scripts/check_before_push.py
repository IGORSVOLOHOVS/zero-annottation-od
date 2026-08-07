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

# pytest exits 5 when it collected no tests at all. This used to be treated as a
# pass here, on the reasoning that an empty tests/ directory is not a failure.
# That reasoning is wrong for the one job this script has: the hosted runner
# gets the same 5, fails the step, and sends the mail. A rename that orphans a
# test file, or a `-k` filter that matches nothing, therefore passed here and
# went red there - which is exactly the sequence this script exists to prevent.
#
# It is now a failure, with an explanation, because the check has to agree with
# CI even when CI is being pedantic.
PYTEST_NO_TESTS = 5


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


def read_workflows() -> str:
    """Every workflow file, concatenated. Read once and passed where needed."""
    if not WORKFLOWS.is_dir():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sorted(WORKFLOWS.glob("*.y*ml"))
    )


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


INSTALL_STEPS = (
    "pip install -e",
    "pip install .",
    # A workflow that installs a requirements file is installing the project's
    # dependencies just as surely as one that installs the package. Leaving this
    # out made the hook report "CI would fail" on three repositories whose only
    # problem was a dependency this machine does not have and the runner does.
    "pip install -r",
    "uv sync",
    "poetry install",
    "setup.py develop",
    "setup.py install",
    "pip install --editable",
)


def ci_installs_the_package(text: str) -> bool:
    """Does any workflow install this project before testing it?

    The answer decides whether a local ModuleNotFoundError is noise or a
    warning. Where CI runs `pip install -e .` the package is importable there
    and not here, so blocking a push would be a false alarm. Where CI installs
    nothing, the tests import the project straight from the checkout - exactly
    as they just failed to do here - and the runner is about to hit the same
    wall.
    """
    return any(step in text for step in INSTALL_STEPS)


def ci_install_command(text: str) -> str | None:
    """The command CI uses to install this project, so it can be suggested.

    Telling someone a check did not run is only half an answer; the other half
    is the one line that makes it run. Returns the first install step found in
    the workflows, or None when the project needs no installing.
    """
    for step in INSTALL_STEPS:
        if step in text:
            return "uv sync --all-extras" if step == "uv sync" else step
    return None


def not_our_fault(output: str, workflows_text: str) -> str:
    """Why a test run failed for a reason a push cannot fix.

    A hook that cries wolf gets bypassed with --no-verify until it stops being
    read at all; a hook that waves failures through is worse, because it says
    "safe to push" and the runner then says otherwise.
    """
    if not ci_installs_the_package(workflows_text):
        return ""
    # "1 error during collection" but "2 errors during collection" - matching
    # the singular alone silently missed every repository with more than one
    # failing test module, which is most of them.
    if "ModuleNotFoundError" in output and "during collection" in output:
        return "the package is not installed here; CI installs it before testing"
    # A library refusing to work because an optional extra is missing is the
    # same situation wearing different words: python-telegram-bot raises
    # RuntimeError rather than ImportError when [rate-limiter] is absent, and
    # the runner has it because requirements.txt pins it.
    if "must be installed via" in output and "pip install" in output:
        return "an optional extra is missing here; CI installs it before testing"
    if "Required test coverage" in output and "Total coverage: 0.00%" in output:
        return "coverage measured nothing because the package is not installed here"
    return ""


def as_ci_invokes_it(command: list[str]) -> list[str]:
    """The command spelled the way a CI step spells it.

    This is not cosmetic. `python -m pytest` puts the working directory on
    sys.path; the `pytest` console script does not. A repository whose code
    sits at the top level therefore imports fine under `-m` and fails with
    ModuleNotFoundError under the bare script - which is what CI runs. Checking
    the easier of the two is worse than not checking at all, because it reports
    success and the runner then reports failure.

    The console script beside this interpreter is preferred over one merely on
    PATH, so a machine with several Pythons still tests the one in use. `-P`
    (3.11+) reproduces the same import behaviour when no script exists.
    """
    tool = command[0]
    scripts = Path(sys.executable).parent
    for candidate in (
        scripts / "Scripts" / f"{tool}.exe",
        scripts / f"{tool}.exe",
        scripts / "bin" / tool,
        scripts / tool,
    ):
        if candidate.is_file():
            return [str(candidate), *command[1:]]

    found = shutil.which(tool)
    if found:
        return [found, *command[1:]]
    # `-P` is what makes this fallback as strict as the console script, and it
    # arrived in 3.11. Asked as a capability rather than as a version number on
    # purpose: this file is copied into repositories with different
    # requires-python values, and a literal `sys.version_info >= (3, 11)` is
    # reported as an outdated version block by every one of them that requires
    # 3.11 or later. sys.flags.safe_path is the flag `-P` sets, so its presence
    # is the same question asked in a way that travels.
    safe_path = ["-P"] if hasattr(sys.flags, "safe_path") else []
    return [sys.executable, *safe_path, "-m", *command]


def run(command: list[str], *, quiet: bool, workflows_text: str = "") -> bool | None:
    """True passed, False failed, None could not be judged on this machine."""
    proc = subprocess.run(
        as_ci_invokes_it(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # the return code is the result, not an error
    )
    label = " ".join(command)
    if proc.returncode == 0:
        print(f"  OK   {label}")
        return True

    output = proc.stdout + proc.stderr
    if command[0] == "pytest" and proc.returncode == PYTEST_NO_TESTS:
        print(f"  FAIL {label}")
        print("       pytest collected no tests and exited 5. The runner will do")
        print("       the same and fail the job. Usually a test file was renamed")
        print("       or moved out of testpaths, or a filter matched nothing.")
        return False

    if command[0] == "pytest" and (reason := not_our_fault(output, workflows_text)):
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


def verdict_of(results: list[bool], skipped: list[str], workflows_text: str) -> int:
    """Say what was checked, what was not, and whether the push is safe.

    The distinction matters more than it looks. Reporting "all clear" when the
    tests were skipped is a lie by omission, and the skipped check is almost
    always the tests - so the runner becomes the first place they execute, which
    is the trip this script exists to avoid.
    """
    if not results:
        print("\nNOT CHECKED: none of CI's checks could run here. Nothing was verified.")
        return 0
    if not all(results):
        print("\nCI would fail on this. Fix it here rather than on a runner.")
        return 1
    if not skipped:
        print("\nall clear - safe to push")
        return 0

    print(f"\n{len(results)} check(s) clean, but {len(skipped)} did NOT run here:")
    for item in skipped:
        print(f"  - {item}")
    if (install := ci_install_command(workflows_text)) is not None:
        print(f"\nRun `{install}` and try again to check them locally.")
    print("Pushing is not blocked - CI installs what is missing and will run them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-hook", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="shorter output, for hook use")
    args = parser.parse_args()

    if args.install_hook:
        return install_hook()

    workflows_text = read_workflows()
    commands = ci_commands()
    if not commands:
        print("NOT CHECKED: CI here runs no lint or test step this script can mirror.")
        print("             Nothing was verified. Pushing is not blocked, but nothing")
        print("             says the code is good either.")
        return 0

    print("running the checks CI runs:")
    results: list[bool] = []
    skipped: list[str] = []
    for command in commands:
        local = localise(command)
        if local is None:
            print(f"  skip {' '.join(command)} (not installed here)")
            skipped.append(" ".join(command))
            continue
        verdict = run(local, quiet=args.quiet, workflows_text=workflows_text)
        if verdict is None:
            skipped.append(" ".join(command))
        else:
            results.append(verdict)

    return verdict_of(results, skipped, workflows_text)


if __name__ == "__main__":
    if shutil.which("git") is None:
        print("git not found")
    raise SystemExit(main())
