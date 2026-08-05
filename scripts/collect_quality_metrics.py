"""Point 8: turn ISO/IEC 25010 from an essay into measured numbers.

ISO/IEC 25010 defines eight product quality characteristics. Most of them cannot
be measured by a script - Usability and Security need judgement. What a script
CAN do is gather the objective evidence that a written assessment then cites, so
the assessment stops being an opinion with no numbers behind it.

    python scripts/collect_quality_metrics.py
    python scripts/collect_quality_metrics.py --output quality-report.json --fail-under 70

Measured here:
  Functional suitability  - test pass rate
  Reliability             - branch coverage
  Maintainability         - cyclomatic complexity, maintainability index, lint debt
  Portability             - declared Python versions and OS matrix in CI
Assessed by hand in docs/quality-iso25010.md, citing this output:
  Performance efficiency, Compatibility, Usability, Security
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# One measured characteristic: a label, a score, and whatever evidence backs it.
# The evidence differs per characteristic, so the values are deliberately loose -
# named alias rather than a bare `dict`, which `mypy --strict` rejects.
Metric = dict[str, object]


def score_of(metric: Metric) -> float:
    """The 0-100 score out of a metric, narrowed from the loose value type."""
    value = metric.get("score", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def label_of(metric: Metric) -> str:
    return str(metric.get("characteristic", "?"))


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(*cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def measure_tests() -> Metric:
    # `-o addopts=` clears the -q already set in pyproject.toml. Without this the
    # two combine into -qq, which suppresses the "N passed" summary line that
    # this function parses, and every run silently reports zero tests.
    proc = run(sys.executable, "-m", "pytest", "-o", "addopts=", "--tb=no", "-q")
    text = proc.stdout + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", text)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", text)) else 0
    total = passed + failed
    return {
        "characteristic": "Functional suitability",
        "tests_passed": passed,
        "tests_failed": failed,
        "pass_rate_percent": round(100 * passed / total, 1) if total else 0.0,
        "score": round(100 * passed / total, 1) if total else 0.0,
    }


def measure_coverage() -> Metric:
    proc = run(
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--cov",
        "--cov-report=json:.coverage.json",
        "--tb=no",
        "-q",
    )
    path = ROOT / ".coverage.json"
    percent = 0.0
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        percent = round(data.get("totals", {}).get("percent_covered", 0.0), 1)
        path.unlink(missing_ok=True)
    elif m := re.search(r"TOTAL.*?(\d+)%", proc.stdout):
        percent = float(m.group(1))
    return {
        "characteristic": "Reliability",
        "branch_coverage_percent": percent,
        "score": percent,
    }


def measure_maintainability() -> Metric:
    """Cyclomatic complexity and maintainability index, via radon."""
    cc = run(sys.executable, "-m", "radon", "cc", str(SRC), "-s", "-j")
    mi = run(sys.executable, "-m", "radon", "mi", str(SRC), "-j")
    lint = run(sys.executable, "-m", "ruff", "check", ".", "--output-format=json")

    complexities: list[int] = []
    worst = None
    if cc.returncode == 0 and cc.stdout.strip():
        for blocks in json.loads(cc.stdout).values():
            for b in blocks if isinstance(blocks, list) else []:
                complexities.append(b["complexity"])
                if worst is None or b["complexity"] > worst["complexity"]:
                    worst = {"name": b["name"], "complexity": b["complexity"]}

    mi_scores = []
    if mi.returncode == 0 and mi.stdout.strip():
        mi_scores = [
            v["mi"] for v in json.loads(mi.stdout).values() if isinstance(v, dict) and "mi" in v
        ]

    violations = 0
    if lint.stdout.strip():
        try:
            violations = len(json.loads(lint.stdout))
        except json.JSONDecodeError:
            violations = 0

    avg_cc = sum(complexities) / len(complexities) if complexities else 0.0
    avg_mi = sum(mi_scores) / len(mi_scores) if mi_scores else 0.0
    # Maintainability index is already 0-100; penalise outstanding lint findings.
    score = max(0.0, min(100.0, avg_mi - min(violations, 20) * 2))
    return {
        "characteristic": "Maintainability",
        "average_cyclomatic_complexity": round(avg_cc, 2),
        "most_complex": worst,
        "average_maintainability_index": round(avg_mi, 1),
        "outstanding_lint_findings": violations,
        "score": round(score, 1),
    }


def measure_portability() -> Metric:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires = (
        m.group(1) if (m := re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject)) else "?"
    )
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""
    operating_systems = sorted(set(re.findall(r"(ubuntu|windows|macos)-latest", ci_text)))
    versions = sorted(set(re.findall(r'"(3\.\d+)"', ci_text)))
    # Two OSes and two interpreter versions is the bar for "portable enough".
    score = min(100.0, 25.0 * len(operating_systems) + 25.0 * len(versions))
    return {
        "characteristic": "Portability",
        "requires_python": requires,
        "ci_operating_systems": operating_systems,
        "ci_python_versions": versions,
        "score": round(score, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if the overall score is below this",
    )
    args = parser.parse_args()

    print("collecting ISO/IEC 25010 evidence...\n", flush=True)
    measured = [
        measure_tests(),
        measure_coverage(),
        measure_maintainability(),
        measure_portability(),
    ]
    overall = round(sum(score_of(m) for m in measured) / len(measured), 1)

    report = {
        "standard": "ISO/IEC 25010:2011 product quality model",
        "measured_characteristics": measured,
        "assessed_by_hand": [
            "Performance efficiency",
            "Compatibility",
            "Usability",
            "Security",
        ],
        "overall_measured_score": overall,
        "note": "Four of the eight characteristics need human judgement; "
        "see docs/quality-iso25010.md, which cites these numbers.",
    }

    for m in measured:
        print(f"  {label_of(m):<24} {score_of(m):>6.1f}")
        for k, v in m.items():
            if k not in ("characteristic", "score"):
                print(f"      {k}: {v}")
    print(f"\n  {'OVERALL (measured)':<24} {overall:>6.1f}")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.output}")

    if args.fail_under is not None and overall < args.fail_under:
        print(f"\nFAIL: {overall} is below the required {args.fail_under}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
