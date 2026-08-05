"""Point 7: profiling, so "it feels slow" becomes a ranked list of functions.

    python scripts/profile_application.py
    python scripts/profile_application.py --paragraphs 2000 --sort tottime
    python scripts/profile_application.py --save profile.out   # open with snakeviz

Benchmarks (benchmarks/) answer "how fast, and did it regress".
This answers "where does the time actually go".
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quality_template.core import analyse_text, top_words

PARAGRAPH = (
    "Quality is not an act, it is a habit. A repository earns trust the same "
    "way: tests that run, documentation that matches the code, and a release "
    "anyone can download and verify. "
)


def workload(paragraphs: int, repeats: int) -> None:
    text = PARAGRAPH * paragraphs
    for _ in range(repeats):
        stats = analyse_text(text)
        top_words(stats, 20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paragraphs", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rows", type=int, default=15, help="how many lines to print")
    parser.add_argument("--sort", default="cumtime", choices=["cumtime", "tottime", "ncalls"])
    parser.add_argument("--save", type=Path, help="write raw pstats for snakeviz")
    args = parser.parse_args()

    print(f"profiling {args.paragraphs} paragraphs x {args.repeats} repeats\n")
    profiler = cProfile.Profile()
    profiler.enable()
    workload(args.paragraphs, args.repeats)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.rows)
    print(buffer.getvalue())

    if args.save:
        stats.dump_stats(str(args.save))
        print(f"raw profile written to {args.save}\n  view it with:  snakeviz {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
