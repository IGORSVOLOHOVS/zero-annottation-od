"""Points 6 and 13: build the downloadable artefact with one command.

    python scripts/build_release_artifact.py --version v1.0.0

Produces, under release/:
  * a single-file executable (.exe on Windows) built with PyInstaller
  * a .zip containing that executable, the README and the LICENSE
  * a .sha256 next to each file, so a download can be verified

The zip is what someone who does not have Python downloads; the checksum is what
makes "I built this" checkable rather than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"
APP_NAME = "zeod"
ENTRY = ROOT / "src" / "zeod" / "cli.py"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_executable(version: str) -> Path:
    # Invoked as a module rather than as the `pyinstaller` console script: pip
    # often installs that script into a per-user Scripts directory that is not
    # on PATH, and then a perfectly good install looks like a missing one.
    probe = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n"
            "  python scripts/install_dependencies.py --dev\n"
            '  pip install -e ".[release]"'
        )
    print(f"PyInstaller {probe.stdout.strip()}")

    work = ROOT / "build" / "pyinstaller"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        APP_NAME,
        "--paths",
        str(ROOT / "src"),
        "--distpath",
        str(RELEASE),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        str(ENTRY),
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    suffix = ".exe" if platform.system() == "Windows" else ""
    built = RELEASE / f"{APP_NAME}{suffix}"
    if not built.is_file():
        raise SystemExit(f"expected {built} but PyInstaller did not produce it")
    return built


def make_zip(executable: Path, version: str) -> Path:
    tag = f"{APP_NAME}-{version}-{platform.system().lower()}-{platform.machine().lower()}"
    archive = RELEASE / f"{tag}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(executable, executable.name)
        for extra in ("README.md", "LICENSE", "CHANGELOG.md"):
            src = ROOT / extra
            if src.is_file():
                zf.write(src, extra)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v0.0.0-dev", help="version tag to stamp into the artefact name")
    parser.add_argument(
        "--skip-exe",
        action="store_true",
        help="package sources only; useful where PyInstaller cannot run",
    )
    args = parser.parse_args()

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)

    produced: list[Path] = []
    if args.skip_exe:
        print("skipping executable build (--skip-exe)")
    else:
        produced.append(build_executable(args.version))

    if produced:
        produced.append(make_zip(produced[0], args.version))

    for path in list(produced):
        checksum = RELEASE / f"{path.name}.sha256"
        checksum.write_text(f"{sha256_of(path)}  {path.name}\n", encoding="utf-8")
        produced.append(checksum)

    print("\nrelease/")
    for path in sorted(RELEASE.iterdir()):
        print(f"  {path.name:<48} {path.stat().st_size / 1024:>10.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
