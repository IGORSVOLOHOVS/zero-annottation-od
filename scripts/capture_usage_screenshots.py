"""Point 11: regenerate the README screenshots instead of taking them by hand.

    python scripts/capture_usage_screenshots.py

Writes docs/screenshots/:
  application-window.png  - the real GUI, captured from its own window
  cli-output.png          - the real CLI output, rendered as a terminal frame

Two details that took a while to get right and are easy to lose:

  * The window is captured with PrintWindow(PW_RENDERFULLCONTENT), not by
    grabbing a screen region. A region grab picks up whatever else is on the
    desktop at that moment; PrintWindow asks the window to redraw itself into an
    offscreen buffer, so nothing else can bleed in.
  * The process declares itself DPI aware first. Without that, GetWindowRect
    returns virtualised coordinates on a scaled display and the capture is
    clipped on the right and bottom.

The GUI capture is Windows-only; the CLI capture works anywhere Playwright runs.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
PW_RENDERFULLCONTENT = 0x00000002
WINDOW_TITLE = "Text Analyser"


def child_env() -> dict[str, str]:
    """Environment for the child process, with the package importable."""
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def make_dpi_aware() -> None:
    """Ask Windows for real pixel coordinates rather than scaled ones."""
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    with contextlib.suppress(AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)


def find_window(title: str, timeout: float) -> int | None:
    """Handle of the first visible window whose title contains `title`."""
    import win32gui

    def matches() -> list[int]:
        hits: list[int] = []

        def visit(hwnd: int, _param: object) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if title.lower() in win32gui.GetWindowText(hwnd).lower():
                hits.append(hwnd)
            return True

        win32gui.EnumWindows(visit, None)
        return hits

    deadline = time.time() + timeout
    while time.time() < deadline:
        found = matches()
        if found:
            return found[0]
        time.sleep(0.5)
    return None


def grab_window(hwnd: int) -> object:
    """Capture a window's own pixels, ignoring anything drawn on top of it."""
    import win32con
    import win32gui
    import win32ui
    from PIL import Image, ImageGrab

    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.8)
    except Exception as exc:
        print(f"  note: could not raise the window ({exc}); capturing it where it is")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    window_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(window_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    printed = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
    if printed:
        info = bitmap.GetInfo()
        image = Image.frombuffer(
            "RGB",
            (info["bmWidth"], info["bmHeight"]),
            bitmap.GetBitmapBits(True),
            "raw",
            "BGRX",
            0,
            1,
        )
    else:
        print("  note: PrintWindow declined; falling back to a screen grab")
        image = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)

    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)
    return image


def capture_gui(out: Path) -> bool:
    make_dpi_aware()
    proc = subprocess.Popen(
        [sys.executable, "-m", "quality_template.cli", "--gui"],
        cwd=ROOT,
        env=child_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        hwnd = find_window(WINDOW_TITLE, timeout=25)
        if hwnd is None:
            return False
        time.sleep(2.0)
        image = grab_window(hwnd)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        print(f"  {out.relative_to(ROOT)}  {image.size[0]}x{image.size[1]}")
        return True
    finally:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            timeout=20,
            check=False,
        )


def capture_cli(out: Path) -> bool:
    """Render the CLI's real output as a terminal-styled image."""
    from playwright.sync_api import sync_playwright
    from rich.console import Console
    from rich.text import Text

    sample = ROOT / "docs" / "sample.txt"
    if not sample.is_file():
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(
            "Quality is not an act, it is a habit. A repository earns trust the "
            "same way: tests that run, documentation that matches the code, and "
            "a release anyone can download and verify.\n",
            encoding="utf-8",
        )

    result = subprocess.run(
        [sys.executable, "-m", "quality_template.cli", str(sample), "-n", "8"],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    body = (result.stdout or result.stderr or "(no output)").rstrip()
    body = f"$ quality-template docs/sample.txt -n 8\n{body}"

    console = Console(record=True, width=88, file=io.StringIO())
    console.print(Text.from_ansi(body))

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "cli.svg"
        console.save_svg(str(svg), title="quality-template")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.goto(svg.as_uri())
            page.locator("svg").screenshot(path=str(out))
            browser.close()
    print(f"  {out.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-gui", action="store_true")
    parser.add_argument("--skip-cli", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("capturing usage screenshots...")
    failed = False

    if not args.skip_cli:
        try:
            capture_cli(OUT / "cli-output.png")
        except Exception as exc:
            print(f"  CLI capture failed: {exc}")
            failed = True

    if not args.skip_gui:
        if sys.platform != "win32":
            print("  GUI capture skipped: implemented for Windows only")
        else:
            try:
                if not capture_gui(OUT / "application-window.png"):
                    print("  GUI capture failed: the window never appeared")
                    failed = True
            except Exception as exc:
                print(f"  GUI capture failed: {exc}")
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
