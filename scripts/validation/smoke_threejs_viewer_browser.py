"""Browser pixel smoke for static Three.js viewer exports.

The command intentionally treats missing browser automation as a validation failure with an
actionable dependency hint. It is a diagnostic smoke gate, not benchmark evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_OK = 0
EXIT_RENDER_FAILED = 1
EXIT_BROWSER_UNAVAILABLE = 2
EXIT_INPUT_ERROR = 3

DEFAULT_BACKGROUND_RGB = (17, 24, 39)
REQUIRED_VIEWER_FILES = ("index.html", "viewer.js", "scene.json")


@dataclass(frozen=True)
class CanvasSmokeResult:
    """Pixel-classification result for a captured viewer canvas."""

    rendered: bool
    distinct_colors: int
    non_background_pixels: int
    reason: str


def classify_canvas_screenshot(
    screenshot_path: str | Path,
    *,
    background_rgb: tuple[int, int, int] = DEFAULT_BACKGROUND_RGB,
    min_non_background_pixels: int = 1,
    min_distinct_colors: int = 2,
) -> CanvasSmokeResult:
    """Classify whether a canvas screenshot contains rendered non-background pixels."""
    image = Image.open(screenshot_path).convert("RGB")
    pixels = list(image.getdata())
    distinct_colors = len(set(pixels))
    non_background_pixels = sum(pixel != background_rgb for pixel in pixels)

    if distinct_colors < min_distinct_colors:
        return CanvasSmokeResult(
            rendered=False,
            distinct_colors=distinct_colors,
            non_background_pixels=non_background_pixels,
            reason="canvas screenshot appears blank: only background color was captured",
        )
    if non_background_pixels < min_non_background_pixels:
        return CanvasSmokeResult(
            rendered=False,
            distinct_colors=distinct_colors,
            non_background_pixels=non_background_pixels,
            reason=(
                "canvas screenshot lacks enough non-background pixels "
                f"({non_background_pixels} < {min_non_background_pixels})"
            ),
        )
    return CanvasSmokeResult(
        rendered=True,
        distinct_colors=distinct_colors,
        non_background_pixels=non_background_pixels,
        reason="canvas screenshot contains non-background scene pixels",
    )


def run_browser_smoke(
    viewer_dir: str | Path,
    screenshot_path: str | Path,
    *,
    width: int = 960,
    height: int = 720,
    timeout_ms: int = 10_000,
) -> None:
    """Open a static viewer in Chromium and capture its canvas screenshot."""
    viewer_dir = Path(viewer_dir)
    screenshot_path = Path(screenshot_path)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    sync_playwright = _load_playwright()
    server = _start_viewer_server(viewer_dir)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                canvas = page.locator("canvas").first
                canvas.wait_for(state="visible", timeout=timeout_ms)
                page.wait_for_function(
                    "() => { const canvas = document.querySelector('canvas');"
                    " return canvas && canvas.width > 0 && canvas.height > 0; }",
                    timeout=timeout_ms,
                )
                canvas.screenshot(path=str(screenshot_path))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def _load_playwright():
    """Load Playwright lazily so ordinary validation imports stay lightweight."""
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _start_viewer_server(viewer_dir: Path) -> http.server.ThreadingHTTPServer:
    """Serve a viewer directory from an ephemeral localhost port."""
    handler = partial(_QuietSimpleHTTPRequestHandler, directory=str(viewer_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class _QuietSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler variant that keeps validation output concise."""

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress per-request logs from the temporary validation server."""


def _validate_viewer_dir(viewer_dir: Path) -> list[str]:
    """Return missing required viewer files."""
    return [name for name in REQUIRED_VIEWER_FILES if not (viewer_dir / name).is_file()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer-dir",
        required=True,
        type=Path,
        help="Directory containing index.html, viewer.js, and scene.json.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("output/validation/threejs_viewer_browser_smoke.png"),
        help="Path for the captured canvas screenshot.",
    )
    parser.add_argument("--width", type=int, default=960, help="Browser viewport width.")
    parser.add_argument("--height", type=int, default=720, help="Browser viewport height.")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10_000,
        help="Browser navigation and canvas wait timeout in milliseconds.",
    )
    parser.add_argument(
        "--min-non-background-pixels",
        type=int,
        default=1,
        help="Minimum non-background pixels required for a rendered canvas.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the browser pixel smoke command."""
    args = parse_args(argv)
    missing_files = _validate_viewer_dir(args.viewer_dir)
    if missing_files:
        print(
            f"Viewer directory {args.viewer_dir} is missing required files: "
            f"{', '.join(missing_files)}",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR

    try:
        run_browser_smoke(
            args.viewer_dir,
            args.screenshot,
            width=args.width,
            height=args.height,
            timeout_ms=args.timeout_ms,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "playwright" and "playwright" not in str(exc):
            raise
        print(
            "Playwright is required for the Three.js browser pixel smoke. "
            "Install it with `uv sync --extra browser`, then run "
            "`uv run python -m playwright install chromium` before this validation command.",
            file=sys.stderr,
        )
        return EXIT_BROWSER_UNAVAILABLE
    except Exception as exc:  # pragma: no cover - browser/runtime dependent
        if _is_browser_dependency_error(exc):
            print(
                "Chromium is required for the Three.js browser pixel smoke. "
                "Run `uv run python -m playwright install chromium` before this validation "
                "command.",
                file=sys.stderr,
            )
            return EXIT_BROWSER_UNAVAILABLE
        print(f"Three.js browser smoke failed while driving Chromium: {exc}", file=sys.stderr)
        return EXIT_RENDER_FAILED

    with contextlib.suppress(FileNotFoundError):
        result = classify_canvas_screenshot(
            args.screenshot,
            min_non_background_pixels=args.min_non_background_pixels,
        )
        if result.rendered:
            print(
                "Three.js browser smoke passed: "
                f"{result.non_background_pixels} non-background pixels, "
                f"{result.distinct_colors} distinct colors."
            )
            return EXIT_OK
        print(f"Three.js browser smoke failed: {result.reason}", file=sys.stderr)
        return EXIT_RENDER_FAILED

    print(
        f"Three.js browser smoke failed: screenshot not written at {args.screenshot}",
        file=sys.stderr,
    )
    return EXIT_RENDER_FAILED


def _is_browser_dependency_error(exc: Exception) -> bool:
    """Return true for Playwright errors caused by a missing browser binary."""
    message = str(exc)
    return "Executable doesn't exist" in message or "playwright install chromium" in message


if __name__ == "__main__":
    raise SystemExit(main())
