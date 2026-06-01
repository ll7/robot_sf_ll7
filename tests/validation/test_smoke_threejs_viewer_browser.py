"""Tests for the browser-backed Three.js viewer smoke command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from scripts.validation import smoke_threejs_viewer_browser as smoke

if TYPE_CHECKING:
    from pathlib import Path


def test_single_color_canvas_screenshot_fails_closed(tmp_path: Path) -> None:
    """A blank canvas screenshot should not count as a rendered viewer."""
    screenshot = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (17, 24, 39)).save(screenshot)

    result = smoke.classify_canvas_screenshot(screenshot, background_rgb=(17, 24, 39))

    assert not result.rendered
    assert result.distinct_colors == 1
    assert "blank" in result.reason


def test_non_background_pixels_pass_canvas_classifier(tmp_path: Path) -> None:
    """A screenshot with scene colors beyond the background should pass the pixel check."""
    screenshot = tmp_path / "scene.png"
    image = Image.new("RGB", (8, 8), (17, 24, 39))
    image.putpixel((4, 4), (34, 197, 94))
    image.save(screenshot)

    result = smoke.classify_canvas_screenshot(screenshot, background_rgb=(17, 24, 39))

    assert result.rendered
    assert result.non_background_pixels == 1


def test_missing_playwright_dependency_fails_closed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The CLI should fail with an actionable hint when Playwright is unavailable."""

    def missing_playwright():
        raise ModuleNotFoundError("No module named 'playwright'")

    viewer_dir = tmp_path / "viewer"
    viewer_dir.mkdir()
    (viewer_dir / "index.html").write_text("<div id='viewer'></div>", encoding="utf-8")
    (viewer_dir / "viewer.js").write_text("", encoding="utf-8")
    (viewer_dir / "scene.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(smoke, "_load_playwright", missing_playwright)

    exit_code = smoke.main(["--viewer-dir", str(viewer_dir)])

    captured = capsys.readouterr()
    assert exit_code == smoke.EXIT_BROWSER_UNAVAILABLE
    assert "Playwright" in captured.err
    assert "uv sync --extra browser" in captured.err
    assert "python -m playwright install chromium" in captured.err


def test_missing_chromium_browser_fails_closed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Installed Playwright without Chromium should report the browser setup command."""

    def missing_chromium(*_args, **_kwargs):
        raise RuntimeError("Executable doesn't exist. Please run playwright install chromium")

    viewer_dir = tmp_path / "viewer"
    viewer_dir.mkdir()
    (viewer_dir / "index.html").write_text("<div id='viewer'></div>", encoding="utf-8")
    (viewer_dir / "viewer.js").write_text("", encoding="utf-8")
    (viewer_dir / "scene.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(smoke, "run_browser_smoke", missing_chromium)

    exit_code = smoke.main(["--viewer-dir", str(viewer_dir)])

    captured = capsys.readouterr()
    assert exit_code == smoke.EXIT_BROWSER_UNAVAILABLE
    assert "Chromium" in captured.err
    assert "python -m playwright install chromium" in captured.err


def test_main_success_accepts_background_override(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The CLI should allow non-default viewer backgrounds and return success."""

    def write_scene_screenshot(_viewer_dir: Path, screenshot_path: Path, **_kwargs) -> None:
        image = Image.new("RGB", (8, 8), (1, 2, 3))
        image.putpixel((4, 4), (34, 197, 94))
        image.save(screenshot_path)

    viewer_dir = tmp_path / "viewer"
    viewer_dir.mkdir()
    (viewer_dir / "index.html").write_text("<div id='viewer'></div>", encoding="utf-8")
    (viewer_dir / "viewer.js").write_text("", encoding="utf-8")
    (viewer_dir / "scene.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(smoke, "run_browser_smoke", write_scene_screenshot)

    exit_code = smoke.main(
        [
            "--viewer-dir",
            str(viewer_dir),
            "--screenshot",
            str(tmp_path / "scene.png"),
            "--background-rgb",
            "1,2,3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == smoke.EXIT_OK
    assert "non-background pixels" in captured.out


def test_corrupt_screenshot_returns_render_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Unreadable screenshots should fail closed without a traceback."""

    def write_corrupt_screenshot(_viewer_dir: Path, screenshot_path: Path, **_kwargs) -> None:
        screenshot_path.write_text("not a png", encoding="utf-8")

    viewer_dir = tmp_path / "viewer"
    viewer_dir.mkdir()
    (viewer_dir / "index.html").write_text("<div id='viewer'></div>", encoding="utf-8")
    (viewer_dir / "viewer.js").write_text("", encoding="utf-8")
    (viewer_dir / "scene.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(smoke, "run_browser_smoke", write_corrupt_screenshot)

    exit_code = smoke.main(
        ["--viewer-dir", str(viewer_dir), "--screenshot", str(tmp_path / "bad.png")]
    )

    captured = capsys.readouterr()
    assert exit_code == smoke.EXIT_RENDER_FAILED
    assert "could not read screenshot" in captured.err


def test_transitive_playwright_import_error_fails_closed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Broken Playwright installs should receive the same setup hint as missing Playwright."""

    def missing_transitive_dependency():
        raise ModuleNotFoundError("No module named 'pyee'")

    viewer_dir = tmp_path / "viewer"
    viewer_dir.mkdir()
    (viewer_dir / "index.html").write_text("<div id='viewer'></div>", encoding="utf-8")
    (viewer_dir / "viewer.js").write_text("", encoding="utf-8")
    (viewer_dir / "scene.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(smoke, "_load_playwright", missing_transitive_dependency)

    exit_code = smoke.main(["--viewer-dir", str(viewer_dir)])

    captured = capsys.readouterr()
    assert exit_code == smoke.EXIT_BROWSER_UNAVAILABLE
    assert "Playwright" in captured.err
