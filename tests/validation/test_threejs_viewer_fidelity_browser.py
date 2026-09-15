"""Browser fidelity tests for the Three.js replay cursor and labels (#9368)."""

from __future__ import annotations

import json
import shutil
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.render.jsonl_playback import PlaybackEpisode
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.threejs_viewer import build_threejs_scene
from scripts.validation import smoke_threejs_viewer_browser as smoke

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("playwright")


def _state(
    timestep: int,
    *,
    time_s: float | None = None,
    ped_heading: float | None = 0.5,
    ped_radius: float | None = 0.31,
) -> VisualizableSimState:
    state = VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((1.0 + timestep, 2.0), 0.25),
        pedestrian_positions=np.array([[3.0, 4.0]]),
        ray_vecs=np.array([]),
        ped_actions=np.array([]),
    )
    if time_s is not None:
        state.time_s = time_s  # type: ignore[attr-defined]
    state.pedestrian_ids = ["ped-a"]  # type: ignore[attr-defined]
    if ped_heading is not None:
        state.pedestrian_headings = np.array([ped_heading])  # type: ignore[attr-defined]
    if ped_radius is not None:
        state.pedestrian_radii = np.array([ped_radius])  # type: ignore[attr-defined]
    return state


def _map() -> MapDefinition:
    zone = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[Obstacle([(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)])],
        robot_spawn_zones=[zone],
        robot_goal_zones=[zone],
        ped_spawn_zones=[zone],
        bounds=[
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 10.0, 8.0, 8.0),
            (0.0, 0.0, 0.0, 8.0),
            (10.0, 10.0, 0.0, 8.0),
        ],
        robot_routes=[],
        ped_goal_zones=[zone],
        ped_crowded_zones=[],
        ped_routes=[],
    )


def _write_export(viewer_dir: Path, scene: dict[str, Any]) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    (viewer_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    package = resources.files("robot_sf.render.web_assets")
    for asset in ("index.html", "viewer.js"):
        with resources.as_file(package.joinpath(asset)) as asset_path:
            shutil.copyfile(asset_path, viewer_dir / asset)


def _require_chromium() -> str:
    """Skip cleanly when no Chromium is available; return the browser version otherwise."""
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            version = browser.version
            browser.close()
    except Exception as exc:
        pytest.skip(f"chromium unavailable for fidelity browser proof: {exc}")
    return version


def _hud_text(viewer_dir: Path, timeout_ms: int = 30_000) -> str:
    """Serve one export, wait for first render, and return the HUD text."""
    from playwright.sync_api import sync_playwright

    browser_version = _require_chromium()
    server = smoke._start_viewer_server(viewer_dir)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 640, "height": 480})
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                page.wait_for_function(
                    "() => document.documentElement.dataset.traceViewerRendered === 'true'",
                    timeout=timeout_ms,
                )
                hud = page.locator("#hud").text_content() or ""
                return f"[chromium {browser_version}] {hud}"
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def _assert_rendered(viewer_dir: Path, screenshot_path: Path) -> None:
    smoke.run_browser_smoke(viewer_dir, screenshot_path, width=640, height=480)
    result = smoke.classify_canvas_screenshot(screenshot_path)
    assert result.rendered, f"fidelity export did not render: {result.reason}"


def test_source_time_cursor_labels_and_renders(tmp_path: Path) -> None:
    """Irregular recorded times drive source-time playback with faithful HUD labels."""
    states = [_state(0, time_s=0.0), _state(1, time_s=0.1), _state(2, time_s=5.0)]
    scene = build_threejs_scene(
        PlaybackEpisode(episode_id=21, states=states), _map(), source="synthetic.jsonl"
    )
    viewer_dir = tmp_path / "viewer"
    _write_export(viewer_dir, scene)

    hud = _hud_text(viewer_dir)

    assert "timing=source-time" in hud
    # Playback runs during load, so the visible frame is whichever source time
    # the cursor reached; it must be one of the three recorded times.
    assert any(f"t={stamp}s" in hud for stamp in ("0.00", "0.10", "5.00"))
    assert "identity=stable" in hud
    assert "geometry=recorded" in hud
    _assert_rendered(viewer_dir, tmp_path / "shot.png")


def test_legacy_timing_label_without_recorded_time(tmp_path: Path) -> None:
    """Missing time fields fall back to frame-index playback and say so."""
    states = [_state(0), _state(1)]
    scene = build_threejs_scene(
        PlaybackEpisode(episode_id=22, states=states), _map(), source="synthetic.jsonl"
    )
    # Simulate a legacy payload without source times.
    for frame in scene["frames"]:
        del frame["time_s"]
    viewer_dir = tmp_path / "viewer"
    _write_export(viewer_dir, scene)

    hud = _hud_text(viewer_dir)

    assert "timing=frame-index (legacy" in hud
    _assert_rendered(viewer_dir, tmp_path / "shot.png")
