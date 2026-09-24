"""Browser tests for the optional 2.5D presentation view (issue #9369)."""

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
from robot_sf.render.presentation_scene import describe_view
from robot_sf.render.sim_state import VisualizableSimState
from robot_sf.render.threejs_viewer import build_threejs_scene
from scripts.validation import smoke_threejs_viewer_browser as smoke

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("playwright")


def _states() -> list[VisualizableSimState]:
    return [
        VisualizableSimState(
            timestep=step,
            robot_action=None,
            robot_pose=((1.0 + step, 2.0), 0.25),
            pedestrian_positions=np.array([[3.0, 4.0]]),
            ray_vecs=np.array([]),
            ped_actions=np.array([]),
        )
        for step in range(3)
    ]


def _map() -> MapDefinition:
    zone = ((-4.0, -2.0), (2.0, -2.0), (2.0, 2.0))
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[Obstacle([(-4.0, -2.0), (-1.0, -2.0), (-1.0, 1.0), (-4.0, 1.0)])],
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
    components = viewer_dir / "components" / "presentation_scene"
    components.mkdir(parents=True, exist_ok=True)
    with resources.as_file(
        package.joinpath("components/presentation_scene/presentation_scene.js")
    ) as asset_path:
        shutil.copyfile(asset_path, components / "presentation_scene.js")


def _require_chromium() -> str:
    """Skip cleanly when no Chromium is available; return the version otherwise."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            version = browser.version
            browser.close()
    except (PlaywrightError, OSError, TimeoutError) as exc:
        pytest.skip(f"chromium unavailable for presentation browser proof: {exc}")
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
    assert result.rendered, f"presentation export did not render: {result.reason}"


def _scene_with_view(preset: str) -> dict[str, Any]:
    scene = build_threejs_scene(
        PlaybackEpisode(episode_id=41, states=_states()), _map(), source="synthetic.jsonl"
    )
    scene["view"] = describe_view(scene, preset)  # type: ignore[arg-type]
    return scene


def test_isometric_preset_labels_and_renders(tmp_path: Path) -> None:
    """The isometric preset discloses symbolic height and renders the scene."""
    viewer_dir = tmp_path / "viewer"
    _write_export(viewer_dir, _scene_with_view("isometric"))

    hud = _hud_text(viewer_dir)

    assert "view=isometric" in hud
    assert "height=symbolic" in hud
    _assert_rendered(viewer_dir, tmp_path / "shot.png")


def test_top_down_default_labels_and_renders(tmp_path: Path) -> None:
    """The default top-down preset keeps the overhead camera and renders."""
    viewer_dir = tmp_path / "viewer"
    _write_export(viewer_dir, _scene_with_view("top-down"))

    hud = _hud_text(viewer_dir)

    assert "view=top-down" in hud
    assert "height=symbolic" not in hud
    _assert_rendered(viewer_dir, tmp_path / "shot.png")
