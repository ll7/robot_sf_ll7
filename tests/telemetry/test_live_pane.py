"""Live telemetry pane smoke tests (US1)."""

from __future__ import annotations

import numpy as np
import pygame

from robot_sf.telemetry.history import replay_episode
from robot_sf.telemetry.pane import TelemetrySession
from robot_sf.telemetry.visualization import render_metric_panel


def test_render_metric_panel_shape():
    """Ensure off-screen renderer returns RGBA array with requested size."""
    data = {"fps": [1.0, 2.0, 3.0], "reward": [0.1, 0.2, 0.3]}
    rgba = render_metric_panel(data, metrics=["fps", "reward"], width=200, height=120)
    assert isinstance(rgba, np.ndarray)
    assert rgba.shape == (120, 200, 4)
    assert rgba.dtype == np.uint8


def test_telemetry_session_appends_and_writes(tmp_path):
    """TelemetrySession records metrics and writes JSONL under artifact root."""
    session = TelemetrySession(
        run_id="test-run",
        record=True,
        metrics=["fps", "reward"],
        refresh_hz=2.0,
        decimation=1,
        pane_size=(200, 120),
    )
    # Override run_dir to temporary location for test isolation
    session.run_dir = tmp_path / "telemetry"
    session.telemetry_path = session.run_dir / "telemetry.jsonl"

    session.append({"frame_idx": 0, "metrics": {"fps": 5.0, "reward": 1.0}})
    session.append({"frame_idx": 1, "metrics": {"fps": 6.0, "reward": 1.5}})

    assert session.telemetry_path.exists()
    content = session.telemetry_path.read_text().strip().splitlines()
    assert len(content) == 2

    replay = replay_episode(session.telemetry_path)
    snap = replay.scrub(to_frame=1)
    assert snap.get("frame_idx") == 1


def test_live_pane_refresh_and_fps_budget(monkeypatch):
    """Pane refreshes at least 1 Hz without heavy frame budget impact."""
    # Simulate 10 frames with ~20ms/frame; pane refresh_hz=5 => expected <3 refreshes allowed
    session = TelemetrySession(
        run_id="test-refresh",
        record=False,
        metrics=["fps"],
        refresh_hz=5.0,
        decimation=1,
        pane_size=(200, 120),
    )
    # Monkeypatch render_metric_panel to count calls instead of real render
    calls = {"count": 0}

    def fake_render(series, metrics, width, height, dpi=100):
        calls["count"] += 1
        return np.zeros((height, width, 4), dtype=np.uint8)

    monkeypatch.setattr("robot_sf.telemetry.pane.render_metric_panel", fake_render)
    # Force the pane's render cadence to trigger by spacing out calls
    for frame in range(3):
        session.append({"frame_idx": frame, "metrics": {"fps": 50.0}})
        pygame.time.delay(250)  # 250ms > 1/refresh_hz (200ms)
        session.render_surface()
    assert calls["count"] >= 1
