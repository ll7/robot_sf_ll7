"""Headless/CI telemetry artifact smoke test (US3)."""

from __future__ import annotations

import json
import os

import pytest

from robot_sf.gym_env.environment_factory import make_robot_env


@pytest.mark.skipif(
    os.environ.get("CI") is None and os.environ.get("SDL_VIDEODRIVER") != "dummy",
    reason="Headless smoke only runs when dummy SDL or CI is set",
)
def test_headless_telemetry_artifacts(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Ensure telemetry JSONL and summary artifact are produced headlessly."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_robot_env(
        debug=False,
        enable_telemetry_panel=False,
        telemetry_record=True,
        telemetry_metrics=["fps", "reward"],
        telemetry_refresh_hz=2.0,
    )
    _obs, _ = env.reset(seed=1)
    for _ in range(10):
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()
    session = getattr(env, "_telemetry_session", None)
    if session is not None:
        session.write_summary()

    telemetry_root = tmp_path / "telemetry"
    assert telemetry_root.exists()
    # Find the newest run dir
    run_dirs = sorted(telemetry_root.iterdir())
    assert run_dirs, "No telemetry run directory created"
    telemetry_jsonl = run_dirs[-1] / "telemetry.jsonl"
    health_json = run_dirs[-1] / "telemetry_health.json"
    summary_png = run_dirs[-1] / "telemetry_summary.png"
    summary_json = run_dirs[-1] / "telemetry_summary.json"
    assert telemetry_jsonl.exists() and telemetry_jsonl.stat().st_size > 0
    assert health_json.exists()
    assert summary_png.exists()
    assert summary_json.exists()
    # Validate JSON lines parse
    lines = telemetry_jsonl.read_text().splitlines()
    assert len(lines) >= 1
    json.loads(lines[0])
