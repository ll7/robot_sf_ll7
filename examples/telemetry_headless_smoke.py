"""Headless telemetry smoke: run a short episode and emit telemetry artifacts.

Usage:
    DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
    uv run python examples/telemetry_headless_smoke.py
"""

from __future__ import annotations

import os

from robot_sf.gym_env.environment_factory import make_robot_env


def main() -> None:
    """Headless smoke run that records telemetry artifacts under output/telemetry/."""
    os.environ.setdefault("ROBOT_SF_ARTIFACT_ROOT", "output")
    env = make_robot_env(
        debug=False,
        enable_telemetry_panel=False,
        telemetry_record=True,
        telemetry_metrics=["fps", "reward"],
        telemetry_refresh_hz=2.0,
    )
    _obs, _ = env.reset(seed=1)
    for _ in range(20):
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    env.close()
    # Emit summary artifacts from telemetry session when available
    session = getattr(env, "_telemetry_session", None)
    if session is not None:
        session.write_summary()


if __name__ == "__main__":
    main()
