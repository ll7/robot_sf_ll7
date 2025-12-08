"""Demo: Live telemetry pane docked in the Pygame window.

Run:
    uv run python examples/advanced/telemetry_pane_demo.py

Outputs:
    - Telemetry JSONL under output/telemetry/<run_id>/telemetry.jsonl (when telemetry_record=True)
    - Pygame window with charts docked (when debug=True)
"""

from __future__ import annotations

import os

from robot_sf.gym_env.environment_factory import make_robot_env


def main() -> None:
    """Run a short live simulation with the telemetry pane enabled."""
    # Ensure artifact root is defined (defaults to output/)
    os.environ.setdefault("ROBOT_SF_ARTIFACT_ROOT", "output")

    env = make_robot_env(
        debug=True,
        enable_telemetry_panel=True,
        telemetry_record=True,
        telemetry_metrics=["fps", "reward", "collisions", "min_ped_distance", "action_norm"],
        telemetry_refresh_hz=2.0,
    )
    _obs, _info = env.reset(seed=42)

    for _ in range(200):
        action = env.action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
