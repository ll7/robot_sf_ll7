"""Headless telemetry smoke: run without rendering and emit telemetry artifacts.

TODO: It is unclear whether this example currently runs correctly.

Usage:
    DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
    uv run python examples/advanced/23_telemetry_headless_smoke.py

Prerequisites:
    - None

Expected Output:
    - Telemetry JSONL under output/telemetry/<run_id>/telemetry.jsonl
    - Summary PNG and JSON under output/telemetry/<run_id>/

Limitations:
    - Headless only; respects SDL_VIDEODRIVER=dummy for CI.

References:
    - specs/343-telemetry-viz/quickstart.md
"""

from __future__ import annotations

from robot_sf.gym_env.environment_factory import make_robot_env


def main() -> None:
    """Run a short headless episode with telemetry recording."""
    env = make_robot_env(
        debug=False,
        enable_telemetry_panel=False,
        telemetry_record=True,
        telemetry_metrics=["fps", "reward", "collisions", "min_ped_distance"],
        telemetry_refresh_hz=2.0,
    )
    _, _info = env.reset(seed=1)
    for step_idx in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            print(f"Episode finished at step {step_idx}")
            break
    env.close()
    # Emit summary artifacts from telemetry session
    paths = env.write_telemetry_summary()
    if paths is not None:
        print(f"Telemetry artifacts written to: {paths}")


if __name__ == "__main__":
    main()
