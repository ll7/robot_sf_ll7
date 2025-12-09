"""Live telemetry pane docked in the Pygame window.

Usage:
    uv run python examples/advanced/22_telemetry_pane.py

Prerequisites:
    - None

Expected Output:
    - Interactive Pygame window with docked telemetry charts (FPS, reward, collisions, etc.)
    - Telemetry JSONL under output/telemetry/<run_id>/telemetry.jsonl

Limitations:
    - Requires a display; run headless with `DISPLAY=` and `SDL_VIDEODRIVER=dummy` for CI.

References:
    - specs/343-telemetry-viz/quickstart.md
"""

from __future__ import annotations

import time

from robot_sf.gym_env.environment_factory import make_robot_env


def main() -> None:
    """Run a short live simulation with the telemetry pane enabled."""
    env = make_robot_env(
        debug=True,
        enable_telemetry_panel=True,
        telemetry_record=True,
        telemetry_metrics=["fps", "reward", "collisions", "min_ped_distance", "action_norm"],
        telemetry_refresh_hz=2.0,
        telemetry_pane_layout="horizontal_split",  # dock below the scene to avoid occlusion
    )
    _, _info = env.reset(seed=42)

    max_runtime_sec = 30
    end_time = time.time() + max_runtime_sec
    step_idx = 0
    while time.time() < end_time:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        step_idx += 1

        if terminated or truncated:
            print(f"Episode finished at step {step_idx}; restarting")
            _, _info = env.reset(seed=step_idx + 1000)
            continue

        # Allow user to close the window at any time
        if getattr(env, "sim_ui", None) and env.sim_ui.is_exit_requested:
            break

    env.close()
    paths = env.write_telemetry_summary()
    if paths is not None:
        print(f"Telemetry artifacts written to: {paths}")


if __name__ == "__main__":
    main()
