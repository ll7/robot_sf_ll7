"""Run a basic Robot SF environment with a random policy.

Usage:
    uv run python examples/quickstart/01_basic_robot.py

Prerequisites:
    - None

Expected Output:
    - Prints the reward and termination status for a short random rollout.
    - Reports the cumulative reward collected during the demo run.

Limitations:
    - Uses a random policy; results vary unless the seed is fixed.
    - Designed for CPU execution; no rendering window is opened.

References:
    - docs/dev_guide.md#quickstart
"""

from __future__ import annotations

from typing import Any

from robot_sf.common.seed import set_global_seed
from robot_sf.gym_env.environment_factory import make_robot_env

STEP_COUNT = 10
SEED = 87234


def run_demo() -> None:
    """Execute a short random rollout in the default robot environment."""

    set_global_seed(SEED)
    env = make_robot_env(debug=False)

    try:
        observation, _ = env.reset()
        print("Environment reset successful.")
        print(f"Initial observation keys: {list(_extract_keys(observation))}")

        total_reward = 0.0
        print("\nRolling out random actions:")
        for step in range(1, STEP_COUNT + 1):
            action = env.action_space.sample()
            result = env.step(action)
            observation, reward, done = _normalize_step(result)
            total_reward += float(reward)

            print(f"Step {step:02d}: reward={reward:.3f} done={done}")

            if done:
                print("Episode finished early; resetting environment.")
                observation, _ = env.reset()

        print("\nDemo complete.")
        print(f"Total reward collected: {total_reward:.3f}")
    finally:
        env.exit()


def _normalize_step(step_result: tuple[Any, ...]) -> tuple[Any, float, bool]:
    """Support both Gym and Gymnasium step signatures."""

    if len(step_result) == 5:
        observation, reward, terminated, truncated, _ = step_result
        return observation, float(reward), bool(terminated or truncated)

    observation, reward, done, _ = step_result
    return observation, float(reward), bool(done)


def _extract_keys(observation: Any) -> list[str]:
    """Return the observation keys when the observation is a mapping."""

    if hasattr(observation, "keys"):
        return list(observation.keys())  # type: ignore[arg-type]
    return []


if __name__ == "__main__":
    run_demo()
