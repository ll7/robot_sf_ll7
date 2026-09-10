"""Implement and run the smallest planner compatible with LocalPlannerProtocol.

Usage:
    uv run python examples/advanced/37_custom_planner_protocol.py [--horizon N] [--format text|json]

Prerequisites:
    - None beyond the repository.

Expected Output:
    - A short headless episode driven by the tutorial planner, with per-step commands
      and a diagnostics summary. ``--format json`` emits the same run summary
      machine-readably.

Limitations:
    - Educational implementation only: constant forward command, no obstacle avoidance,
      no benchmark or safety claim. Not benchmark evidence.

References:
    - robot_sf/planner/protocol.py
    - docs/contributing_planner.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from robot_sf.common.seed import set_global_seed
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.planner.protocol import (
    PLANNER_TYPE_KEY,
    LocalPlannerProtocol,
    normalize_planner_diagnostics,
)

LINEAR_BOUNDS = (0.0, 1.0)
ANGULAR_BOUNDS = (-1.2, 1.2)
SEED = 11
HORIZON = 20
PLANNER_TYPE = "tutorial_constant_forward"


def validate_command(command: object) -> tuple[float, float]:
    """Validate a planner command against the declared action bounds.

    Args:
        command: Candidate ``(linear_speed, angular_rate)`` command.

    Returns:
        The validated command as finite floats.

    Raises:
        ValueError: If the shape is wrong, a component is non-finite, or a
            component leaves the declared bounds. Commands are never clipped
            silently: out-of-bounds output is an explicit error.
    """
    if not isinstance(command, (tuple, list)) or len(command) != 2:
        raise ValueError(f"Command must be a (linear, angular) pair, got: {command!r}")
    linear, angular = float(command[0]), float(command[1])
    if not math.isfinite(linear) or not math.isfinite(angular):
        raise ValueError(f"Command components must be finite, got: {command!r}")
    if not LINEAR_BOUNDS[0] <= linear <= LINEAR_BOUNDS[1]:
        raise ValueError(f"Linear speed {linear} outside bounds {LINEAR_BOUNDS}.")
    if not ANGULAR_BOUNDS[0] <= angular <= ANGULAR_BOUNDS[1]:
        raise ValueError(f"Angular rate {angular} outside bounds {ANGULAR_BOUNDS}.")
    return linear, angular


class TutorialPlanner:
    """Deterministic stateless goal-directed tutorial planner (educational only)."""

    def __init__(self, linear_speed: float = 0.3, angular_rate: float = 0.0) -> None:
        """Create the planner with one constant validated command.

        Args:
            linear_speed: Forward speed in m/s within bounds.
            angular_rate: Angular rate in rad/s within bounds.
        """
        self._command = validate_command((linear_speed, angular_rate))
        self._seed: int | None = None
        self._steps = 0
        self._closed = False

    def reset(self, *, seed: int | None = None) -> None:
        """Reset step count and record the seed (planner holds no RNG state)."""
        self._seed = seed
        self._steps = 0

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the constant validated command and count the step."""
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a mapping.")
        self._steps += 1
        return validate_command(self._command)

    def diagnostics(self) -> dict[str, Any]:
        """Return execution diagnostics carrying the planner type."""
        return normalize_planner_diagnostics(
            {
                PLANNER_TYPE_KEY: PLANNER_TYPE,
                "steps_planned": self._steps,
                "command": list(self._command),
                "seed": self._seed,
            },
            fallback_planner_type=PLANNER_TYPE,
        )

    def close(self) -> None:
        """Release held resources (idempotent; the planner holds none)."""
        self._closed = True


assert isinstance(TutorialPlanner(), LocalPlannerProtocol)


def _step_budget(default: int) -> int:
    """Return a smaller rollout budget when the example runs in smoke mode."""
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:  # pragma: no cover - defensive guard
            pass
    return default


def run_episode(horizon: int) -> dict[str, Any]:
    """Run one short headless episode with the tutorial planner.

    Args:
        horizon: Bounded step budget for the episode.

    Returns:
        Run summary with steps, total reward, diagnostics, and seed.
    """
    set_global_seed(SEED)
    planner = TutorialPlanner()
    planner.reset(seed=SEED)
    env = make_robot_env(debug=False)
    total_reward = 0.0
    steps = 0
    try:
        observation, _ = env.reset()
        for _ in range(horizon):
            command = planner.plan(observation)
            observation, reward, terminated, truncated, _ = env.step(command)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        planner.close()
        env.close()
    return {
        "planner_type": PLANNER_TYPE,
        "horizon": horizon,
        "steps": steps,
        "total_reward": total_reward,
        "seed": SEED,
        "diagnostics": planner.diagnostics(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the tutorial.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Parsed arguments with horizon and output format.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the tutorial episode and print the summary.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    summary = run_episode(_step_budget(max(1, args.horizon)))
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"Tutorial planner ran {summary['steps']} steps, total reward {summary['total_reward']:.3f}."
        )
        print(f"Diagnostics: {summary['diagnostics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
