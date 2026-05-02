#!/usr/bin/env python3
"""Probe a narrow Robot SF wrapper path for the upstream social-jym SARL policy.

The probe is intentionally optional and dependency-isolated. It does not add JAX or social-jym to
Robot SF runtime dependencies; callers must provide those packages in the execution environment.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter


@dataclass
class SocialJymWrapperReport:
    """Structured result from the social-jym wrapper feasibility probe."""

    issue: int
    verdict: str
    wrapper_boundary: str
    source_environment: str
    source_policy: str
    source_scenario: str
    source_humans_policy: str
    projection_policy: str
    steps_executed: int
    latest_source_action_xy: list[float] | None
    latest_robot_command_vw: list[float] | None
    latest_robot_action: list[float] | None
    latest_heading_error_rad: float | None
    observation_keys: list[str]
    failure_reason: str | None = None


@dataclass
class SocialJymParityReport:
    """Structured result from the controlled social-jym wrapper parity probe."""

    issue: int
    verdict: str
    source_policy: str
    controlled_state: str
    observation_max_abs_error: float
    robot_goal_max_abs_error: float
    time_abs_error: float
    vnet_input_max_abs_error: float | None
    projection_cases: list[dict[str, float | list[float]]]
    benchmark_boundary: str
    failure_reason: str | None = None


def _as_float_array(value: Any, *, name: str, min_size: int) -> np.ndarray:
    """Return ``value`` as a flat float array with at least ``min_size`` entries."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < min_size:
        msg = f"Missing or undersized observation field: {name}"
        raise ValueError(msg)
    return arr


def _extract_socnav_blocks(
    obs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return robot, goal, and pedestrian blocks from nested or flat SocNav observations."""
    robot = obs.get("robot")
    goal = obs.get("goal")
    pedestrians = obs.get("pedestrians")
    if isinstance(robot, dict) and isinstance(goal, dict) and isinstance(pedestrians, dict):
        return robot, goal, pedestrians
    return (
        {
            "position": obs.get("robot_position"),
            "heading": obs.get("robot_heading"),
            "speed": obs.get("robot_speed"),
            "radius": obs.get("robot_radius", [0.3]),
        },
        {"current": obs.get("goal_current")},
        {
            "positions": obs.get("pedestrians_positions", []),
            "velocities": obs.get("pedestrians_velocities", []),
            "count": obs.get("pedestrians_count", [0]),
            "radius": obs.get("pedestrians_radius", [0.3]),
        },
    )


def _pedestrian_count(pedestrians: dict[str, Any], positions: np.ndarray) -> int:
    """Resolve the active pedestrian count from an observation block."""
    count_arr = np.asarray(pedestrians.get("count", [positions.shape[0]])).reshape(-1)
    count = int(count_arr[0]) if count_arr.size else int(positions.shape[0])
    return max(0, min(count, int(positions.shape[0])))


def build_social_jym_policy_inputs(
    obs: dict[str, Any],
    *,
    max_humans: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Map a Robot SF SocNav observation into the minimal upstream SARL input contract.

    Returns:
        tuple: ``(social_obs, social_info, observation_keys)`` where ``social_obs`` has shape
        ``(max_humans + 1, 6)`` and ``social_info`` contains ``robot_goal``.
    """
    robot, goal, pedestrians = _extract_socnav_blocks(obs)
    robot_pos = _as_float_array(robot.get("position"), name="robot.position", min_size=2)[:2]
    robot_heading = _as_float_array(robot.get("heading", [0.0]), name="robot.heading", min_size=1)[
        0
    ]
    robot_speed = _as_float_array(robot.get("speed", [0.0, 0.0]), name="robot.speed", min_size=1)
    goal_current = _as_float_array(goal.get("current"), name="goal.current", min_size=2)[:2]
    robot_radius = _as_float_array(robot.get("radius", [0.3]), name="robot.radius", min_size=1)[0]

    positions = np.asarray(pedestrians.get("positions", []), dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 2) if positions.size else np.zeros((0, 2), dtype=float)
    velocities = np.asarray(pedestrians.get("velocities", []), dtype=float)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 2) if velocities.size else np.zeros((0, 2), dtype=float)
    count = min(_pedestrian_count(pedestrians, positions), max(1, int(max_humans)))
    if velocities.shape[0] < count:
        velocities = np.pad(velocities, ((0, count - velocities.shape[0]), (0, 0)))

    radius_raw = np.asarray(pedestrians.get("radius", [0.3]), dtype=float).reshape(-1)
    human_radius = float(radius_raw[0]) if radius_raw.size else 0.3

    social_obs = np.zeros((max(1, int(max_humans)) + 1, 6), dtype=np.float32)
    for idx in range(count):
        social_obs[idx, 0:2] = positions[idx, 0:2]
        social_obs[idx, 2:4] = velocities[idx, 0:2]
        social_obs[idx, 4] = human_radius

    # social-jym holonomic SocialNav encodes robot velocity in columns 2:4 and uses column 5 as
    # padding/heading. We preserve heading anyway so projection diagnostics can use the same value.
    social_obs[-1, 0:2] = robot_pos
    if robot_speed.size >= 2:
        social_obs[-1, 2:4] = robot_speed[:2]
    else:
        social_obs[-1, 2] = float(robot_speed[0])
    social_obs[-1, 4] = robot_radius
    social_obs[-1, 5] = robot_heading

    social_info = {
        "robot_goal": goal_current.astype(np.float32),
        # Upstream SARL traces the reward path during JAX compilation even when the action
        # selection epsilon forces random exploration, so the source-style time field is required.
        "time": np.asarray(0.0, dtype=np.float32),
    }
    keys = sorted(str(key) for key in obs.keys())
    return social_obs, social_info, keys


def _controlled_robot_sf_observation() -> dict[str, Any]:
    """Return a simple Robot SF SocNav observation with one human in absolute coordinates."""
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=np.float32),
            "heading": np.asarray([0.0], dtype=np.float32),
            "speed": np.asarray([0.2, 0.0], dtype=np.float32),
            "radius": np.asarray([0.3], dtype=np.float32),
        },
        "goal": {"current": np.asarray([2.0, 0.0], dtype=np.float32)},
        "pedestrians": {
            "positions": np.asarray([[1.0, 0.5]], dtype=np.float32),
            "velocities": np.asarray([[0.0, -0.1]], dtype=np.float32),
            "count": np.asarray([1], dtype=np.int32),
            "radius": np.asarray([0.3], dtype=np.float32),
        },
    }


def _controlled_source_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the matched source-shaped SARL observation and info for the controlled state."""
    source_obs = np.asarray(
        [
            [1.0, 0.5, 0.0, -0.1, 0.3, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    source_info = {
        "robot_goal": np.asarray([2.0, 0.0], dtype=np.float32),
        "time": np.asarray(0.0, dtype=np.float32),
    }
    return source_obs, source_info


def project_holonomic_action_to_unicycle(
    action_xy: np.ndarray,
    *,
    robot_heading: float,
    dt: float,
    max_linear_speed: float = 1.0,
    max_angular_speed: float = 2.0,
) -> tuple[float, float, float]:
    """Project a source holonomic velocity into a conservative Robot SF ``(v, omega)`` command."""
    action = np.asarray(action_xy, dtype=float).reshape(-1)
    if action.size < 2:
        msg = "social-jym action must contain at least vx, vy"
        raise ValueError(msg)
    speed = float(np.linalg.norm(action[:2]))
    if speed <= 1e-9:
        return 0.0, 0.0, 0.0
    desired_heading = math.atan2(float(action[1]), float(action[0]))
    heading_error = (desired_heading - float(robot_heading) + math.pi) % (2 * math.pi) - math.pi
    linear = min(float(max_linear_speed), speed) * max(0.0, math.cos(heading_error))
    angular = max(
        -float(max_angular_speed),
        min(float(max_angular_speed), heading_error / max(float(dt), 1e-6)),
    )
    return float(linear), float(angular), float(heading_error)


@contextmanager
def _optional_repo_on_path(repo_root: Path | None):
    """Temporarily prepend an optional upstream checkout to ``sys.path``."""
    if repo_root is None:
        yield
        return
    import sys

    repo_str = str(repo_root.resolve())
    old_path = list(sys.path)
    sys.path.insert(0, repo_str)
    try:
        yield
    finally:
        sys.path[:] = old_path


class SocialJymSARLWrapper:
    """Thin wrapper around upstream SARL action selection for one-step smoke probing."""

    def __init__(
        self, *, repo_root: Path | None = None, max_humans: int = 1, seed: int = 0
    ) -> None:
        """Initialize upstream modules and randomly initialized SARL parameters."""
        self.max_humans = max(1, int(max_humans))
        with _optional_repo_on_path(repo_root):
            import jax.numpy as jnp
            from jax import random
            from socialjym.policies.sarl import SARL
            from socialjym.utils.rewards.socialnav_rewards.dummy_reward import DummyReward

        self.jnp = jnp
        self.random = random
        self.reward = DummyReward(kinematics="holonomic", time_limit=5.0)
        self.policy = SARL(self.reward, dt=0.25, kinematics="holonomic")
        self.params = self.policy.model.init(
            random.PRNGKey(int(seed)),
            jnp.zeros((self.max_humans, self.policy.vnet_input_size)),
        )
        self.key = random.PRNGKey(int(seed) + 1)

    def act(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a source holonomic action and adapter metadata for a Robot SF observation."""
        social_obs, social_info, observation_keys = build_social_jym_policy_inputs(
            obs,
            max_humans=self.max_humans,
        )
        action, self.key, _vnet_input, _action_values = self.policy.act(
            self.key,
            self.jnp.asarray(social_obs),
            {
                "robot_goal": self.jnp.asarray(social_info["robot_goal"]),
                "time": self.jnp.asarray(social_info["time"]),
            },
            self.params,
            1.0,
        )
        action_np = np.asarray(action, dtype=float)
        return action_np, {
            "observation_keys": observation_keys,
            "upstream_policy": "socialjym.policies.sarl.SARL",
            "source_action_xy": [float(x) for x in action_np[:2]],
        }


def run_parity_probe(*, repo_root: Path | None = None) -> SocialJymParityReport:
    """Compare wrapper-built SARL inputs against a matched source-shaped control input."""
    source_obs, source_info = _controlled_source_inputs()
    wrapper_obs, wrapper_info, _keys = build_social_jym_policy_inputs(
        _controlled_robot_sf_observation(),
        max_humans=1,
    )
    observation_error = float(np.max(np.abs(wrapper_obs - source_obs)))
    goal_error = float(np.max(np.abs(wrapper_info["robot_goal"] - source_info["robot_goal"])))
    time_error = float(abs(float(wrapper_info["time"]) - float(source_info["time"])))

    vnet_error: float | None = None
    failure_reason: str | None = None
    try:
        wrapper = SocialJymSARLWrapper(repo_root=repo_root, max_humans=1, seed=0)
        source_vnet = wrapper.policy.batch_compute_vnet_input(
            wrapper.jnp.asarray(source_obs[-1]),
            wrapper.jnp.asarray(source_obs[0:1]),
            {
                "robot_goal": wrapper.jnp.asarray(source_info["robot_goal"]),
                "time": wrapper.jnp.asarray(source_info["time"]),
            },
        )
        wrapper_vnet = wrapper.policy.batch_compute_vnet_input(
            wrapper.jnp.asarray(wrapper_obs[-1]),
            wrapper.jnp.asarray(wrapper_obs[0:1]),
            {
                "robot_goal": wrapper.jnp.asarray(wrapper_info["robot_goal"]),
                "time": wrapper.jnp.asarray(wrapper_info["time"]),
            },
        )
        vnet_error = float(np.max(np.abs(np.asarray(wrapper_vnet) - np.asarray(source_vnet))))
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    projection_cases: list[dict[str, float | list[float]]] = []
    for action in (
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    ):
        linear, angular, heading_error = project_holonomic_action_to_unicycle(
            action,
            robot_heading=0.0,
            dt=0.25,
        )
        source_speed = float(np.linalg.norm(action))
        projection_cases.append(
            {
                "source_action_xy": [float(action[0]), float(action[1])],
                "source_speed": source_speed,
                "projected_linear": float(linear),
                "projected_angular": float(angular),
                "heading_error_rad": float(heading_error),
                "instant_linear_speed_loss": float(source_speed - linear),
            }
        )

    parity_pass = (
        observation_error <= 1e-6
        and goal_error <= 1e-6
        and time_error <= 1e-6
        and vnet_error is not None
        and vnet_error <= 1e-6
    )
    return SocialJymParityReport(
        issue=907,
        verdict="controlled input parity passed"
        if parity_pass
        else "controlled input parity blocked",
        source_policy="socialjym.policies.sarl.SARL",
        controlled_state="one_robot_one_human_holonomic_socialnav",
        observation_max_abs_error=observation_error,
        robot_goal_max_abs_error=goal_error,
        time_abs_error=time_error,
        vnet_input_max_abs_error=vnet_error,
        projection_cases=projection_cases,
        benchmark_boundary=(
            "Controlled SARL input parity is not trained-policy provenance or benchmark support; "
            "holonomic-to-unicycle projection still loses lateral/reverse semantics."
        ),
        failure_reason=failure_reason,
    )


def run_probe(
    *,
    repo_root: Path | None,
    seed: int,
    max_steps: int,
) -> SocialJymWrapperReport:
    """Run one Robot SF loop using a random-action upstream SARL wrapper."""
    cfg = RobotSimulationConfig()
    cfg.observation_mode = ObservationMode.SOCNAV_STRUCT
    env = make_robot_env(config=cfg, debug=False)
    latest_source_action: list[float] | None = None
    latest_robot_command: list[float] | None = None
    latest_robot_action: list[float] | None = None
    latest_heading_error: float | None = None
    observation_keys: list[str] = []
    steps = 0
    try:
        obs, _info = env.reset(seed=int(seed))
        time_step = float(env.env_config.sim_config.time_per_step_in_secs)
        adapter = PlannerActionAdapter(env.simulator.robots[0], env.action_space, time_step)
        wrapper = SocialJymSARLWrapper(repo_root=repo_root, max_humans=1, seed=seed)
        for _ in range(max(1, int(max_steps))):
            source_action, meta = wrapper.act(obs)
            robot, _goal, _pedestrians = _extract_socnav_blocks(obs)
            robot_heading = _as_float_array(
                robot.get("heading", [0.0]), name="robot.heading", min_size=1
            )[0]
            linear, angular, heading_error = project_holonomic_action_to_unicycle(
                source_action,
                robot_heading=robot_heading,
                dt=time_step,
            )
            robot_action = np.asarray(adapter.from_velocity_command((linear, angular)), dtype=float)
            obs, _reward, terminated, truncated, _step_info = env.step(robot_action)
            latest_source_action = meta["source_action_xy"]
            latest_robot_command = [float(linear), float(angular)]
            latest_robot_action = [float(x) for x in robot_action.reshape(-1)]
            latest_heading_error = float(heading_error)
            observation_keys = list(meta["observation_keys"])
            steps += 1
            if terminated or truncated:
                break
    except Exception as exc:
        return SocialJymWrapperReport(
            issue=905,
            verdict="wrapper prototype blocked",
            wrapper_boundary="No benchmark-facing support; wrapper smoke failed before viability.",
            source_environment="socialjym.envs.socialnav.SocialNav",
            source_policy="socialjym.policies.sarl.SARL",
            source_scenario="circular_crossing",
            source_humans_policy="hsfm",
            projection_policy="heading_safe_holonomic_xy_to_unicycle_vw",
            steps_executed=steps,
            latest_source_action_xy=latest_source_action,
            latest_robot_command_vw=latest_robot_command,
            latest_robot_action=latest_robot_action,
            latest_heading_error_rad=latest_heading_error,
            observation_keys=observation_keys,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        env.close()

    return SocialJymWrapperReport(
        issue=905,
        verdict="wrapper prototype viable",
        wrapper_boundary=(
            "One random-action upstream SARL wrapper can drive a real Robot SF step loop, but this "
            "is not source-policy parity, trained-policy evidence, or benchmark support."
        ),
        source_environment="socialjym.envs.socialnav.SocialNav",
        source_policy="socialjym.policies.sarl.SARL",
        source_scenario="circular_crossing",
        source_humans_policy="hsfm",
        projection_policy="heading_safe_holonomic_xy_to_unicycle_vw",
        steps_executed=steps,
        latest_source_action_xy=latest_source_action,
        latest_robot_command_vw=latest_robot_command,
        latest_robot_action=latest_robot_action,
        latest_heading_error_rad=latest_heading_error,
        observation_keys=observation_keys,
        failure_reason=None,
    )


def _render_markdown(report: SocialJymWrapperReport) -> str:
    """Render a compact Markdown report for the probe."""
    lines = [
        "# Social-Jym Wrapper Probe",
        "",
        f"- Issue: #{report.issue}",
        f"- Verdict: `{report.verdict}`",
        f"- Source environment: `{report.source_environment}`",
        f"- Source policy: `{report.source_policy}`",
        f"- Source scenario: `{report.source_scenario}`",
        f"- Source humans policy: `{report.source_humans_policy}`",
        f"- Projection policy: `{report.projection_policy}`",
        f"- Steps executed in Robot SF: `{report.steps_executed}`",
        f"- Latest source action xy: `{report.latest_source_action_xy}`",
        f"- Latest Robot SF command vw: `{report.latest_robot_command_vw}`",
        f"- Latest Robot SF action: `{report.latest_robot_action}`",
        f"- Latest heading error rad: `{report.latest_heading_error_rad}`",
        f"- Observation keys: `{report.observation_keys}`",
        "",
        "## Boundary",
        "",
        f"- {report.wrapper_boundary}",
        "- This remains an exploratory wrapper smoke result.",
        "- It does not add JAX or social-jym to Robot SF runtime dependencies.",
        "- It must not be reported as benchmark success without later parity and fail-closed "
        "readiness proof.",
    ]
    if report.failure_reason:
        lines.extend(["", "## Failure", "", f"- `{report.failure_reason}`"])
    return "\n".join(lines) + "\n"


def _render_parity_markdown(report: SocialJymParityReport) -> str:
    """Render a compact Markdown report for the controlled parity probe."""
    lines = [
        "# Social-Jym SARL Wrapper Parity Probe",
        "",
        f"- Issue: #{report.issue}",
        f"- Verdict: `{report.verdict}`",
        f"- Source policy: `{report.source_policy}`",
        f"- Controlled state: `{report.controlled_state}`",
        f"- Observation max abs error: `{report.observation_max_abs_error}`",
        f"- Robot goal max abs error: `{report.robot_goal_max_abs_error}`",
        f"- Time abs error: `{report.time_abs_error}`",
        f"- VNet input max abs error: `{report.vnet_input_max_abs_error}`",
        "",
        "## Projection Cases",
        "",
    ]
    for case in report.projection_cases:
        lines.append(f"- `{case}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"- {report.benchmark_boundary}",
            "- This result must remain non-benchmark evidence until trained-policy provenance and "
            "benchmark readiness are proven separately.",
        ]
    )
    if report.failure_reason:
        lines.extend(["", "## Failure", "", f"- `{report.failure_reason}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("wrapper", "parity"), default="wrapper")
    parser.add_argument("--repo-root", type=Path, default=Path("output/repos/social-jym"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/benchmarks/external/social_jym_issue905/wrapper_probe.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("output/benchmarks/external/social_jym_issue905/wrapper_probe.md"),
    )
    return parser.parse_args()


def main() -> int:
    """Run the wrapper probe and write JSON/Markdown reports."""
    args = parse_args()
    if args.mode == "parity":
        parity_report = run_parity_probe(repo_root=args.repo_root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(parity_report), indent=2, sort_keys=True), encoding="utf-8"
        )
        args.output_markdown.write_text(_render_parity_markdown(parity_report), encoding="utf-8")
        print(json.dumps(asdict(parity_report), indent=2, sort_keys=True))
        return 0 if parity_report.verdict == "controlled input parity passed" else 1

    report = run_probe(repo_root=args.repo_root, seed=args.seed, max_steps=args.max_steps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.verdict == "wrapper prototype viable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
