#!/usr/bin/env python3
"""Probe a thin Robot SF wrapper around the upstream Social-Navigation-PyEnvs ORCA policy."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator


def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _as_array(value: Any, *, pad: int, fill: float = 0.0) -> np.ndarray:
    """Return a 1D float array padded or cropped to the requested length."""
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.size >= pad:
        return arr[:pad]
    out = np.full((pad,), fill, dtype=float)
    out[: arr.size] = arr
    return out


def _as_matrix(value: Any, *, cols: int) -> np.ndarray:
    """Return a 2D float array with the requested column count."""
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.size == 0:
        return np.zeros((0, cols), dtype=float)
    arr = arr.reshape(-1, cols)
    return arr


@contextmanager
def _upstream_import_context(repo_root: Path) -> Iterator[None]:
    """Temporarily prepend the upstream repo root and restore the import state."""
    repo_str = str(repo_root)
    original_path = list(sys.path)
    sys.path.insert(0, repo_str)
    try:
        yield
    finally:
        sys.path[:] = original_path


class SocialNavigationPyEnvsORCAWrapper:
    """Thin adapter from Robot SF SocNav observations to upstream ORCA policy inference."""

    def __init__(
        self,
        repo_root: Path,
        *,
        preferred_speed: float = 1.0,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
    ) -> None:
        """Load the upstream ORCA policy and fixed state/action types from a checked-out repo."""
        self.repo_root = repo_root
        self.preferred_speed = float(preferred_speed)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        with _upstream_import_context(repo_root):
            action_mod = importlib.import_module("crowd_nav.utils.action")
            state_mod = importlib.import_module("crowd_nav.utils.state")
            policy_mod = importlib.import_module("crowd_nav.policy_no_train.orca")
        self._ActionXY = action_mod.ActionXY
        self._FullState = state_mod.FullState
        self._ObservableState = state_mod.ObservableState
        self._JointState = state_mod.JointState
        self._policy = policy_mod.ORCA()

    def _joint_state(self, observation: dict[str, Any]) -> Any:
        if "robot" in observation:
            robot_state = observation.get("robot", {})
            goal_state = observation.get("goal", {})
            ped_state = observation.get("pedestrians", {})
            robot_pos = _as_array(robot_state.get("position"), pad=2)
            goal_pos = _as_array(goal_state.get("current"), pad=2)
            heading = float(_as_array(robot_state.get("heading"), pad=1)[0])
            speed = float(_as_array(robot_state.get("speed"), pad=1)[0])
            radius = float(_as_array(robot_state.get("radius"), pad=1, fill=0.3)[0])
            ped_positions = _as_matrix(ped_state.get("positions"), cols=2)
            ped_velocities = _as_matrix(ped_state.get("velocities"), cols=2)
            ped_count = (
                int(_as_array(ped_state.get("count"), pad=1)[0])
                if ped_state.get("count") is not None
                else ped_positions.shape[0]
            )
            ped_radius_value = float(_as_array(ped_state.get("radius"), pad=1, fill=0.3)[0])
        else:
            robot_pos = _as_array(observation.get("robot_position"), pad=2)
            goal_pos = _as_array(observation.get("goal_current"), pad=2)
            heading = float(_as_array(observation.get("robot_heading"), pad=1)[0])
            speed = float(_as_array(observation.get("robot_speed"), pad=1)[0])
            radius = float(_as_array(observation.get("robot_radius"), pad=1, fill=0.3)[0])
            ped_positions = _as_matrix(observation.get("pedestrians_positions"), cols=2)
            ped_velocities = _as_matrix(observation.get("pedestrians_velocities"), cols=2)
            ped_count = (
                int(_as_array(observation.get("pedestrians_count"), pad=1)[0])
                if observation.get("pedestrians_count") is not None
                else ped_positions.shape[0]
            )
            ped_radius_value = float(
                _as_array(observation.get("pedestrians_radius"), pad=1, fill=0.3)[0]
            )
        vx = float(speed * math.cos(heading))
        vy = float(speed * math.sin(heading))
        self_state = self._FullState(
            float(robot_pos[0]),
            float(robot_pos[1]),
            vx,
            vy,
            radius,
            float(goal_pos[0]),
            float(goal_pos[1]),
            self.preferred_speed,
            heading,
        )
        ped_count = min(ped_count, ped_positions.shape[0], ped_velocities.shape[0])

        humans = [
            self._ObservableState(
                float(ped_positions[i, 0]),
                float(ped_positions[i, 1]),
                float(ped_velocities[i, 0]),
                float(ped_velocities[i, 1]),
                ped_radius_value,
            )
            for i in range(ped_count)
        ]
        return self._JointState(self_state, humans)

    def act(
        self, observation: dict[str, Any], *, time_step: float
    ) -> tuple[float, float, dict[str, Any]]:
        """Return a projected `(v, w)` command plus adapter metadata."""
        joint_state = self._joint_state(observation)
        self._policy.time_step = float(time_step)
        action = self._policy.predict(joint_state)
        if not isinstance(action, self._ActionXY):
            raise TypeError(f"Unexpected upstream action type: {type(action)}")

        vx = float(action.vx)
        vy = float(action.vy)
        speed = float(np.hypot(vx, vy))
        if "robot" in observation:
            current_heading = float(
                _as_array(observation.get("robot", {}).get("heading"), pad=1)[0]
            )
        else:
            current_heading = float(_as_array(observation.get("robot_heading"), pad=1)[0])
        desired_heading = float(math.atan2(vy, vx)) if speed > 1e-8 else current_heading
        heading_error = _normalize_angle(desired_heading - current_heading)
        dt = max(float(time_step), 1e-6)
        linear = float(
            np.clip(speed * max(0.0, math.cos(heading_error)), 0.0, self.max_linear_speed)
        )
        angular = float(
            np.clip(heading_error / dt, -self.max_angular_speed, self.max_angular_speed)
        )
        return (
            linear,
            angular,
            {
                "upstream_action_xy": [vx, vy],
                "projected_command_vw": [linear, angular],
                "heading_error_rad": heading_error,
                "projection_policy": "heading_safe_velocity_to_unicycle_vw",
            },
        )


@dataclass
class WrapperProbeReport:
    """Structured report for the wrapper probe."""

    issue: int
    repo_root: str
    verdict: str
    projection_policy: str
    wrapper_boundary: str
    upstream_policy: str
    steps_executed: int
    latest_robot_command: list[float]
    latest_upstream_action_xy: list[float]
    latest_heading_error_rad: float
    observation_keys: list[str]


def run_probe(repo_root: Path, *, seed: int, max_steps: int) -> WrapperProbeReport:
    """Run a minimal Robot SF rollout with the upstream ORCA policy wrapped."""
    wrapper = SocialNavigationPyEnvsORCAWrapper(repo_root)
    env = make_robot_env(
        config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        debug=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        action_adapter = PlannerActionAdapter(
            robot=env.simulator.robots[0],
            action_space=env.action_space,
            time_step=env.env_config.sim_config.time_per_step_in_secs,
        )
        latest_meta: dict[str, Any] | None = None
        steps_executed = 0
        for _ in range(max_steps):
            linear, angular, latest_meta = wrapper.act(
                obs, time_step=env.env_config.sim_config.time_per_step_in_secs
            )
            action = action_adapter.from_velocity_command((linear, angular))
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps_executed += 1
            if terminated or truncated:
                break

        if latest_meta is None:
            raise RuntimeError("Wrapper probe executed zero steps.")
        return WrapperProbeReport(
            issue=642,
            repo_root=str(repo_root),
            verdict="wrapper prototype viable",
            projection_policy=str(latest_meta["projection_policy"]),
            wrapper_boundary=(
                "Map Robot SF SocNav structured observation into upstream JointState, "
                "run upstream ORCA predict(), then project ActionXY into unicycle_vw."
            ),
            upstream_policy="crowd_nav.policy_no_train.orca.ORCA",
            steps_executed=steps_executed,
            latest_robot_command=[float(x) for x in latest_meta["projected_command_vw"]],
            latest_upstream_action_xy=[float(x) for x in latest_meta["upstream_action_xy"]],
            latest_heading_error_rad=float(latest_meta["heading_error_rad"]),
            observation_keys=sorted(str(key) for key in obs.keys()),
        )
    finally:
        env.close()


def _render_markdown(report: WrapperProbeReport) -> str:
    return "\n".join(
        [
            "# Social-Navigation-PyEnvs ORCA Wrapper Probe",
            "",
            f"- Verdict: `{report.verdict}`",
            f"- Issue: `#{report.issue}`",
            f"- Upstream policy: `{report.upstream_policy}`",
            f"- Repo root: `{report.repo_root}`",
            f"- Steps executed: `{report.steps_executed}`",
            f"- Projection policy: `{report.projection_policy}`",
            "",
            "## Boundary",
            "",
            f"- {report.wrapper_boundary}",
            "",
            "## Latest Command",
            "",
            f"- Upstream ActionXY: `{report.latest_upstream_action_xy}`",
            f"- Projected command (v, w): `{report.latest_robot_command}`",
            f"- Heading error (rad): `{report.latest_heading_error_rad}`",
            "",
            "## Observation Keys",
            "",
            f"- `{report.observation_keys}`",
            "",
            "## Interpretation",
            "",
            "- This proves a thin wrapper around the upstream non-trainable ORCA policy can drive a real Robot SF step loop.",
            "- It does not prove learned-path parity or paper-benchmark readiness.",
            "",
        ]
    )


def main() -> None:
    """Run the wrapper probe and write JSON/Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()

    report = run_probe(args.repo_root.resolve(), seed=args.seed, max_steps=args.max_steps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
