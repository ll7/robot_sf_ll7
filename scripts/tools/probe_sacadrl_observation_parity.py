#!/usr/bin/env python3
"""Probe observation-level parity for the Robot SF SACADRL adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robot_sf.planner.socnav import (
    _SACADRL_STATE_ORDER,
    SACADRLPlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.sensor.socnav_observation import SocNavObservationFusion

ISSUE_NUMBER = 663
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_UPSTREAM_REPO_ROOT = Path("output/repos/gym-collision-avoidance")
DEFAULT_SIDE_ENV_PYTHON = Path(
    "output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python"
)
UPSTREAM_REPO_URL = "https://github.com/mit-acl/gym-collision-avoidance"
EXAMPLE_PATH = "gym_collision_avoidance/experiments/src/example.py"


@dataclass(frozen=True)
class CommandResult:
    """Structured result for one subprocess stage."""

    name: str
    command: list[str]
    returncode: int | None
    failure_summary: str | None
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class CaseResult:
    """One parity case outcome."""

    name: str
    verdict: str
    max_abs_diff: float
    component_max_abs_diff: dict[str, float]
    notes: list[str]


@dataclass(frozen=True)
class ProbeReport:
    """Structured report for issue #663."""

    issue: int
    repo_root: str
    repo_remote_url: str
    side_env_python: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    cases: list[CaseResult]
    source_contract: dict[str, Any]
    commands: list[CommandResult]


def _run_command(name: str, command: list[str], cwd: Path, timeout_seconds: int) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_raw = exc.stdout or ""
        stderr_raw = exc.stderr or ""
        return CommandResult(
            name=name,
            command=command,
            returncode=None,
            failure_summary=f"command exceeded timeout ({timeout_seconds}s)",
            stdout_tail=(
                stdout_raw.decode("utf-8", errors="replace")
                if isinstance(stdout_raw, bytes)
                else stdout_raw
            )[-4000:],
            stderr_tail=(
                stderr_raw.decode("utf-8", errors="replace")
                if isinstance(stderr_raw, bytes)
                else stderr_raw
            )[-4000:],
        )
    failure_summary = (
        None if result.returncode == 0 else _detect_failure_summary(result.stdout, result.stderr)
    )
    return CommandResult(
        name=name,
        command=command,
        returncode=result.returncode,
        failure_summary=failure_summary,
        stdout_tail=result.stdout[-4000:],
        stderr_tail=result.stderr[-4000:],
    )


def _detect_failure_summary(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else "unknown failure"


def _parse_json_stdout(result: CommandResult) -> dict[str, Any]:
    lines = [line for line in result.stdout_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON payload found in command output for {result.name}")


def _validate_paths(repo_root: Path, side_env_python: Path) -> None:
    required = [repo_root / "README.md", repo_root / EXAMPLE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream files: {', '.join(missing)}")
    if not side_env_python.exists():
        raise FileNotFoundError(f"Side-environment interpreter missing: {side_env_python}")


def _upstream_live_payload_script() -> str:
    return """
import json
import os

os.environ['MPLBACKEND'] = 'Agg'
os.environ['GYM_CONFIG_CLASS'] = 'Example'
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import matplotlib as mpl
_original_use = mpl.use

def patched_use(backend, *args, **kwargs):
    if backend == 'TkAgg':
        return _original_use('Agg', *args, **kwargs)
    return _original_use(backend, *args, **kwargs)

mpl.use = patched_use
import gym
import tensorflow.compat.v1 as tf
from gym_collision_avoidance.envs import test_cases as tc
import gym_collision_avoidance.envs.visualize as viz

viz.animate_episode = lambda *args, **kwargs: None
gym.logger.set_level(40)
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.Session().__enter__()

env = gym.make('CollisionAvoidance-v0')
agents = tc.get_testcase_two_agents()
for agent in agents:
    if hasattr(agent.policy, 'initialize_network'):
        agent.policy.initialize_network()
env.set_agents(agents)
reset_out = env.reset()
obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
internal = obs[1]
keys = [
    'num_other_agents',
    'dist_to_goal',
    'heading_ego_frame',
    'pref_speed',
    'radius',
    'other_agents_states',
]
payload = {}
for key in keys:
    value = internal[key]
    payload[key] = value.tolist() if hasattr(value, 'tolist') else value
print(json.dumps({'native_state': payload}))
"""


def _extract_source_contract() -> dict[str, Any]:
    return {
        "learned_policy": "GA3C_CADRL",
        "state_order": list(_SACADRL_STATE_ORDER),
        "action_space": "speed_delta_heading",
        "kinematics": "unicycle_like_speed_plus_delta_heading",
        "reference_boundary": (
            "This issue checks observation-to-network-input parity. It does not make "
            "a benchmark-quality claim by itself."
        ),
    }


def _as_float_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _flatten_native_state(native_state: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = {
        "num_other_agents": _as_float_array([native_state["num_other_agents"]]),
        "dist_to_goal": _as_float_array([native_state["dist_to_goal"]]),
        "heading_ego_frame": _as_float_array([native_state["heading_ego_frame"]]),
        "pref_speed": _as_float_array([native_state["pref_speed"]]),
        "radius": _as_float_array([native_state["radius"]]),
        "other_agents_states": _as_float_array(native_state["other_agents_states"]),
    }
    vec_obs = np.array([], dtype=np.float32)
    for state in _SACADRL_STATE_ORDER:
        vec_obs = np.hstack([vec_obs, components[state].flatten()])
    return np.expand_dims(vec_obs, axis=0), components


def _inverse_rotate_to_ego(robot_heading: float, velocities_world: np.ndarray) -> np.ndarray:
    cos_h = np.cos(robot_heading)
    sin_h = np.sin(robot_heading)
    ego = np.zeros_like(velocities_world, dtype=np.float32)
    if velocities_world.size:
        ego[:, 0] = cos_h * velocities_world[:, 0] + sin_h * velocities_world[:, 1]
        ego[:, 1] = -sin_h * velocities_world[:, 0] + cos_h * velocities_world[:, 1]
    return ego


def _shared_ped_radius(other_states: np.ndarray, active_count: int) -> tuple[float, list[str]]:
    notes: list[str] = []
    if active_count <= 0:
        return 0.3, notes
    active = other_states[:active_count, 4]
    first = float(active[0])
    if not np.allclose(active, first):
        notes.append("active upstream other-agent rows use non-shared pedestrian radii")
    return first, notes


def _robot_sf_observation_from_native_state(
    native_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    other_states = _as_float_array(native_state["other_agents_states"])
    active_count = int(native_state["num_other_agents"])
    heading = float(native_state["heading_ego_frame"])
    dist_to_goal = float(native_state["dist_to_goal"])
    ped_radius, notes = _shared_ped_radius(other_states, active_count)

    ped_positions = np.zeros((other_states.shape[0], 2), dtype=np.float32)
    ped_positions[:, 0] = other_states[:, 0]
    ped_positions[:, 1] = other_states[:, 1]
    ped_vel_world = np.zeros((other_states.shape[0], 2), dtype=np.float32)
    ped_vel_world[:, 0] = other_states[:, 2]
    ped_vel_world[:, 1] = other_states[:, 3]
    ped_vel_ego = _inverse_rotate_to_ego(heading, ped_vel_world)

    observation = {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([native_state["radius"]], dtype=np.float32),
        },
        "goal": {
            "current": np.array([dist_to_goal, 0.0], dtype=np.float32),
            "next": np.zeros(2, dtype=np.float32),
        },
        "pedestrians": {
            "positions": ped_positions,
            "velocities": ped_vel_ego,
            "count": np.array([float(active_count)], dtype=np.float32),
            "radius": np.array([ped_radius], dtype=np.float32),
        },
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }
    return observation, notes


def _component_diffs(
    local_vec: np.ndarray, native_components: dict[str, np.ndarray]
) -> dict[str, float]:
    offset = 0
    diffs: dict[str, float] = {}
    for state in _SACADRL_STATE_ORDER:
        flat = native_components[state].astype(np.float32).flatten()
        local = local_vec.flatten()[offset : offset + flat.size]
        diffs[state] = float(np.max(np.abs(local - flat))) if flat.size else 0.0
        offset += flat.size
    return diffs


def _run_native_roundtrip_case(name: str, native_state: dict[str, Any]) -> CaseResult:
    expected_vec, native_components = _flatten_native_state(native_state)
    observation, notes = _robot_sf_observation_from_native_state(native_state)
    other_rows = int(np.asarray(native_state["other_agents_states"]).shape[0])
    config = SocNavPlannerConfig(
        sacadrl_pref_speed=float(native_state["pref_speed"]),
        sacadrl_max_other_agents=other_rows,
    )
    adapter = SACADRLPlannerAdapter(config=config, allow_fallback=True)
    local_vec, _pref_speed, _dist_to_goal = adapter._build_network_input(observation)
    component_diff = _component_diffs(local_vec, native_components)
    return CaseResult(
        name=name,
        verdict=(
            "parity reproduced"
            if float(np.max(np.abs(local_vec - expected_vec))) <= 1e-6
            else "material mismatch"
        ),
        max_abs_diff=float(np.max(np.abs(local_vec - expected_vec))),
        component_max_abs_diff=component_diff,
        notes=notes,
    )


def _run_controlled_rotated_case() -> CaseResult:
    robot_radius = 0.35
    ped_radius = 0.25
    rows = []
    for p_parallel, p_orth, v_parallel, v_orth in (
        (1.5, -0.2, 0.3, 0.1),
        (2.2, 0.6, -0.4, 0.2),
        (3.0, -0.7, 0.0, -0.3),
    ):
        rel = np.array([p_parallel, p_orth], dtype=np.float32)
        dist_2_other = float(np.linalg.norm(rel) - robot_radius - ped_radius)
        rows.append(
            [
                p_parallel,
                p_orth,
                v_parallel,
                v_orth,
                ped_radius,
                robot_radius + ped_radius,
                dist_2_other,
            ]
        )
    native_state = {
        "num_other_agents": 3,
        "dist_to_goal": 4.0,
        "heading_ego_frame": 0.6,
        "pref_speed": 0.8,
        "radius": robot_radius,
        "other_agents_states": rows,
    }
    return _run_native_roundtrip_case("controlled_rotated_multi_agent", native_state)


def _run_socnav_fusion_case() -> CaseResult:
    robot = SimpleNamespace(
        pose=(np.array([1.0, 1.0], dtype=np.float32), np.pi / 2),
        current_speed=np.array([0.0, 0.0], dtype=np.float32),
        config=SimpleNamespace(radius=0.3),
        state=SimpleNamespace(velocity_xy=np.array([0.0, 0.0], dtype=np.float32)),
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[2.0, 1.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0]], dtype=np.float32),
        robots=[robot],
        goal_pos=[np.array([3.0, 1.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    env_config = SimpleNamespace(
        predictive_foresight_enabled=False,
        sim_config=SimpleNamespace(ped_radius=0.25),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    observation = fusion.next_obs()
    adapter = SACADRLPlannerAdapter(
        config=SocNavPlannerConfig(sacadrl_pref_speed=1.0, sacadrl_max_other_agents=4),
        allow_fallback=True,
    )
    local_vec, _pref_speed, _dist_to_goal = adapter._build_network_input(observation)
    expected = {
        "num_other_agents": 1.0,
        "dist_to_goal": 2.0,
        "heading_ego_frame": float(np.pi / 2),
        "pref_speed": 1.0,
        "radius": 0.3,
        "other_agents_states": np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 0.25, 0.55, 0.45],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    expected_vec, native_components = _flatten_native_state(expected)
    component_diff = _component_diffs(local_vec, native_components)
    return CaseResult(
        name="robot_sf_socnav_observation_fusion",
        verdict=(
            "parity reproduced"
            if float(np.max(np.abs(local_vec - expected_vec))) <= 1e-6
            else "material mismatch"
        ),
        max_abs_diff=float(np.max(np.abs(local_vec - expected_vec))),
        component_max_abs_diff=component_diff,
        notes=["Robot SF observation builder emits pedestrian velocities in ego frame."],
    )


def run_probe(repo_root: Path, side_env_python: Path, timeout_seconds: int) -> ProbeReport:
    """Run the issue #663 parity probe."""
    repo_root = repo_root.resolve()
    side_env_python = (
        side_env_python if side_env_python.is_absolute() else Path.cwd() / side_env_python
    )
    _validate_paths(repo_root, side_env_python)

    upstream_result = _run_command(
        "upstream_live_native_state",
        [str(side_env_python), "-c", _upstream_live_payload_script()],
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    commands = [upstream_result]
    if upstream_result.returncode != 0:
        return ProbeReport(
            issue=ISSUE_NUMBER,
            repo_root=str(repo_root),
            repo_remote_url=UPSTREAM_REPO_URL,
            side_env_python=str(side_env_python),
            verdict="parity blocked",
            failure_stage=upstream_result.name,
            failure_summary=upstream_result.failure_summary,
            cases=[],
            source_contract=_extract_source_contract(),
            commands=commands,
        )

    try:
        upstream_payload = _parse_json_stdout(upstream_result)
    except ValueError as exc:
        return ProbeReport(
            issue=ISSUE_NUMBER,
            repo_root=str(repo_root),
            repo_remote_url=UPSTREAM_REPO_URL,
            side_env_python=str(side_env_python),
            verdict="parity blocked",
            failure_stage=upstream_result.name,
            failure_summary=str(exc),
            cases=[],
            source_contract=_extract_source_contract(),
            commands=commands,
        )
    live_case = _run_native_roundtrip_case(
        "live_upstream_two_agents_reset",
        upstream_payload["native_state"],
    )
    controlled_case = _run_controlled_rotated_case()
    fusion_case = _run_socnav_fusion_case()
    cases = [live_case, controlled_case, fusion_case]
    worst_diff = max(case.max_abs_diff for case in cases)
    verdict = (
        "adapter observation mapping reproduced in controlled cases"
        if worst_diff <= 1e-6
        else "adapter observation mapping has material mismatch"
    )
    return ProbeReport(
        issue=ISSUE_NUMBER,
        repo_root=str(repo_root),
        repo_remote_url=UPSTREAM_REPO_URL,
        side_env_python=str(side_env_python),
        verdict=verdict,
        failure_stage=None,
        failure_summary=None,
        cases=cases,
        source_contract=_extract_source_contract(),
        commands=commands,
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        f"# Issue {report.issue} SACADRL Observation Parity Probe",
        "",
        f"Verdict: `{report.verdict}`",
        "",
        "## Reference",
        f"- upstream repo: {report.repo_remote_url}",
        f"- upstream checkout: `{report.repo_root}`",
        f"- side-environment python: `{report.side_env_python}`",
        "",
        "## Case summary",
    ]
    for case in report.cases:
        lines.extend(
            [
                f"### {case.name}",
                f"- verdict: `{case.verdict}`",
                f"- max abs diff: `{case.max_abs_diff:.8f}`",
            ]
        )
        for component, value in case.component_max_abs_diff.items():
            lines.append(f"- {component}: `{value:.8f}`")
        for note in case.notes:
            lines.append(f"- note: {note}")
        lines.append("")
    if report.failure_summary:
        lines.extend(
            [
                "## Failure",
                f"- stage: `{report.failure_stage}`",
                f"- summary: {report.failure_summary}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            (
                "- The current SACADRL adapter now appears observation-faithful at the "
                "network-input level on the tested live upstream reset and controlled cases."
                if report.verdict == "adapter observation mapping reproduced in controlled cases"
                else "- The current SACADRL adapter still has a material observation-mapping mismatch."
            ),
            "- The Robot SF SocNav structured observation builder provides pedestrian velocities in ego frame, and the adapter rotates them back into world-frame before constructing the goal-frame CADRL state.",
            "- Remaining benchmark weakness should therefore be interpreted as a planner/scenario-performance question unless a later parity case contradicts this result.",
            "",
            "## Recommendation",
            (
                "- Keep `sacadrl` as benchmarkable CADRL-family evidence, but evaluate its quality on benchmark outcomes separately from adapter-faithfulness."
                if report.verdict == "adapter observation mapping reproduced in controlled cases"
                else "- Do not treat current `sacadrl` benchmark numbers as source-faithful until the mapping mismatch is fixed."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the parity probe CLI and persist JSON/Markdown artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_UPSTREAM_REPO_ROOT)
    parser.add_argument("--side-env-python", type=Path, default=DEFAULT_SIDE_ENV_PYTHON)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = run_probe(args.repo_root, args.side_env_python, args.timeout_seconds)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report.verdict, "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
