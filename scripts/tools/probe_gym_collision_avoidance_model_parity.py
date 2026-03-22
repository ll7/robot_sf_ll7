#!/usr/bin/env python3
"""Probe model-level parity between upstream GA3C-CADRL and the local SACADRL wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ISSUE_NUMBER = 661
DEFAULT_TIMEOUT_SECONDS = 180
COMMAND_OUTPUT_TAIL_LENGTH = 4000
DEFAULT_SIDE_ENV_PYTHON = Path(
    "output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python"
)
UPSTREAM_REPO_URL = "https://github.com/mit-acl/gym-collision-avoidance"
EXAMPLE_PATH = "gym_collision_avoidance/experiments/src/example.py"
DEFAULT_UPSTREAM_REPO_ROOT = Path("output/repos/gym-collision-avoidance")
CHECKPOINT_PREFIX_RELATIVE = Path(
    "gym_collision_avoidance/envs/policies/GA3C_CADRL/checkpoints/IROS18/network_01900000"
)


@dataclass(frozen=True)
class CommandResult:
    """Structured result for one parity-stage command."""

    name: str
    command: list[str]
    returncode: int | None
    failure_summary: str | None
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class ProbeReport:
    """Structured parity report."""

    issue: int
    repo_root: str
    repo_remote_url: str
    side_env_python: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    parity_summary: dict[str, Any]
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
        return CommandResult(
            name=name,
            command=command,
            returncode=None,
            failure_summary=f"command exceeded timeout ({timeout_seconds}s)",
            stdout_tail=(exc.stdout or "")[-COMMAND_OUTPUT_TAIL_LENGTH:],
            stderr_tail=(exc.stderr or "")[-COMMAND_OUTPUT_TAIL_LENGTH:],
        )
    failure_summary = (
        None if result.returncode == 0 else _detect_failure_summary(result.stdout, result.stderr)
    )
    return CommandResult(
        name=name,
        command=command,
        returncode=result.returncode,
        failure_summary=failure_summary,
        stdout_tail=result.stdout[-COMMAND_OUTPUT_TAIL_LENGTH:],
        stderr_tail=result.stderr[-COMMAND_OUTPUT_TAIL_LENGTH:],
    )


def _detect_failure_summary(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if "no such file or directory" in lowered and "network_01900000.meta" in lowered:
        return "checkpoint prefix not found"
    if "failed to import tkagg backend" in lowered:
        return "headless compatibility shim missing or incomplete"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else "unknown failure"


def _validate_paths(repo_root: Path, side_env_python: Path) -> None:
    required = [repo_root / "README.md", repo_root / EXAMPLE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream files: {', '.join(missing)}")
    if not side_env_python.exists():
        raise FileNotFoundError(f"Side-environment interpreter missing: {side_env_python}")


def _upstream_payload_script(checkpoint_prefix_relative: Path = CHECKPOINT_PREFIX_RELATIVE) -> str:
    checkpoint_prefix_relative_str = checkpoint_prefix_relative.as_posix()
    return """
import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
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
from gym_collision_avoidance.envs import Config, test_cases as tc
import gym_collision_avoidance.envs.visualize as viz

viz.animate_episode = lambda *args, **kwargs: None
gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Example'
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
internal_obs = obs[1]
vec_obs = np.array([])
for state in Config.STATES_IN_OBS:
    if state not in Config.STATES_NOT_USED_IN_POLICY:
        vec_obs = np.hstack([vec_obs, internal_obs[state].flatten()])
vec_obs = np.expand_dims(vec_obs, axis=0)
policy = agents[1].policy
pred = policy.nn.predict_p(vec_obs)[0]
action_idx = int(np.argmax(pred))
raw_action = policy.possible_actions.actions[action_idx]
action = np.array([internal_obs['pref_speed'] * raw_action[0], raw_action[1]])
print(json.dumps({
    'vec_obs': vec_obs.tolist(),
    'upstream_probs': pred.tolist(),
    'upstream_argmax': action_idx,
    'upstream_raw_action': raw_action.tolist(),
    'upstream_final_action': action.tolist(),
    'upstream_actions': policy.possible_actions.actions.tolist(),
    'checkpoint_prefix': str((Path.cwd() / '__CHECKPOINT_PREFIX_RELATIVE__').resolve()),
    'obs_shape': list(vec_obs.shape),
    'states_used': [state for state in Config.STATES_IN_OBS if state not in Config.STATES_NOT_USED_IN_POLICY],
}))
""".replace("__CHECKPOINT_PREFIX_RELATIVE__", checkpoint_prefix_relative_str)


def _local_payload_script(payload_file: Path) -> str:
    return f"""
import json
from pathlib import Path
import numpy as np
from robot_sf.planner.socnav import _SACADRLModel, _sacadrl_actions
payload = json.loads(Path({str(payload_file)!r}).read_text())
obs = np.asarray(payload['vec_obs'], dtype=np.float32)
model = _SACADRLModel(Path(payload['checkpoint_prefix']))
local_probs = model.predict(obs)[0]
local_actions = _sacadrl_actions()
local_idx = int(np.argmax(local_probs))
print(json.dumps({{
    'local_argmax': local_idx,
    'local_raw_action': local_actions[local_idx].tolist(),
    'prob_max_abs_diff': float(np.max(np.abs(local_probs - np.asarray(payload['upstream_probs'], dtype=np.float32)))),
    'actions_max_abs_diff': float(np.max(np.abs(local_actions - np.asarray(payload['upstream_actions'], dtype=np.float32)))),
}}))
"""


def _parse_json_stdout(result: CommandResult) -> dict[str, Any]:
    lines = [line for line in result.stdout_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON payload found in command output for {result.name}")


def _extract_source_contract() -> dict[str, Any]:
    return {
        "learned_policy": "GA3C_CADRL",
        "action_space": "speed_delta_heading",
        "checkpoint_family": CHECKPOINT_PREFIX_RELATIVE.as_posix().split("envs/policies/", 1)[1],
        "kinematics": "unicycle_like_speed_plus_delta_heading",
        "parity_boundary": (
            "This issue checks model-level parity on native upstream observations. "
            "Robot SF observation mapping and benchmark performance remain separate questions."
        ),
    }


def run_probe(repo_root: Path, side_env_python: Path, timeout_seconds: int) -> ProbeReport:
    """Run the upstream-native versus local-model parity probe."""
    repo_root = repo_root.resolve()
    side_env_python = (
        side_env_python if side_env_python.is_absolute() else Path.cwd() / side_env_python
    )
    _validate_paths(repo_root, side_env_python)

    upstream_result = _run_command(
        "upstream_native_observation_and_policy",
        [str(side_env_python), "-c", _upstream_payload_script()],
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
            parity_summary={},
            source_contract=_extract_source_contract(),
            commands=commands,
        )

    payload_file = Path(
        "output/benchmarks/external/gym_collision_avoidance_model_parity/payload.json"
    )
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    upstream_payload = _parse_json_stdout(upstream_result)
    payload_file.write_text(json.dumps(upstream_payload, indent=2), encoding="utf-8")

    local_result = _run_command(
        "local_sacadrl_model_parity",
        ["uv", "run", "python", "-c", _local_payload_script(payload_file.resolve())],
        cwd=Path.cwd(),
        timeout_seconds=timeout_seconds,
    )
    commands.append(local_result)
    if local_result.returncode != 0:
        return ProbeReport(
            issue=ISSUE_NUMBER,
            repo_root=str(repo_root),
            repo_remote_url=UPSTREAM_REPO_URL,
            side_env_python=str(side_env_python),
            verdict="parity blocked",
            failure_stage=local_result.name,
            failure_summary=local_result.failure_summary,
            parity_summary=upstream_payload,
            source_contract=_extract_source_contract(),
            commands=commands,
        )

    local_payload = _parse_json_stdout(local_result)
    parity_summary = {
        "upstream_argmax": int(upstream_payload["upstream_argmax"]),
        "local_argmax": int(local_payload["local_argmax"]),
        "prob_max_abs_diff": float(local_payload["prob_max_abs_diff"]),
        "actions_max_abs_diff": float(local_payload["actions_max_abs_diff"]),
        "obs_shape": upstream_payload["obs_shape"],
        "states_used": upstream_payload["states_used"],
        "upstream_final_action": upstream_payload["upstream_final_action"],
        "local_raw_action": local_payload["local_raw_action"],
    }
    verdict = "native-model parity reproduced"
    if (
        parity_summary["upstream_argmax"] != parity_summary["local_argmax"]
        or parity_summary["prob_max_abs_diff"] > 1e-6
        or parity_summary["actions_max_abs_diff"] > 1e-6
    ):
        verdict = "material model mismatch"

    return ProbeReport(
        issue=ISSUE_NUMBER,
        repo_root=str(repo_root),
        repo_remote_url=UPSTREAM_REPO_URL,
        side_env_python=str(side_env_python),
        verdict=verdict,
        failure_stage=None,
        failure_summary=None,
        parity_summary=parity_summary,
        source_contract=_extract_source_contract(),
        commands=commands,
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        "# gym-collision-avoidance Model Parity Probe",
        "",
        f"Issue: `#{report.issue}`",
        f"Verdict: `{report.verdict}`",
        "",
        "## Summary",
        "",
        "This probe compares a live upstream GA3C-CADRL native observation against the current local "
        "Robot SF `_SACADRLModel` and action table.",
        "",
        f"- upstream repo: `{report.repo_remote_url}`",
        f"- repo root: `{report.repo_root}`",
        f"- side-env python: `{report.side_env_python}`",
        "",
    ]
    if report.failure_stage is not None:
        lines.extend(
            [
                "## Current result",
                "",
                f"- blocking stage: `{report.failure_stage}`",
                f"- failure summary: `{report.failure_summary}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Parity summary",
                "",
                f"- upstream argmax: `{report.parity_summary['upstream_argmax']}`",
                f"- local argmax: `{report.parity_summary['local_argmax']}`",
                f"- prob max abs diff: `{report.parity_summary['prob_max_abs_diff']:.8f}`",
                f"- action-table max abs diff: `{report.parity_summary['actions_max_abs_diff']:.8f}`",
                f"- native obs shape: `{report.parity_summary['obs_shape']}`",
                f"- states used: `{report.parity_summary['states_used']}`",
                "",
            ]
        )
        if report.verdict == "native-model parity reproduced":
            lines.extend(
                [
                    "- The current local `_SACADRLModel` matches upstream checkpoint inference on the tested native source observation.",
                    "- The remaining open question is Robot SF observation mapping and benchmark behavior, not model loading parity.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "- The current local SACADRL wrapper diverges from upstream even on native source observations.",
                    "- That means wrapper/parity work must fix model-level assumptions before benchmark evaluation.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Source contract",
            "",
            f"- learned policy: `{report.source_contract['learned_policy']}`",
            f"- action space: `{report.source_contract['action_space']}`",
            f"- checkpoint family: `{report.source_contract['checkpoint_family']}`",
            f"- kinematics: `{report.source_contract['kinematics']}`",
            f"- parity boundary: `{report.source_contract['parity_boundary']}`",
            "",
            "## Commands",
            "",
        ]
    )
    for command in report.commands:
        lines.extend(
            [
                f"### `{command.name}`",
                "",
                f"- returncode: `{command.returncode}`",
                f"- failure: `{command.failure_summary}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the model-parity probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_UPSTREAM_REPO_ROOT)
    parser.add_argument("--side-env-python", type=Path, default=DEFAULT_SIDE_ENV_PYTHON)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = run_probe(args.repo_root, args.side_env_python, args.timeout_seconds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report.verdict, "failure_summary": report.failure_summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
