#!/usr/bin/env python3
"""Probe the gym-collision-avoidance source harness and emit a blocked-or-runnable report."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_simple_assignment(text: str, attribute: str) -> Any | None:
    guarded_match = re.search(
        rf'if not hasattr\(self, "{attribute}"\):\s*\n\s*self\.{attribute}\s*=\s*(.+)',
        text,
    )
    if guarded_match:
        expr = guarded_match.group(1).split("#", 1)[0].strip()
        try:
            return ast.literal_eval(expr)
        except (ValueError, SyntaxError):
            if re.fullmatch(r"self\.MAX_NUM_AGENTS_IN_ENVIRONMENT\s*-\s*1", expr):
                max_agents = _extract_simple_assignment(text, "MAX_NUM_AGENTS_IN_ENVIRONMENT")
                if isinstance(max_agents, (int, float)):
                    return int(max_agents - 1)
            return None
    match = re.search(rf"self\.{attribute}\s*=\s*(.+)", text)
    if not match:
        return None
    expr = match.group(1).split("#", 1)[0].strip()
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return None


def _extract_default_policies(test_cases_path: Path) -> list[str]:
    text = _safe_read_text(test_cases_path)
    match = re.search(r"def get_testcase_two_agents\(policies=(\[[^\]]+\])\):", text)
    if not match:
        return []
    try:
        return list(ast.literal_eval(match.group(1)))
    except (ValueError, SyntaxError):
        return []


def _extract_discrete_actions(network_path: Path) -> int | None:
    text = _safe_read_text(network_path)
    match = re.search(r"Define\s+(\d+)\s+choices of actions", text)
    if match:
        return int(match.group(1))
    if "There are 11 choices of actions" in text:
        return 11
    return None


def _extract_contract(
    config_path: Path,
    test_cases_path: Path,
    ga3c_policy_path: Path,
    network_path: Path,
) -> dict[str, Any]:
    config_text = _safe_read_text(config_path)
    states_in_obs = _extract_simple_assignment(config_text, "STATES_IN_OBS") or []
    states_not_used = _extract_simple_assignment(config_text, "STATES_NOT_USED_IN_POLICY") or []
    dt = _extract_simple_assignment(config_text, "DT")
    max_agents = _extract_simple_assignment(config_text, "MAX_NUM_OTHER_AGENTS_OBSERVED")

    ga3c_text = _safe_read_text(ga3c_policy_path)
    action_semantics = None
    if "action = np.array([pref_speed*raw_action[0], raw_action[1]])" in ga3c_text:
        action_semantics = "speed_delta_heading"

    return {
        "example_default_policies": _extract_default_policies(test_cases_path),
        "observation_states_in_obs": states_in_obs,
        "observation_states_not_used_in_policy": states_not_used,
        "observation_encoding": "flattened_dict_obs_excluding_states_not_used_in_policy",
        "max_num_other_agents_observed": max_agents,
        "dt_seconds": dt,
        "learned_policy": "GA3C_CADRL",
        "action_space": action_semantics,
        "discrete_action_count": _extract_discrete_actions(network_path),
        "checkpoint_family": "GA3C_CADRL/checkpoints/IROS18/network_01900000",
        "kinematics": "unicycle_like_speed_plus_delta_heading",
    }


def _detect_failure_summary(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in [stderr, stdout] if part)
    missing_module = re.search(r"No module named '([^']+)'", combined)
    if missing_module:
        return f"missing python dependency: {missing_module.group(1)}"
    if combined.strip():
        return combined.strip().splitlines()[-1][:240]
    return "unknown failure"


@dataclass
class CommandResult:
    """Structured result for one attempted upstream command."""

    name: str
    command: list[str]
    returncode: int | None
    failure_summary: str | None
    stdout_tail: str
    stderr_tail: str


@dataclass
class ProbeReport:
    """Structured report for the gym-collision-avoidance source-harness probe."""

    issue: int
    repo_remote_url: str
    repo_root: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    timeout_seconds: int
    required_files: dict[str, str]
    source_contract: dict[str, Any]
    commands: list[CommandResult]


def _run_command(name: str, command: list[str], cwd: Path, timeout_seconds: int) -> CommandResult:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        failure_summary = (
            None if proc.returncode == 0 else _detect_failure_summary(proc.stdout, proc.stderr)
        )
        return CommandResult(
            name=name,
            command=command,
            returncode=proc.returncode,
            failure_summary=failure_summary,
            stdout_tail=proc.stdout[-4000:],
            stderr_tail=proc.stderr[-4000:],
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=command,
            returncode=None,
            failure_summary=f"command exceeded timeout ({timeout_seconds}s)",
            stdout_tail=(exc.stdout or "")[-4000:],
            stderr_tail=(exc.stderr or "")[-4000:],
        )


def _validate_required_files(repo_root: Path) -> dict[str, str]:
    required = {
        "readme": "README.md",
        "package_init": "gym_collision_avoidance/__init__.py",
        "config": "gym_collision_avoidance/envs/config.py",
        "test_cases": "gym_collision_avoidance/envs/test_cases.py",
        "example": "gym_collision_avoidance/experiments/src/example.py",
        "test_file": "gym_collision_avoidance/tests/test_collision_avoidance.py",
        "ga3c_policy": "gym_collision_avoidance/envs/policies/GA3CCADRLPolicy.py",
        "ga3c_network": "gym_collision_avoidance/envs/policies/GA3C_CADRL/network.py",
        "ga3c_checkpoint_meta": (
            "gym_collision_avoidance/envs/policies/GA3C_CADRL/checkpoints/IROS18/"
            "network_01900000.meta"
        ),
        "ga3c_checkpoint_index": (
            "gym_collision_avoidance/envs/policies/GA3C_CADRL/checkpoints/IROS18/"
            "network_01900000.index"
        ),
        "ga3c_checkpoint_data": (
            "gym_collision_avoidance/envs/policies/GA3C_CADRL/checkpoints/IROS18/"
            "network_01900000.data-00000-of-00001"
        ),
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, rel_path in required.items():
        candidate = repo_root / rel_path
        if not candidate.exists():
            missing.append(rel_path)
        else:
            resolved[key] = str(candidate)
    if missing:
        raise FileNotFoundError(
            f"gym-collision-avoidance checkout is missing required files: {', '.join(missing)}"
        )
    return resolved


def run_probe(repo_root: Path, timeout_seconds: int) -> ProbeReport:
    """Attempt the upstream example and learned-policy path, then summarize the result."""
    required_files = _validate_required_files(repo_root)
    source_contract = _extract_contract(
        Path(required_files["config"]),
        Path(required_files["test_cases"]),
        Path(required_files["ga3c_policy"]),
        Path(required_files["ga3c_network"]),
    )

    commands = [
        _run_command(
            "upstream_example",
            [sys.executable, "gym_collision_avoidance/experiments/src/example.py"],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "learned_policy_import",
            [
                sys.executable,
                "-c",
                (
                    "from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import "
                    "GA3CCADRLPolicy; policy = GA3CCADRLPolicy(); policy.initialize_network(); "
                    "print('ga3c_ready')"
                ),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "pytest_example_collection",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "gym_collision_avoidance/tests/test_collision_avoidance.py",
                "-k",
                "test_example_script",
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
    ]

    first_failure = next((result for result in commands if result.returncode != 0), None)
    verdict = "source harness reproducible" if first_failure is None else "source harness blocked"
    return ProbeReport(
        issue=639,
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        repo_root=str(repo_root),
        verdict=verdict,
        failure_stage=first_failure.name if first_failure else None,
        failure_summary=first_failure.failure_summary if first_failure else None,
        timeout_seconds=timeout_seconds,
        required_files=required_files,
        source_contract=source_contract,
        commands=commands,
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        "# gym-collision-avoidance Source Harness Probe",
        "",
        f"Verdict: `{report.verdict}`",
        "",
        "## Source root",
        "",
        f"- repo root: `{report.repo_root}`",
        f"- upstream repo: {report.repo_remote_url}",
        f"- timeout seconds: `{report.timeout_seconds}`",
        "",
        "## Commands attempted",
        "",
    ]
    for result in report.commands:
        lines.extend(
            [
                f"### {result.name}",
                "",
                f"- command: `{' '.join(result.command)}`",
                f"- return code: `{result.returncode}`",
                f"- failure summary: `{result.failure_summary or 'none'}`",
                "",
            ]
        )

    contract = report.source_contract
    lines.extend(
        [
            "## Extracted source contract",
            "",
            f"- example default policies: `{contract.get('example_default_policies', [])}`",
            f"- observation states in obs: `{contract.get('observation_states_in_obs', [])}`",
            f"- states not used in policy: `{contract.get('observation_states_not_used_in_policy', [])}`",
            f"- observation encoding: `{contract.get('observation_encoding', 'unknown')}`",
            f"- dt seconds: `{contract.get('dt_seconds', 'unknown')}`",
            f"- max other agents observed: `{contract.get('max_num_other_agents_observed', 'unknown')}`",
            f"- learned policy: `{contract.get('learned_policy', 'unknown')}`",
            f"- action space: `{contract.get('action_space', 'unknown')}`",
            f"- discrete action count: `{contract.get('discrete_action_count', 'unknown')}`",
            f"- checkpoint family: `{contract.get('checkpoint_family', 'unknown')}`",
            f"- kinematics: `{contract.get('kinematics', 'unknown')}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if report.verdict == "source harness reproducible":
        lines.append(
            "- The upstream example and learned-policy import both run in the current environment."
        )
        lines.append("- A Robot SF wrapper becomes justified as the next step.")
    else:
        lines.append(
            "- The source harness is blocked before planner-level parity can be evaluated in practice."
        )
        lines.append(
            "- Wrapper work is not yet justified; the next step would be a side-environment "
            "reproduction only."
        )
    return "\n".join(lines) + "\n"


def _write_optional(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("output/repos/gym-collision-avoidance"),
        help="Path to the checked-out gym-collision-avoidance repository.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main() -> int:
    """Run the source-harness probe and emit JSON plus optional Markdown artifacts."""
    args = _build_parser().parse_args()
    report = run_probe(args.repo_root, timeout_seconds=args.timeout_seconds)
    payload = json.dumps(asdict(report), indent=2)
    markdown = _render_markdown(report)
    _write_optional(args.output_json, payload + "\n")
    _write_optional(args.output_md, markdown)
    print(payload)
    return 0 if report.verdict == "source harness reproducible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
