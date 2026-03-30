#!/usr/bin/env python3
"""Probe gym-collision-avoidance in an isolated side environment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ISSUE_NUMBER = 641
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_SIDE_ENV_PYTHON_RELATIVE = Path(
    "output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python"
)
UPSTREAM_REPO_URL = "https://github.com/mit-acl/gym-collision-avoidance"


@dataclass(frozen=True)
class CommandResult:
    """Structured result for a side-environment command."""

    name: str
    command: list[str]
    returncode: int | None
    failure_summary: str | None
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class ProbeReport:
    """Structured report for the side-environment reproduction attempt."""

    issue: int
    repo_root: str
    repo_remote_url: str
    side_env_python: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
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
        stdout_tail = (
            stdout_raw.decode("utf-8", errors="replace")
            if isinstance(stdout_raw, bytes)
            else stdout_raw
        )
        stderr_tail = (
            stderr_raw.decode("utf-8", errors="replace")
            if isinstance(stderr_raw, bytes)
            else stderr_raw
        )
        return CommandResult(
            name=name,
            command=command,
            returncode=None,
            failure_summary=f"command exceeded timeout ({timeout_seconds}s)",
            stdout_tail=stdout_tail[-4000:],
            stderr_tail=stderr_tail[-4000:],
        )

    failure_summary = None
    if result.returncode != 0:
        failure_summary = _detect_failure_summary(result.stdout, result.stderr)
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
    lowered = text.lower()
    if "failed to import tkagg backend" in lowered:
        return "upstream macOS visualization path forces TkAgg backend"
    if "no module named 'gym'" in lowered:
        return "missing python dependency: gym"
    if "no module named 'scipy'" in lowered:
        return "missing python dependency: scipy"
    if "ga3c_ready" in text:
        return "unexpected failure after GA3C initialization"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else "unknown failure"


def _validate_paths(repo_root: Path, side_env_python: Path) -> None:
    """Validate required upstream files and the side-environment interpreter.

    Args:
        repo_root: Path to the checked-out upstream `gym-collision-avoidance` repository.
        side_env_python: Path to the isolated side-environment interpreter to execute.

    Raises:
        FileNotFoundError: If the upstream README/example/test/policy files are missing, or if the
            side-environment interpreter path does not exist.
        PermissionError: If the side-environment interpreter exists but is not executable.
    """
    required_files = [
        repo_root / "README.md",
        repo_root / "gym_collision_avoidance/experiments/src/example.py",
        repo_root / "gym_collision_avoidance/tests/test_collision_avoidance.py",
        repo_root / "gym_collision_avoidance/envs/policies/GA3CCADRLPolicy.py",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream files: {', '.join(missing)}")
    if not side_env_python.exists():
        raise FileNotFoundError(f"Side-environment interpreter missing: {side_env_python}")
    if not side_env_python.is_file() or not os.access(side_env_python, os.X_OK):
        raise PermissionError(f"Side-environment interpreter is not executable: {side_env_python}")


def _versions_script() -> str:
    return (
        "import gym, json, tensorflow as tf; "
        "print(json.dumps({'gym': gym.__version__, 'tensorflow': tf.__version__}))"
    )


def _ga3c_script() -> str:
    return (
        "from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy; "
        "policy = GA3CCADRLPolicy(); "
        "policy.initialize_network(); "
        "print('ga3c_ready')"
    )


def _blocker_category(failure_summary: str | None) -> str:
    if failure_summary is None:
        return "success"
    lowered = failure_summary.lower()
    if "tkagg" in lowered:
        return "visualization_backend"
    if "missing python dependency: gym" in lowered:
        return "missing_gym"
    if "missing python dependency: scipy" in lowered:
        return "missing_scipy"
    if "command exceeded timeout" in lowered:
        return "timeout"
    if "unexpected failure after ga3c initialization" in lowered:
        return "post_ga3c_failure"
    return "other_runtime_failure"


def _extract_source_contract(failure_summary: str | None) -> dict[str, Any]:
    blocker_category = _blocker_category(failure_summary)
    blocker_by_category = {
        "success": "no remaining blocker in the isolated side environment",
        "visualization_backend": "visualization/backend rather than learned-policy runtime",
        "missing_gym": "missing python dependency: gym",
        "missing_scipy": "missing python dependency: scipy",
        "timeout": "timeout/hang during side-environment execution",
        "post_ga3c_failure": "post-initialization runtime failure after GA3C import",
        "other_runtime_failure": "other upstream/runtime failure in the isolated side environment",
    }
    goal_by_category = {
        "success": "legacy runtime isolated from main robot_sf_ll7 stack with source-harness parity demonstrated",
        "visualization_backend": "legacy runtime isolated from main robot_sf_ll7 stack with a remaining visualization/backend blocker",
        "missing_gym": "legacy runtime isolation attempt still blocked on restoring the gym dependency in the side environment",
        "missing_scipy": "legacy runtime isolation attempt still blocked on restoring the scipy dependency in the side environment",
        "timeout": "legacy runtime isolation attempt remains inconclusive because the side-environment commands timed out",
        "post_ga3c_failure": "legacy runtime isolation reaches learned-policy initialization but still blocks later in the upstream runtime",
        "other_runtime_failure": "legacy runtime isolation remains blocked by another upstream/runtime failure class",
    }
    return {
        "learned_policy": "GA3C_CADRL",
        "action_space": "speed_delta_heading",
        "checkpoint_family": "GA3C_CADRL/checkpoints/IROS18/network_01900000",
        "kinematics": "unicycle_like_speed_plus_delta_heading",
        "side_environment_goal": goal_by_category[blocker_category],
        "remaining_blocker_class": blocker_by_category[blocker_category],
    }


def _resolve_side_env_python(repo_root: Path, side_env_python: Path | None) -> Path:
    if side_env_python is None:
        return repo_root / DEFAULT_SIDE_ENV_PYTHON_RELATIVE
    return side_env_python if side_env_python.is_absolute() else Path.cwd() / side_env_python


def run_probe(repo_root: Path, side_env_python: Path | None, timeout_seconds: int) -> ProbeReport:
    """Run the side-environment reproduction probe."""
    repo_root = repo_root.resolve()
    side_env_python = _resolve_side_env_python(repo_root, side_env_python)
    _validate_paths(repo_root, side_env_python)

    commands = [
        _run_command(
            "side_env_versions",
            [str(side_env_python), "-c", _versions_script()],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "learned_policy_import",
            [str(side_env_python), "-c", _ga3c_script()],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "upstream_example",
            [str(side_env_python), "gym_collision_avoidance/experiments/src/example.py"],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "pytest_example_collection",
            [
                str(side_env_python),
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "-q",
                "gym_collision_avoidance/tests/test_collision_avoidance.py",
                "-k",
                "test_example_script",
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
    ]

    blocking = next(
        (
            result
            for result in commands
            if result.name
            in {
                "side_env_versions",
                "learned_policy_import",
                "upstream_example",
                "pytest_example_collection",
            }
            and result.returncode != 0
        ),
        None,
    )
    verdict = (
        "source harness reproducible in side environment"
        if blocking is None
        else "source harness still blocked"
    )
    return ProbeReport(
        issue=ISSUE_NUMBER,
        repo_root=str(repo_root),
        repo_remote_url=UPSTREAM_REPO_URL,
        side_env_python=str(side_env_python),
        verdict=verdict,
        failure_stage=None if blocking is None else blocking.name,
        failure_summary=None if blocking is None else blocking.failure_summary,
        source_contract=_extract_source_contract(
            None if blocking is None else blocking.failure_summary
        ),
        commands=commands,
    )


def _interpretation_lines(report: ProbeReport) -> list[str]:
    blocker_category = _blocker_category(report.failure_summary)
    if blocker_category == "success":
        return [
            "- The isolated side environment reproduces the learned-policy path and the upstream harness end-to-end.",
            "- Legacy main-runtime incompatibilities are no longer blocking source-harness parity for this recipe.",
            "- Full source-harness parity is now demonstrated in the isolated side environment.",
            "- A wrapper/parity issue is justified after preserving this exact runtime recipe.",
        ]
    if blocker_category == "visualization_backend":
        return [
            "- The legacy runtime problem from `#639` is substantially narrowed.",
            "- `gym`, TensorFlow, and the GA3C-CADRL learned-policy import path now reproduce successfully in the side environment.",
            "- The remaining blocker is the upstream macOS visualization path forcing `TkAgg`, not missing CADRL-family runtime dependencies.",
            "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
            "- The evidence now supports a narrower follow-up around non-visual or headless upstream reproduction, not a generic legacy-runtime rescue.",
        ]
    if blocker_category == "missing_gym":
        return [
            "- The isolated side environment still fails before source-harness parity because `gym` is missing.",
            "- The probe does not yet demonstrate learned-policy or full upstream harness reproducibility.",
            "- The remaining blocker is dependency restoration inside the side environment, not Robot SF integration logic.",
            "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
        ]
    if blocker_category == "missing_scipy":
        return [
            "- The isolated side environment still fails before source-harness parity because `scipy` is missing.",
            "- The probe does not yet demonstrate learned-policy or full upstream harness reproducibility.",
            "- The remaining blocker is dependency restoration inside the side environment, not Robot SF integration logic.",
            "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
        ]
    if blocker_category == "timeout":
        return [
            "- The side-environment probe remains inconclusive because one of the upstream commands timed out.",
            "- The current evidence does not isolate whether the blocker is environmental setup, runtime performance, or an upstream hang.",
            "- A longer timeout or narrower reproduction step is needed before claiming source-harness parity.",
            "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
        ]
    if blocker_category == "post_ga3c_failure":
        return [
            "- The side environment gets past learned-policy initialization but still fails later in the upstream runtime.",
            "- The remaining blocker is after GA3C-CADRL bootstrapping, not the basic dependency restore itself.",
            "- A narrower follow-up should inspect the post-initialization runtime path before claiming parity.",
            "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
        ]
    return [
        "- The side environment is still blocked by an upstream/runtime failure outside the known TkAgg path.",
        "- The failure summary should be treated as the current best evidence for the remaining blocker class.",
        "- A narrower follow-up should inspect the exact failing command before claiming parity.",
        "- A Robot SF wrapper is still not justified from full source-harness parity yet.",
    ]


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        "# gym-collision-avoidance Side-Environment Probe",
        "",
        f"Issue: `#{report.issue}`",
        f"Verdict: `{report.verdict}`",
        "",
        "## Summary",
        "",
        "This probe reruns the `gym-collision-avoidance` learned-policy source harness in a side "
        "environment that restores legacy `gym` without touching the main Robot SF runtime.",
        "",
        f"- upstream repo: `{report.repo_remote_url}`",
        f"- repo root: `{report.repo_root}`",
        f"- side-env python: `{report.side_env_python}`",
        "",
        "## Current result",
        "",
    ]
    if report.failure_stage is not None:
        lines.extend(
            [
                f"- first blocking stage: `{report.failure_stage}`",
                f"- failure summary: `{report.failure_summary}`",
            ]
        )
    else:
        lines.append("- all side-environment source-harness commands succeeded")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *_interpretation_lines(report),
        ]
    )

    lines.extend(
        [
            "",
            "## Extracted source contract",
            "",
            f"- learned policy: `{report.source_contract['learned_policy']}`",
            f"- action space: `{report.source_contract['action_space']}`",
            f"- checkpoint family: `{report.source_contract['checkpoint_family']}`",
            f"- kinematics: `{report.source_contract['kinematics']}`",
            f"- blocker class after side-env restore: `{report.source_contract['remaining_blocker_class']}`",
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
                "```bash",
                shlex.join(command.command),
                "```",
                "",
                f"- returncode: `{command.returncode}`",
                f"- failure: `{command.failure_summary}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the side-environment probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--side-env-python", type=Path, default=None)
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
