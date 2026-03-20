#!/usr/bin/env python3
"""Probe headless upstream reproduction for gym-collision-avoidance."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ISSUE_NUMBER = 659
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_SIDE_ENV_PYTHON = Path(
    "output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python"
)
UPSTREAM_REPO_URL = "https://github.com/mit-acl/gym-collision-avoidance"
EXAMPLE_PATH = "gym_collision_avoidance/experiments/src/example.py"


@dataclass(frozen=True)
class CommandResult:
    """Structured result for one headless reproduction stage."""

    name: str
    command: list[str]
    returncode: int | None
    failure_summary: str | None
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class ProbeReport:
    """Structured headless-reproduction report."""

    issue: int
    repo_root: str
    repo_remote_url: str
    side_env_python: str
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    source_contract: dict[str, Any]
    shims: list[str]
    commands: list[CommandResult]


def _validate_paths(repo_root: Path, side_env_python: Path) -> None:
    required = [repo_root / "README.md", repo_root / EXAMPLE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required upstream files: {', '.join(missing)}")
    if not side_env_python.exists():
        raise FileNotFoundError(f"Side-environment interpreter missing: {side_env_python}")


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
            stdout_tail=(exc.stdout or "")[-4000:],
            stderr_tail=(exc.stderr or "")[-4000:],
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
    if "module 'numpy' has no attribute 'bool8'" in lowered:
        return "legacy gym passive checker expects numpy.bool8"
    if "unable to download 'ffmpeg-osx-v3.2.4'" in lowered:
        return "legacy moviepy/imageio ffmpeg download path blocks headless reset animation"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else "unknown failure"


def _headless_script(repo_root: Path, *, patch_bool8: bool, skip_animation: bool) -> str:
    return f"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
if {patch_bool8!r} and not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import matplotlib as mpl
_original_use = mpl.use

def patched_use(backend, *args, **kwargs):
    if backend == 'TkAgg':
        return _original_use('Agg', *args, **kwargs)
    return _original_use(backend, *args, **kwargs)

mpl.use = patched_use
if {skip_animation!r}:
    import gym_collision_avoidance.envs.visualize as viz
    viz.animate_episode = lambda *args, **kwargs: None
from runpy import run_path
run_path({EXAMPLE_PATH!r}, run_name='__main__')
"""


def _extract_source_contract() -> dict[str, Any]:
    return {
        "learned_policy": "GA3C_CADRL",
        "action_space": "speed_delta_heading",
        "checkpoint_family": "GA3C_CADRL/checkpoints/IROS18/network_01900000",
        "kinematics": "unicycle_like_speed_plus_delta_heading",
        "headless_boundary": (
            "Planner logic is unchanged. Compatibility shims only neutralize visualization/backend "
            "assumptions and one NumPy alias mismatch in the legacy gym passive checker."
        ),
    }


def run_probe(repo_root: Path, side_env_python: Path, timeout_seconds: int) -> ProbeReport:
    """Run the staged headless upstream reproduction probe."""
    repo_root = repo_root.resolve()
    side_env_python = (
        side_env_python if side_env_python.is_absolute() else Path.cwd() / side_env_python
    )
    _validate_paths(repo_root, side_env_python)

    stages = [
        (
            "headless_tkagg_redirect",
            _headless_script(repo_root, patch_bool8=False, skip_animation=False),
        ),
        (
            "headless_plus_numpy_bool8_alias",
            _headless_script(repo_root, patch_bool8=True, skip_animation=False),
        ),
        (
            "headless_plus_numpy_bool8_alias_no_animation",
            _headless_script(repo_root, patch_bool8=True, skip_animation=True),
        ),
    ]
    commands = [
        _run_command(
            name,
            [str(side_env_python), "-c", script],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        )
        for name, script in stages
    ]

    final = commands[-1]
    verdict = (
        "headless source harness reproducible"
        if final.returncode == 0
        else "still blocked beyond visualization"
    )
    return ProbeReport(
        issue=ISSUE_NUMBER,
        repo_root=str(repo_root),
        repo_remote_url=UPSTREAM_REPO_URL,
        side_env_python=str(side_env_python),
        verdict=verdict,
        failure_stage=None if final.returncode == 0 else final.name,
        failure_summary=None if final.returncode == 0 else final.failure_summary,
        source_contract=_extract_source_contract(),
        shims=[
            "redirect upstream matplotlib.use('TkAgg') to Agg",
            "restore numpy.bool8 alias for gym 0.26 passive checker",
            "disable final animate_episode video export for headless reproduction",
        ],
        commands=commands,
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        "# gym-collision-avoidance Headless Reproduction Probe",
        "",
        f"Issue: `#{report.issue}`",
        f"Verdict: `{report.verdict}`",
        "",
        "## Summary",
        "",
        "This probe keeps the isolated side environment from `#641` and removes only non-planner "
        "headless blockers from the upstream example path.",
        "",
        f"- upstream repo: `{report.repo_remote_url}`",
        f"- repo root: `{report.repo_root}`",
        f"- side-env python: `{report.side_env_python}`",
        "",
        "## Compatibility shims",
        "",
    ]
    lines.extend(f"- {shim}" for shim in report.shims)
    lines.extend(["", "## Current result", ""])
    if report.verdict == "headless source harness reproducible":
        lines.extend(
            [
                "- the upstream example reaches `All agents finished!` / `Experiment over.`",
                "- the remaining workaround surface is explicitly limited to headless compatibility shims",
                "- a wrapper/parity issue is now justified if those shims stay explicit",
            ]
        )
    else:
        lines.extend(
            [
                f"- first blocking stage: `{report.failure_stage}`",
                f"- failure summary: `{report.failure_summary}`",
            ]
        )
    lines.extend(["", "## Extracted source contract", ""])
    lines.extend(
        [
            f"- learned policy: `{report.source_contract['learned_policy']}`",
            f"- action space: `{report.source_contract['action_space']}`",
            f"- checkpoint family: `{report.source_contract['checkpoint_family']}`",
            f"- kinematics: `{report.source_contract['kinematics']}`",
            f"- headless boundary: `{report.source_contract['headless_boundary']}`",
            "",
            "## Command results",
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
    """Run the headless-reproduction probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
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
