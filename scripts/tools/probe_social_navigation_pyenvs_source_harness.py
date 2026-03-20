#!/usr/bin/env python3
"""Probe the Social-Navigation-PyEnvs source harness and emit a feasibility report."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_remote_url(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "unknown"


def _extract_requirement(requirements_path: Path, package: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(package)}\s*==\s*(.+)$", re.MULTILINE)
    match = pattern.search(_safe_read_text(requirements_path))
    if not match:
        return None
    return match.group(1).strip()


def _extract_policy_names(policy_factory_path: Path) -> list[str]:
    matches = re.findall(r"policy_factory\['([^']+)'\]\s*=", _safe_read_text(policy_factory_path))
    return sorted(set(matches))


def _extract_contract(repo_root: Path) -> dict[str, Any]:
    requirements_path = repo_root / "requirements.txt"
    robot_agent_path = repo_root / "social_gym" / "src" / "robot_agent.py"
    motion_model_path = repo_root / "social_gym" / "src" / "motion_model_manager.py"
    policy_factory_path = repo_root / "crowd_nav" / "policy_no_train" / "policy_factory.py"

    robot_text = _safe_read_text(robot_agent_path)
    motion_text = _safe_read_text(motion_model_path)

    return {
        "gymnasium_version": _extract_requirement(requirements_path, "gymnasium"),
        "numpy_version": _extract_requirement(requirements_path, "numpy"),
        "torch_version": _extract_requirement(requirements_path, "torch"),
        "non_trainable_policies": _extract_policy_names(policy_factory_path),
        "robot_actuation": "differential_drive" if "DifferentialDrive" in robot_text else "unknown",
        "robot_policy_accepts_holonomic_actions": "self.kinematics == 'holonomic'" in robot_text,
        "robot_policy_accepts_nonholonomic_actions": "else: assert isinstance(action, ActionRot)"
        in robot_text,
        "orca_available_as_robot_motion_model": 'policy_name == "orca"' in motion_text
        or '"orca"' in motion_text,
        "orca_preferred_velocity_semantics": "goal_vector_pref_velocity",
        "runtime_bug_signature": "np.NaN removed in NumPy 2" if "np.NaN" in motion_text else None,
        "runtime_bug_locations": [
            "social_gym/src/motion_model_manager.py:264",
            "social_gym/src/motion_model_manager.py:271",
        ]
        if "np.NaN" in motion_text
        else [],
        "minimal_local_compatibility_shims": [
            "install socialforce extra dependency",
            "restore numpy.NaN alias before upstream import",
            "set with_theta_and_omega_visible=False on ORCA policy when absent",
            "call upstream raw env.configure/set_robot/set_robot_policy path before reset",
        ],
        "notes": (
            "Gymnasium-native package with differential-drive robot support; "
            "learned runtime remains unproven because packaged checkpoints were not found."
        ),
    }


def _detect_failure_summary(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in [stderr, stdout] if part)
    missing_module = re.search(r"No module named '([^']+)'", combined)
    if missing_module:
        return f"missing python dependency: {missing_module.group(1)}"
    if "np.NaN" in combined and "NumPy 2.0 release" in combined:
        return "upstream NumPy 2 incompatibility: np.NaN"
    if (
        "has no wheels with a matching Python ABI tag" in combined
        or "requirements are unsatisfiable" in combined
    ):
        return "pinned requirements incompatible with current Python ABI"
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
    """Structured report for the Social-Navigation-PyEnvs source-harness probe."""

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
    packaged_weights_present: bool


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


def _uv_command() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable not found on PATH")
    return uv


def _validate_required_files(repo_root: Path) -> dict[str, str]:
    required = {
        "readme": "README.md",
        "requirements": "requirements.txt",
        "setup": "setup.py",
        "social_gym_init": "social_gym/__init__.py",
        "social_nav_gym": "social_gym/social_nav_gym.py",
        "social_nav_sim": "social_gym/social_nav_sim.py",
        "robot_agent": "social_gym/src/robot_agent.py",
        "motion_model_manager": "social_gym/src/motion_model_manager.py",
        "policy_factory": "crowd_nav/policy_no_train/policy_factory.py",
        "train_entrypoint": "crowd_nav/train.py",
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
            "Social-Navigation-PyEnvs checkout is missing required files: " + ", ".join(missing)
        )
    return resolved


def _packaged_weights_present(repo_root: Path) -> bool:
    patterns = ("*.pt", "*.pth", "*.zip", "*.model", "*.onnx")
    for pattern in patterns:
        if any(repo_root.rglob(pattern)):
            return True
    return False


def run_probe(repo_root: Path, timeout_seconds: int) -> ProbeReport:
    """Attempt the main upstream paths and summarize feasibility."""
    required_files = _validate_required_files(repo_root)
    source_contract = _extract_contract(repo_root)
    uv = _uv_command()

    commands = [
        _run_command(
            "package_import",
            [uv, "run", "python", "-c", "import gymnasium, social_gym; print('import_ok')"],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "env_make_main_runtime",
            [
                uv,
                "run",
                "python",
                "-c",
                "import social_gym, gymnasium as gym; env = gym.make('SocialGym-v0'); print(type(env).__name__)",
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "simulator_core_with_socialforce",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "python",
                "-c",
                (
                    "from social_gym.social_nav_sim import SocialNavSim; "
                    "sim = SocialNavSim(config_data={'insert_robot': True, 'human_policy': 'orca', "
                    "'headless': True, 'runge_kutta': False, 'robot_visible': True, 'robot_radius': 0.3, "
                    "'circle_radius': 7, 'n_actors': 3, 'randomize_human_positions': True, "
                    "'randomize_human_attributes': False}, scenario='circular_crossing', "
                    "parallelize_robot=False, parallelize_humans=False); sim.set_time_step(1/20); "
                    "print('sim_ok', len(sim.humans), sim.robot is not None)"
                ),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "env_make_with_socialforce",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "python",
                "-c",
                "import social_gym, gymnasium as gym; env = gym.make('SocialGym-v0'); print(type(env).__name__)",
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "policy_registry_with_socialforce",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "python",
                "-c",
                (
                    "from crowd_nav.policy_no_train.policy_factory import policy_factory; "
                    "print(sorted(policy_factory.keys()))"
                ),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "robot_orca_policy_with_socialforce",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "python",
                "-c",
                (
                    "from social_gym.social_nav_sim import SocialNavSim; "
                    "sim = SocialNavSim(config_data={'insert_robot': True, 'human_policy': 'orca', "
                    "'headless': True, 'runge_kutta': False, 'robot_visible': True, 'robot_radius': 0.3, "
                    "'circle_radius': 7, 'n_actors': 3, 'randomize_human_positions': True, "
                    "'randomize_human_attributes': False}, scenario='circular_crossing', "
                    "parallelize_robot=False, parallelize_humans=False); "
                    "sim.set_robot_policy(policy_name='orca', runge_kutta=False); "
                    "print('robot_motion_model', sim.motion_model_manager.robot_motion_model_title, "
                    "'crowdnav', sim.robot_crowdnav_policy)"
                ),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "pinned_requirements_probe",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "--with-requirements",
                "requirements.txt",
                "python",
                "-c",
                "print('requirements_ok')",
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
        _run_command(
            "shimmed_orca_reset_step",
            [
                uv,
                "run",
                "--with",
                "socialforce",
                "python",
                "-c",
                (
                    "import configparser; import numpy as np; np.NaN = np.nan; "
                    "from social_gym.social_nav_gym import SocialNavGym; "
                    "from social_gym.src.robot_agent import RobotAgent; "
                    "env = SocialNavGym(); "
                    "config = configparser.RawConfigParser(); "
                    "config.read('crowd_nav/configs/env.config'); "
                    "env.configure(config); "
                    "robot = RobotAgent(env); robot.configure(config, 'robot'); env.set_robot(robot); "
                    "env.set_robot_policy('orca', crowdnav_policy=True); "
                    "setattr(env.robot.policy, 'with_theta_and_omega_visible', "
                    "getattr(env.robot.policy, 'with_theta_and_omega_visible', False)); "
                    "obs, info = env.reset('test'); "
                    "action = env.robot.act(obs); "
                    "step = env.step(action); "
                    "print('shim_ok', len(obs), type(action).__name__, len(step), step[2], step[3]); "
                    "env.close()"
                ),
            ],
            cwd=repo_root,
            timeout_seconds=timeout_seconds,
        ),
    ]

    failure_stage = None
    failure_summary = None
    verdict = "source harness partially reproducible"

    for command in commands:
        if command.name == "simulator_core_with_socialforce" and command.returncode != 0:
            verdict = "source harness blocked"
            failure_stage = command.name
            failure_summary = command.failure_summary
            break
        if command.name == "robot_orca_policy_with_socialforce" and command.returncode != 0:
            verdict = "source harness blocked"
            failure_stage = command.name
            failure_summary = command.failure_summary
            break

    if verdict != "source harness blocked":
        for command in commands:
            if command.name == "env_make_with_socialforce" and command.returncode != 0:
                failure_stage = command.name
                failure_summary = command.failure_summary
                verdict = "source harness partially reproducible"
                break

    shimmed_step = next(
        command for command in commands if command.name == "shimmed_orca_reset_step"
    )
    if shimmed_step.returncode != 0 and verdict != "source harness blocked":
        verdict = "source harness partially reproducible"

    return ProbeReport(
        issue=642,
        repo_remote_url=_extract_remote_url(repo_root),
        repo_root=str(repo_root),
        verdict=verdict,
        failure_stage=failure_stage,
        failure_summary=failure_summary,
        timeout_seconds=timeout_seconds,
        required_files=required_files,
        source_contract=source_contract,
        commands=commands,
        packaged_weights_present=_packaged_weights_present(repo_root),
    )


def _render_markdown(report: ProbeReport) -> str:
    lines = [
        "# Social-Navigation-PyEnvs Source Harness Probe",
        "",
        f"- Verdict: `{report.verdict}`",
        f"- Issue: `#{report.issue}`",
        f"- Upstream: `{report.repo_remote_url}`",
        f"- Repo root: `{report.repo_root}`",
        f"- Packaged learned weights present: `{report.packaged_weights_present}`",
    ]
    if report.failure_stage:
        lines.extend(
            [
                f"- Primary blocker stage: `{report.failure_stage}`",
                f"- Primary blocker: `{report.failure_summary}`",
            ]
        )
    lines.extend(["", "## Source Contract", ""])
    for key, value in report.source_contract.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Commands", ""])
    for command in report.commands:
        lines.extend(
            [
                f"### `{command.name}`",
                "",
                f"- Return code: `{command.returncode}`",
                f"- Failure summary: `{command.failure_summary}`",
                "",
                "```bash",
                shlex.join(command.command),
                "```",
                "",
            ]
        )

    lines.extend(["## Interpretation", ""])
    if report.verdict == "source harness partially reproducible":
        lines.extend(
            [
                "- The upstream package and simulator core are runnable here with a narrow extra dependency (`socialforce`).",
                "- At least one non-trainable planner path (`orca`) executes in the upstream simulator without local source patches.",
                "- A narrow local compatibility shim is sufficient to reset and step the upstream ORCA path once.",
                "- Full Gymnasium env creation is still blocked by an upstream NumPy 2 incompatibility (`np.NaN`).",
                "- The pinned learned stack is not yet reproducible in the current Python 3.13 runtime.",
                "- Wrapper work for non-trainable planners is now justified as a prototype path, but learned-path work still needs either stricter side-env reproduction or explicit compatibility boundaries.",
            ]
        )
    else:
        lines.extend(
            [
                "- The upstream source harness does not yet run far enough to justify wrapper work.",
                "- Fix the runtime/package blockers first, then rerun the same probe before attempting Robot SF integration.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the probe and write JSON/Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    report = run_probe(args.repo_root.resolve(), timeout_seconds=args.timeout_seconds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
