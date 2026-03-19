#!/usr/bin/env python3
"""Probe the SoNIC source harness and emit a reproducible blocked-or-runnable report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_args_defaults(arguments_path: Path) -> dict[str, Any]:
    module = _load_module(arguments_path, f"sonic_args_{arguments_path.stem}")
    get_args = getattr(module, "get_args", None)
    if not callable(get_args):
        return {}
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(arguments_path)]
        args = get_args()
    finally:
        sys.argv = original_argv
    return vars(args)


def _extract_contract(config_path: Path) -> dict[str, Any]:
    module = _load_module(config_path, f"sonic_config_{config_path.stem}")
    config_cls = getattr(module, "Config", None)
    if config_cls is None:
        return {}
    config = config_cls()
    return {
        "robot_policy": getattr(getattr(config, "robot", object()), "policy", None),
        "human_policy": getattr(getattr(config, "humans", object()), "policy", None),
        "robot_sensor": getattr(getattr(config, "robot", object()), "sensor", None),
        "predict_method": getattr(getattr(config, "sim", object()), "predict_method", None),
        "action_kinematics": getattr(getattr(config, "action_space", object()), "kinematics", None),
        "env_use_wrapper": getattr(getattr(config, "env", object()), "use_wrapper", None),
    }


def _extract_docker_base_image(dockerfile_path: Path) -> str | None:
    for line in _safe_read_text(dockerfile_path).splitlines():
        stripped = line.strip()
        if stripped.startswith("FROM "):
            return stripped.split(maxsplit=1)[1]
    return None


def _read_requirements(requirements_path: Path) -> list[str]:
    lines: list[str] = []
    for line in _safe_read_text(requirements_path).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _default_checkpoint(checkpoints_dir: Path) -> str:
    candidates = sorted(path.name for path in checkpoints_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoints_dir}")
    return candidates[-1]


def _detect_failure_summary(stderr: str) -> str:
    missing_module = re.search(r"No module named '([^']+)'", stderr)
    if missing_module:
        return f"missing python dependency: {missing_module.group(1)}"
    if "Failed to initialize NVML" in stderr:
        return "gpu runtime unavailable"
    if stderr.strip():
        first_line = stderr.strip().splitlines()[0]
        return first_line[:240]
    return "unknown failure"


@dataclass
class ProbeReport:
    """Structured result for one SoNIC source-harness probe."""

    issue: int
    repo_remote_url: str
    repo_root: str
    model_name: str
    checkpoint: str
    command: list[str]
    docker_base_image: str | None
    source_contract: dict[str, Any]
    training_defaults: dict[str, Any]
    requirements_files: dict[str, list[str]]
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    timeout_seconds: int


def run_probe(
    repo_root: Path, model_name: str, checkpoint: str | None, timeout_seconds: int
) -> ProbeReport:
    """Run a fail-fast probe against the upstream SoNIC source test entrypoint."""
    test_path = repo_root / "test.py"
    model_root = repo_root / "trained_models" / model_name
    arguments_path = model_root / "arguments.py"
    config_path = model_root / "configs" / "config.py"
    checkpoints_dir = model_root / "checkpoints"
    dockerfile_path = repo_root / "Dockerfile"
    requirements_paths = {
        "gst_updated": repo_root / "gst_updated" / "requirements.txt",
        "python_rvo2": repo_root / "Python-RVO2" / "requirements.txt",
    }

    required_paths = [
        test_path,
        arguments_path,
        config_path,
        checkpoints_dir,
        dockerfile_path,
        *requirements_paths.values(),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return ProbeReport(
            issue=626,
            repo_remote_url="https://github.com/tasl-lab/SoNIC-Social-Nav",
            repo_root=str(repo_root),
            model_name=model_name,
            checkpoint=checkpoint or "",
            command=[],
            docker_base_image=None,
            source_contract={},
            training_defaults={},
            requirements_files={},
            verdict="source harness blocked",
            failure_stage="missing_assets",
            failure_summary=", ".join(missing),
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            timeout_seconds=timeout_seconds,
        )

    resolved_checkpoint = checkpoint or _default_checkpoint(checkpoints_dir)
    checkpoint_path = checkpoints_dir / resolved_checkpoint
    if not checkpoint_path.exists():
        return ProbeReport(
            issue=626,
            repo_remote_url="https://github.com/tasl-lab/SoNIC-Social-Nav",
            repo_root=str(repo_root),
            model_name=model_name,
            checkpoint=resolved_checkpoint,
            command=[],
            docker_base_image=_extract_docker_base_image(dockerfile_path),
            source_contract=_extract_contract(config_path),
            training_defaults=_load_args_defaults(arguments_path),
            requirements_files={
                name: _read_requirements(path) for name, path in requirements_paths.items()
            },
            verdict="source harness blocked",
            failure_stage="missing_checkpoint",
            failure_summary=str(checkpoint_path),
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            timeout_seconds=timeout_seconds,
        )

    command = [
        sys.executable,
        "test.py",
        "--model_dir",
        f"trained_models/{model_name}",
        "--test_model",
        resolved_checkpoint,
    ]

    try:
        proc = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        verdict = (
            "source harness reproducible" if proc.returncode == 0 else "source harness blocked"
        )
        failure_stage = None if proc.returncode == 0 else "source_entrypoint"
        failure_summary = None if proc.returncode == 0 else _detect_failure_summary(proc.stderr)
        returncode = proc.returncode
        stdout_tail = proc.stdout[-4000:]
        stderr_tail = proc.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        verdict = "source harness blocked"
        failure_stage = "timeout"
        failure_summary = f"entrypoint exceeded timeout ({timeout_seconds}s)"
        returncode = None
        stdout_tail = (exc.stdout or "")[-4000:]
        stderr_tail = (exc.stderr or "")[-4000:]

    return ProbeReport(
        issue=626,
        repo_remote_url="https://github.com/tasl-lab/SoNIC-Social-Nav",
        repo_root=str(repo_root),
        model_name=model_name,
        checkpoint=resolved_checkpoint,
        command=command,
        docker_base_image=_extract_docker_base_image(dockerfile_path),
        source_contract=_extract_contract(config_path),
        training_defaults=_load_args_defaults(arguments_path),
        requirements_files={
            name: _read_requirements(path) for name, path in requirements_paths.items()
        },
        verdict=verdict,
        failure_stage=failure_stage,
        failure_summary=failure_summary,
        returncode=returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        timeout_seconds=timeout_seconds,
    )


def _render_markdown(report: ProbeReport) -> str:
    contract = report.source_contract
    defaults = report.training_defaults
    lines = [
        "# SoNIC Source Harness Probe",
        "",
        f"- Issue: `#{report.issue}`",
        f"- Repo remote: `{report.repo_remote_url}`",
        f"- Model: `{report.model_name}`",
        f"- Checkpoint: `{report.checkpoint}`",
        f"- Verdict: `{report.verdict}`",
        "",
        "## Invocation",
        "",
        "```bash",
        "cd " + report.repo_root,
        " ".join(report.command) if report.command else "# no command executed",
        "```",
        "",
        "## Source Contract",
        "",
        f"- `robot_policy`: `{contract.get('robot_policy')}`",
        f"- `human_policy`: `{contract.get('human_policy')}`",
        f"- `robot_sensor`: `{contract.get('robot_sensor')}`",
        f"- `predict_method`: `{contract.get('predict_method')}`",
        f"- `action_kinematics`: `{contract.get('action_kinematics')}`",
        f"- `env_use_wrapper`: `{contract.get('env_use_wrapper')}`",
        f"- `env_name` default from arguments: `{defaults.get('env_name')}`",
        "",
        "## Runtime Assumptions",
        "",
        f"- Docker base image: `{report.docker_base_image}`",
        f"- `gst_updated` requirements count: `{len(report.requirements_files.get('gst_updated', []))}`",
        f"- `Python-RVO2` requirements count: `{len(report.requirements_files.get('python_rvo2', []))}`",
        "",
        "## Probe Result",
        "",
        f"- `failure_stage`: `{report.failure_stage}`",
        f"- `failure_summary`: `{report.failure_summary}`",
        f"- `returncode`: `{report.returncode}`",
        f"- `timeout_seconds`: `{report.timeout_seconds}`",
        "",
        "## stderr tail",
        "",
        "```text",
        report.stderr_tail or "(empty)",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the SoNIC source-harness probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repository_root() / "output" / "repos" / "SoNIC-Social-Nav",
        help="Path to the upstream SoNIC checkout.",
    )
    parser.add_argument(
        "--model-name",
        default="SoNIC_GST",
        help="Model directory under trained_models/ to probe.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint file name. Defaults to the latest .pt file in checkpoints/.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for the source entrypoint probe.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Optional Markdown report path."
    )
    args = parser.parse_args()

    report = run_probe(
        repo_root=args.repo_root.resolve(),
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        timeout_seconds=args.timeout_seconds,
    )
    payload = asdict(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if report.verdict == "source harness reproducible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
