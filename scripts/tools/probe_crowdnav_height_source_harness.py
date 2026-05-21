#!/usr/bin/env python3
"""Probe the CrowdNav HEIGHT source harness and emit a blocked-or-runnable report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def _repository_root() -> Path:
    """Resolve the local repository root."""
    return Path(__file__).resolve().parents[2]


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    """Load a Python module from a source path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_metadata_extract(extractor: Any, *args: Any) -> Any:
    """Run best-effort source metadata extraction."""
    try:
        return extractor(*args)
    except (Exception, SystemExit):
        return None


def _optional_attr(value: Any, name: str) -> Any:
    """Return an optional attribute without failing when the parent is ``None``."""
    return getattr(value, name, None) if value is not None else None


def _read_requirements(requirements_path: Path) -> list[str]:
    """Read non-comment requirement lines."""
    if not requirements_path.exists():
        return []
    lines: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _extract_contract(config_path: Path) -> dict[str, Any]:
    """Extract the HEIGHT source config contract when importable."""
    module = _load_module(config_path, f"height_config_{config_path.stem}")
    config_cls = getattr(module, "Config", None)
    if config_cls is None:
        return {}
    config = config_cls()
    env_config = getattr(config, "env", None)
    robot_config = getattr(config, "robot", None)
    sim_config = getattr(config, "sim", None)
    action_space_config = getattr(config, "action_space", None)
    return {
        "env_name": _optional_attr(env_config, "env_name"),
        "scenario": _optional_attr(env_config, "scenario"),
        "mode": _optional_attr(env_config, "mode"),
        "robot_policy": _optional_attr(robot_config, "policy"),
        "human_num": _optional_attr(sim_config, "human_num"),
        "static_obs": _optional_attr(sim_config, "static_obs"),
        "action_space_kinematics": _optional_attr(action_space_config, "kinematics"),
    }


def _detect_failure_summary(stderr: str) -> str:
    """Summarize source-harness stderr."""
    missing_module = re.search(r"No module named '([^']+)'", stderr)
    if missing_module:
        return f"missing python dependency: {missing_module.group(1)}"
    if stderr.strip():
        return stderr.strip().splitlines()[0][:240]
    return "unknown failure"


@dataclass
class ProbeReport:
    """Structured result for one HEIGHT source-harness probe."""

    issue: int
    repo_remote_url: str
    repo_root: str
    model_dir: str
    checkpoint: str
    checkpoint_status: str
    command: list[str]
    source_contract: dict[str, Any]
    requirements: list[str]
    verdict: str
    failure_stage: str | None
    failure_summary: str | None
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    timeout_seconds: int
    finished_at_utc: str
    total_runtime_sec: float


def run_probe(
    repo_root: Path,
    model_dir: str,
    checkpoint: str,
    timeout_seconds: int,
    *,
    issue: int = 1394,
    repo_remote_url: str = "https://github.com/Shuijing725/CrowdNav_HEIGHT",
) -> ProbeReport:
    """Run a fail-fast probe against the upstream HEIGHT source test entrypoint."""
    started_at = perf_counter()
    test_path = repo_root / "test.py"
    config_path = repo_root / "crowd_nav" / "configs" / "config.py"
    requirements_path = repo_root / "requirements.txt"
    model_path = repo_root / model_dir
    checkpoint_path = model_path / "checkpoints" / checkpoint

    required_paths = [test_path, config_path, requirements_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return ProbeReport(
            issue=issue,
            repo_remote_url=repo_remote_url,
            repo_root=str(repo_root),
            model_dir=model_dir,
            checkpoint=checkpoint,
            checkpoint_status="not_checked",
            command=[],
            source_contract={},
            requirements=[],
            verdict="source harness blocked",
            failure_stage="missing_assets",
            failure_summary=", ".join(missing),
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            timeout_seconds=timeout_seconds,
            finished_at_utc=datetime.now(UTC).isoformat(),
            total_runtime_sec=perf_counter() - started_at,
        )

    checkpoint_status = "present" if checkpoint_path.exists() else "missing_local_checkpoint"
    command = [
        sys.executable,
        "test.py",
        "--model_dir",
        model_dir,
        "--test_model",
        checkpoint,
        "--cpu",
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
        issue=issue,
        repo_remote_url=repo_remote_url,
        repo_root=str(repo_root),
        model_dir=model_dir,
        checkpoint=checkpoint,
        checkpoint_status=checkpoint_status,
        command=command,
        source_contract=_safe_metadata_extract(_extract_contract, config_path) or {},
        requirements=_read_requirements(requirements_path),
        verdict=verdict,
        failure_stage=failure_stage,
        failure_summary=failure_summary,
        returncode=returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        timeout_seconds=timeout_seconds,
        finished_at_utc=datetime.now(UTC).isoformat(),
        total_runtime_sec=perf_counter() - started_at,
    )


def _render_markdown(report: ProbeReport) -> str:
    """Render a Markdown report for the HEIGHT source-harness probe."""
    contract = report.source_contract
    lines = [
        "# CrowdNav HEIGHT Source Harness Probe",
        "",
        f"- Issue: `#{report.issue}`",
        f"- Repo remote: `{report.repo_remote_url}`",
        f"- Model dir: `{report.model_dir}`",
        f"- Checkpoint: `{report.checkpoint}`",
        f"- Checkpoint status: `{report.checkpoint_status}`",
        f"- Verdict: `{report.verdict}`",
        "",
        "## Invocation",
        "",
        "```bash",
        f"cd {shlex.quote(report.repo_root)}",
        shlex.join(report.command) if report.command else "# no command executed",
        "```",
        "",
        "## Source Contract",
        "",
        f"- `env_name`: `{contract.get('env_name')}`",
        f"- `scenario`: `{contract.get('scenario')}`",
        f"- `mode`: `{contract.get('mode')}`",
        f"- `robot_policy`: `{contract.get('robot_policy')}`",
        f"- `human_num`: `{contract.get('human_num')}`",
        f"- `static_obs`: `{contract.get('static_obs')}`",
        f"- `action_space_kinematics`: `{contract.get('action_space_kinematics')}`",
        "",
        "## Runtime Assumptions",
        "",
        f"- Requirements count: `{len(report.requirements)}`",
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
    """Run the HEIGHT source-harness probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repository_root() / "output" / "repos" / "CrowdNav_HEIGHT",
    )
    parser.add_argument("--model-dir", default="trained_models/HEIGHT")
    parser.add_argument("--checkpoint", default="237400.pt")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--issue", type=int, default=1394)
    parser.add_argument(
        "--repo-remote-url", default="https://github.com/Shuijing725/CrowdNav_HEIGHT"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    report = run_probe(
        repo_root=args.repo_root.resolve(),
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        timeout_seconds=args.timeout_seconds,
        issue=args.issue,
        repo_remote_url=args.repo_remote_url,
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
