"""Collect Slurm accounting evidence for CPU/GPU utilization investigations."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SSTAT_FORMAT = "JobID,AveCPU,MaxRSS,MaxVMSize,AveRSS"
_SACCT_FORMAT = (
    "JobID,JobName%40,Partition,State,Elapsed,AllocCPUS,AveCPU,TotalCPU,"
    "MaxRSS,ReqMem,MaxVMSize,ExitCode"
)


def _run_command(command: list[str]) -> dict[str, Any]:
    """Run one Slurm command and return a JSON-serializable result."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "missing_tool": command[0],
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def collect_job_reports(job_ids: list[str], *, include_seff: bool = True) -> dict[str, Any]:
    """Collect sstat/sacct/seff evidence for representative Slurm job IDs."""
    jobs: list[dict[str, Any]] = []
    for job_id in job_ids:
        commands = {
            "sstat": _run_command(["sstat", "-j", f"{job_id}.batch", f"--format={_SSTAT_FORMAT}"]),
            "sacct": _run_command(["sacct", "-j", job_id, f"--format={_SACCT_FORMAT}", "-P"]),
        }
        if include_seff:
            commands["seff"] = _run_command(["seff", job_id])
        jobs.append({"job_id": job_id, "commands": commands})
    return {
        "schema_version": "slurm_utilization_evidence.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "job_ids": job_ids,
        "jobs": jobs,
    }


def _markdown_command_block(command_name: str, result: dict[str, Any]) -> list[str]:
    """Render one command result for a Markdown report."""
    command = " ".join(str(part) for part in result.get("command", []))
    lines = [f"### `{command_name}`", "", f"- command: `{command}`"]
    if result.get("missing_tool"):
        lines.append(f"- missing_tool: `{result['missing_tool']}`")
    else:
        lines.append(f"- returncode: `{result.get('returncode')}`")
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    if stdout:
        lines.extend(["", "stdout:", "", "```text", stdout, "```"])
    if stderr:
        lines.extend(["", "stderr:", "", "```text", stderr, "```"])
    return lines


def _to_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from collected evidence."""
    lines = [
        "# Slurm Utilization Evidence",
        "",
        f"- schema_version: `{report['schema_version']}`",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- job_ids: `{', '.join(report['job_ids'])}`",
        "",
        "## Interpretation Checklist",
        "",
        "- Compare `AllocCPUS` against `AveCPU` and `TotalCPU` in `sacct`.",
        "- Compare `ReqMem` against `MaxRSS` and `MaxVMSize`.",
        "- Use `sstat` for live batch-step CPU and memory values while jobs are running.",
        "- Treat missing Slurm tools as an environment blocker, not utilization evidence.",
        "",
    ]
    for job in report["jobs"]:
        lines.extend([f"## Job `{job['job_id']}`", ""])
        for command_name, result in job["commands"].items():
            lines.extend(_markdown_command_block(command_name, result))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    """Write JSON and Markdown reports."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_to_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run Slurm utilization evidence collection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_ids", nargs="+", help="Slurm job IDs to inspect.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/slurm/slurm_utilization_evidence.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("output/slurm/slurm_utilization_evidence.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--no-seff", action="store_true", help="Skip optional seff collection.")
    args = parser.parse_args(argv)

    report = collect_job_reports(args.job_ids, include_seff=not args.no_seff)
    write_reports(report, output_json=args.output_json, output_md=args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
