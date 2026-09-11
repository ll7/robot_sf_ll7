#!/usr/bin/env python3
"""Static, credential-safe audit for SLURM launcher scripts and wrappers.

Checks launch scripts and campaign wrappers for stale partitions, missing job
identities, timeouts, unsafe output paths, hardcoded private paths, unbounded
arrays, conflicting GPU requests, stale module commands, missing preflight,
and non-portable resume logic.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot-sf-slurm-launcher-audit.v1"

SBATCH_DIRECTIVE_RE = re.compile(
    r"^#SBATCH\s+(?:--(?P<long_key>[a-zA-Z0-9_-]+)(?:=(?P<long_val>\S+))?|-(?P<short_key>[a-zA-Z])(?:\s+(?P<short_val>\S+))?)"
)

PRIVATE_PATH_PATTERNS = [
    (re.compile(r"/hpc/(?:gpfs\d*/)?scratch/\S+"), "hardcoded HPC scratch path"),
    (
        re.compile(r"/home/(?!(?:runner|circleci)\b)[a-zA-Z0-9._-]{2,}/\S+"),
        "hardcoded home directory",
    ),
    (
        re.compile(r"/scratch/(?!(?:tmp|local)\b)[a-zA-Z0-9._-]{2,}/\S+"),
        "hardcoded scratch directory",
    ),
]

STALE_PARTITION_TOKENS = {
    "epyc-gpu-test",
    "test",
    "obsolete",
    "dead",
    "legacy",
}

SHORT_OPTION_MAP = {
    "J": "job-name",
    "p": "partition",
    "t": "time",
    "c": "cpus-per-task",
    "o": "output",
    "e": "error",
    "a": "array",
    "N": "nodes",
    "n": "ntasks",
}


@dataclass
class Violation:
    """Individual rule violation detected in a SLURM launcher."""

    code: str
    severity: str
    message: str


@dataclass
class LauncherAuditRow:
    """Audit summary and parsed directives for one SLURM launcher."""

    path: str
    job_name: str | None = None
    partition: str | None = None
    time_limit: str | None = None
    cpus_per_task: int | None = None
    gpus: str | None = None
    memory: str | None = None
    output_pattern: str | None = None
    error_pattern: str | None = None
    is_array: bool = False
    array_spec: str | None = None
    has_preflight: bool = False
    has_resume: bool = False
    has_harvest: bool = False
    status: str = "pass"
    violations: list[Violation] = field(default_factory=list)


def _parse_sbatch_line(line: str, directives: dict[str, str]) -> None:
    match = SBATCH_DIRECTIVE_RE.match(line.strip())
    if match:
        if match.group("long_key"):
            directives[match.group("long_key")] = match.group("long_val") or "true"
        elif match.group("short_key"):
            short = match.group("short_key")
            directives[SHORT_OPTION_MAP.get(short, short)] = match.group("short_val") or "true"


def _parse_wrapper_flags(
    line: str, directives: dict[str, str], wrapper_flags: dict[str, str]
) -> None:
    patterns = [
        (r'JOB_NAME=["\']?([a-zA-Z0-9_-]+)["\']?', "job-name"),
        (r'SLURM_TIME=["\']?([0-9:-]+)["\']?', "time"),
        (r'DEFAULT_PARTITION=["\']?([a-zA-Z0-9_-]+)["\']?', "partition"),
        (r'--output=([^\s"\']+)', "output"),
    ]
    for pattern, key in patterns:
        if not directives.get(key) and key not in wrapper_flags:
            m = re.search(pattern, line)
            if m:
                wrapper_flags[key] = m.group(1)


def _extract_directives_and_flags(lines: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Parse #SBATCH directives and common wrapper variables from script lines."""
    directives: dict[str, str] = {}
    wrapper_flags: dict[str, str] = {}

    for line in lines:
        _parse_sbatch_line(line, directives)
        _parse_wrapper_flags(line, directives, wrapper_flags)

    return directives, wrapper_flags


def _extract_resources(
    directives: dict[str, str], wrapper_flags: dict[str, str], content: str
) -> tuple[int | None, str | None, str | None, str | None, bool]:
    """Extract CPU, GPU, memory, and array configuration."""
    cpus_raw = directives.get("cpus-per-task") or directives.get("cpus")
    cpus = int(cpus_raw) if cpus_raw and cpus_raw.isdigit() else None

    gpus = directives.get("gres") or directives.get("gpus")
    if not gpus:
        gres_match = re.search(r"--gres=gpu:([0-9]+)", content)
        if gres_match:
            gpus = f"gpu:{gres_match.group(1)}"

    memory = directives.get("mem") or directives.get("mem-per-cpu")
    array_spec = directives.get("array")
    if not array_spec:
        arr_match = re.search(r"--array=([^\s\"']+)", content)
        if arr_match:
            array_spec = arr_match.group(1)
    is_array = bool(array_spec or "SLURM_ARRAY_TASK_ID" in content)

    return cpus, gpus, memory, array_spec, is_array


def _extract_capabilities(lines: list[str], content: str) -> tuple[bool, bool, bool]:
    """Detect preflight, portable resume, and harvest capabilities."""
    body_lines = [line for line in lines if not line.strip().startswith("#")]
    body_content = "\n".join(body_lines)

    has_preflight = bool(
        re.search(r"\bpreflight\b", body_content, re.IGNORECASE)
        or re.search(r"\bcanary\b", body_content, re.IGNORECASE)
        or "check_checkpoint" in body_content
        or "verify_slurm" in body_content
        or "if [[ ! -f" in body_content
        or "if [ ! -f" in body_content
        or "is required" in body_content.lower()
        or re.search(r":\?[^}]+required", content)
    )

    has_resume = bool("--resume-receipt" in content or "RESUME_RECEIPT" in content)
    has_harvest = bool(
        "harvest" in content.lower() or "rsync" in content.lower() or "finalize" in content.lower()
    )
    return has_preflight, has_resume, has_harvest


def _check_paths_and_identity(row: LauncherAuditRow, content: str) -> None:
    """Validate job name, timeout, output pattern, and path escape rules."""
    if not row.job_name:
        row.violations.append(
            Violation(
                "MISSING_JOB_NAME",
                "error",
                "Launcher lacks a declared #SBATCH --job-name or wrapper identity.",
            )
        )
    if not row.time_limit:
        row.violations.append(
            Violation(
                "MISSING_TIMEOUT",
                "error",
                "Launcher lacks a declared #SBATCH --time or wrapper timeout.",
            )
        )
    if not row.output_pattern:
        row.violations.append(
            Violation(
                "UNSAFE_OUTPUT_PATH",
                "error",
                "Missing #SBATCH --output directive; logs must target output/slurm/.",
            )
        )
    else:
        clean = row.output_pattern.replace('"', "").replace("'", "")
        prefix = "${ROBOT_SF_ARTIFACT_ROOT:-$REPO_ROOT/output}/slurm/"
        if not (clean.startswith("output/slurm/") or clean.startswith(prefix)):
            row.violations.append(
                Violation(
                    "UNSAFE_OUTPUT_PATH",
                    "error",
                    f"Log output '{row.output_pattern}' must write into output/slurm/.",
                )
            )

    for pattern, desc in PRIVATE_PATH_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            row.violations.append(
                Violation("HARDCODED_PRIVATE_PATH", "error", f"Contains {desc}: {matches[0]}")
            )


def _check_resources_and_runtime(row: LauncherAuditRow, content: str, body: str) -> None:
    """Validate partition, GPU configuration, array bounds, modules, and preflight."""
    if row.partition and row.partition.lower() in STALE_PARTITION_TOKENS:
        row.violations.append(
            Violation(
                "STALE_PARTITION",
                "error",
                f"Partition '{row.partition}' is a known stale or prohibited queue.",
            )
        )
    if (
        row.partition
        and "cpu" in row.partition.lower()
        and row.gpus
        and row.gpus not in ("0", "none", "gpu:0")
    ):
        row.violations.append(
            Violation(
                "CONFLICTING_GPU_REQUEST",
                "error",
                f"Partition '{row.partition}' is CPU-only but GPU requested ({row.gpus}).",
            )
        )
    if ("--gpus=0" in content or "GPUS=0" in content) and (
        "--gres=gpu:1" in content or "gpus=1" in content.lower()
    ):
        row.violations.append(
            Violation(
                "CONFLICTING_GPU_REQUEST",
                "error",
                "Contradictory GPU request detected (gpus=0 and gres=gpu:1).",
            )
        )

    if row.is_array and not row.array_spec:
        has_bounds = bool(
            re.search(r"(?:SLURM_ARRAY_TASK_ID|TASK_ID).*?(?:>=|-ge|<|>)", content)
            or "out of range" in content
        )
        if not has_bounds:
            row.violations.append(
                Violation(
                    "UNBOUNDED_ARRAY",
                    "error",
                    "Array script references task ID without bounds or range check.",
                )
            )

    if re.search(r"module\s+load\s+(?:anaconda|micromamba|condaforge)", body) or re.search(
        r"conda\s+activate", body
    ):
        row.violations.append(
            Violation(
                "STALE_MODULE_COMMAND",
                "warning",
                "References stale conda module activation instead of uv/.venv.",
            )
        )

    if re.search(r"--resume(?:-dir)?\s+/[a-zA-Z0-9_./-]+", content) or (
        re.search(r"--resume\b", content) and not row.has_resume
    ):
        row.violations.append(
            Violation(
                "NON_PORTABLE_RESUME",
                "error",
                "Non-portable resume command found with raw path instead of receipt.",
            )
        )

    if ("train_" in content or "run_benchmark" in content) and not row.has_preflight:
        row.violations.append(
            Violation(
                "MISSING_PREFLIGHT", "warning", "Workload script lacks preflight verification step."
            )
        )


def parse_slurm_launcher(path: Path) -> LauncherAuditRow:
    """Parse and audit one SLURM launcher or wrapper script."""
    content = path.read_text(encoding="utf-8", errors="replace")
    if "moved to the private operations overlay" in content:
        return LauncherAuditRow(
            path=path.as_posix(), job_name="private_ops_shim", status="pass", violations=[]
        )

    lines = content.splitlines()
    directives, wrapper_flags = _extract_directives_and_flags(lines)

    job_name = directives.get("job-name") or wrapper_flags.get("job-name")
    partition = directives.get("partition") or wrapper_flags.get("partition")
    time_limit = directives.get("time") or wrapper_flags.get("time")
    output_pattern = directives.get("output") or wrapper_flags.get("output")
    error_pattern = directives.get("error") or wrapper_flags.get("error")

    cpus, gpus, memory, array_spec, is_array = _extract_resources(
        directives, wrapper_flags, content
    )
    has_preflight, has_resume, has_harvest = _extract_capabilities(lines, content)

    row = LauncherAuditRow(
        path=path.as_posix(),
        job_name=job_name,
        partition=partition,
        time_limit=time_limit,
        cpus_per_task=cpus,
        gpus=gpus,
        memory=memory,
        output_pattern=output_pattern,
        error_pattern=error_pattern,
        is_array=is_array,
        array_spec=array_spec,
        has_preflight=has_preflight,
        has_resume=has_resume,
        has_harvest=has_harvest,
    )

    body_lines = [line for line in lines if not line.strip().startswith("#")]
    body = "\n".join(body_lines)

    _check_paths_and_identity(row, content)
    _check_resources_and_runtime(row, content, body)

    errors = [v for v in row.violations if v.severity == "error"]
    warnings = [v for v in row.violations if v.severity == "warning"]
    row.status = "fail" if errors else ("warn" if warnings else "pass")

    return row


def audit_launchers(paths: list[Path]) -> dict[str, Any]:
    """Audit multiple SLURM launcher scripts and compile an audit report."""
    rows: list[LauncherAuditRow] = []
    total_errors = 0
    total_warnings = 0
    blockers_by_code: dict[str, int] = {}

    for p in sorted(paths):
        row = parse_slurm_launcher(p)
        rows.append(row)
        for v in row.violations:
            if v.severity == "error":
                total_errors += 1
                blockers_by_code[v.code] = blockers_by_code.get(v.code, 0) + 1
            elif v.severity == "warning":
                total_warnings += 1

    passed = sum(1 for r in rows if r.status == "pass")
    failed = sum(1 for r in rows if r.status == "fail")
    warned = sum(1 for r in rows if r.status == "warn")

    return {
        "schema": SCHEMA_VERSION,
        "ok": total_errors == 0,
        "summary": {
            "total": len(rows),
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "blockers_by_code": blockers_by_code,
        },
        "launchers": [
            {
                **asdict(r),
                "violations": [asdict(v) for v in r.violations],
            }
            for r in rows
        ],
    }


def _collect_target_paths(file_args: list[Path]) -> list[Path]:
    """Resolve file arguments or discover default SLURM directory launchers."""
    paths: list[Path] = []
    if file_args:
        for f in file_args:
            if f.is_dir():
                paths.extend(f.glob("**/*.sl"))
                paths.extend(f.glob("**/*.sbatch"))
                paths.extend(f.glob("**/*.slurm"))
                paths.extend(f.glob("**/submit_*.sh"))
            elif f.is_file():
                paths.append(f)
    else:
        slurm_dir = Path("SLURM")
        if slurm_dir.is_dir():
            paths.extend(slurm_dir.glob("**/*.sl"))
            paths.extend(slurm_dir.glob("**/*.sbatch"))
            paths.extend(slurm_dir.glob("**/*.slurm"))
            paths.extend(slurm_dir.glob("submit_*.sh"))
    return sorted(set(paths))


def _print_text_report(report: dict[str, Any]) -> None:
    """Print human-readable CLI summary and launcher status."""
    summary = report["summary"]
    print(f"SLURM Launcher Audit ({SCHEMA_VERSION})")
    print(
        f"Total: {summary['total']} | Passed: {summary['passed']} | Warned: {summary['warned']} | Failed: {summary['failed']}"
    )
    if summary["blockers_by_code"]:
        print("\nBlockers by code:")
        for code, count in sorted(summary["blockers_by_code"].items()):
            print(f"  - {code}: {count}")

    print("\nDetails:")
    for launcher in report["launchers"]:
        icon = (
            "✅"
            if launcher["status"] == "pass"
            else ("⚠️" if launcher["status"] == "warn" else "❌")
        )
        print(
            f"{icon} {launcher['path']} (job={launcher['job_name']}, time={launcher['time_limit']}, part={launcher['partition']})"
        )
        for v in launcher["violations"]:
            print(f"    [{v['severity'].upper()}] {v['code']}: {v['message']}")


def main() -> int:
    """CLI entrypoint for launcher audit."""
    parser = argparse.ArgumentParser(
        description="Static audit for SLURM launchers and campaign wrappers."
    )
    parser.add_argument(
        "files", nargs="*", type=Path, help="Specific files or directories to audit."
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if any error violations are found."
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    parser.add_argument("--output", type=Path, help="Write JSON audit receipt to file.")
    args = parser.parse_args()

    paths = _collect_target_paths(args.files)
    report = audit_launchers(paths)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report)

    return 1 if args.check and not report["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
