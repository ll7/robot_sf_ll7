#!/usr/bin/env python3
"""Plan and safely submit queued SLURM training jobs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

QUEUE_SCHEMA_VERSION = "slurm-submission-queue.v1"
ALLOWED_STATUSES = {"planned", "ready_to_submit", "blocked", "superseded"}
REQUIRED_FIELDS = (
    "id",
    "status",
    "issue",
    "objective",
    "target",
    "resources",
    "seeds",
    "output_root",
    "log_path",
    "priority_reason",
    "auto_submit",
)


class QueueValidationError(ValueError):
    """Raised when the submission queue is malformed."""


class SubmissionBlockedError(RuntimeError):
    """Raised when a queued job is unsafe to submit."""


@dataclass(frozen=True)
class QueueEntry:
    """One planned SLURM training submission."""

    queue_id: str
    status: str
    issue: int | str
    objective: str
    config: str | None
    launcher: str | None
    target: dict[str, Any]
    resources: dict[str, Any]
    seeds: list[int | str]
    output_root: str
    log_path: str
    priority_reason: str
    auto_submit: bool
    job_name: str
    wrapper: str
    wrapper_args: list[str]
    sbatch_args: list[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], repo_root: Path) -> QueueEntry:
        """Validate and build one queue entry from YAML data."""
        missing = [field for field in REQUIRED_FIELDS if _is_missing(payload.get(field))]
        if missing:
            raise QueueValidationError(f"{payload.get('id', '<unknown>')}: missing {missing[0]}")

        queue_id = str(payload["id"])
        status = payload["status"]
        if status not in ALLOWED_STATUSES:
            raise QueueValidationError(
                f"{queue_id}: status must be one of {sorted(ALLOWED_STATUSES)}"
            )

        config = payload.get("config")
        launcher = payload.get("launcher")
        _validate_entry_paths(queue_id, config=config, launcher=launcher, repo_root=repo_root)

        target = _require_mapping(payload, "target", queue_id)
        if _is_missing(target.get("cluster")):
            raise QueueValidationError(f"{queue_id}: target.cluster is required")
        resources = _require_mapping(payload, "resources", queue_id)
        seeds = _require_list(payload, "seeds", queue_id)
        auto_submit = payload["auto_submit"]
        if not isinstance(auto_submit, bool):
            raise QueueValidationError(f"{queue_id}: auto_submit must be boolean")

        wrapper_default = "scripts/dev/sbatch_use_max_time.sh"
        wrapper_raw = payload.get("wrapper")
        if wrapper_raw is not None and not isinstance(wrapper_raw, str):
            raise QueueValidationError(f"{queue_id}: wrapper must be a string")
        wrapper = wrapper_raw or wrapper_default
        wrapper_path = repo_root / wrapper
        if not wrapper_path.is_file():
            raise QueueValidationError(f"{queue_id}: wrapper not found: {wrapper}")
        if not os.access(wrapper_path, os.X_OK):
            raise QueueValidationError(f"{queue_id}: wrapper is not executable: {wrapper}")
        sbatch_args = _optional_string_list(payload, "sbatch_args", queue_id)
        wrapper_args = _optional_string_list(payload, "wrapper_args", queue_id)

        return cls(
            queue_id=queue_id,
            status=status,
            issue=payload["issue"],
            objective=str(payload["objective"]),
            config=str(config) if config else None,
            launcher=str(launcher) if launcher else None,
            target=target,
            resources=resources,
            seeds=seeds,
            output_root=str(payload["output_root"]),
            log_path=str(payload["log_path"]),
            priority_reason=str(payload["priority_reason"]),
            auto_submit=auto_submit,
            job_name=str(payload.get("job_name") or f"gse-{payload['issue']}-{queue_id}"),
            wrapper=wrapper,
            wrapper_args=wrapper_args,
            sbatch_args=sbatch_args,
        )


@dataclass(frozen=True)
class SubmissionPlanResult:
    """Result of a dry-run or submit pass."""

    manifest_path: Path
    jobs: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


def load_queue(queue_path: Path, *, repo_root: Path) -> list[QueueEntry]:
    """Load and validate queued SLURM submissions."""
    try:
        payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QueueValidationError(f"cannot read queue: {exc}") from exc
    except yaml.YAMLError as exc:
        raise QueueValidationError(f"invalid queue YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise QueueValidationError("queue YAML must be a mapping")
    if payload.get("schema_version") != QUEUE_SCHEMA_VERSION:
        raise QueueValidationError(
            f"expected schema_version {QUEUE_SCHEMA_VERSION!r}, found {payload.get('schema_version')!r}"
        )
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        raise QueueValidationError("entries must be a list")

    entries: list[QueueEntry] = []
    seen: set[str] = set()
    for entry_payload in entries_payload:
        if not isinstance(entry_payload, dict):
            raise QueueValidationError("entry must be a mapping")
        entry = QueueEntry.from_mapping(entry_payload, repo_root)
        if entry.queue_id in seen:
            raise QueueValidationError(f"duplicate id: {entry.queue_id}")
        seen.add(entry.queue_id)
        entries.append(entry)
    return entries


def _validate_entry_paths(
    queue_id: str,
    *,
    config: Any,
    launcher: Any,
    repo_root: Path,
) -> None:
    """Validate that an entry names at least one existing execution surface."""
    if _is_missing(config) and _is_missing(launcher):
        raise QueueValidationError(f"{queue_id}: either config or launcher is required")
    if config is not None and not isinstance(config, str):
        raise QueueValidationError(f"{queue_id}: config must be a string")
    if launcher is not None and not isinstance(launcher, str):
        raise QueueValidationError(f"{queue_id}: launcher must be a string")
    if config and not (repo_root / config).is_file():
        raise QueueValidationError(f"{queue_id}: config not found: {config}")
    if launcher and not (repo_root / launcher).is_file():
        raise QueueValidationError(f"{queue_id}: launcher not found: {launcher}")


def _require_mapping(payload: dict[str, Any], field: str, queue_id: str) -> dict[str, Any]:
    """Return a required mapping field."""
    value = payload[field]
    if not isinstance(value, dict):
        raise QueueValidationError(f"{queue_id}: {field} must be a mapping")
    return value


def _require_list(payload: dict[str, Any], field: str, queue_id: str) -> list[Any]:
    """Return a required non-empty list field."""
    value = payload[field]
    if not isinstance(value, list) or not value:
        raise QueueValidationError(f"{queue_id}: {field} must be a non-empty list")
    return value


def _optional_string_list(payload: dict[str, Any], field: str, queue_id: str) -> list[str]:
    """Return an optional list of strings."""
    value = payload.get(field)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(arg, str) for arg in value):
        raise QueueValidationError(f"{queue_id}: {field} must be a list of strings")
    return value


def deterministic_experiment_id(entry: QueueEntry, *, commit: str) -> str:
    """Return a stable experiment id for duplicate detection and reporting."""
    seed_part = "seed" + "-".join(str(seed) for seed in entry.seeds)
    cluster = _slug(str(entry.target["cluster"]))
    return f"{_slug(entry.queue_id)}_{seed_part}_{commit[:7]}_{cluster}"


def plan_submissions(
    *,
    queue_path: Path,
    repo_root: Path,
    submit: bool,
    now: str | None = None,
) -> SubmissionPlanResult:
    """Dry-run or submit safe queue entries and write a manifest."""
    repo_root = repo_root.resolve()
    entries = load_queue(queue_path, repo_root=repo_root)
    git_state = _git_state(repo_root)
    timestamp = now or _timestamp()
    manifest_dir = repo_root / "output" / "slurm" / "submissions"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if submit and not _local_allows_slurm(repo_root):
        raise SubmissionBlockedError("local.machine.md does not allow_slurm_submission")

    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in entries:
        if entry.status != "ready_to_submit":
            skipped.append(
                {
                    "queue_id": entry.queue_id,
                    "status": "skipped",
                    "reason": f"entry status is {entry.status}",
                }
            )
            continue
        if submit and not entry.auto_submit:
            raise SubmissionBlockedError(f"{entry.queue_id}: auto_submit is false")
        duplicate_evidence = _local_duplicate_evidence(entry, repo_root=repo_root)
        if not duplicate_evidence:
            active_jobs = _query_active_jobs(repo_root, submit=submit)
            recent_jobs = _query_recent_jobs(repo_root, submit=submit)
            duplicate_evidence.extend(
                _scheduler_duplicate_evidence(
                    entry,
                    active_jobs=active_jobs,
                    recent_jobs=recent_jobs,
                )
            )
        if submit:
            _assert_submit_allowed(entry, duplicate_evidence=duplicate_evidence)

        command = _build_command(entry, repo_root=repo_root, dry_run=not submit)
        job = _job_manifest(entry, git_state=git_state, command=command, repo_root=repo_root)
        job["duplicate_check"] = duplicate_evidence
        if submit:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            job["stdout"] = completed.stdout.strip()
            job["stderr"] = completed.stderr.strip()
            job["exit_code"] = completed.returncode
            if completed.returncode != 0:
                raise SubmissionBlockedError(
                    f"{entry.queue_id}: wrapper failed with exit {completed.returncode}"
                )
            job["slurm_job_id"] = _parse_job_id(completed.stdout)
            job["status"] = "submitted"
        else:
            job["status"] = "dry_run"
            job["slurm_job_id"] = "not_submitted"
        jobs.append(job)

    manifest = {
        "schema_version": "slurm-submission-manifest.v1",
        "created_at": timestamp,
        "mode": "submit" if submit else "dry-run",
        "queue_path": str(queue_path),
        "repository": str(repo_root),
        "git": git_state,
        "jobs": jobs,
        "skipped": skipped,
    }
    manifest_path = manifest_dir / f"{timestamp}_training_submission_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return SubmissionPlanResult(manifest_path=manifest_path, jobs=jobs, skipped=skipped)


def _assert_submit_allowed(entry: QueueEntry, *, duplicate_evidence: list[str]) -> None:
    """Raise when a queue entry may not auto-submit."""
    if not entry.auto_submit:
        raise SubmissionBlockedError(f"{entry.queue_id}: auto_submit is false")
    if duplicate_evidence:
        raise SubmissionBlockedError(f"{entry.queue_id}: {'; '.join(duplicate_evidence)}")


def _job_manifest(
    entry: QueueEntry,
    *,
    git_state: dict[str, Any],
    command: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the manifest row for one queue entry."""
    return {
        "queue_id": entry.queue_id,
        "issue": entry.issue,
        "objective": entry.objective,
        "experiment_id": deterministic_experiment_id(entry, commit=git_state["commit"]),
        "branch": git_state["branch"],
        "commit": git_state["commit"],
        "dirty_tree": git_state["dirty"],
        "config": entry.config,
        "launcher": entry.launcher,
        "target": entry.target,
        "resources": entry.resources,
        "seeds": entry.seeds,
        "output_root": entry.output_root,
        "log_path": entry.log_path,
        "priority_reason": entry.priority_reason,
        "job_name": entry.job_name,
        "command": [str(_display_path(arg, repo_root)) for arg in command],
    }


def _build_command(entry: QueueEntry, *, repo_root: Path, dry_run: bool) -> list[str]:
    """Return the wrapper command for one queue entry."""
    command = [str(repo_root / entry.wrapper)]
    if dry_run:
        command.append("--dry-run")
    command.extend(entry.wrapper_args)
    command.extend(["--sbatch-arg", f"--job-name={entry.job_name}"])
    command.extend(entry.sbatch_args)
    if entry.launcher:
        command.append(entry.launcher)
    return command


def _local_duplicate_evidence(entry: QueueEntry, *, repo_root: Path) -> list[str]:
    """Return local filesystem reasons this entry appears to duplicate a run."""
    evidence: list[str] = []
    output_root = repo_root / entry.output_root
    if output_root.exists():
        evidence.append(f"output_root exists: {entry.output_root}")
    evidence.extend(_manifest_duplicate_evidence(entry, repo_root=repo_root))
    return evidence


def _manifest_duplicate_evidence(entry: QueueEntry, *, repo_root: Path) -> list[str]:
    """Return duplicate evidence from prior generated submission manifests."""
    manifest_dir = repo_root / "output" / "slurm" / "submissions"
    if not manifest_dir.exists():
        return []

    evidence: list[str] = []
    for manifest_path in sorted(manifest_dir.glob("*_training_submission_manifest.yaml")):
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(manifest, dict):
            continue
        jobs = manifest.get("jobs", [])
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if not isinstance(job, dict) or job.get("status") != "submitted":
                continue
            if job.get("queue_id") == entry.queue_id:
                evidence.append(f"prior manifest submitted queue id: {manifest_path.name}")
                return evidence
            if job.get("output_root") == entry.output_root:
                evidence.append(f"prior manifest used output_root: {manifest_path.name}")
                return evidence
    return evidence


def _parse_squeue_job_names(output: str) -> list[str]:
    """Extract job names from squeue whitespace-separated output (column 2)."""
    names: list[str] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in ("JOBID",):
            names.append(parts[1])
    return names


def _parse_sacct_job_names(output: str) -> list[str]:
    """Extract job names from sacct -P pipe-separated output (field 2)."""
    names: list[str] = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] not in ("JobID",):
            names.append(fields[1])
    return names


def _scheduler_duplicate_evidence(
    entry: QueueEntry,
    *,
    active_jobs: str,
    recent_jobs: str,
) -> list[str]:
    """Return scheduler reasons this entry appears to duplicate an existing run."""
    evidence: list[str] = []
    active_names = _parse_squeue_job_names(active_jobs)
    recent_names = _parse_sacct_job_names(recent_jobs)
    if entry.job_name and entry.job_name in active_names:
        evidence.append(f"active job with name {entry.job_name}")
    if entry.job_name and entry.job_name in recent_names:
        evidence.append(f"recent job with name {entry.job_name}")
    if entry.job_name.startswith("gse-"):
        for name in active_names:
            if name.startswith("gse-") and name != entry.job_name:
                evidence.append(f"another active gse job: {name}")
                break
    return evidence


def _query_active_jobs(repo_root: Path, *, submit: bool) -> str:
    """Return live SLURM queue text, or empty text in dry-run when unavailable."""
    if shutil.which("squeue") is None:
        if submit:
            raise SubmissionBlockedError("squeue is required for submit mode")
        return ""
    completed = subprocess.run(
        ["squeue", "--me", "--format=%i %j %T %P %M %l %R"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if submit:
            raise SubmissionBlockedError(f"squeue failed: {completed.stderr.strip()}")
        return ""
    return completed.stdout


def _query_recent_jobs(repo_root: Path, *, submit: bool) -> str:
    """Return recent accounting text, or empty text in dry-run when unavailable."""
    if shutil.which("sacct") is None:
        if submit:
            raise SubmissionBlockedError("sacct is required for submit mode")
        return ""
    completed = subprocess.run(
        [
            "sacct",
            "-u",
            os.environ.get("USER", ""),
            "--starttime",
            "now-3days",
            "--format=JobID,JobName%64,State,ExitCode,Start,End",
            "-P",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if submit:
            raise SubmissionBlockedError(f"sacct failed: {completed.stderr.strip()}")
        return ""
    return completed.stdout


def _local_allows_slurm(repo_root: Path) -> bool:
    """Return whether local.machine.md explicitly permits SLURM submission."""
    machine_context = repo_root / "local.machine.md"
    if not machine_context.exists():
        return False
    text = machine_context.read_text(encoding="utf-8")
    match = re.search(r"^\s*(?:-\s*)?allow_slurm_submission:\s*(\S+)\s*$", text, re.MULTILINE)
    if match is None:
        return False
    return match.group(1).lower() == "true"


def _git_state(repo_root: Path) -> dict[str, Any]:
    """Return branch, commit, and dirty tree summary for the manifest."""
    branch = _run_git(repo_root, ["git", "branch", "--show-current"]) or "unknown"
    commit = _run_git(repo_root, ["git", "rev-parse", "HEAD"]) or "unknown"
    dirty = _run_git(repo_root, ["git", "status", "--short"])
    return {
        "branch": branch,
        "commit": commit,
        "dirty": dirty.splitlines() if dirty else [],
    }


def _run_git(repo_root: Path, command: list[str]) -> str:
    """Run a git command and return stripped stdout, tolerating unavailable Git in tests."""
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _parse_job_id(stdout: str) -> str:
    """Extract the scheduler job id from sbatch output."""
    match = re.search(r"Submitted batch job\s+(\S+)", stdout)
    return match.group(1) if match else "unknown"


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _slug(value: str) -> str:
    """Return a conservative slug suitable for experiment ids."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "unknown"


def _display_path(value: str, repo_root: Path) -> str:
    """Display absolute wrapper paths as repo-relative when possible."""
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return value


def _is_missing(value: Any) -> bool:
    """Return whether a required value is absent or empty."""
    return value is None or value in ("", [], {})


def main(argv: list[str] | None = None) -> int:
    """Run the queue helper CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("experiments/submission_queue.yaml"),
        help="Path to the SLURM submission queue YAML.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Write a manifest without submitting.")
    mode.add_argument("--submit", action="store_true", help="Submit eligible queue entries.")
    args = parser.parse_args(argv)

    submit = args.submit
    try:
        result = plan_submissions(
            queue_path=args.queue,
            repo_root=Path.cwd(),
            submit=submit,
        )
    except (QueueValidationError, SubmissionBlockedError) as exc:
        print(f"submit_training_jobs: {exc}", file=sys.stderr)
        return 1

    print(f"manifest: {result.manifest_path}")
    print(f"jobs: {len(result.jobs)} skipped: {len(result.skipped)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
