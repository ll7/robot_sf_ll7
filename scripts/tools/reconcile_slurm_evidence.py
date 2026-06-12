#!/usr/bin/env python3
"""Reconcile SLURM submission manifests against compact evidence bundles.

The tool is intentionally metadata-only: it maps queue seed requests to manifest
status and evidence-preservation status, then reports duplicates plus conservative
warnings and errors.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "slurm-evidence-reconciler.v1"


RUNNING_STATES = {
    "running",
    "pending",
    "requeued",
    "submitted_running",
    "in_progress",
}
COMPLETED_STATES = {
    "completed",
    "success",
    "completed_pending_artifact_promotion",
}
FAILED_STATES = {
    "failed",
    "error",
    "timeout",
    "cancelled",
    "failed_closed",
    "rerun_required",
    "finalization_failed",
}
EVIDENCE_PRESERVE_STATES = {"completed", "success"}

RAW_ARTIFACT_LIKE_SUFFIXES = {".zip", ".pt", ".pth", ".ckpt", ".jsonl", ".out", ".err"}


def _normalize_status(value: Any) -> str:
    """Normalize a status-like value into lower snake case."""
    return str(value).strip().lower().replace("-", "_")


def _coerce_seed(value: Any) -> int | None:
    """Return a stable integer seed value when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _coerce_seed_list(value: Any) -> list[int]:
    """Normalize all seed-like values to a deterministic integer list."""
    if isinstance(value, list):
        return sorted({seed for raw in value if (seed := _coerce_seed(raw)) is not None})
    if value is None:
        return []
    if isinstance(value, (int, str)):
        seed = _coerce_seed(value)
        return [seed] if seed is not None else []
    return []


def _coerce_csv_value(value: Any) -> str:
    """Return a normalized string for CSV/JSON scalar comparison."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML map from disk with deterministic errors."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read YAML {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping in {path}")
    return payload


@dataclass(frozen=True)
class QueueEntry:
    """One queue row to reconcile."""

    queue_id: str
    seeds: tuple[int, ...]
    status: str
    issue: str | int | None = None
    excluded_seeds: tuple[int, ...] = ()


def _extract_excluded_seeds(payload: dict[str, Any]) -> tuple[int, ...]:
    """Extract explicit excluded seeds from a queue or manifest payload."""
    explicit = (
        payload.get("excluded") or payload.get("excluded_seeds") or payload.get("exclude") or []
    )
    return tuple(_coerce_seed_list(explicit))


def load_queue(path: Path) -> list[QueueEntry]:
    """Load queue entries with minimal validation."""
    payload = _load_yaml(path)
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError("queue does not contain entries list")
    entries: list[QueueEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("queue entry is not a mapping")
        queue_id = _coerce_csv_value(raw_entry.get("id")).strip()
        if not queue_id:
            raise RuntimeError("queue entry is missing id")
        seeds = _coerce_seed_list(raw_entry.get("seeds"))
        if not seeds:
            raise RuntimeError(f"{queue_id}: seeds must contain at least one numeric value")
        entries.append(
            QueueEntry(
                queue_id=queue_id,
                seeds=tuple(seeds),
                status=_normalize_status(raw_entry.get("status", "planned")),
                issue=raw_entry.get("issue"),
                excluded_seeds=_extract_excluded_seeds(raw_entry),
            )
        )
    return entries


@dataclass(frozen=True)
class ManifestJob:
    """One SLURM manifest job row."""

    queue_id: str
    status: str
    slurm_job_id: str | None
    seeds: tuple[int, ...]
    experiment_id: str | None = None
    scheduler_state: str | None = None
    ledger: str | None = None
    source_path: str = ""


def _extract_state_from_payload(
    payload: dict[str, Any], *, include_status: bool = True
) -> str | None:
    """Extract a scheduler/status-like state from common keys."""
    candidates = [
        payload.get("scheduler"),
        payload.get("scheduler_state"),
        payload.get("scheduler_status"),
        payload.get("ledger"),
        payload.get("ledger_status"),
        payload.get("run_status"),
    ]
    if include_status:
        candidates = [payload.get("status"), payload.get("state"), *candidates]
    else:
        candidates = [payload.get("state"), *candidates]
    for candidate in candidates:
        if isinstance(candidate, dict):
            nested = candidate.get("status") or candidate.get("state")
            if isinstance(nested, str):
                return _normalize_status(nested)
        elif isinstance(candidate, str):
            return _normalize_status(candidate)
    return None


def load_submission_manifests(paths: list[Path]) -> tuple[list[ManifestJob], list[str], list[str]]:
    """Load all manifest jobs and collect non-fatal queue-path warnings."""
    jobs: list[ManifestJob] = []
    errors: list[str] = []
    warnings: list[str] = []
    for manifest_path in paths:
        try:
            payload = _load_yaml(manifest_path)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        raw_jobs = payload.get("jobs", [])
        if not isinstance(raw_jobs, list):
            continue
        for raw_job in raw_jobs:
            if not isinstance(raw_job, dict):
                warnings.append(f"{manifest_path}: manifest job entry is not a mapping; skipping")
                continue
            queue_id = _coerce_csv_value(raw_job.get("queue_id")).strip()
            if not queue_id:
                warnings.append(f"{manifest_path}: manifest job has no queue_id; skipping")
                continue
            status = _normalize_status(raw_job.get("status", ""))
            seeds = _coerce_seed_list(raw_job.get("seeds"))
            if not seeds:
                # Jobs without explicit seeds can still be observed if they carry queue_id.
                seeds = []
            jobs.append(
                ManifestJob(
                    queue_id=queue_id,
                    status=status,
                    slurm_job_id=_coerce_csv_value(raw_job.get("slurm_job_id")) or None,
                    seeds=tuple(seeds),
                    experiment_id=_coerce_csv_value(raw_job.get("experiment_id")) or None,
                    scheduler_state=_extract_state_from_payload(raw_job, include_status=False),
                    ledger=_extract_state_from_payload(raw_job.get("ledger", {}))
                    if isinstance(raw_job.get("ledger"), dict)
                    else None,
                    source_path=str(manifest_path),
                )
            )
    return jobs, errors, warnings


@dataclass(frozen=True)
class EvidenceRow:
    """One compact evidence row."""

    queue_id: str | None
    seed: int | None
    job_id: str | None
    wandb_url: str | None
    durable_pointer: str | None
    claim_boundary: str | None
    has_checksum: bool
    status: str | None
    source_path: str


def _is_raw_artifact_like(path: Path) -> bool:
    """Detect raw artifact-like paths that should not alone satisfy evidence preservation."""
    lower = path.name.lower()
    posix_path = path.as_posix().lower()
    if lower in {"wandb", "model_cache"} or "output/model_cache" in posix_path:
        return True
    if lower in {"wandb", "output", "artifacts", "runs"} and path.is_dir():
        return True
    if any(lower.endswith(suffix) for suffix in RAW_ARTIFACT_LIKE_SUFFIXES):
        return True
    if "/wandb/" in posix_path:
        return True
    if "/output/model_cache/" in posix_path:
        return True
    return False


def _extract_pointer(payload: dict[str, Any]) -> str | None:
    """Extract a durable pointer URL from common compact-evidence keys."""
    keys = [
        "wandb_url",
        "wandb_run_url",
        "artifact_uri",
        "artifact_url",
        "dvc_uri",
        "s3_uri",
    ]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Some rows omit explicit pointer keys and include the pointer inside generic text.
    for key in ("url", "uri", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().startswith(
            ("http://", "https://", "s3://", "gs://")
        ):
            return value.strip()
    return None


def _extract_claim_boundary(payload: dict[str, Any]) -> str | None:
    """Extract a claim-boundary field when present."""
    for key in ("claim_boundary", "claim-boundary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _has_checksum(payload: dict[str, Any]) -> bool:
    """Return whether the payload row carries at least one checksum/checksum-like field."""
    for key in payload.keys():
        if "sha" in key.lower() or "checksum" in key.lower():
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _iter_json_rows(data: Any) -> list[dict[str, Any]]:
    """Yield normalized row-like mappings from a JSON payload."""
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            rows.extend(item for item in data["rows"] if isinstance(item, dict))
        elif isinstance(data.get("payload"), dict) and isinstance(
            data["payload"].get("rows"), list
        ):
            rows.extend(item for item in data["payload"]["rows"] if isinstance(item, dict))
        elif all(key in data for key in ("seed", "job_id")) or "seed" in data:
            rows.append(data)
    if isinstance(data, list):
        rows.extend(item for item in data if isinstance(item, dict))
    return rows


def _iter_csv_rows(path: Path) -> list[dict[str, str]]:
    """Yield row dictionaries from a compact CSV file."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _coerce_csv_value(value) for key, value in row.items()})
    return rows


def load_evidence_rows(path: Path) -> tuple[list[EvidenceRow], list[str]]:
    """Load compact evidence rows and report raw-artifact-like inputs."""
    evidence_rows: list[EvidenceRow] = []
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"evidence root does not exist: {path}")
        return evidence_rows, warnings
    for item in sorted(path.rglob("*")):
        if item.is_dir():
            if _is_raw_artifact_like(item):
                warnings.append(f"raw artifact-like directory input: {item}")
            continue
        if _is_raw_artifact_like(item):
            warnings.append(f"raw artifact-like file input: {item}")
            continue
        try:
            evidence_rows.extend(_load_evidence_from_path(item))
        except RuntimeError as exc:
            warnings.append(str(exc))
    return evidence_rows, warnings


def _infer_manifest_seed_status(job: ManifestJob) -> str:
    """Infer a compact status from manifest/job fields."""
    if job.status == "excluded":
        return "excluded"
    state = job.ledger or job.scheduler_state or job.status
    norm = _normalize_status(state)
    if norm in COMPLETED_STATES:
        return "completed"
    if norm in FAILED_STATES:
        return "failed"
    if norm in RUNNING_STATES:
        return "running"
    if job.status == "submitted" and bool(job.slurm_job_id):
        return "submitted"
    return job.status or "planned"


def _seed_status_rank(status: str) -> int:
    """Rank statuses from lowest priority to highest-confidence for deterministic merge."""
    return {
        "planned": 0,
        "submitted": 10,
        "running": 20,
        "completed": 30,
        "evidence_preserved": 40,
        "failed": 50,
        "excluded": 60,
    }.get(status, 0)


def _find_matching_evidence_for_seed(
    evidence_rows: list[EvidenceRow],
    queue_id: str,
    seed: int,
    slurm_job_id: str | None,
) -> list[EvidenceRow]:
    """Find evidence rows that match queue/seed and optionally slurm job id."""
    candidates = [row for row in evidence_rows if row.seed == seed and row.queue_id == queue_id]
    if not candidates:
        candidates = [row for row in evidence_rows if row.seed == seed]
    if slurm_job_id:
        exact = [row for row in candidates if row.job_id and row.job_id == slurm_job_id]
        if exact:
            return exact
        return [row for row in candidates if not row.job_id]
    return candidates


def _is_evidence_row_preserved(row: EvidenceRow) -> bool:
    """Conservative condition for a row to count as preserved evidence."""
    pointer = row.wandb_url or row.durable_pointer
    return bool(pointer and row.claim_boundary and row.has_checksum)


def _apply_manifest_jobs(
    *, status: str, sources: list[str], candidate_jobs: list[ManifestJob]
) -> tuple[str, list[str], list[str]]:
    """Merge manifest job statuses into the current queue-seed state."""
    job_ids: list[str] = []
    for job in candidate_jobs:
        candidate_status = _infer_manifest_seed_status(job)
        candidate_rank = _seed_status_rank(candidate_status)
        if candidate_rank > _seed_status_rank(status):
            status = candidate_status
            sources = [f"manifest:{job.source_path}"]
        elif candidate_status and candidate_rank == _seed_status_rank(status):
            sources.append(f"manifest:{job.source_path}")
        if job.slurm_job_id:
            job_ids.append(job.slurm_job_id)
    return status, sources, job_ids


def _build_evidence_row(row: dict[str, Any], source_path: Path) -> EvidenceRow:
    """Build a normalized evidence row from a dict payload."""
    queue_id = _coerce_csv_value(row.get("queue_id") or row.get("experiment_id"))
    if not queue_id:
        queue_id = _coerce_csv_value(row.get("exp_id"))
    seed = _coerce_seed(row.get("seed") or row.get("seed_id"))
    if seed is None and isinstance(row.get("seeds"), str):
        seed = _coerce_seed(row["seeds"].split(",")[0])
    wandb_url = _coerce_csv_value(row.get("wandb_url") or row.get("wandb_run_url")) or None
    return EvidenceRow(
        queue_id=queue_id or None,
        seed=seed,
        job_id=_coerce_csv_value(row.get("job_id") or row.get("run_id") or row.get("slurm_job_id")),
        wandb_url=wandb_url,
        durable_pointer=_extract_pointer(row),
        claim_boundary=_extract_claim_boundary(row),
        has_checksum=_has_checksum(row),
        status=_normalize_status(row.get("status")) if row.get("status") is not None else None,
        source_path=source_path.as_posix(),
    )


def _load_evidence_from_json(path: Path) -> list[EvidenceRow]:
    """Load evidence rows from a single JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: malformed JSON {exc}") from exc
    rows = _iter_json_rows(payload)
    return [_build_evidence_row(row, path) for row in rows]


def _load_evidence_from_csv(path: Path) -> list[EvidenceRow]:
    """Load evidence rows from a single CSV file."""
    rows: list[EvidenceRow] = []
    try:
        csv_rows = _iter_csv_rows(path)
    except OSError as exc:
        raise RuntimeError(f"cannot read CSV {path}: {exc}") from exc
    for raw_row in csv_rows:
        if not raw_row:
            continue
        rows.append(_build_evidence_row(raw_row, path))
    return rows


def _load_evidence_from_path(path: Path) -> list[EvidenceRow]:
    """Load evidence rows from one file with the supported codecs."""
    if path.suffix.lower() == ".json":
        return _load_evidence_from_json(path)
    if path.suffix.lower() == ".csv":
        return _load_evidence_from_csv(path)
    return []


def _build_seed_row(
    queue_entry: QueueEntry,
    seed: int,
    candidate_jobs: list[ManifestJob],
    evidence_rows: list[EvidenceRow],
) -> dict[str, Any]:
    """Build one deterministic queue-seed row."""
    status = "planned"
    sources: list[str] = ["queue"]
    notes: list[str] = []
    if seed in queue_entry.excluded_seeds:
        status = "excluded"
        sources.append("queue.excluded_seeds")

    status, sources, job_ids = _apply_manifest_jobs(
        status=status,
        sources=sources,
        candidate_jobs=candidate_jobs,
    )

    matching_evidence = _find_matching_evidence_for_seed(
        evidence_rows,
        queue_entry.queue_id,
        seed,
        job_ids[0] if job_ids else None,
    )
    if any(row.status == "excluded" for row in matching_evidence):
        status = "excluded"
        sources.append("evidence.excluded")
    if status in COMPLETED_STATES:
        if any(_is_evidence_row_preserved(row) for row in matching_evidence):
            status = "evidence_preserved"
            sources.append("evidence")
        else:
            notes.append("completed but not preserved in compact evidence")
            notes.append("missing compact preserved evidence rows for completed seed")
    if status in {"completed", "evidence_preserved"}:
        if matching_evidence and all(
            not row.wandb_url and not row.durable_pointer for row in matching_evidence
        ):
            notes.append("missing wandb link or durable pointer on completed/preserved candidate")

    return {
        "queue_id": queue_entry.queue_id,
        "seed": seed,
        "issue": queue_entry.issue,
        "status": status,
        "sources": sorted(set(sources)),
        "job_ids": sorted(set(job_ids)),
        "notes": sorted(set(notes)),
    }


def _build_duplicate_observations(manifest_jobs: list[ManifestJob]) -> dict[str, list[str]]:
    """Index duplicate queue_id/seed combinations across manifests."""
    duplicate_observations: dict[str, list[str]] = {}
    bucket: dict[str, list[ManifestJob]] = {}
    for job in manifest_jobs:
        seeds = job.seeds if job.seeds else (None,)
        for seed in seeds:
            key = f"{job.queue_id}::{seed}"
            bucket.setdefault(key, []).append(job)
    for key, jobs in bucket.items():
        if len(jobs) > 1:
            duplicate_observations[key] = sorted(
                [f"{job.source_path}:{job.status}" for job in jobs]
            )
    return duplicate_observations


def _build_duplicate_experiments(
    manifest_jobs: list[ManifestJob],
) -> dict[str, list[str]]:
    """Collect duplicated manifest experiment ids."""
    duplicate_experiments: dict[str, list[str]] = {}
    for job in manifest_jobs:
        if not job.experiment_id:
            continue
        duplicate_experiments.setdefault(job.experiment_id, []).append(job.source_path)
    return {
        experiment_id: sorted(set(paths))
        for experiment_id, paths in duplicate_experiments.items()
        if len(paths) > 1
    }


def _build_errors_and_warnings(
    *,
    queue_entries: list[QueueEntry],
    manifest_errors: list[str],
    manifest_warnings: list[str],
    evidence_warnings: list[str],
    duplicate_experiments: dict[str, list[str]],
    duplicate_observations: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Build deterministic warning and error outputs."""
    queue_ids = [entry.queue_id for entry in queue_entries]
    duplicates = sorted({entry for entry in queue_ids if queue_ids.count(entry) > 1})
    errors: list[str] = []
    errors.extend(manifest_errors)
    errors.extend(f"duplicate queue_id in queue: {dup}" for dup in duplicates)
    errors.extend(
        f"duplicate experiment_id across manifests: {experiment_id} in {', '.join(paths)}"
        for experiment_id, paths in sorted(duplicate_experiments.items())
    )
    warnings = sorted(set(manifest_warnings + evidence_warnings))
    if duplicates:
        warnings.append(f"duplicate queue_id in queue: {', '.join(duplicates)}")
    if duplicate_experiments:
        warnings.append(f"duplicate experiment_id in manifests: {', '.join(duplicate_experiments)}")
    if duplicate_observations:
        for key, values in sorted(duplicate_observations.items()):
            warnings.append(f"duplicate queue_id/seed observation: {key} from {', '.join(values)}")
    return errors, warnings


def reconcile(
    *, queue_path: Path, submission_manifests: list[Path], evidence_root: Path
) -> dict[str, Any]:
    """Compute a deterministic status ledger for queue-seed pairs."""
    queue_entries = load_queue(queue_path)
    manifest_jobs, manifest_errors, manifest_warnings = load_submission_manifests(
        submission_manifests
    )
    evidence_rows, evidence_warnings = load_evidence_rows(evidence_root)
    duplicate_experiments = _build_duplicate_experiments(manifest_jobs)
    duplicate_observations = _build_duplicate_observations(manifest_jobs)

    rows: list[dict[str, Any]] = []
    for queue_entry in queue_entries:
        for seed in queue_entry.seeds:
            candidate_jobs = [
                job
                for job in manifest_jobs
                if job.queue_id == queue_entry.queue_id and (not job.seeds or seed in job.seeds)
            ]
            row = _build_seed_row(
                queue_entry=queue_entry,
                seed=seed,
                candidate_jobs=candidate_jobs,
                evidence_rows=evidence_rows,
            )
            rows.append(row)

    errors, warnings = _build_errors_and_warnings(
        queue_entries=queue_entries,
        manifest_errors=manifest_errors,
        manifest_warnings=manifest_warnings,
        evidence_warnings=evidence_warnings,
        duplicate_experiments=duplicate_experiments,
        duplicate_observations=duplicate_observations,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "queue_path": str(queue_path),
        "submission_manifests": sorted({str(path) for path in submission_manifests}),
        "evidence_root": str(evidence_root),
        "observations": sorted(rows, key=lambda row: (row["queue_id"], row["seed"])),
        "duplicate_ids": {
            "manifest_experiment_ids": sorted(duplicate_experiments),
            "queue_seed_observations": {
                key: sorted(set(values)) for key, values in sorted(duplicate_observations.items())
            },
        },
        "errors": sorted(set(errors)),
        "warnings": warnings,
    }


def _format_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown reconciliation summary."""
    lines = [
        "# SLURM Evidence Reconciliation",
        "",
        f"- queue: `{report['queue_path']}`",
        f"- evidence root: `{report['evidence_root']}`",
        "",
        "| queue_id | seed | status | notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["observations"]:
        notes = "; ".join(row["notes"])
        lines.append(f"| {row['queue_id']} | {row['seed']} | {row['status']} | {notes} |")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("experiments/submission_queue.yaml"),
        help="Path to queue YAML to reconcile.",
    )
    parser.add_argument(
        "--submission-manifest",
        action="append",
        default=[],
        help="Submission manifest path to include. May be repeated.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("docs/context/evidence"),
        help="Evidence root directory containing compact evidence summaries.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional markdown output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the reconciler."""
    args = _parse_args(argv)
    try:
        report = reconcile(
            queue_path=args.queue,
            submission_manifests=[Path(path) for path in args.submission_manifest],
            evidence_root=args.evidence_root,
        )
    except RuntimeError as exc:
        print(f"reconcile_slurm_evidence: {exc}")
        return 1

    output = json.dumps(report, sort_keys=True, indent=2)
    if args.json:
        print(output)
    else:
        print(f"schema_version={report['schema_version']}")
        status_counts: dict[str, int] = {}
        for row in report["observations"]:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        print("observations=", len(report["observations"]))
        print("status_counts=", status_counts)
        if report["errors"]:
            print("errors:")
            for error in report["errors"]:
                print(f"- {error}")
        if report["warnings"]:
            print("warnings:")
            for warning in report["warnings"]:
                print(f"- {warning}")
        print("markdown=")
        print(_format_markdown(report))

    if args.markdown:
        args.markdown.write_text(_format_markdown(report), encoding="utf-8")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
