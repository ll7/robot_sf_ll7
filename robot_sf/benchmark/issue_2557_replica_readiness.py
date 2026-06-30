"""Read-only readiness checks for issue #2557 fixed-seed replica artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue-2557-replica-readiness.v1"
DEFAULT_MANIFEST = Path(
    "docs/context/evidence/issue_2557_reward_curriculum_partial_2026-06-08/seed_summary.json"
)
DEFAULT_CONFIG_TEMPLATE = (
    "configs/training/ppo/ablations/"
    "expert_ppo_issue_2557_reward_curriculum_promotion_10m_env22_eval_aligned_"
    "large_capacity_seed{seed}_fixed.yaml"
)
REQUIRED_ROW_FIELDS = (
    "seed",
    "job_id",
    "partition",
    "run_id",
    "eval_step",
    "success_rate",
    "collision_rate",
    "snqi",
    "wandb_url",
    "run_summary_sha256",
    "eval_timeline_sha256",
    "source_config_sha256",
)
REQUIRED_SOURCE_FIELDS = ("branch", "commit", "seeds")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WANDB_URL_RE = re.compile(r"^https://wandb\.ai/[^/]+/[^/]+/runs/[^/]+$")


@dataclass(frozen=True)
class ReplicaReadinessResult:
    """Structured artifact-readiness verdict for one compact replica manifest."""

    manifest_path: str
    ready: bool
    allow_partial: bool
    expected_seeds: tuple[int, ...]
    present_seeds: tuple[int, ...]
    missing_seeds: tuple[int, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return stable JSON-serializable report payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": self.manifest_path,
            "ready": self.ready,
            "allow_partial": self.allow_partial,
            "expected_seeds": list(self.expected_seeds),
            "present_seeds": list(self.present_seeds),
            "missing_seeds": list(self.missing_seeds),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return payload


def _parse_expected_seeds(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    seeds: set[int] = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        if "-" in value:
            start_raw, end_raw = value.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"seed range start exceeds end: {value}")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(value))
    return tuple(sorted(seeds))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _check_row_required_fields(row: dict[str, Any], *, index: int) -> list[str]:
    return [f"rows[{index}] missing {field}" for field in REQUIRED_ROW_FIELDS if field not in row]


def _check_row_numeric_fields(row: dict[str, Any], *, seed: Any) -> list[str]:
    blockers: list[str] = []
    for field in ("job_id", "eval_step"):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            blockers.append(f"seed {seed}: {field} must be an integer")

    for field in ("success_rate", "collision_rate", "snqi"):
        if not _is_finite_number(row.get(field)):
            blockers.append(f"seed {seed}: {field} must be a finite number")
    return blockers


def _check_row_provenance_fields(row: dict[str, Any], *, seed: Any) -> list[str]:
    blockers: list[str] = []
    wandb_url = row.get("wandb_url")
    if not isinstance(wandb_url, str) or not WANDB_URL_RE.match(wandb_url):
        blockers.append(f"seed {seed}: wandb_url must be a W&B run URL")

    for field in ("run_summary_sha256", "eval_timeline_sha256", "source_config_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or not SHA256_RE.match(value):
            blockers.append(f"seed {seed}: {field} must be a sha256 hex digest")
    return blockers


def _check_row(row: Any, *, index: int) -> tuple[int | None, list[str]]:
    blockers: list[str] = []
    if not isinstance(row, dict):
        return None, [f"rows[{index}] must be an object"]

    blockers.extend(_check_row_required_fields(row, index=index))

    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        blockers.append(f"rows[{index}].seed must be an integer")
        seed_value: int | None = None
    else:
        seed_value = seed

    blockers.extend(_check_row_numeric_fields(row, seed=seed))

    partition = row.get("partition")
    if partition not in {"a30", "l40s"}:
        blockers.append(f"seed {seed}: partition must be a30 or l40s")

    blockers.extend(_check_row_provenance_fields(row, seed=seed))
    return seed_value, blockers


def _check_source_worktrees(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    sources = payload.get("source_worktrees")
    if not isinstance(sources, list) or not sources:
        return ["source_worktrees must be a non-empty list"]
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            blockers.append(f"source_worktrees[{index}] must be an object")
            continue
        for field in REQUIRED_SOURCE_FIELDS:
            if field not in source:
                blockers.append(f"source_worktrees[{index}] missing {field}")
        commit = source.get("commit")
        if not isinstance(commit, str) or not re.match(r"^[0-9a-f]{40}$", commit):
            blockers.append(f"source_worktrees[{index}].commit must be a 40-char git sha")
        seeds = source.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        ):
            blockers.append(f"source_worktrees[{index}].seeds must be non-empty integer list")
    return blockers


def _check_manifest_boundary(
    payload: dict[str, Any], *, allow_partial: bool
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("issue_2557_reward_curriculum"):
        blockers.append("schema_version must be an issue_2557_reward_curriculum manifest")

    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        blockers.append("claim_boundary must describe the evidence boundary")
    elif ("partial" in claim_boundary or "not_full" in claim_boundary) and not allow_partial:
        blockers.append(
            "manifest is explicitly partial; rerun with allow_partial only for diagnostics"
        )

    incomplete = payload.get("incomplete_or_pending_seeds")
    if isinstance(incomplete, list) and incomplete and not allow_partial:
        blockers.append("manifest still lists incomplete_or_pending_seeds")
    elif isinstance(incomplete, list) and incomplete:
        warnings.append("manifest lists incomplete_or_pending_seeds; diagnostic-only readiness")
    return blockers, warnings


def _check_rows(payload: dict[str, Any]) -> tuple[list[int], int, list[str]]:
    blockers: list[str] = []
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return [], 0, ["rows must be a non-empty list"]

    present: list[int] = []
    for index, row in enumerate(rows):
        seed, row_blockers = _check_row(row, index=index)
        blockers.extend(row_blockers)
        if seed is not None:
            present.append(seed)

    duplicate_seeds = sorted({seed for seed in present if present.count(seed) > 1})
    for seed in duplicate_seeds:
        blockers.append(f"seed {seed}: duplicate manifest row")
    return present, len(rows), blockers


def _check_aggregate(payload: dict[str, Any], *, row_count: int) -> list[str]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return ["aggregate must be an object"]
    if aggregate.get("count") != row_count:
        return ["aggregate.count must match number of rows"]
    return []


def _check_expected_configs(
    *,
    root: Path,
    expected_seeds: tuple[int, ...],
    config_template: str,
) -> list[str]:
    blockers: list[str] = []
    for seed in expected_seeds:
        config = root / config_template.format(seed=seed)
        if not config.is_file():
            blockers.append(f"seed {seed}: expected config missing: {_repo_relative(config, root)}")
    return blockers


def assess_issue_2557_replica_readiness(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
    expected_seeds: tuple[int, ...] = (),
    config_template: str = DEFAULT_CONFIG_TEMPLATE,
    allow_partial: bool = False,
) -> ReplicaReadinessResult:
    """Assess whether a compact issue #2557 seed-replica manifest is analysis-ready.

    The checker is read-only and does not inspect Slurm, W&B, raw logs, checkpoints, or videos.
    It validates the compact manifest and local config/provenance pointers needed before a
    downstream analyzer can safely promote or interpret the artifact bundle.

    Returns:
        Structured readiness verdict with blockers, warnings, and seed coverage metadata.
    """

    root = (repo_root or Path.cwd()).resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    payload = _load_manifest(manifest_path)

    blockers, warnings = _check_manifest_boundary(payload, allow_partial=allow_partial)
    blockers.extend(_check_source_worktrees(payload))
    present, row_count, row_blockers = _check_rows(payload)
    blockers.extend(row_blockers)

    present_seeds = tuple(sorted(set(present)))
    missing_seeds = tuple(seed for seed in expected_seeds if seed not in present_seeds)
    for seed in missing_seeds:
        blockers.append(f"seed {seed}: missing replica row")

    blockers.extend(_check_aggregate(payload, row_count=row_count))
    blockers.extend(
        _check_expected_configs(
            root=root,
            expected_seeds=expected_seeds,
            config_template=config_template,
        )
    )

    metadata = {
        "manifest_schema_version": payload.get("schema_version"),
        "claim_boundary": payload.get("claim_boundary"),
        "row_count": row_count,
        "incomplete_or_pending_seeds": payload.get("incomplete_or_pending_seeds")
        if isinstance(payload.get("incomplete_or_pending_seeds"), list)
        else [],
    }
    return ReplicaReadinessResult(
        manifest_path=_repo_relative(manifest_path, root),
        ready=not blockers,
        allow_partial=allow_partial,
        expected_seeds=expected_seeds,
        present_seeds=present_seeds,
        missing_seeds=missing_seeds,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the issue #2557 artifact-readiness checker.

    Returns:
        Process exit code, where 0 means ready and 1 means not ready.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST), help="Compact seed manifest JSON."
    )
    parser.add_argument(
        "--expected-seeds",
        default="501-508",
        help="Comma/range list of fixed seeds that must have manifest rows.",
    )
    parser.add_argument(
        "--config-template",
        default=DEFAULT_CONFIG_TEMPLATE,
        help="Repo-relative format string for expected per-seed configs; must contain {seed}.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit explicitly partial manifests for diagnostic inspection only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = parser.parse_args(argv)

    result = assess_issue_2557_replica_readiness(
        Path(args.manifest),
        expected_seeds=_parse_expected_seeds(args.expected_seeds),
        config_template=args.config_template,
        allow_partial=args.allow_partial,
    )
    payload = result.to_payload()
    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        status = "ready" if result.ready else "not_ready"
        sys.stdout.write(f"issue #2557 replica artifact readiness: {status}\n")
        for blocker in result.blockers:
            sys.stdout.write(f"- blocker: {blocker}\n")
        for warning in result.warnings:
            sys.stdout.write(f"- warning: {warning}\n")
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
