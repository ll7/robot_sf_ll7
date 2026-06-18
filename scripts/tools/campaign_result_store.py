#!/usr/bin/env python3
"""Create and validate compact canonical campaign result stores.

The result store is intentionally small: episode rows live in Parquet, while
summary, analysis, claim-card, and reproduction metadata stay in reviewable
text formats. Raw campaign artifacts remain outside git unless separately
promoted through the artifact policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

SCHEMA_VERSION = "campaign-result-store.v1"
REQUIRED_EPISODE_FIELDS = (
    "run_id",
    "episode_id",
    "planner",
    "scenario_id",
    "scenario_family",
    "seed",
    "row_status",
    "artifact_uri",
    "artifact_sha256",
)
REQUIRED_STORE_FILES = (
    "episodes.parquet",
    "summary.json",
    "analysis.json",
    "claim_card.yaml",
    "reproduction.md",
    "tables/manifest.json",
    "figures/manifest.json",
)
ROW_STATUS_VALUES = (
    "native",
    "adapter",
    "diagnostic_only",
    "fallback",
    "degraded",
    "unavailable",
    "failed",
)


@dataclass(frozen=True, slots=True)
class ResultStoreValidation:
    """Validation result for a campaign result store."""

    ok: bool
    errors: list[str]


def write_result_store(
    output_dir: Path,
    episode_rows: Iterable[Mapping[str, Any]],
    *,
    study_id: str,
    command: str,
    source_commit: str | None = None,
    claim_card: Mapping[str, Any] | None = None,
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact result store and return its summary payload."""
    rows = list(episode_rows)
    errors = _validate_episode_rows(rows)
    if errors:
        raise ValueError("; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.DataFrame(rows)
    episodes.to_parquet(output_dir / "episodes.parquet", index=False)

    summary = _summary_payload(
        rows, study_id=study_id, command=command, source_commit=source_commit
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "analysis_status": "fixture_or_preliminary",
        **(dict(analysis or {})),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    claim_payload = {
        "schema_version": "campaign-claim-card.v1",
        "study_id": study_id,
        "claim_status": "not_reviewed",
        "claim_boundary": (
            "not paper-facing until durable artifacts, exclusions, and claim review pass"
        ),
        **(dict(claim_card or {})),
    }
    (output_dir / "claim_card.yaml").write_text(
        yaml.safe_dump(claim_payload, sort_keys=False),
        encoding="utf-8",
    )
    _write_empty_output_manifest(output_dir / "tables", role="tables", study_id=study_id)
    _write_empty_output_manifest(output_dir / "figures", role="figures", study_id=study_id)
    (output_dir / "reproduction.md").write_text(
        _reproduction_markdown(study_id=study_id, command=command),
        encoding="utf-8",
    )
    return summary


def validate_result_store(output_dir: Path) -> ResultStoreValidation:
    """Validate the required files and episode-row contract for a result store."""
    errors: list[str] = []
    for filename in REQUIRED_STORE_FILES:
        if not (output_dir / filename).is_file():
            errors.append(f"missing required result-store file: {filename}")
    parquet_path = output_dir / "episodes.parquet"
    if parquet_path.is_file():
        errors.extend(_validate_episode_parquet(parquet_path))
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        errors.extend(_validate_summary_file(summary_path))
    return ResultStoreValidation(ok=not errors, errors=errors)


def _validate_episode_parquet(parquet_path: Path) -> list[str]:
    """Return validation errors for the episode Parquet surface."""
    try:
        episodes = pd.read_parquet(parquet_path)
    except Exception as exc:
        # Pandas may surface engine-specific exceptions for corrupt or unsupported Parquet files.
        return [f"episodes.parquet could not be read: {exc}"]
    missing = [field for field in REQUIRED_EPISODE_FIELDS if field not in episodes.columns]
    if missing:
        return [f"episodes.parquet missing required columns: {missing}"]
    invalid_statuses = sorted(set(episodes["row_status"].dropna()) - set(ROW_STATUS_VALUES))
    if invalid_statuses:
        return [f"episodes.parquet has invalid row_status values: {invalid_statuses}"]
    for field in ("artifact_uri", "artifact_sha256"):
        missing_values = episodes[field].isna() | (episodes[field].astype(str).str.len() == 0)
        if bool(missing_values.any()):
            return [f"episodes.parquet has missing artifact provenance field: {field}"]
    return []


def _validate_summary_file(summary_path: Path) -> list[str]:
    """Return validation errors for the summary JSON surface."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"summary.json could not be read: {exc}"]
    if summary.get("schema_version") != SCHEMA_VERSION:
        return [
            f"summary.json schema_version must be {SCHEMA_VERSION!r}, "
            f"found {summary.get('schema_version')!r}"
        ]
    return []


def _validate_episode_rows(rows: list[Mapping[str, Any]]) -> list[str]:
    """Return validation errors for in-memory episode rows."""
    if not rows:
        return ["episode_rows must be non-empty"]
    errors: list[str] = []
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_EPISODE_FIELDS if field not in row]
        if missing:
            errors.append(f"row {index} missing required fields: {missing}")
            continue
        row_status = str(row["row_status"])
        if row_status not in ROW_STATUS_VALUES:
            errors.append(f"row {index} invalid row_status: {row_status!r}")
        for field in ("artifact_uri", "artifact_sha256"):
            if not str(row[field]).strip():
                errors.append(f"row {index} missing artifact provenance field: {field}")
    return errors


def _summary_payload(
    rows: list[Mapping[str, Any]],
    *,
    study_id: str,
    command: str,
    source_commit: str | None,
) -> dict[str, Any]:
    """Build deterministic summary metadata for episode rows."""
    row_status_counts: dict[str, int] = {}
    planners: set[str] = set()
    run_ids: set[str] = set()
    scenario_families: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        row_status = str(row["row_status"])
        row_status_counts[row_status] = row_status_counts.get(row_status, 0) + 1
        planners.add(str(row["planner"]))
        run_ids.add(str(row["run_id"]))
        scenario_families.add(str(row["scenario_family"]))
        seeds.add(int(row["seed"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "command": command,
        "source_commit": source_commit,
        "episode_count": len(rows),
        "run_count": len(run_ids),
        "run_ids": sorted(run_ids),
        "planner_count": len(planners),
        "scenario_family_count": len(scenario_families),
        "seed_count": len(seeds),
        "row_status_counts": dict(sorted(row_status_counts.items())),
    }


def _write_empty_output_manifest(output_dir: Path, *, role: str, study_id: str) -> None:
    """Write a placeholder manifest for derived report output directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": f"campaign-result-store-{role}-manifest.v1",
        "study_id": study_id,
        "role": role,
        "status": "empty_until_report_generation",
        "files": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reproduction_markdown(*, study_id: str, command: str) -> str:
    """Return a tiny reproduction record."""
    return (
        f"# Reproduction: {study_id}\n\n"
        "Run the frozen command from the protocol record and regenerate this result store:\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Result-store directory to validate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate a result store from the command line."""
    args = _parse_args(argv)
    result = validate_result_store(args.output_dir)
    if result.errors:
        for error in result.errors:
            print(error)
        return 1
    print(f"validated campaign result store: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
