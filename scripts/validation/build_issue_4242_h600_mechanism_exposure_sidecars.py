#!/usr/bin/env python3
"""Build retained h600 mechanism/exposure sidecars for issue #4242."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
    FailureMechanismTaxonomyError,
    unknown_failure_mechanism_record,
    validate_failure_mechanism_record,
)
from robot_sf.benchmark.interaction_exposure import (
    compute_interaction_exposure_fields,
    not_derivable_interaction_exposure,
)

SCHEMA_VERSION = "issue_4242_h600_mechanism_exposure_backfill.v1"
DEFAULT_SOURCE_MANIFEST = Path(
    "docs/context/evidence/issue_3810_h600_interpretation_2026-07/source_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3810_h600_interpretation_2026-07")
MECHANISM_SIDECAR = "h600_mechanism_labels_sidecar.csv"
EXPOSURE_SIDECAR = "h600_interaction_exposure_sidecar.csv"
BACKFILL_MANIFEST = "h600_mechanism_exposure_backfill_manifest.json"
BACKFILL_REPORT = "h600_mechanism_exposure_backfill_report.md"

IDENTIFIER_FIELDS = (
    "job_id",
    "run_label",
    "campaign_id",
    "episode_id",
    "scenario_id",
    "planner_key",
    "seed",
    "repeat_index",
)

EXPOSURE_FIELDS = (
    "interaction_exposure_schema_version",
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
    "interaction_exposure_radius_m",
    "interaction_exposure_steps",
    "interaction_exposure_denominator_steps",
    "robot_motion_steps_before_first_clearance",
    "robot_motion_denominator_steps_before_first_clearance",
    "first_clearance_reason",
    "interaction_exposure_status",
    "low_exposure_success_threshold",
)


class BackfillBuildError(RuntimeError):
    """Raised when the retained h600 sidecar builder cannot read its manifest."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BackfillBuildError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _identifiers(row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    campaign = run.get("campaign") if isinstance(run.get("campaign"), dict) else {}
    return {
        "job_id": run.get("job_id", ""),
        "run_label": run.get("run_label", ""),
        "campaign_id": campaign.get("campaign_id", ""),
        "episode_id": row.get("episode_id", ""),
        "scenario_id": row.get("scenario_id", ""),
        "planner_key": row.get("planner_key", row.get("planner_id", row.get("algo", ""))),
        "seed": row.get("seed", ""),
        "repeat_index": row.get("repeat_index", row.get("episode_index", "")),
    }


def _load_json_field(row: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = row.get(name)
        if not value:
            continue
        return json.loads(value)
    return None


def _mechanism_from_row(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    candidate = {field: row.get(field, "") for field in REQUIRED_MECHANISM_FIELDS}
    if any(str(value).strip() for value in candidate.values()):
        try:
            mechanism = validate_failure_mechanism_record(candidate)
        except FailureMechanismTaxonomyError as exc:
            mechanism = unknown_failure_mechanism_record(f"not_derivable_invalid_mechanism: {exc}")
            return mechanism, "not_derivable_invalid_mechanism"
        if mechanism["mechanism_evidence_mode"] in TRACE_VERIFIED_EVIDENCE_MODES:
            return mechanism, "computed_from_retained_trace"
        mechanism = unknown_failure_mechanism_record("not_derivable_non_trace_verified_mechanism")
        return mechanism, "not_derivable_non_trace_verified_mechanism"
    return unknown_failure_mechanism_record(
        "not_derivable_missing_trace"
    ), "not_derivable_missing_trace"


def _exposure_from_row(
    row: dict[str, Any],
    *,
    exposure_radius_m: float,
    low_exposure_success_threshold: float,
) -> tuple[dict[str, Any], str]:
    native_status = str(row.get("interaction_exposure_status", "")).strip()
    native_values = {
        field: row.get(field, "")
        for field in EXPOSURE_FIELDS
        if field in row and str(row.get(field, "")).strip()
    }
    if native_values and native_status == "computed":
        backfilled = {field: row.get(field, "") for field in EXPOSURE_FIELDS}
        return backfilled, "computed_from_retained_trace"

    try:
        robot_positions = _load_json_field(row, ("robot_positions_json", "robot_trace_json"))
        pedestrian_positions = _load_json_field(
            row,
            ("pedestrian_positions_json", "pedestrian_trace_json"),
        )
    except json.JSONDecodeError:
        return not_derivable_interaction_exposure("malformed"), "malformed"

    if robot_positions is None or pedestrian_positions is None:
        return not_derivable_interaction_exposure(
            "not_derivable_missing_trace"
        ), "not_derivable_missing_trace"

    dt = float(row.get("dt", row.get("time_step_s", 1.0)) or 1.0)
    fields = compute_interaction_exposure_fields(
        robot_positions=robot_positions,
        pedestrian_positions=pedestrian_positions,
        dt=dt,
        exposure_radius_m=exposure_radius_m,
        low_exposure_success_threshold=low_exposure_success_threshold,
        success=_truthy(row.get("success")),
    )
    return fields, "computed_from_retained_trace"


def _run_rows(run: dict[str, Any], *, repo_root: Path) -> tuple[Path, list[dict[str, str]]]:
    reports_dir = _repo_path(Path(str(run["reports_dir"])), repo_root=repo_root)
    seed_rows_path = _repo_path(
        Path(str(run.get("seed_episode_rows") or reports_dir / "seed_episode_rows.csv")),
        repo_root=repo_root,
    )
    if not seed_rows_path.exists():
        return seed_rows_path, []
    return seed_rows_path, _read_csv(seed_rows_path)


def build_sidecars(
    *,
    source_manifest: Path,
    output_dir: Path,
    generated_at: str,
    exposure_radius_m: float,
    low_exposure_success_threshold: float,
    repo_root: Path,
) -> dict[str, Any]:
    """Build h600 sidecar CSVs and compact manifest/report outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _repo_path(source_manifest, repo_root=repo_root)
    manifest = _read_json(manifest_path)
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise BackfillBuildError("source manifest must contain a runs list")

    mechanism_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    mechanism_statuses: Counter[str] = Counter()
    exposure_statuses: Counter[str] = Counter()

    for run in runs:
        if not isinstance(run, dict):
            continue
        seed_rows_path, rows = _run_rows(run, repo_root=repo_root)
        run_mechanism_statuses: Counter[str] = Counter()
        run_exposure_statuses: Counter[str] = Counter()
        for row in rows:
            identifiers = _identifiers(row, run)
            mechanism, mechanism_status = _mechanism_from_row(row)
            exposure, exposure_status = _exposure_from_row(
                row,
                exposure_radius_m=exposure_radius_m,
                low_exposure_success_threshold=low_exposure_success_threshold,
            )
            mechanism_statuses[mechanism_status] += 1
            exposure_statuses[exposure_status] += 1
            run_mechanism_statuses[mechanism_status] += 1
            run_exposure_statuses[exposure_status] += 1
            mechanism_rows.append(
                {
                    **identifiers,
                    **mechanism,
                    "mechanism_backfill_status": mechanism_status,
                }
            )
            exposure_rows.append(
                {
                    **identifiers,
                    **exposure,
                    "interaction_exposure_backfill_status": exposure_status,
                }
            )
        run_summaries.append(
            {
                "job_id": run.get("job_id", ""),
                "run_label": run.get("run_label", ""),
                "seed_episode_rows": str(seed_rows_path.relative_to(repo_root))
                if seed_rows_path.is_relative_to(repo_root)
                else str(seed_rows_path),
                "retained_episode_rows": len(rows),
                "mechanism_status_counts": dict(sorted(run_mechanism_statuses.items())),
                "interaction_exposure_status_counts": dict(sorted(run_exposure_statuses.items())),
                "status": "ok" if rows else "blocked_missing_seed_episode_rows",
            }
        )

    mechanism_path = output_dir / MECHANISM_SIDECAR
    exposure_path = output_dir / EXPOSURE_SIDECAR
    backfill_manifest_path = output_dir / BACKFILL_MANIFEST
    report_path = output_dir / BACKFILL_REPORT

    _write_csv(
        mechanism_path,
        mechanism_rows,
        [*IDENTIFIER_FIELDS, *REQUIRED_MECHANISM_FIELDS, "mechanism_backfill_status"],
    )
    _write_csv(
        exposure_path,
        exposure_rows,
        [*IDENTIFIER_FIELDS, *EXPOSURE_FIELDS, "interaction_exposure_backfill_status"],
    )

    backfill_manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4242,
        "generated_at": generated_at,
        "claim_boundary": (
            "Retained h600 sidecars only; rows are trace-derived when retained traces/native "
            "fields exist, otherwise explicitly not derivable. No geometry-only mechanism "
            "labels, imputation, benchmark ranking, or paper/dissertation claims."
        ),
        "source_manifest": str(source_manifest),
        "outputs": [MECHANISM_SIDECAR, EXPOSURE_SIDECAR, BACKFILL_REPORT, BACKFILL_MANIFEST],
        "retained_episode_rows": len(mechanism_rows),
        "mechanism_status_counts": dict(sorted(mechanism_statuses.items())),
        "interaction_exposure_status_counts": dict(sorted(exposure_statuses.items())),
        "runs": run_summaries,
    }
    _write_json(backfill_manifest_path, backfill_manifest)
    _write_report(report_path, backfill_manifest)

    manifest["generated_outputs"] = sorted(
        set(manifest.get("generated_outputs") or [])
        | {MECHANISM_SIDECAR, EXPOSURE_SIDECAR, BACKFILL_MANIFEST, BACKFILL_REPORT}
    )
    manifest["issue_4242_h600_sidecars"] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "mechanism_sidecar": MECHANISM_SIDECAR,
        "interaction_exposure_sidecar": EXPOSURE_SIDECAR,
        "backfill_manifest": BACKFILL_MANIFEST,
        "backfill_report": BACKFILL_REPORT,
        "mechanism_status_counts": dict(sorted(mechanism_statuses.items())),
        "interaction_exposure_status_counts": dict(sorted(exposure_statuses.items())),
    }
    _write_json(output_dir / "source_manifest.json", manifest)
    _write_sha256sums(output_dir)
    return backfill_manifest


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Issue #4242 h600 mechanism/exposure sidecar backfill",
        "",
        "Plain-language summary: this compact sidecar records whether retained h600 "
        "episode rows can support trace-verified mechanism and interaction-exposure fields.",
        "",
        "Evidence status: diagnostic-only schema closure. This is not a benchmark ranking, "
        "paper-facing claim, dissertation claim, or imputation pass.",
        "",
        "## Status counts",
        "",
        f"- Retained episode rows: {manifest['retained_episode_rows']}",
        f"- Mechanism statuses: `{json.dumps(manifest['mechanism_status_counts'], sort_keys=True)}`",
        "- Interaction-exposure statuses: "
        f"`{json.dumps(manifest['interaction_exposure_status_counts'], sort_keys=True)}`",
        "",
        "## Run summary",
        "",
        "| job_id | run_label | rows | mechanism statuses | exposure statuses |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for run in manifest["runs"]:
        lines.append(
            f"| {run['job_id']} | {run['run_label']} | {run['retained_episode_rows']} | "
            f"`{json.dumps(run['mechanism_status_counts'], sort_keys=True)}` | "
            f"`{json.dumps(run['interaction_exposure_status_counts'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            manifest["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_sha256sums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{_sha256(path)}  {path.as_posix()}")
    (output_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    """Run the retained h600 mechanism/exposure sidecar builder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at", default="now")
    parser.add_argument("--exposure-radius-m", type=float, default=2.0)
    parser.add_argument("--low-exposure-success-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    generated_at = _now() if args.generated_at == "now" else args.generated_at
    repo_root = Path.cwd()
    summary = build_sidecars(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        generated_at=generated_at,
        exposure_radius_m=args.exposure_radius_m,
        low_exposure_success_threshold=args.low_exposure_success_threshold,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "retained_episode_rows": summary["retained_episode_rows"],
                "mechanism_status_counts": summary["mechanism_status_counts"],
                "interaction_exposure_status_counts": summary["interaction_exposure_status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
