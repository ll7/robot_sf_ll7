#!/usr/bin/env python3
"""Fail-closed registration check for a salvaged trace-capable h600 campaign.

This checker reads a completed camera-ready campaign in place and writes only a
small receipt.  It never copies raw episode data, submits compute, or upgrades
the campaign to benchmark or paper evidence.  A passing receipt says only that
the campaign is structurally ready for the issue #4206 mechanism cross-cut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
)

SCHEMA_VERSION = "issue_5248_salvaged_trace_rerun_registration.v1"
READY_STATUS = "ready_for_issue_4206_reanalysis"
BLOCKED_STATUS = "blocked_campaign_registration"
SUMMARY_RELATIVE_PATH = Path("reports/campaign_summary.json")
EPISODE_ROWS_RELATIVE_PATH = Path("reports/seed_episode_rows.csv")
UNKNOWN_LABELS = {"", "unknown", "not_derivable"}


def _sha256(path: Path) -> str:
    """Return a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or raise a concise validation error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _load_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    """Load compact episode rows and their header, preserving every row."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path.name} has no CSV header")
            return list(reader), set(reader.fieldnames)
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc


def _load_trace_contract(path: Path) -> tuple[set[str], float]:
    """Load trace modes and minimum labeled fraction from the preregistration contract."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read preregistration config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid preregistration config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("preregistration config must be a mapping")
    mechanism = (payload.get("required_outputs") or {}).get("failure_mechanism")
    if not isinstance(mechanism, dict):
        raise ValueError("preregistration config lacks failure-mechanism contract")
    modes = set(mechanism.get("trace_verified_evidence_modes") or ())
    fraction = mechanism.get("min_trace_verified_labeled_fraction")
    if modes != set(TRACE_VERIFIED_EVIDENCE_MODES):
        raise ValueError("preregistration trace-verified modes drift from the canonical schema")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise ValueError("preregistration minimum trace-labeled fraction must be in (0, 1]")
    return modes, float(fraction)


def _campaign_block(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the nested camera-ready campaign block, fail-closed."""

    campaign = summary.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("campaign_summary.json must contain a campaign object")
    return campaign


def _trace_labeled(row: dict[str, str], *, trace_modes: set[str]) -> bool:
    """Return whether a row has a usable trace-verified taxonomy label."""

    return (
        str(row.get("mechanism_schema_version") or "").strip() == MECHANISM_SCHEMA_VERSION
        and str(row.get("mechanism_label") or "").strip().lower() not in UNKNOWN_LABELS
        and str(row.get("mechanism_confidence") or "").strip()
        in {"observed_mechanism", "supported_hypothesis"}
        and str(row.get("mechanism_evidence_mode") or "").strip() in trace_modes
        and bool(str(row.get("mechanism_evidence_uri") or "").strip())
    )


def build_registration_receipt(
    *,
    campaign_root: Path,
    job_id: str,
    expected_total_episodes: int,
    preregistration_config: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Check a campaign and return a receipt without promoting its conclusions."""

    blockers: list[str] = []
    source_files: dict[str, dict[str, Any]] = {}
    summary_path = campaign_root / SUMMARY_RELATIVE_PATH
    rows_path = campaign_root / EPISODE_ROWS_RELATIVE_PATH
    summary: dict[str, Any] | None = None
    rows: list[dict[str, str]] = []
    header: set[str] = set()
    rows_loaded = False
    trace_modes: set[str] = set()
    min_fraction: float | None = None

    try:
        trace_modes, min_fraction = _load_trace_contract(preregistration_config)
    except ValueError as exc:
        blockers.append(str(exc))

    try:
        summary = _load_json(summary_path)
        source_files[SUMMARY_RELATIVE_PATH.as_posix()] = {
            "sha256": _sha256(summary_path),
            "size_bytes": summary_path.stat().st_size,
        }
        campaign = _campaign_block(summary)
        if campaign.get("total_episodes") != expected_total_episodes:
            blockers.append(
                "campaign.total_episodes must equal "
                f"{expected_total_episodes}; got {campaign.get('total_episodes')!r}"
            )
        if campaign.get("campaign_execution_status") != "completed":
            blockers.append(
                "campaign.campaign_execution_status must be 'completed'; got "
                f"{campaign.get('campaign_execution_status')!r}"
            )
    except ValueError as exc:
        blockers.append(str(exc))

    try:
        rows, header = _load_rows(rows_path)
        rows_loaded = True
        source_files[EPISODE_ROWS_RELATIVE_PATH.as_posix()] = {
            "sha256": _sha256(rows_path),
            "size_bytes": rows_path.stat().st_size,
        }
        if len(rows) != expected_total_episodes:
            blockers.append(
                f"seed_episode_rows.csv must contain {expected_total_episodes} rows; got {len(rows)}"
            )
        missing_fields = [field for field in REQUIRED_MECHANISM_FIELDS if field not in header]
        if missing_fields:
            blockers.append(f"seed_episode_rows.csv missing mechanism fields: {missing_fields}")
    except ValueError as exc:
        blockers.append(str(exc))

    trace_labeled_rows = (
        sum(_trace_labeled(row, trace_modes=trace_modes) for row in rows) if trace_modes else 0
    )
    trace_labeled_fraction = trace_labeled_rows / len(rows) if rows else 0.0
    if rows_loaded and min_fraction is not None and trace_labeled_fraction < min_fraction:
        blockers.append(
            "trace-verified labeled fraction must meet preregistration minimum "
            f"{min_fraction:.3f}; got {trace_labeled_fraction:.3f}"
        )

    status = READY_STATUS if not blockers else BLOCKED_STATUS
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "issue": 5248,
        "job_id": str(job_id),
        "status": status,
        "claim_boundary": (
            "Registration readiness only: this receipt verifies campaign structure and trace-label "
            "coverage for the issue #4206 reanalysis. It does not establish benchmark, planner, "
            "paper, or dissertation claims."
        ),
        "campaign": {
            "total_episodes_expected": expected_total_episodes,
            "total_episodes_observed": (
                _campaign_block(summary).get("total_episodes") if summary is not None else None
            ),
            "campaign_execution_status": (
                _campaign_block(summary).get("campaign_execution_status")
                if summary is not None
                else None
            ),
            "episode_row_count": len(rows),
        },
        "trace_labels": {
            "required_schema_version": MECHANISM_SCHEMA_VERSION,
            "required_fields": list(REQUIRED_MECHANISM_FIELDS),
            "trace_verified_evidence_modes": sorted(trace_modes),
            "minimum_labeled_fraction": min_fraction,
            "trace_labeled_rows": trace_labeled_rows,
            "trace_labeled_fraction": trace_labeled_fraction,
        },
        "source_files": source_files,
        "blockers": blockers,
        "next_action": (
            "Run the issue #4206 mechanism cross-cut builder against this campaign after this receipt "
            "is ready."
            if status == READY_STATUS
            else "Resolve the listed source-artifact or trace-label blockers before reanalysis."
        ),
    }


def _render_markdown(receipt: dict[str, Any]) -> str:
    """Render a compact, human-reviewable registration receipt."""

    campaign = receipt["campaign"]
    trace_labels = receipt["trace_labels"]
    lines = [
        "# Salvaged trace-capable h600 registration receipt",
        "",
        f"- Status: `{receipt['status']}`",
        f"- Job: `{receipt['job_id']}`",
        f"- Claim boundary: {receipt['claim_boundary']}",
        "",
        "| Check | Observed |",
        "| --- | --- |",
        f"| Total episodes | {campaign['total_episodes_observed']} / expected {campaign['total_episodes_expected']} |",
        f"| Execution status | `{campaign['campaign_execution_status']}` |",
        f"| Episode rows | {campaign['episode_row_count']} |",
        f"| Trace-labeled rows | {trace_labels['trace_labeled_rows']} ({trace_labels['trace_labeled_fraction']:.3f}) |",
        f"| Minimum trace-labeled fraction | {trace_labels['minimum_labeled_fraction']} |",
        "",
    ]
    if receipt["blockers"]:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in receipt["blockers"])
    else:
        lines.extend(["## Next action", "", f"{receipt['next_action']}"])
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-total-episodes", type=int, default=6480)
    parser.add_argument(
        "--preregistration-config",
        type=Path,
        default=Path("configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the receipt and return nonzero when registration is blocked."""

    args = _parse_args(argv)
    if args.expected_total_episodes <= 0:
        raise ValueError("--expected-total-episodes must be positive")
    receipt = build_registration_receipt(
        campaign_root=args.campaign_root,
        job_id=args.job_id,
        expected_total_episodes=args.expected_total_episodes,
        preregistration_config=args.preregistration_config,
        generated_at=args.generated_at,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "registration.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "registration.md").write_text(_render_markdown(receipt), encoding="utf-8")
    print(f"status: {receipt['status']}")
    return 0 if receipt["status"] == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
