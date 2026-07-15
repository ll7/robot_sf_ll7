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


def _public_path(path: Path) -> str:
    """Return a repo-root-relative key for an evidence path.

    Avoids embedding absolute host paths in receipts (see friction fixes #5677,
    #5679): the registration receipt must stay portable across hosts. Falls back
    to the file name when no repo anchor is present.
    """

    resolved = path.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


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
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < float(fraction) <= 1
    ):
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


def _load_mechanism_sidecar(path: Path) -> tuple[list[dict[str, str]], str]:
    """Load a derived mechanism-label sidecar CSV, skipping leading comment lines.

    The #4831 derivation builder emits trace-verified labels as a *post-hoc* sidecar
    (the campaign's own ``seed_episode_rows.csv`` always carries ``unknown`` labels
    because mechanism labels are derived after the run, not written at run time).
    This loader lets the registration checker consume that derivation so the receipt
    reflects the campaign's true label coverage instead of the pre-derivation rows.
    """

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            # The sidecar may start with one or more ``# ...`` review-marker
            # comment lines. Rewind to the first non-comment line so the CSV
            # header is detected regardless of its column order.
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    handle.seek(position)
                    break
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read mechanism sidecar {path.name}: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"invalid CSV in mechanism sidecar {path.name}: {exc}") from exc
    if "episode_id" not in (fieldnames or []):
        raise ValueError(f"mechanism sidecar {path.name} must contain an episode_id column")
    missing = [field for field in REQUIRED_MECHANISM_FIELDS if field not in (fieldnames or [])]
    if missing:
        raise ValueError(f"mechanism sidecar {path.name} missing taxonomy fields: {missing}")
    return rows, _sha256(path)


def _index_sidecar_by_episode(sidecar_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index sidecar rows by ``episode_id``, keeping the first occurrence."""

    index: dict[str, dict[str, str]] = {}
    for sidecar_row in sidecar_rows:
        episode_id = str(sidecar_row.get("episode_id") or "").strip()
        if episode_id:
            index.setdefault(episode_id, sidecar_row)
    return index


def _load_sidecar_overlay(
    mechanism_sidecar: Path,
    *,
    source_files: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], str | None, str | None, str | None]:
    """Load an optional mechanism sidecar and record its provenance.

    Returns ``(rows, sha256, public_path, load_error)``. A load error is reported
    to the caller rather than raised so the receipt still records the sidecar's
    path and the fact that it could not be consumed.
    """

    sidecar_public = _public_path(mechanism_sidecar)
    try:
        rows, sha256 = _load_mechanism_sidecar(mechanism_sidecar)
    except (ValueError, OSError) as exc:
        return [], None, sidecar_public, str(exc)
    source_files[sidecar_public] = {
        "sha256": sha256,
        "size_bytes": mechanism_sidecar.stat().st_size,
    }
    return rows, sha256, sidecar_public, None


def _compute_trace_coverage(
    rows: list[dict[str, str]],
    *,
    trace_modes: set[str],
    sidecar_by_episode: dict[str, dict[str, str]],
) -> tuple[int, int, int, float, float]:
    """Return trace-labeled counts with and without the sidecar overlay.

    The sidecar overlay counts a raw row as trace-labeled when either the raw
    row itself qualifies or a derived sidecar label for its ``episode_id``
    qualifies. Returns ``(with_sidecar, without_sidecar, matched, frac, frac_raw)``.
    """

    def _row_labeled(row: dict[str, str]) -> bool:
        if _trace_labeled(row, trace_modes=trace_modes):
            return True
        episode_id = str(row.get("episode_id") or "").strip()
        sidecar_row = sidecar_by_episode.get(episode_id)
        return sidecar_row is not None and _trace_labeled(sidecar_row, trace_modes=trace_modes)

    without = (
        sum(_trace_labeled(row, trace_modes=trace_modes) for row in rows) if trace_modes else 0
    )
    with_sidecar = sum(_row_labeled(row) for row in rows) if trace_modes else 0
    matched = (
        sum(1 for row in rows if str(row.get("episode_id") or "").strip() in sidecar_by_episode)
        if rows and sidecar_by_episode
        else 0
    )
    denom = len(rows) if rows else 0
    return (
        with_sidecar,
        without,
        matched,
        with_sidecar / denom if denom else 0.0,
        without / denom if denom else 0.0,
    )


def _validate_campaign_summary(
    summary_path: Path,
    *,
    expected_total_episodes: int,
    source_files: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and structurally validate the campaign summary, returning (summary, blockers)."""

    blockers: list[str] = []
    summary: dict[str, Any] | None = None
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
    return summary, blockers


def _validate_episode_rows(
    rows_path: Path,
    *,
    expected_total_episodes: int,
    source_files: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], bool, list[str]]:
    """Load and validate episode rows, returning (rows, rows_loaded, blockers)."""

    blockers: list[str] = []
    rows: list[dict[str, str]] = []
    rows_loaded = False
    try:
        rows, header = _load_rows(rows_path)
        rows_loaded = True
        source_files[EPISODE_ROWS_RELATIVE_PATH.as_posix()] = {
            "sha256": _sha256(rows_path),
            "size_bytes": rows_path.stat().st_size,
        }
        if len(rows) != expected_total_episodes:
            blockers.append(
                f"seed_episode_rows.csv must contain {expected_total_episodes} rows; "
                f"got {len(rows)}"
            )
        missing_fields = [field for field in REQUIRED_MECHANISM_FIELDS if field not in header]
        if missing_fields:
            blockers.append(f"seed_episode_rows.csv missing mechanism fields: {missing_fields}")
    except ValueError as exc:
        blockers.append(str(exc))
    return rows, rows_loaded, blockers


def build_registration_receipt(
    *,
    campaign_root: Path,
    job_id: str,
    expected_total_episodes: int,
    preregistration_config: Path,
    generated_at: str | None = None,
    mechanism_sidecar: Path | None = None,
) -> dict[str, Any]:
    """Check a campaign and return a receipt without promoting its conclusions.

    When ``mechanism_sidecar`` is provided, the checker also consumes the post-hoc
    derived mechanism labels (e.g. the #4831 trace-verified sidecar) and overlays
    them on the campaign's raw episode rows before computing trace-label coverage.
    The campaign's own rows always carry ``unknown`` labels because mechanism labels
    are derived after the run; the sidecar is the only place a trace-verified label
    can appear. The receipt records both the raw-row and sidecar-augmented coverage
    so a reviewer can see exactly what the sidecar contributed.
    """

    blockers: list[str] = []
    source_files: dict[str, dict[str, Any]] = {}
    summary_path = campaign_root / SUMMARY_RELATIVE_PATH
    rows_path = campaign_root / EPISODE_ROWS_RELATIVE_PATH
    summary: dict[str, Any] | None = None
    rows: list[dict[str, str]] = []
    rows_loaded = False
    trace_modes: set[str] = set()
    min_fraction: float | None = None

    try:
        trace_modes, min_fraction = _load_trace_contract(preregistration_config)
    except ValueError as exc:
        blockers.append(str(exc))

    summary, summary_blockers = _validate_campaign_summary(
        summary_path,
        expected_total_episodes=expected_total_episodes,
        source_files=source_files,
    )
    blockers.extend(summary_blockers)

    rows, rows_loaded, rows_blockers = _validate_episode_rows(
        rows_path,
        expected_total_episodes=expected_total_episodes,
        source_files=source_files,
    )
    blockers.extend(rows_blockers)

    # Optional post-hoc derived mechanism-label sidecar (#4831 trace-verified
    # derivation). The campaign's own rows are pre-derivation (always ``unknown``);
    # the sidecar is where a trace-verified label lives. Loading errors are blockers,
    # not a silent skip, so a stale/malformed sidecar cannot mask a real gap.
    sidecar_rows: list[dict[str, str]] = []
    sidecar_sha256: str | None = None
    sidecar_key: str | None = None
    sidecar_load_error: str | None = None
    if mechanism_sidecar is not None:
        sidecar_rows, sidecar_sha256, sidecar_key, sidecar_load_error = _load_sidecar_overlay(
            mechanism_sidecar, source_files=source_files
        )
        if sidecar_load_error is not None:
            blockers.append(sidecar_load_error)

    sidecar_by_episode = _index_sidecar_by_episode(sidecar_rows)
    (
        trace_labeled_rows,
        trace_labeled_rows_without_sidecar,
        sidecar_matched_rows,
        trace_labeled_fraction,
        trace_labeled_fraction_without_sidecar,
    ) = _compute_trace_coverage(
        rows, trace_modes=trace_modes, sidecar_by_episode=sidecar_by_episode
    )
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
            "trace_labeled_rows_without_sidecar": trace_labeled_rows_without_sidecar,
            "trace_labeled_fraction_without_sidecar": trace_labeled_fraction_without_sidecar,
        },
        "mechanism_sidecar": (
            None
            if mechanism_sidecar is None
            else {
                "path": sidecar_key,
                "sha256": sidecar_sha256,
                "row_count": len(sidecar_rows),
                "rows_matched_in_campaign": sidecar_matched_rows,
                "load_error": sidecar_load_error,
            }
        ),
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
    sidecar = receipt.get("mechanism_sidecar")
    if sidecar:
        frac_no_sidecar = trace_labels.get("trace_labeled_fraction_without_sidecar")
        lines.extend(
            [
                "## Mechanism-label sidecar (post-hoc derivation)",
                "",
                f"- Path: `{sidecar.get('path')}`",
                f"- SHA-256: `{sidecar.get('sha256')}`",
                (
                    f"- Sidecar rows: {sidecar.get('row_count')} "
                    f"(matched {sidecar.get('rows_matched_in_campaign')} campaign rows)"
                ),
                (
                    f"- Raw-row trace-labeled fraction (without sidecar): {frac_no_sidecar:.3f}"
                    if isinstance(frac_no_sidecar, float)
                    else "- Raw-row trace-labeled fraction: unavailable"
                ),
            ]
        )
        if sidecar.get("load_error"):
            lines.append(f"- Load error: {sidecar['load_error']}")
        lines.append("")
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
    parser.add_argument(
        "--mechanism-sidecar",
        type=Path,
        default=None,
        help="Optional post-hoc derived mechanism-label sidecar CSV (e.g. the #4831 "
        "trace-verified sidecar) to overlay on the campaign's pre-derivation rows.",
    )
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
        mechanism_sidecar=args.mechanism_sidecar,
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
