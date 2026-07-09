#!/usr/bin/env python3
"""Derive trace-verified failure-mechanism labels from h600 rerun episode JSONL.

Reads campaign episode JSONL files and derives failure-mechanism labels from
event_ledger, outcome, and safety predicates. Produces mechanism sidecars
that can be consumed by the #4206 crosscut builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    validate_failure_mechanism_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = "issue_4831_mechanism_label.v1"
# Evidence-hygiene marker required on every new docs/context/evidence artifact
# (see scripts/ci/pr_contract_check.py: AI-GENERATED / NEEDS-REVIEW convention).
REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"
_CSV_MARKER_COMMENT = f"# {REVIEW_MARKER}"
_README_MARKER_COMMENT = "<!-- AI-GENERATED (robot_sf#4831) - NEEDS-REVIEW -->"
OUTPUT_DIR_DEFAULT = Path("docs/context/evidence/issue_4831_trace_verified_failure_mechanisms")
CAMPAIGN_ROOT_DEFAULT = Path(
    "output/benchmarks/camera_ready/issue4206_trace_capable_h600_rerun_20260704"
)


class DerivationError(ValueError):
    """Raised when mechanism derivation encounters irrecoverable input error."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DerivationError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv_path(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"{_CSV_MARKER_COMMENT}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_jsonl_path(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def _public_path(path: Path) -> str:
    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _is_failure(outcome: Mapping[str, Any]) -> bool:
    return not bool(outcome.get("route_complete", False))


def _is_collision(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome.get("collision_event", False))


def _is_timeout(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome.get("timeout_event", False))


def _get_surrogate(el: Mapping[str, Any], name: str) -> bool | None:
    surrogate = el.get("surrogate_events")
    if not isinstance(surrogate, dict):
        return None
    val = surrogate.get(name)
    if val is None or isinstance(val, dict):
        return None
    return bool(val)


def _get_event_ledger(ep: dict[str, Any]) -> dict[str, Any]:
    return ep.get("event_ledger") or {}


def _get_metrics(ep: dict[str, Any]) -> dict[str, Any]:
    return ep.get("metrics") or {}


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _derive_mechanism_label(row: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0915
    """Derive a trace-verified mechanism label from episode trace surfaces.

    Returns a validated taxonomy record, or an unknown record with explicit caveat.
    """
    outcome = row.get("outcome") or {}
    if not _is_failure(outcome):
        return None  # type: ignore[return-value]

    event_ledger = _get_event_ledger(row)
    metrics = _get_metrics(row)

    has_collision = _is_collision(outcome)
    has_timeout = _is_timeout(outcome)

    near_miss_count = _to_float(metrics.get("near_misses")) or 0.0
    total_collision_count = _to_float(metrics.get("total_collision_count")) or 0.0
    stalled_time = _to_float(metrics.get("stalled_time")) or 0.0
    avg_speed = _to_float(metrics.get("avg_speed")) or 0.0
    path_efficiency = _to_float(metrics.get("path_efficiency")) or 0.0

    surrogate_late_evasive = _get_surrogate(event_ledger, "late_evasive") or False
    surrogate_clearance_breach = _get_surrogate(event_ledger, "clearance_breach") or False
    surrogate_oscillation = _get_surrogate(event_ledger, "oscillation") or False

    label = "unknown"
    confidence = "unknown"
    evidence_mode = "unknown"
    caveat = "not_derivable_from_trace_surfaces"
    evidence_uri = ""

    if has_collision:
        evidence_uri = f"event_ledger.collision_events({total_collision_count})"

        if surrogate_late_evasive and surrogate_clearance_breach:
            label = "proxemic_or_clearance_tradeoff"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = (
                "collision with late_evasive and clearance_breach surrogate events "
                "indicates insufficient clearance response"
            )
        elif total_collision_count > 0:
            label = "proxemic_or_clearance_tradeoff"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = (
                f"collision with {total_collision_count} event(s); "
                "clearance tradeoff derived from collision evidence"
            )
        else:
            label = "proxemic_or_clearance_tradeoff"
            confidence = "weak_hypothesis"
            evidence_mode = "paired_trace"
            caveat = "collision outcome present but limited trace detail"

    elif has_timeout:
        evidence_uri = "outcome.timeout_event"

        if avg_speed < 0.01:
            label = "static_deadlock_or_local_minimum"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = f"timeout with avg_speed={avg_speed:.4f}; static deadlock suspected"
        elif stalled_time > 10.0:
            label = "static_deadlock_or_local_minimum"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = f"timeout with stalled_time={stalled_time:.1f}s; robot stalled before timeout"
        elif path_efficiency < 0.2:
            label = "static_deadlock_or_local_minimum"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = (
                f"timeout with path_efficiency={path_efficiency:.4f}; "
                "severely limited progress indicates local minimum"
            )
        elif surrogate_oscillation:
            label = "dynamic_phase_or_order_sensitivity"
            confidence = "supported_hypothesis"
            evidence_mode = "paired_trace"
            caveat = "timeout with oscillatory behavior; negotiation failure or order sensitivity"
        else:
            label = "time_budget_artifact"
            confidence = "weak_hypothesis"
            evidence_mode = "paired_trace"
            caveat = (
                f"timeout with moderate path_efficiency={path_efficiency:.4f}; "
                "may be time budget constraint rather than deadlock"
            )

    # Failure without explicit collision or timeout marker
    elif surrogate_oscillation:
        label = "dynamic_phase_or_order_sensitivity"
        confidence = "supported_hypothesis"
        evidence_mode = "paired_trace"
        caveat = "oscillation surrogate without collision/timeout"
    elif near_miss_count > 5:
        label = "proxemic_or_clearance_tradeoff"
        confidence = "weak_hypothesis"
        evidence_mode = "paired_trace"
        caveat = f"high near_miss count ({near_miss_count}) without collision event"
    else:
        label = "unknown"
        confidence = "unknown"
        evidence_mode = "unknown"
        caveat = "failure outcome but insufficient trace surfaces for derivation"

    record = {
        "mechanism_schema_version": MECHANISM_SCHEMA_VERSION,
        "mechanism_label": label,
        "mechanism_confidence": confidence,
        "mechanism_evidence_mode": evidence_mode,
        "mechanism_evidence_uri": evidence_uri,
        "mechanism_case_id": row.get("episode_id") or "",
        "mechanism_caveat": caveat,
    }

    try:
        validate_failure_mechanism_record(record)
    except ValueError as e:
        record = {
            "mechanism_schema_version": MECHANISM_SCHEMA_VERSION,
            "mechanism_label": "unknown",
            "mechanism_confidence": "unknown",
            "mechanism_evidence_mode": "unknown",
            "mechanism_evidence_uri": "",
            "mechanism_case_id": row.get("episode_id") or "",
            "mechanism_caveat": f"validation_failed: {e}",
        }

    return record


def _blocked_input_summary(
    campaign_root: Path, output_dir: Path, generated_at: str
) -> dict[str, Any]:
    """Write a blocked-input marker and return the blocked summary.

    Phase 1 input audit: a missing/empty campaign must produce an explicit
    ``blocked_missing_input_artifacts`` status and stop before derivation,
    rather than emitting a vacuous "completed" packet (anti-fabrication guard
    for the issue's "blocks cleanly until hydrated" criterion).
    """
    reason = (
        "no episode rows found under runs/*/episodes.jsonl"
        if campaign_root.is_dir()
        else "campaign_root is not a directory"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "input_status.json",
        [
            {
                "schema_version": SCHEMA_VERSION,
                "issue": 4831,
                "generated_at": generated_at,
                "status": "blocked_missing_input_artifacts",
                "campaign_root": str(campaign_root),
                "reason": reason,
            }
        ],
    )
    return {
        "status": "blocked_missing_input_artifacts",
        "issue": 4831,
        "generated_at": generated_at,
        "reason": reason,
    }


def _scan_episode_jsonl(runs_dir: Path) -> list[dict[str, Any]]:
    """Scan all planner run directories for episodes.jsonl files."""
    all_episodes: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return all_episodes

    for planner_dir in sorted(runs_dir.iterdir()):
        if not planner_dir.is_dir():
            continue
        episodes_path = planner_dir / "episodes.jsonl"
        if not episodes_path.is_file():
            continue
        episodes = _read_jsonl(episodes_path)
        for ep in episodes:
            ep["_planner_run"] = planner_dir.name
            ep["_episodes_file"] = _public_path(episodes_path)
        all_episodes.extend(episodes)
    return all_episodes


def build_mechanism_sidecar(  # noqa: C901
    campaign_root: Path,
    output_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Build the trace-verified mechanism-label sidecar.

    Args:
        campaign_root: Campaign output root with runs/{planner}/episodes.jsonl
        output_dir: Output directory for sidecar artifacts
        generated_at: ISO 8601 timestamp

    Returns:
        Summary metadata dictionary.
    """
    runs_dir = campaign_root / "runs"
    manifest_path = campaign_root / "campaign_manifest.json"

    all_episodes = _scan_episode_jsonl(runs_dir)
    total = len(all_episodes)

    # Phase 1 input audit: block cleanly when campaign artifacts are absent.
    # A missing/empty campaign must NOT produce a vacuous "completed" packet,
    # because that would look indistinguishable from a genuine zero-failure run
    # (anti-fabrication guard for the issue's "blocks cleanly until hydrated" criterion).
    if not campaign_root.is_dir() or total == 0:
        return _blocked_input_summary(campaign_root, output_dir, generated_at)

    failure_episodes = []
    success_episodes = 0
    skipped_unavailable: int = 0

    for ep in all_episodes:
        outcome = ep.get("outcome") or {}
        planner_key = ep.get("algo") or ep.get("_planner_run", "").split("__")[0]

        if planner_key == "guarded_ppo" and ep.get("status") != "ok":
            skipped_unavailable += 1
            continue

        if _is_failure(outcome):
            failure_episodes.append(ep)
        else:
            success_episodes += 1

    labeled: list[dict[str, Any]] = []
    unlabeled: list[dict[str, Any]] = []

    for ep in failure_episodes:
        record = _derive_mechanism_label(ep)
        if record is None:
            continue  # Shouldn't happen for failure episodes

        episode_id = ep.get("episode_id", "")
        scenario_id = ep.get("scenario_id", "")
        seed = ep.get("seed", "")
        planner_key_derived = ep.get("algo") or (
            ep.get("_planner_run", "").split("__")[0] if ep.get("_planner_run") else ""
        )

        base_row = {
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "planner_key": planner_key_derived,
            "seed": seed,
            "source_run": ep.get("_planner_run", ""),
            "source_file": ep.get("_episodes_file", ""),
            "review_marker": REVIEW_MARKER,
        }

        label = record.get("mechanism_label", "unknown")
        if label == "unknown" and record.get("mechanism_confidence") == "unknown":
            unlabeled.append({**base_row, **record})
        else:
            labeled.append({**base_row, **record})

    label_counts = Counter(r.get("mechanism_label", "unknown") for r in labeled)
    confidence_counts = Counter(r.get("mechanism_confidence", "unknown") for r in labeled)
    planner_failures: dict[str, dict] = defaultdict(lambda: {"total": 0, "labeled": 0})

    for ep in failure_episodes:
        pk = ep.get("algo") or (
            ep.get("_planner_run", "").split("__")[0] if ep.get("_planner_run") else ""
        )
        planner_failures[pk]["total"] += 1

    for r in labeled:
        pk = r.get("planner_key", "")
        planner_failures[pk]["labeled"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    mechanism_fieldnames = [
        "episode_id",
        "scenario_id",
        "planner_key",
        "seed",
        "source_run",
        "source_file",
        "mechanism_schema_version",
        "mechanism_label",
        "mechanism_confidence",
        "mechanism_evidence_mode",
        "mechanism_evidence_uri",
        "mechanism_case_id",
        "mechanism_caveat",
    ]

    full_labeled = list(labeled)
    full_labeled.extend(unlabeled)

    _write_csv_path(output_dir / "mechanism_labels.csv", mechanism_fieldnames, full_labeled)
    _write_jsonl_path(output_dir / "mechanism_labels.jsonl", full_labeled)

    residual_path = output_dir / "residual_unlabeled.jsonl"
    if unlabeled:
        _write_jsonl_path(residual_path, unlabeled)
    else:
        # No residual rows: keep the file empty of data but carry the
        # evidence-hygiene marker as a JSONL comment line.
        residual_path.write_text(f"{_CSV_MARKER_COMMENT}\n", encoding="utf-8")

    coverage = {
        "review_marker": REVIEW_MARKER,
        "schema_version": SCHEMA_VERSION,
        "issue": 4831,
        "generated_at": generated_at,
        "campaign_root": _public_path(campaign_root),
        "manifest_path": _public_path(manifest_path) if manifest_path.exists() else "",
        "total_episodes": total,
        "success_episodes": success_episodes,
        "failure_episodes": len(failure_episodes),
        "skipped_unavailable": skipped_unavailable,
        "labeled_failure_count": len(labeled),
        "unlabeled_failure_count": len(unlabeled),
        "coverage_fraction": len(labeled) / len(failure_episodes) if failure_episodes else 0.0,
        "label_distribution": dict(label_counts),
        "confidence_distribution": dict(confidence_counts),
        "planner_failure_counts": {k: dict(v) for k, v in sorted(planner_failures.items())},
        "guarded_ppo_status": "accepted_unavailable",
    }
    _write_jsonl(output_dir / "label_coverage.json", [coverage])

    selection_manifest = {
        "review_marker": REVIEW_MARKER,
        "failure_selection_policy": "collision_or_timeout_or_failure_to_progress",
        "selected_failure_episode_count": len(failure_episodes),
        "excluded_success_episode_count": success_episodes,
        "excluded_accepted_unavailable": skipped_unavailable,
        "guarded_ppo_excluded": True,
    }
    _write_jsonl(output_dir / "selection_manifest.json", [selection_manifest])

    summary = {
        "status": "completed",
        "issue": 4831,
        "generated_at": generated_at,
        "labeled_count": len(labeled),
        "failure_count": len(failure_episodes),
        "coverage_fraction": coverage["coverage_fraction"],
    }

    _write_readmes(output_dir, coverage, summary)
    return summary


def _write_readmes(
    output_dir: Path, coverage: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    lines = [
        _README_MARKER_COMMENT,
        "",
        "# Issue #4831: Trace-Verified Failure-Mechanism Labels",
        "",
        f"Generated: {coverage.get('generated_at', summary.get('generated_at', 'N/A'))}",
        "",
        f"Campaign root: {coverage.get('campaign_root', 'N/A')}",
        "",
        f"- Total episodes: {coverage.get('total_episodes', 0)}",
        f"- Success: {coverage.get('success_episodes', 0)}",
        f"- Failure: {coverage.get('failure_episodes', 0)}",
        f"- Labeled: {coverage.get('labeled_failure_count', 0)}",
        f"- Unlabeled residual: {coverage.get('unlabeled_failure_count', 0)}",
        f"- Coverage: {coverage.get('coverage_fraction', 0):.2%}",
        "",
        "## guarded_ppo",
        "",
        "guarded_ppo is excluded from derivation (accepted-unavailable status).",
        "",
        "## Label Distribution",
        "",
    ]
    for label, count in sorted(
        (coverage.get("label_distribution") or {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"- {label}: {count}")

    lines += [
        "",
        "## Evidence Mode",
        "All derived labels use `paired_trace` evidence from event_ledger, outcome,",
        "and safety predicates in the episode JSONL records.",
        "",
        "## Claim Boundary",
        "This packet derives failure-mechanism labels from trace surfaces available in",
        "the episode JSONL. Confidence levels reflect the strength of derivation evidence.",
        "Labels are diagnostic, not paper-facing claims.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the mechanism-label derivation CLI."""
    parser = argparse.ArgumentParser(
        description="Derive trace-verified failure-mechanism labels from episode JSONL"
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=CAMPAIGN_ROOT_DEFAULT,
        help="Campaign output root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for mechanism sidecars",
    )
    parser.add_argument(
        "--generated-at",
        default="now",
        help="ISO 8601 timestamp (default: now)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args(argv or sys.argv[1:])

    generated_at = (
        datetime.now(UTC).isoformat(timespec="seconds")
        if args.generated_at == "now"
        else str(args.generated_at)
    )

    campaign_root = Path(args.campaign_root).resolve()
    if not campaign_root.is_dir():
        print(f"error: campaign root not found: {campaign_root}", file=sys.stderr)
        return 1

    try:
        summary = build_mechanism_sidecar(campaign_root, args.output_dir, generated_at)
    except DerivationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if summary.get("status") == "blocked_missing_input_artifacts":
        print(
            f"status: {summary.get('status')}: {summary.get('reason')}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"status: {summary.get('status')}")
        print(f"failure_count: {summary.get('failure_count')}")
        print(f"labeled_count: {summary.get('labeled_count')}")
        print(f"coverage: {summary.get('coverage_fraction', 0):.2%}")
        print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
