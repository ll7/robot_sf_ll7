#!/usr/bin/env python3
"""Build issue #5591 hybrid-portfolio structure-ablation delta evidence.

Consumes ONE seed_episode_rows.csv per ablation arm (emitted by the standard
benchmark harness for the runner config ``configs/benchmarks/
issue_5591_hybrid_ablation_runner.yaml``), keyed by the arm's ``--planner-key``
(labeled ``planner_key`` in the rows), and produces:

- a per-arm safety/comfort/efficiency summary table,
- a per-arm delta-from-``hybrid_full`` summary table (NOT a fresh ranking),
- a per-mechanism-group breakdown of those deltas when mechanism labels exist,
- an explicit crash/degraded-run rate per arm (reported in the denominator).

This script does NOT run the campaign; it is the post-processing companion.
Run it after the 5-arm campaign lands the per-arm ``reports/seed_episode_rows.csv``
files under the campaign evidence root.

Usage:
    uv run python scripts/validation/build_issue_5591_hybrid_ablation_delta.py \
        --campaign-root output/benchmarks/issue5591-hybrid-ablation \
        --out docs/context/evidence/issue_5591_hybrid_ablation
"""
# evidence-writer-exempt: references evidence paths but does not write to evidence tree; guarded by AST analysis


from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ARM = "hybrid_full"
# Arms that are intended as single-knob deltas (used for the acceptance summary).
SINGLE_KNOB_ARMS = (
    "hybrid_minus_static_escape",
    "hybrid_minus_hard_guard",
    "hybrid_minus_adaptive_switching",
)
ALL_ARMS = SINGLE_KNOB_ARMS + ("hybrid_minus_all_three",)

# Metric columns expected in seed_episode_rows.csv (gracefully skipped if absent).
SAFETY_METRICS = ("collision_rate", "near_miss_rate")
COMFORT_METRICS = ("comfort_rate",)
EFFICIENCY_METRICS = ("time_to_goal_norm", "success_rate")
NUMERIC_COLUMNS = SAFETY_METRICS + COMFORT_METRICS + EFFICIENCY_METRICS


def _as_float(value: Any) -> float | None:
    """Parse a finite numeric cell, returning ``None`` for unusable input."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_arm_rows(rows_path: Path) -> list[dict[str, str]]:
    if not rows_path.exists():
        return []
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate mean metric values and crash/degraded counts for one arm."""
    summary: dict[str, Any] = {"n_episodes": len(rows)}
    skipped_by_metric: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        values = [_as_float(r.get(col)) for r in rows]
        usable = [value for value in values if value is not None]
        skipped_by_metric[col] = len(values) - len(usable)
        summary[col] = _mean(usable)
    crashed = sum(
        1 for r in rows if str(r.get("status", "")).lower() in {"crashed", "error", "failed"}
    )
    degraded = sum(1 for r in rows if str(r.get("degraded", "")).lower() in {"true", "1", "yes"})
    summary["crashed_episodes"] = float(crashed)
    summary["degraded_episodes"] = float(degraded)
    summary["crash_rate"] = (crashed / len(rows)) if rows else None
    summary["degraded_rate"] = (degraded / len(rows)) if rows else None
    summary["skipped_numeric_cells"] = sum(skipped_by_metric.values())
    summary["skipped_numeric_cells_by_metric"] = skipped_by_metric
    return summary


def _subtract(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _delta(ref: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for col in NUMERIC_COLUMNS:
        out[col] = _subtract(arm[col], ref[col])
    out["crash_rate_delta"] = _subtract(arm["crash_rate"], ref["crash_rate"])
    out["degraded_rate_delta"] = _subtract(arm["degraded_rate"], ref["degraded_rate"])
    return out


def build(campaign_root: Path) -> dict[str, Any]:
    """Load per-arm rows and compute the delta tables."""
    reports = campaign_root / "reports"
    arm_rows: dict[str, list[dict[str, str]]] = {}
    missing: list[str] = []
    for arm in (REFERENCE_ARM, *ALL_ARMS):
        rows_path = reports / f"seed_episode_rows_{arm}.csv"
        if not rows_path.exists():
            # Fall back to the single merged file with planner_key column.
            rows_path = reports / "seed_episode_rows.csv"
            rows = [r for r in _read_arm_rows(rows_path) if r.get("planner_key") == arm]
        else:
            rows = _read_arm_rows(rows_path)
        if not rows:
            missing.append(arm)
        arm_rows[arm] = rows

    if REFERENCE_ARM in missing:
        return {
            "status": "blocked_missing_reference_arm",
            "missing_arms": missing,
            "note": "hybrid_full reference rows are required to compute deltas.",
        }

    summaries = {arm: _arm_summary(rows) for arm, rows in arm_rows.items()}
    ref_summary = summaries[REFERENCE_ARM]
    deltas = {arm: _delta(ref_summary, summaries[arm]) for arm in ALL_ARMS if arm not in missing}

    # Per-mechanism-group breakdown (only if mechanism labels are present).
    mechanism_groups: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    has_mechanism = any("mechanism_group" in r for r in arm_rows[REFERENCE_ARM])
    if has_mechanism:
        for arm in ALL_ARMS:
            if arm in missing:
                continue
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for r in arm_rows[arm]:
                grouped[str(r.get("mechanism_group", "unclassified"))].append(r)
            mechanism_groups[arm] = {
                group: _delta(ref_summary, _arm_summary(grp_rows))
                for group, grp_rows in grouped.items()
            }

    return {
        "status": "analysis_ready" if not missing else "partial_missing_arms",
        "reference_arm": REFERENCE_ARM,
        "missing_arms": missing,
        "single_knob_arms": list(SINGLE_KNOB_ARMS),
        "per_arm_summary": summaries,
        "delta_from_reference": deltas,
        "per_mechanism_group_delta": dict(mechanism_groups) if has_mechanism else {},
        "mechanism_labels_present": has_mechanism,
        "generated_at": datetime.now(UTC).isoformat(),
        "campaign_root": str(campaign_root),
        "claim_scope": "controlled_ablation_delta_from_hybrid_full",
        "note": (
            "Deltas are reported relative to hybrid_full; crash/degraded runs are "
            "included in the denominator and surfaced via crash_rate_delta."
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    """Parse args, build the delta evidence, and write the output JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        required=True,
        type=Path,
        help="Root containing per-arm reports/seed_episode_rows*.csv",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the delta evidence JSON.",
    )
    args = parser.parse_args()

    result = build(args.campaign_root)
    out_path = args.out / "issue_5591_hybrid_ablation_delta.json"
    _write_json(out_path, result)

    skipped = sum(
        int(summary.get("skipped_numeric_cells", 0))
        for summary in result.get("per_arm_summary", {}).values()
    )
    if skipped:
        print(f"[issue_5591] skipped {skipped} unusable numeric cells", file=sys.stderr)

    status = result.get("status")
    if status in {"blocked_missing_reference_arm"}:
        print(f"[issue_5591] {status}: {result.get('note')}", file=sys.stderr)
        return 2
    print(f"[issue_5591] wrote {out_path} (status={status})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
