#!/usr/bin/env python3
"""Compare two camera-ready campaign summaries at planner-metric level."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_METRICS: tuple[str, ...] = (
    "success_mean",
    "collisions_mean",
    "near_misses_mean",
    "snqi_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
    "comfort_exposure_mean",
    "jerk_mean",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _read_summary(campaign_root: Path) -> dict[str, Any]:
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _planner_rows_by_key(summary_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary_payload.get("planner_rows")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        planner_key = row.get("planner_key")
        if isinstance(planner_key, str) and planner_key:
            out[planner_key] = row
    return out


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Camera-Ready Campaign Comparison",
        "",
        f"- Base campaign: `{payload['base_campaign_id']}`",
        f"- Candidate campaign: `{payload['candidate_campaign_id']}`",
        "",
        "## Planner Deltas",
        "",
        "| planner | metric | base | candidate | delta(candidate-base) |",
        "|---|---|---:|---:|---:|",
    ]
    for planner in payload["planner_deltas"]:
        planner_key = planner["planner_key"]
        for metric, values in planner["metrics"].items():
            lines.append(
                "| "
                f"{planner_key} | {metric} | {values['base']:.4f} | {values['candidate']:.4f} | "
                f"{values['delta']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def compare_campaigns(base_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Return planner-metric deltas for two campaigns."""
    base_summary = _read_summary(base_root)
    candidate_summary = _read_summary(candidate_root)
    base_rows = _planner_rows_by_key(base_summary)
    candidate_rows = _planner_rows_by_key(candidate_summary)

    planner_deltas: list[dict[str, Any]] = []
    for planner_key in sorted(set(base_rows) & set(candidate_rows)):
        base_row = base_rows[planner_key]
        candidate_row = candidate_rows[planner_key]
        metrics: dict[str, dict[str, float]] = {}
        for metric in _METRICS:
            base_value = _safe_float(base_row.get(metric))
            candidate_value = _safe_float(candidate_row.get(metric))
            if base_value is None or candidate_value is None:
                continue
            metrics[metric] = {
                "base": base_value,
                "candidate": candidate_value,
                "delta": candidate_value - base_value,
            }
        planner_deltas.append(
            {
                "planner_key": planner_key,
                "base_status": str(base_row.get("status", "unknown")),
                "candidate_status": str(candidate_row.get("status", "unknown")),
                "base_episodes": int(base_row.get("episodes", 0) or 0),
                "candidate_episodes": int(candidate_row.get("episodes", 0) or 0),
                "metrics": metrics,
            }
        )

    return {
        "base_campaign_id": str(
            (base_summary.get("campaign") or {}).get("campaign_id", base_root.name),
        ),
        "candidate_campaign_id": str(
            (candidate_summary.get("campaign") or {}).get("campaign_id", candidate_root.name),
        ),
        "base_campaign_root": str(base_root),
        "candidate_campaign_root": str(candidate_root),
        "planner_deltas": planner_deltas,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-campaign-root", type=Path, required=True)
    parser.add_argument("--candidate-campaign-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> int:
    """CLI entry point."""
    args = _build_parser().parse_args()
    payload = compare_campaigns(
        args.base_campaign_root.resolve(), args.candidate_campaign_root.resolve()
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
