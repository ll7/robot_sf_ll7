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
    missing_in_base = payload.get("missing_in_base", [])
    missing_in_candidate = payload.get("missing_in_candidate", [])
    lines.extend(["", "## Coverage Gaps", ""])
    if missing_in_base:
        lines.append(f"- Missing in base campaign: `{', '.join(missing_in_base)}`")
    if missing_in_candidate:
        lines.append(f"- Missing in candidate campaign: `{', '.join(missing_in_candidate)}`")
    if not missing_in_base and not missing_in_candidate:
        lines.append("- No planner coverage gaps.")
    lines.append("")
    return "\n".join(lines) + "\n"


def compare_campaigns(base_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Return planner-metric deltas for two campaigns."""
    base_summary = _read_summary(base_root)
    candidate_summary = _read_summary(candidate_root)
    base_rows = _planner_rows_by_key(base_summary)
    candidate_rows = _planner_rows_by_key(candidate_summary)

    common_planners = sorted(set(base_rows) & set(candidate_rows))
    missing_in_base = sorted(set(candidate_rows) - set(base_rows))
    missing_in_candidate = sorted(set(base_rows) - set(candidate_rows))

    planner_deltas: list[dict[str, Any]] = []
    for planner_key in common_planners:
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
        "missing_in_base": missing_in_base,
        "missing_in_candidate": missing_in_candidate,
    }


def _resolve_safe_output_path(path: Path, safe_root: Path) -> Path:
    """Resolve output path and require that it stays inside safe_root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(safe_root):
        raise ValueError(f"Unsafe output path outside {safe_root}: {path}")
    return resolved


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
    safe_root = Path.cwd().resolve()
    output_json = _resolve_safe_output_path(args.output_json, safe_root)
    output_md = _resolve_safe_output_path(args.output_md, safe_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {"output_json": str(output_json), "output_md": str(output_md)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
