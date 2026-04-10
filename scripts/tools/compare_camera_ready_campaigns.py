#!/usr/bin/env python3
"""Compare two camera-ready campaign summaries at planner-metric level."""

from __future__ import annotations

import argparse
import hashlib
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
    if math.isnan(parsed) or math.isinf(parsed):
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


def _planner_row_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical signature for a planner summary row."""
    metrics: dict[str, float] = {}
    for metric in _METRICS:
        value = _safe_float(row.get(metric))
        if value is not None:
            metrics[metric] = value
    return {
        "status": str(row.get("status", "unknown")),
        "episodes": int(row.get("episodes", 0) or 0),
        "metrics": metrics,
    }


def _planner_row_digest(signature: dict[str, Any]) -> str:
    """Return a stable digest for a planner row signature."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Camera-Ready Campaign Comparison",
        "",
        f"- Base campaign: `{payload['base_campaign_id']}`",
        f"- Candidate campaign: `{payload['candidate_campaign_id']}`",
        (
            "- Reproducibility verdict: "
            f"`{payload.get('reproducibility', {}).get('status', 'unknown')}`"
        ),
        "",
        "## Planner Deltas",
        "",
        "| planner | base_status | candidate_status | base_episodes | candidate_episodes | exact_match | metric | base | candidate | delta(candidate-base) |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for planner in payload["planner_deltas"]:
        planner_key = planner["planner_key"]
        base_status = planner.get("base_status", "unknown")
        candidate_status = planner.get("candidate_status", "unknown")
        base_episodes = int(planner.get("base_episodes", 0) or 0)
        candidate_episodes = int(planner.get("candidate_episodes", 0) or 0)
        exact_match = "yes" if planner.get("exact_match") else "no"
        metrics = planner.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            for metric, values in metrics.items():
                lines.append(
                    "| "
                    f"{planner_key} | {base_status} | {candidate_status} | "
                    f"{base_episodes} | {candidate_episodes} | {exact_match} | {metric} | "
                    f"{values['base']:.4f} | {values['candidate']:.4f} | {values['delta']:.4f} |"
                )
            continue
        lines.append(
            "| "
            f"{planner_key} | {base_status} | {candidate_status} | "
            f"{base_episodes} | {candidate_episodes} | {exact_match} | N/A | N/A | N/A | N/A |"
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
    reproducibility = payload.get("reproducibility")
    if isinstance(reproducibility, dict):
        lines.extend(["", "## Reproducibility", ""])
        lines.append(f"- Status: `{reproducibility.get('status', 'unknown')}`")
        exact = reproducibility.get("exact_match_planners", [])
        mismatched = reproducibility.get("mismatched_planners", [])
        if exact:
            lines.append(f"- Exact-match planners: `{', '.join(map(str, exact))}`")
        if mismatched:
            lines.append(f"- Mismatched planners: `{', '.join(map(str, mismatched))}`")
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
    exact_match_planners: list[str] = []
    mismatched_planners: list[str] = []
    for planner_key in common_planners:
        base_row = base_rows[planner_key]
        candidate_row = candidate_rows[planner_key]
        base_signature = _planner_row_signature(base_row)
        candidate_signature = _planner_row_signature(candidate_row)
        exact_match = base_signature == candidate_signature
        if exact_match:
            exact_match_planners.append(planner_key)
        else:
            mismatched_planners.append(planner_key)
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
                "exact_match": exact_match,
                "base_signature_sha256": _planner_row_digest(base_signature),
                "candidate_signature_sha256": _planner_row_digest(candidate_signature),
                "metrics": metrics,
            }
        )

    reproducibility_status = "reproduced"
    if missing_in_base or missing_in_candidate or mismatched_planners:
        reproducibility_status = "drift_detected"

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
        "reproducibility": {
            "status": reproducibility_status,
            "exact_match_planners": exact_match_planners,
            "mismatched_planners": mismatched_planners,
            "coverage_complete": not missing_in_base and not missing_in_candidate,
        },
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
    parser.add_argument(
        "--require-identical",
        action="store_true",
        help="Exit non-zero when the comparison reports drift instead of exact reproducibility.",
    )
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
    if args.require_identical and payload.get("reproducibility", {}).get("status") != "reproduced":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
