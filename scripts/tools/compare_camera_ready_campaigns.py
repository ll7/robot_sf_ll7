#!/usr/bin/env python3
"""Compare two camera-ready campaign summaries at planner, scenario, and family level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_METRICS: tuple[str, ...] = (
    "success_mean",
    "unfinished_mean",
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
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("'"):
            value = value[1:]
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    """Return a metric value, including derived route-incomplete metrics."""
    if metric == "unfinished_mean":
        success = _safe_float(row.get("success_mean"))
        if success is None:
            return None
        return 1.0 - success
    return _safe_float(row.get(metric))


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
        value = _metric_value(row, metric)
        if value is not None:
            metrics[metric] = value
    return {
        "status": str(row.get("status") or "unknown"),
        "episodes": int(row.get("episodes") or 0),
        "metrics": metrics,
    }


def _planner_row_digest(signature: dict[str, Any]) -> str:
    """Return a stable digest for a planner row signature."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_optional_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact when present, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _breakdown_rows_by_key(
    rows: list[dict[str, str]], key_fields: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in key_fields)
        if all(key):
            out[key] = row
    return out


def _row_episodes(row: dict[str, Any]) -> int:
    value = _safe_float(row.get("episodes"))
    if value is None:
        return 0
    return int(value)


def _breakdown_row_signature(row: dict[str, Any], key_fields: tuple[str, ...]) -> dict[str, Any]:
    """Return a canonical signature for a scenario or family breakdown row."""
    metrics: dict[str, float] = {}
    for metric in _METRICS:
        value = _metric_value(row, metric)
        if value is not None:
            metrics[metric] = value
    return {
        "key": {field: str(row.get(field) or "") for field in key_fields},
        "episodes": _row_episodes(row),
        "metrics": metrics,
    }


def _key_to_payload(key_fields: tuple[str, ...], key: tuple[str, ...]) -> dict[str, str]:
    return {field: key[index] for index, field in enumerate(key_fields)}


def _compare_breakdown_artifact(
    base_root: Path,
    candidate_root: Path,
    *,
    filename: str,
    key_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Compare optional scenario/family breakdown CSV artifacts."""
    base_rows = _breakdown_rows_by_key(
        _read_optional_csv_rows(base_root / "reports" / filename), key_fields
    )
    candidate_rows = _breakdown_rows_by_key(
        _read_optional_csv_rows(candidate_root / "reports" / filename), key_fields
    )

    common_keys = sorted(set(base_rows) & set(candidate_rows))
    missing_in_base = sorted(set(candidate_rows) - set(base_rows))
    missing_in_candidate = sorted(set(base_rows) - set(candidate_rows))

    deltas: list[dict[str, Any]] = []
    for key in common_keys:
        base_row = base_rows[key]
        candidate_row = candidate_rows[key]
        base_signature = _breakdown_row_signature(base_row, key_fields)
        candidate_signature = _breakdown_row_signature(candidate_row, key_fields)
        metrics: dict[str, dict[str, float]] = {}
        for metric in _METRICS:
            base_value = _metric_value(base_row, metric)
            candidate_value = _metric_value(candidate_row, metric)
            if base_value is None or candidate_value is None:
                continue
            metrics[metric] = {
                "base": base_value,
                "candidate": candidate_value,
                "delta": candidate_value - base_value,
            }
        row_payload: dict[str, Any] = _key_to_payload(key_fields, key)
        row_payload.update(
            {
                "base_episodes": _row_episodes(base_row),
                "candidate_episodes": _row_episodes(candidate_row),
                "exact_match": base_signature == candidate_signature,
                "base_signature_sha256": _planner_row_digest(base_signature),
                "candidate_signature_sha256": _planner_row_digest(candidate_signature),
                "metrics": metrics,
            }
        )
        deltas.append(row_payload)

    return {
        "deltas": deltas,
        "missing_in_base": [_key_to_payload(key_fields, key) for key in missing_in_base],
        "missing_in_candidate": [_key_to_payload(key_fields, key) for key in missing_in_candidate],
    }


def _success_delta_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    metrics = row.get("metrics")
    success = metrics.get("success_mean") if isinstance(metrics, dict) else None
    delta = success.get("delta") if isinstance(success, dict) else 0.0
    try:
        abs_delta = abs(float(delta))
    except (TypeError, ValueError):
        abs_delta = 0.0
    key_bits = [
        str(row.get("planner_key", "")),
        str(row.get("scenario_family", "")),
        str(row.get("scenario_id", "")),
    ]
    return (-abs_delta, "|".join(key_bits))


def _append_breakdown_markdown(
    lines: list[str],
    *,
    title: str,
    deltas: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    missing_in_base: list[dict[str, str]],
    missing_in_candidate: list[dict[str, str]],
    row_limit: int = 40,
) -> None:
    if not deltas and not missing_in_base and not missing_in_candidate:
        return
    lines.extend(["", f"## {title}", ""])
    lines.append(
        f"- Complete machine-readable deltas are in the JSON artifact; showing up to {row_limit} "
        "rows sorted by absolute success delta."
    )
    if missing_in_base:
        lines.append(f"- Rows missing in base: `{len(missing_in_base)}`")
    if missing_in_candidate:
        lines.append(f"- Rows missing in candidate: `{len(missing_in_candidate)}`")
    lines.append("")
    header = " | ".join(
        [
            *key_fields,
            "base_episodes",
            "candidate_episodes",
            "metric",
            "base",
            "candidate",
            "delta(candidate-base)",
        ]
    )
    alignment = " | ".join(
        ["---"] * len(key_fields) + ["---:", "---:", "---", "---:", "---:", "---:"]
    )
    lines.append(f"| {header} |")
    lines.append(f"| {alignment} |")
    for row in sorted(deltas, key=_success_delta_sort_key)[:row_limit]:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            values = [str(row.get(field, "")) for field in key_fields]
            lines.append(
                "| "
                + " | ".join(
                    [
                        *values,
                        str(row.get("base_episodes", 0)),
                        str(row.get("candidate_episodes", 0)),
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                    ]
                )
                + " |"
            )
            continue
        for metric, metric_values in metrics.items():
            values = [str(row.get(field, "")) for field in key_fields]
            lines.append(
                "| "
                + " | ".join(
                    [
                        *values,
                        str(row.get("base_episodes", 0)),
                        str(row.get("candidate_episodes", 0)),
                        metric,
                        f"{metric_values['base']:.4f}",
                        f"{metric_values['candidate']:.4f}",
                        f"{metric_values['delta']:.4f}",
                    ]
                )
                + " |"
            )


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
    _append_breakdown_markdown(
        lines,
        title="Scenario Deltas",
        deltas=payload.get("scenario_deltas", []),
        key_fields=("planner_key", "scenario_family", "scenario_id"),
        missing_in_base=payload.get("scenario_missing_in_base", []),
        missing_in_candidate=payload.get("scenario_missing_in_candidate", []),
    )
    _append_breakdown_markdown(
        lines,
        title="Scenario Family Deltas",
        deltas=payload.get("scenario_family_deltas", []),
        key_fields=("planner_key", "scenario_family"),
        missing_in_base=payload.get("scenario_family_missing_in_base", []),
        missing_in_candidate=payload.get("scenario_family_missing_in_candidate", []),
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def compare_campaigns(base_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Return planner-metric deltas for two campaigns."""
    base_summary = _read_summary(base_root)
    candidate_summary = _read_summary(candidate_root)
    base_rows = _planner_rows_by_key(base_summary)
    candidate_rows = _planner_rows_by_key(candidate_summary)
    scenario_comparison = _compare_breakdown_artifact(
        base_root,
        candidate_root,
        filename="scenario_breakdown.csv",
        key_fields=("planner_key", "scenario_family", "scenario_id"),
    )
    scenario_family_comparison = _compare_breakdown_artifact(
        base_root,
        candidate_root,
        filename="scenario_family_breakdown.csv",
        key_fields=("planner_key", "scenario_family"),
    )

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
            base_value = _metric_value(base_row, metric)
            candidate_value = _metric_value(candidate_row, metric)
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
        "scenario_deltas": scenario_comparison["deltas"],
        "scenario_missing_in_base": scenario_comparison["missing_in_base"],
        "scenario_missing_in_candidate": scenario_comparison["missing_in_candidate"],
        "scenario_family_deltas": scenario_family_comparison["deltas"],
        "scenario_family_missing_in_base": scenario_family_comparison["missing_in_base"],
        "scenario_family_missing_in_candidate": scenario_family_comparison["missing_in_candidate"],
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
