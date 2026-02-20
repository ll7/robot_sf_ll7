#!/usr/bin/env python3
"""Analyze a camera-ready benchmark campaign for consistency and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlannerDiagnostics:
    """Derived diagnostics for one planner run."""

    planner_key: str
    algo: str
    episodes_summary: int
    episodes_file: int
    success_mean_episodes: float
    collision_mean_episodes: float
    snqi_mean_episodes: float | None
    status_counts: dict[str, int]
    preflight_status: str
    adapter_summary_status: str | None
    adapter_episode_statuses: dict[str, int]
    absolute_map_path_count: int
    runtime_sec: float
    episodes_per_second: float
    wall_time_mean_sec: float
    wall_time_p95_sec: float
    slowest_scenarios: list[dict[str, Any]]
    findings: list[str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Path to output/benchmarks/camera_ready/<campaign_id> directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for analysis JSON. Defaults to reports/campaign_analysis.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output path for analysis Markdown. Defaults to reports/campaign_analysis.md.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Absolute tolerance for numeric row-vs-episode consistency checks.",
    )
    return parser


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: list[float], q: float) -> float:
    """Return quantile in [0, 1] with deterministic nearest-rank interpolation."""
    if not values:
        return 0.0
    q_clamped = min(1.0, max(0.0, float(q)))
    ordered = sorted(values)
    rank = round((len(ordered) - 1) * q_clamped)
    return float(ordered[rank])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _get_repository_root() -> Path:
    """Resolve repository root without importing benchmark runtime modules."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = proc.stdout.strip()
        if text:
            return Path(text).resolve()
    except (OSError, subprocess.CalledProcessError):
        pass
    return Path.cwd().resolve()


def _planner_row_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("planner_rows")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("planner_key")
        if isinstance(key, str) and key:
            out[key] = row
    return out


def _resolve_safe_episodes_path(campaign_root: Path, raw_path: str) -> Path:
    """Resolve episodes path while preventing traversal outside trusted roots."""
    candidate_path = Path(raw_path)
    repo_root = _get_repository_root()
    trusted_roots = (campaign_root.resolve(), repo_root.resolve())

    if candidate_path.is_absolute():
        resolved = candidate_path.resolve()
        if any(resolved.is_relative_to(root) for root in trusted_roots):
            return resolved
        raise ValueError(
            f"Unsafe absolute episodes_path outside trusted roots: {candidate_path}",
        )

    for base in trusted_roots:
        resolved = (base / candidate_path).resolve()
        if not resolved.is_relative_to(base):
            continue
        if resolved.exists():
            return resolved
    # Keep deterministic fallback for diagnostics while still rejecting traversal.
    fallback = (trusted_roots[0] / candidate_path).resolve()
    if fallback.is_relative_to(trusted_roots[0]):
        return fallback
    raise ValueError(f"Unsafe relative episodes_path: {raw_path}")


def _analyze_planner(  # noqa: C901
    run_entry: dict[str, Any],
    row_entry: dict[str, Any] | None,
    campaign_root: Path,
    *,
    tolerance: float,
) -> PlannerDiagnostics:
    planner = run_entry.get("planner", {}) if isinstance(run_entry, dict) else {}
    planner_key = str(planner.get("key", "unknown"))
    algo = str(planner.get("algo", "unknown"))
    summary = run_entry.get("summary", {}) if isinstance(run_entry, dict) else {}
    summary_written = int(summary.get("written", 0))
    preflight_status = str((summary.get("preflight") or {}).get("status", "unknown"))
    adapter_summary_status = (
        (summary.get("algorithm_metadata_contract") or {}).get("adapter_impact") or {}
    ).get("status")
    runtime_sec = float(run_entry.get("runtime_sec", summary.get("runtime_sec", 0.0)) or 0.0)
    eps_per_sec = float(summary.get("episodes_per_second", 0.0) or 0.0)

    episodes_rel = run_entry.get("episodes_path")
    rel_text = str(episodes_rel or "")
    episodes_path = _resolve_safe_episodes_path(campaign_root, rel_text)
    episodes = _read_jsonl(episodes_path)
    episodes_n = len(episodes)

    metrics_success = [
        float(bool((entry.get("metrics") or {}).get("success"))) for entry in episodes
    ]
    metrics_collisions = [
        float((entry.get("metrics") or {}).get("collisions", 0.0) or 0.0) for entry in episodes
    ]
    snqi_values = [_safe_float((entry.get("metrics") or {}).get("snqi")) for entry in episodes]
    snqi_clean = [value for value in snqi_values if value is not None]

    status_counts = Counter(str(entry.get("status", "unknown")) for entry in episodes)
    adapter_episode_statuses = Counter(
        str(
            ((entry.get("algorithm_metadata") or {}).get("adapter_impact") or {}).get(
                "status",
                "unknown",
            ),
        )
        for entry in episodes
    )
    absolute_map_path_count = sum(
        1
        for entry in episodes
        if Path(
            str(
                ((entry.get("scenario_params") or {}).get("map_file"))
                if isinstance(entry.get("scenario_params"), dict)
                else "",
            ),
        ).is_absolute()
    )
    wall_times = [
        float(entry.get("wall_time_sec", 0.0) or 0.0)
        for entry in episodes
        if _safe_float(entry.get("wall_time_sec")) is not None
    ]
    wall_time_mean_sec = _mean(wall_times)
    wall_time_p95_sec = _percentile(wall_times, 0.95)
    per_scenario_wall_times: dict[str, list[float]] = {}
    for entry in episodes:
        scenario_id = str(entry.get("scenario_id", "unknown"))
        wall_time = _safe_float(entry.get("wall_time_sec"))
        if wall_time is None:
            continue
        per_scenario_wall_times.setdefault(scenario_id, []).append(float(wall_time))
    slowest_scenarios: list[dict[str, Any]] = []
    for scenario_id, values in per_scenario_wall_times.items():
        slowest_scenarios.append(
            {
                "scenario_id": scenario_id,
                "episodes": len(values),
                "mean_wall_time_sec": _mean(values),
                "p95_wall_time_sec": _percentile(values, 0.95),
            }
        )
    slowest_scenarios.sort(key=lambda item: float(item["mean_wall_time_sec"]), reverse=True)
    slowest_scenarios = slowest_scenarios[:5]

    findings: list[str] = []
    if episodes_n != summary_written:
        findings.append(
            f"episode count mismatch: summary.written={summary_written}, episodes.jsonl={episodes_n}",
        )
    success_mean = _mean(metrics_success)
    collision_mean = _mean(metrics_collisions)
    snqi_mean = _mean(snqi_clean) if snqi_clean else None

    if row_entry is not None:
        row_success = _safe_float(row_entry.get("success_mean"))
        row_collision = _safe_float(row_entry.get("collision_mean"))
        row_snqi = _safe_float(row_entry.get("snqi_mean"))
        if row_success is not None and abs(row_success - success_mean) > tolerance:
            findings.append(
                f"success_mean mismatch: row={row_success:.6f}, episodes={success_mean:.6f}",
            )
        if row_collision is not None and abs(row_collision - collision_mean) > tolerance:
            findings.append(
                f"collision_mean mismatch: row={row_collision:.6f}, episodes={collision_mean:.6f}",
            )
        if row_snqi is not None and snqi_mean is not None and abs(row_snqi - snqi_mean) > tolerance:
            findings.append(
                f"snqi_mean mismatch: row={row_snqi:.6f}, episodes={snqi_mean:.6f}",
            )

    # Known signal: run summary preflight may report pending while episodes emit complete.
    if (
        isinstance(adapter_summary_status, str)
        and adapter_summary_status == "pending"
        and adapter_episode_statuses.get("complete", 0) > 0
    ):
        findings.append(
            "adapter impact status mismatch: summary=pending but episodes contain complete",
        )
    if absolute_map_path_count > 0:
        findings.append(
            f"non-portable provenance: {absolute_map_path_count} episodes use absolute map_file paths",
        )

    return PlannerDiagnostics(
        planner_key=planner_key,
        algo=algo,
        episodes_summary=summary_written,
        episodes_file=episodes_n,
        success_mean_episodes=success_mean,
        collision_mean_episodes=collision_mean,
        snqi_mean_episodes=snqi_mean,
        status_counts=dict(status_counts),
        preflight_status=preflight_status,
        adapter_summary_status=(
            str(adapter_summary_status) if isinstance(adapter_summary_status, str) else None
        ),
        adapter_episode_statuses=dict(adapter_episode_statuses),
        absolute_map_path_count=absolute_map_path_count,
        runtime_sec=runtime_sec,
        episodes_per_second=eps_per_sec,
        wall_time_mean_sec=wall_time_mean_sec,
        wall_time_p95_sec=wall_time_p95_sec,
        slowest_scenarios=slowest_scenarios,
        findings=findings,
    )


def _build_markdown_report(payload: dict[str, Any]) -> str:
    campaign = payload.get("campaign", {})
    planners = payload.get("planners", [])
    runtime_hotspots = payload.get("runtime_hotspots", {})
    findings = payload.get("findings", [])
    lines = [
        "# Camera-Ready Campaign Analysis",
        "",
        f"- Campaign ID: `{campaign.get('campaign_id', 'unknown')}`",
        f"- Campaign root: `{campaign.get('campaign_root', 'unknown')}`",
        f"- Runtime sec: `{campaign.get('runtime_sec', 0.0)}`",
        f"- Episodes/sec: `{campaign.get('episodes_per_second', 0.0)}`",
        "",
        "## Planner Diagnostics",
        "",
        (
            "| planner | algo | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | "
            "abs map paths | runtime(s) | eps/s |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for planner in planners:
        lines.append(
            "| "
            f"{planner.get('planner_key')} | {planner.get('algo')} | {planner.get('preflight_status')} | "
            f"{planner.get('episodes_file')} | {planner.get('success_mean_episodes'):.4f} | "
            f"{planner.get('collision_mean_episodes'):.4f} | "
            f"{planner.get('snqi_mean_episodes') if planner.get('snqi_mean_episodes') is not None else 'nan'} | "
            f"{planner.get('absolute_map_path_count', 0)} | "
            f"{planner.get('runtime_sec'):.4f} | {planner.get('episodes_per_second'):.4f} |"
        )

    lines.extend(["", "## Runtime Hotspots", ""])
    slowest_planners = runtime_hotspots.get("slowest_planners", [])
    if isinstance(slowest_planners, list) and slowest_planners:
        lines.append("| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |")
        lines.append("|---|---:|---:|---:|")
        for item in slowest_planners:
            lines.append(
                "| "
                f"{item.get('planner_key')} | {float(item.get('runtime_sec', 0.0)):.4f} | "
                f"{float(item.get('wall_time_mean_sec', 0.0)):.4f} | "
                f"{float(item.get('wall_time_p95_sec', 0.0)):.4f} |"
            )
        lines.append("")
        for item in slowest_planners:
            top_scenarios = item.get("top_scenarios", [])
            if not top_scenarios:
                continue
            lines.append(f"- `{item.get('planner_key')}` top slow scenarios:")
            for scenario in top_scenarios:
                lines.append(
                    "  - "
                    f"`{scenario.get('scenario_id')}` mean={float(scenario.get('mean_wall_time_sec', 0.0)):.4f}s "
                    f"p95={float(scenario.get('p95_wall_time_sec', 0.0)):.4f}s "
                    f"(episodes={int(scenario.get('episodes', 0))})"
                )
    else:
        lines.append("- No runtime hotspot data available.")

    lines.extend(["", "## Findings", ""])
    if findings:
        for item in findings:
            lines.append(f"- {item}")
    else:
        lines.append("- No inconsistencies detected by automated checks.")
    return "\n".join(lines) + "\n"


def analyze_campaign(
    campaign_root: Path,
    *,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Analyze one camera-ready campaign and return diagnostics payload."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary_payload = _read_json(summary_path)
    campaign = summary_payload.get("campaign", {}) if isinstance(summary_payload, dict) else {}
    run_entries = summary_payload.get("runs", [])
    row_map = _planner_row_index(summary_payload)

    diagnostics: list[PlannerDiagnostics] = []
    findings: list[str] = []
    for entry in run_entries:
        if not isinstance(entry, dict):
            continue
        planner_key = str((entry.get("planner") or {}).get("key", "unknown"))
        diag = _analyze_planner(
            entry,
            row_map.get(planner_key),
            campaign_root,
            tolerance=tolerance,
        )
        diagnostics.append(diag)
        for finding in diag.findings:
            findings.append(f"{planner_key}: {finding}")
        if diag.preflight_status == "fallback":
            findings.append(
                f"{planner_key}: preflight status is fallback (experimental degraded mode)",
            )

    diagnostics_payload = [diag.__dict__ for diag in diagnostics]
    diagnostics_payload.sort(key=lambda item: str(item.get("planner_key", "")))
    findings.sort()
    slowest_planners = []
    for item in sorted(
        diagnostics_payload, key=lambda x: float(x.get("runtime_sec", 0.0)), reverse=True
    ):
        slowest_planners.append(
            {
                "planner_key": item.get("planner_key"),
                "runtime_sec": float(item.get("runtime_sec", 0.0)),
                "wall_time_mean_sec": float(item.get("wall_time_mean_sec", 0.0)),
                "wall_time_p95_sec": float(item.get("wall_time_p95_sec", 0.0)),
                "top_scenarios": list(item.get("slowest_scenarios", []))[:3],
            }
        )
    slowest_planners = slowest_planners[:3]

    return {
        "campaign": {
            "campaign_id": campaign.get("campaign_id", "unknown"),
            "campaign_root": str(campaign_root),
            "runtime_sec": _safe_float(campaign.get("runtime_sec")) or 0.0,
            "episodes_per_second": _safe_float(campaign.get("episodes_per_second")) or 0.0,
            "summary_json": str(summary_path),
        },
        "planners": diagnostics_payload,
        "runtime_hotspots": {
            "slowest_planners": slowest_planners,
        },
        "findings": findings,
    }


def main() -> int:
    """CLI entry point for campaign analysis."""
    args = _build_parser().parse_args()
    campaign_root = args.campaign_root.resolve()
    analysis = analyze_campaign(campaign_root, tolerance=float(args.tolerance))

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else (campaign_root / "reports" / "campaign_analysis.json").resolve()
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else (campaign_root / "reports" / "campaign_analysis.md").resolve()
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown_report(analysis), encoding="utf-8")
    print(json.dumps({"analysis_json": str(output_json), "analysis_md": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
