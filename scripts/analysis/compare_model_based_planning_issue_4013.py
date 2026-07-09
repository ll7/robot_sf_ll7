#!/usr/bin/env python3
"""Build issue #4013 learned-prediction MPC diagnostic comparison reports.

The report consumes real benchmark episode JSONL files. It does not run a benchmark
campaign, train a predictor, or promote paper-facing claims.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import read_jsonl

SCHEMA_VERSION = "issue_4013.model_based_planning_comparison.v1"
EXPECTED_EVIDENCE_TIER = "diagnostic-only"
EXPECTED_CLAIM_BOUNDARY = (
    "diagnostic matched-scenario comparison; not paper-grade benchmark evidence"
)
REQUIRED_ROLES = ("learned_prediction_mpc", "cv_prediction_mpc", "model_free_baseline")
WORLD_MODEL_EXCLUSIONS = (
    "dreamerv3",
    "planet",
    "td_mpc2",
    "large_generative_world_model",
    "paper_grade_claim",
)


def build_report_from_config(config_path: str | Path) -> dict[str, Any]:
    """Build and write the issue #4013 comparison report from a YAML config."""
    config_path = Path(config_path).resolve()
    config = _load_yaml_mapping(config_path)
    output_json = _resolve_output_path(config_path, config.get("output_json"))
    output_markdown = _resolve_output_path(config_path, config.get("output_markdown"))

    blockers = _config_blockers(config)
    runs = _load_runs(config_path, config)
    rows = {role: _summarize_run(role, episodes, path) for role, episodes, path in runs}
    blockers.extend(_comparison_blockers(rows))
    fallback_summary = _fallback_summary(rows)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4013,
        "evidence_tier": str(config.get("evidence_tier", "")),
        "claim_boundary": str(config.get("claim_boundary", "")),
        "status": "diagnostic_ready" if not blockers else "diagnostic_blocked",
        "paired_seed_count": _paired_seed_count(rows),
        "scenario_ids": _paired_scenario_ids(rows),
        "roles": rows,
        "fallback_degraded_rows": fallback_summary,
        "world_model_exclusions": list(WORLD_MODEL_EXCLUSIONS),
        "blockers": blockers,
        "closure_criteria": _closure_criteria(rows, fallback_summary, blockers),
    }
    _write_json(output_json, report)
    output_markdown.write_text(_format_markdown(report), encoding="utf-8")
    return report


def _load_runs(
    config_path: Path,
    config: Mapping[str, Any],
) -> list[tuple[str, list[dict[str, Any]], Path]]:
    runs = config.get("runs")
    if not isinstance(runs, list):
        return []
    loaded: list[tuple[str, list[dict[str, Any]], Path]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        role = str(run.get("role", ""))
        episodes_path = _resolve_path(config_path, run.get("episodes_jsonl"))
        loaded.append((role, read_jsonl(episodes_path), episodes_path))
    return loaded


def _summarize_run(role: str, episodes: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    fallback_rows = [_fallback_reason(record) for record in episodes]
    non_fallback = [
        record for record, reason in zip(episodes, fallback_rows, strict=True) if reason is None
    ]
    metrics = [_metrics(record) for record in non_fallback]
    keys = sorted({_episode_key(record) for record in non_fallback})
    return {
        "role": role,
        "episodes_path": str(path),
        "episodes": len(episodes),
        "evidence_episodes": len(non_fallback),
        "excluded_fallback_or_degraded": len(episodes) - len(non_fallback),
        "fallback_reasons": sorted({reason for reason in fallback_rows if reason}),
        "episode_keys": keys,
        "algorithms": sorted(
            {str(record.get("algo", "")) for record in episodes if record.get("algo")}
        ),
        "algorithm_metadata_statuses": sorted(
            {
                str(_algorithm_metadata(record).get("status", "missing"))
                for record in episodes
                if _algorithm_metadata(record)
            }
        ),
        "success_rate": _mean([float(m.get("success", 0.0)) for m in metrics]),
        "collision_rate": _mean([float(m.get("collision", 0.0)) for m in metrics]),
        "near_miss_rate": _mean([float(m.get("near_miss", 0.0)) for m in metrics]),
        "mean_min_clearance_m": _mean([_finite_float(m.get("min_clearance_m")) for m in metrics]),
        "mean_time_to_goal_s": _mean([_finite_float(m.get("time_to_goal_s")) for m in metrics]),
        "mean_wall_time_sec": _mean(
            [_finite_float(record.get("wall_time_sec")) for record in non_fallback]
        ),
    }


def _comparison_blockers(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    missing_roles = [role for role in REQUIRED_ROLES if role not in rows]
    blockers.extend(f"missing role: {role}" for role in missing_roles)
    for role in REQUIRED_ROLES:
        row = rows.get(role)
        if row is None:
            continue
        if int(row["episodes"]) == 0:
            blockers.append(f"{role} has no episode rows")
        if int(row["evidence_episodes"]) == 0:
            blockers.append(f"{role} has no non-fallback evidence rows")
        if int(row["excluded_fallback_or_degraded"]) > 0:
            blockers.append(f"{role} has fallback/degraded rows excluded")
    if any(role not in rows for role in REQUIRED_ROLES):
        return blockers
    key_sets = [set(rows[role]["episode_keys"]) for role in REQUIRED_ROLES]
    shared = set.intersection(*key_sets)
    if not shared:
        blockers.append("required roles have no matched scenario/seed episodes")
    for role, keys in zip(REQUIRED_ROLES, key_sets, strict=True):
        if keys != shared:
            blockers.append(f"{role} episode keys are not paired with all required roles")
    return blockers


def _config_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if config.get("schema_version") != SCHEMA_VERSION:
        blockers.append("schema_version mismatch")
    if config.get("issue") != 4013:
        blockers.append("issue must be 4013")
    if config.get("evidence_tier") != EXPECTED_EVIDENCE_TIER:
        blockers.append("evidence_tier must be diagnostic-only")
    if config.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        blockers.append("claim_boundary mismatch")
    if not isinstance(config.get("runs"), list):
        blockers.append("runs must be a list")
    return blockers


def _fallback_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    excluded = sum(int(row["excluded_fallback_or_degraded"]) for row in rows.values())
    included = 0
    return {"excluded": excluded, "included_as_non_evidence": included}


def _closure_criteria(
    rows: Mapping[str, Mapping[str, Any]],
    fallback_summary: Mapping[str, int],
    blockers: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "criterion": "short-horizon predictor checkpoint loads without fallback",
            "met": "learned_prediction_mpc" in rows
            and int(rows["learned_prediction_mpc"]["evidence_episodes"]) > 0
            and int(rows["learned_prediction_mpc"]["excluded_fallback_or_degraded"]) == 0,
            "evidence": rows.get("learned_prediction_mpc", {}).get("episodes_path"),
        },
        {
            "criterion": "model-based action selection runs on smoke scenario",
            "met": _paired_seed_count(rows) > 0 and "learned_prediction_mpc" in rows,
            "evidence": _paired_scenario_ids(rows),
        },
        {
            "criterion": "comparison includes cv_prediction_mpc and one model-free baseline",
            "met": all(role in rows for role in REQUIRED_ROLES),
            "evidence": list(rows),
        },
        {
            "criterion": "fallback/degraded rows are excluded from evidence",
            "met": int(fallback_summary["included_as_non_evidence"]) == 0,
            "evidence": dict(fallback_summary),
        },
        {
            "criterion": "claim boundary excludes large world-model and paper-grade claims",
            "met": not blockers or "claim_boundary mismatch" not in blockers,
            "evidence": EXPECTED_CLAIM_BOUNDARY,
        },
    ]


def _paired_seed_count(rows: Mapping[str, Mapping[str, Any]]) -> int:
    if any(role not in rows for role in REQUIRED_ROLES):
        return 0
    key_sets = [set(rows[role]["episode_keys"]) for role in REQUIRED_ROLES]
    return len(set.intersection(*key_sets))


def _paired_scenario_ids(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if any(role not in rows for role in REQUIRED_ROLES):
        return []
    key_sets = [set(rows[role]["episode_keys"]) for role in REQUIRED_ROLES]
    shared = set.intersection(*key_sets)
    return sorted({key.split("::", 1)[0] for key in shared})


def _metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    success = metrics.get("success", outcome.get("success", 0.0))
    collision = metrics.get(
        "collision", outcome.get("collision_event", metrics.get("collisions", 0.0))
    )
    near_miss = metrics.get("near_miss", metrics.get("near_misses", 0.0))
    return {
        "success": _boolish_rate(success),
        "collision": _boolish_rate(collision),
        "near_miss": _boolish_rate(near_miss),
        "min_clearance_m": metrics.get("min_clearance_m"),
        "time_to_goal_s": metrics.get("time_to_goal_s", metrics.get("time_to_goal")),
    }


def _fallback_reason(record: Mapping[str, Any]) -> str | None:
    metadata = _algorithm_metadata(record)
    status = str(metadata.get("status", "")).lower()
    if status and status not in {"ok", "ready", "success"}:
        return f"algorithm_metadata.status={status}"
    fallback_reason = metadata.get("fallback_reason")
    if fallback_reason:
        return f"algorithm_metadata.fallback_reason={fallback_reason}"
    fallback_flags = (
        record.get("fallback_or_degraded"),
        record.get("degraded"),
        record.get("fallback"),
    )
    if any(bool(flag) for flag in fallback_flags):
        return "record fallback_or_degraded flag"
    return None


def _algorithm_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("algorithm_metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _episode_key(record: Mapping[str, Any]) -> str:
    scenario_id = record.get("scenario_id")
    seed = record.get("seed")
    if scenario_id is None or seed is None:
        raise ValueError("episode record is missing required 'scenario_id' or 'seed'")
    return f"{scenario_id}::{seed}"


def _boolish_rate(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = _finite_float(value)
    return 0.0 if number is None else float(number)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _resolve_path(config_path: Path, raw_path: Any) -> Path:
    if not raw_path:
        raise ValueError(f"{config_path} contains an empty required path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"required input does not exist: {path}")
    return path


def _resolve_output_path(config_path: Path, raw_path: Any) -> Path:
    if not raw_path:
        raise ValueError(f"{config_path} contains an empty output path")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_markdown(report: Mapping[str, Any]) -> str:
    rows = report["roles"]
    assert isinstance(rows, Mapping)
    lines = [
        "# Issue #4013 Learned-Prediction MPC Diagnostic Comparison",
        "",
        f"- Claim boundary: {report['claim_boundary']}",
        f"- Evidence tier: {report['evidence_tier']}",
        f"- Status: {report['status']}",
        f"- Paired seed count: {report['paired_seed_count']}",
        "- World-model exclusions: " + ", ".join(report["world_model_exclusions"]),
        "",
        "| Role | Episodes | Evidence episodes | Success rate | Collision rate | Near-miss rate | Excluded fallback/degraded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in REQUIRED_ROLES:
        row = rows.get(role, {})
        lines.append(
            "| {role} | {episodes} | {evidence} | {success} | {collision} | {near_miss} | {excluded} |".format(
                role=role,
                episodes=row.get("episodes", 0),
                evidence=row.get("evidence_episodes", 0),
                success=_format_number(row.get("success_rate")),
                collision=_format_number(row.get("collision_rate")),
                near_miss=_format_number(row.get("near_miss_rate")),
                excluded=row.get("excluded_fallback_or_degraded", 0),
            )
        )
    blockers = report["blockers"]
    assert isinstance(blockers, Sequence)
    lines.extend(["", "## Blockers"])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Closure Criteria"])
    criteria = report["closure_criteria"]
    assert isinstance(criteria, Sequence)
    for item in criteria:
        assert isinstance(item, Mapping)
        status = "met" if item["met"] else "not met"
        lines.append(f"- {status}: {item['criterion']}")
    lines.append("")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the configured report and return a fail-closed exit code."""
    args = parse_args(argv)
    report = build_report_from_config(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "diagnostic_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
