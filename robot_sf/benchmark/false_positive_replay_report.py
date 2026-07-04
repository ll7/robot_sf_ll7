"""Issue #3300 false-positive actor-injection replay report helpers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.robustness_delta import (
    build_robustness_delta_report,
    load_episode_jsonl,
    write_report_json,
)

SCHEMA_VERSION = "false_positive_actor_injection_replay.v1"
ISSUE = 3300
CLAIM_BOUNDARY = (
    "CPU-local observation-quality replay smoke for false-positive actor injection. "
    "Diagnostic only; not a full benchmark campaign, hardware sensor model, or paper-facing claim."
)

CLASS_OBSERVED = "observed"
CLASS_SCENARIO_TOO_WEAK = "scenario_too_weak"
CLASS_BLOCKED_UNAVAILABLE = "blocked_unavailable"
CLASS_TRACE_ONLY_DIAGNOSTIC = "trace_only_diagnostic"

if TYPE_CHECKING:
    from pathlib import Path

_PREDECLARED_METRICS = (
    "min_distance_m",
    "clearance_m",
    "near_miss",
    "collision",
    "route_progress",
    "route_complete",
    "runtime_s",
    "stop_step",
    "yield_step",
)


def build_false_positive_replay_report(
    *,
    nominal_jsonl: Path,
    perturbed_jsonl: Path,
    replay_mode: str = "executable",
    nominal_rows: Sequence[Mapping[str, Any]] | None = None,
    perturbed_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a paired false-positive actor-injection replay report.

    Args:
        nominal_jsonl: Nominal replay episode JSONL.
        perturbed_jsonl: False-positive actor-injection replay episode JSONL.
        replay_mode: ``executable`` for live replay evidence or ``trace_derived`` for
            trace-only diagnostics.
        nominal_rows: Optional parsed nominal rows for tests.
        perturbed_rows: Optional parsed perturbed rows for tests.

    Returns:
        JSON-serializable report with per-episode deltas and diagnostic classification.
    """
    clean_rows = (
        list(nominal_rows) if nominal_rows is not None else load_episode_jsonl(nominal_jsonl)
    )
    noisy_rows = (
        list(perturbed_rows) if perturbed_rows is not None else load_episode_jsonl(perturbed_jsonl)
    )
    delta_report = build_robustness_delta_report(
        nominal_jsonl=nominal_jsonl,
        perturbed_jsonl=perturbed_jsonl,
        nominal_rows=clean_rows,
        perturbed_rows=noisy_rows,
    )
    per_episode_deltas = _paired_episode_deltas(
        clean_rows,
        noisy_rows,
        delta_report["pairing"]["pair_keys"],
    )
    injection_summary = _injection_summary(noisy_rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "claim_boundary": CLAIM_BOUNDARY,
        "replay_mode": replay_mode,
        "inputs": {
            "nominal_jsonl": nominal_jsonl.as_posix(),
            "perturbed_jsonl": perturbed_jsonl.as_posix(),
        },
        "pairing": delta_report["pairing"],
        "planner_rows": delta_report["planner_rows"],
        "per_episode_deltas": per_episode_deltas,
        "injection_summary": injection_summary,
    }
    report["classification"] = classify_false_positive_replay(report)
    return report


def classify_false_positive_replay(report: Mapping[str, Any]) -> dict[str, Any]:
    """Classify #3300 replay result without promoting benchmark claims.

    Returns:
        A JSON-serializable label and reason pair.
    """
    replay_mode = str(report.get("replay_mode", "executable"))
    if replay_mode == "trace_derived":
        return {
            "label": CLASS_TRACE_ONLY_DIAGNOSTIC,
            "reason": "report was built from trace-derived diagnostics, not executable replay",
        }

    injection_summary = _mapping(report.get("injection_summary"))
    if int(injection_summary.get("pedestrians_added", 0) or 0) <= 0:
        return {
            "label": CLASS_BLOCKED_UNAVAILABLE,
            "reason": "no perturbed episode recorded pedestrians_added > 0",
        }

    if _has_predeclared_delta(report):
        return {
            "label": CLASS_OBSERVED,
            "reason": "false-positive injection changed at least one predeclared replay outcome",
        }

    return {
        "label": CLASS_SCENARIO_TOO_WEAK,
        "reason": "false-positive injection occurred but pinned smoke outcomes did not change",
    }


def format_false_positive_replay_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown issue #3300 replay report.

    Returns:
        Markdown report text.
    """
    classification = _mapping(report.get("classification"))
    injection = _mapping(report.get("injection_summary"))
    pairing = _mapping(report.get("pairing"))
    lines = [
        "# Issue #3300 False-Positive Actor-Injection Replay",
        "",
        "## Claim Boundary",
        str(report.get("claim_boundary", CLAIM_BOUNDARY)),
        "",
        "## Classification",
        f"- Label: `{classification.get('label', '')}`",
        f"- Reason: {classification.get('reason', '')}",
        f"- Replay mode: `{report.get('replay_mode', '')}`",
        "",
        "## Pairing",
        f"- Paired rows: `{pairing.get('paired_rows', 0)}`",
        f"- Unmatched nominal rows: `{pairing.get('unmatched_nominal_rows', 0)}`",
        f"- Unmatched perturbed rows: `{pairing.get('unmatched_perturbed_rows', 0)}`",
        "",
        "## Injection Summary",
        f"- Pedestrians added: `{injection.get('pedestrians_added', 0)}`",
        f"- Steps with noise: `{injection.get('steps_with_noise', 0)}`",
        f"- Perturbation profiles: `{', '.join(injection.get('profiles', []))}`",
        f"- Perturbation hashes: `{', '.join(injection.get('hashes', []))}`",
        "",
        "## Episode Deltas",
        "",
        "| planner | scenario | seed | pedestrians added | changed fields |",
        "|---|---|---:|---:|---|",
    ]
    for row in report.get("per_episode_deltas", []):
        if not isinstance(row, Mapping):
            continue
        changed = ", ".join(str(field) for field in row.get("changed_fields", []))
        lines.append(
            "| {planner} | {scenario} | {seed} | {added} | {changed} |".format(
                planner=row.get("planner_key", ""),
                scenario=row.get("scenario_id", ""),
                seed=row.get("seed", ""),
                added=row.get("pedestrians_added", 0),
                changed=changed or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "- CPU replay smoke only.",
            "- False-positive effects are reported separately from other observation noise.",
            "- No full benchmark campaign, Slurm/GPU submission, or paper-facing claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_false_positive_replay_markdown(report: Mapping[str, Any], path: Path) -> None:
    """Write Markdown issue #3300 report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_false_positive_replay_markdown(report), encoding="utf-8")


def write_false_positive_replay_json(report: Mapping[str, Any], path: Path) -> None:
    """Write JSON issue #3300 report."""
    write_report_json(report, path)


def write_false_positive_replay_csv(report: Mapping[str, Any], path: Path) -> None:
    """Write per-episode false-positive actor-injection delta CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "planner_key",
        "scenario_id",
        "seed",
        "pedestrians_added",
        "steps_with_noise",
        "nominal_actor_count",
        "perturbed_actor_count",
        "changed_fields",
        "metric_delta",
        "selected_action_nominal",
        "selected_action_perturbed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("per_episode_deltas", []):
            if not isinstance(row, Mapping):
                continue
            actor_counts = _mapping(row.get("observed_actor_counts"))
            action = _mapping(row.get("selected_action"))
            writer.writerow(
                {
                    "planner_key": row.get("planner_key"),
                    "scenario_id": row.get("scenario_id"),
                    "seed": row.get("seed"),
                    "pedestrians_added": row.get("pedestrians_added"),
                    "steps_with_noise": row.get("steps_with_noise"),
                    "nominal_actor_count": actor_counts.get("nominal"),
                    "perturbed_actor_count": actor_counts.get("perturbed"),
                    "changed_fields": ";".join(
                        str(field) for field in row.get("changed_fields", [])
                    ),
                    "metric_delta": json.dumps(row.get("metric_delta", {}), sort_keys=True),
                    "selected_action_nominal": action.get("nominal"),
                    "selected_action_perturbed": action.get("perturbed"),
                }
            )


def _paired_episode_deltas(
    nominal_rows: Sequence[Mapping[str, Any]],
    perturbed_rows: Sequence[Mapping[str, Any]],
    pair_keys: Sequence[str],
) -> list[dict[str, Any]]:
    nominal_index = {_pair_key(row, pair_keys): row for row in nominal_rows}
    perturbed_index = {_pair_key(row, pair_keys): row for row in perturbed_rows}
    deltas: list[dict[str, Any]] = []
    for key in sorted(set(nominal_index) & set(perturbed_index)):
        nominal = nominal_index[key]
        perturbed = perturbed_index[key]
        metric_delta = _metric_delta(nominal, perturbed)
        action_delta = _action_delta(nominal, perturbed)
        changed_fields = sorted([field for field, value in metric_delta.items() if value != 0])
        if action_delta["changed"]:
            changed_fields.append("selected_action")
        deltas.append(
            {
                "planner_key": _planner_identity(perturbed) or _planner_identity(nominal),
                "scenario_id": _first_present(perturbed, nominal, "scenario_id"),
                "seed": _first_present(perturbed, nominal, "seed"),
                "pedestrians_added": _noise_stat(perturbed, "pedestrians_added"),
                "steps_with_noise": _noise_stat(perturbed, "steps_with_noise"),
                "observed_actor_counts": {
                    "nominal": _actor_count(nominal),
                    "perturbed": _actor_count(perturbed),
                },
                "metric_delta": metric_delta,
                "selected_action": action_delta,
                "changed_fields": changed_fields,
            }
        )
    return deltas


def _injection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = {"pedestrians_added": 0, "steps_with_noise": 0}
    profiles: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        stats["pedestrians_added"] += _noise_stat(row, "pedestrians_added")
        stats["steps_with_noise"] += _noise_stat(row, "steps_with_noise")
        profile = _noise_profile(row)
        if profile:
            profiles.add(profile)
        noise_hash = row.get("observation_noise_hash")
        if noise_hash:
            hashes.add(str(noise_hash))
    return {**stats, "profiles": sorted(profiles), "hashes": sorted(hashes)}


def _metric_delta(nominal: Mapping[str, Any], perturbed: Mapping[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for field in _PREDECLARED_METRICS:
        clean = _metric_value(nominal, field)
        noisy = _metric_value(perturbed, field)
        if clean is None and noisy is None:
            continue
        deltas[field] = _number(noisy) - _number(clean)
    return deltas


def _action_delta(nominal: Mapping[str, Any], perturbed: Mapping[str, Any]) -> dict[str, Any]:
    clean = _selected_action(nominal)
    noisy = _selected_action(perturbed)
    return {"nominal": clean, "perturbed": noisy, "changed": clean != noisy}


def _has_predeclared_delta(report: Mapping[str, Any]) -> bool:
    for row in report.get("planner_rows", []):
        if not isinstance(row, Mapping):
            continue
        if _number(row.get("success_delta")) != 0.0 or _number(row.get("collision_delta")) != 0.0:
            return True
    for row in report.get("per_episode_deltas", []):
        if not isinstance(row, Mapping):
            continue
        if row.get("changed_fields"):
            return True
    return False


def _pair_key(row: Mapping[str, Any], pair_keys: Sequence[str]) -> tuple[tuple[str, Any], ...]:
    values: list[tuple[str, Any]] = []
    for key in pair_keys:
        if key == "planner_identity":
            value = _planner_identity(row)
        else:
            value = _get_nested(row, key)
        if not isinstance(value, str | int | float | bool) and value is not None:
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        values.append((key, value))
    return tuple(values)


def _planner_identity(row: Mapping[str, Any]) -> str:
    """Return stable planner identity while preserving valid falsy identifiers."""
    for key in ("planner_key", "planner", "algo", "scenario_params.algo"):
        value = _get_nested(row, key)
        if value is not None:
            return str(value)
    return "unknown"


def _metric_value(row: Mapping[str, Any], field: str) -> Any:
    for prefix in ("metrics", "outcome", "summary"):
        value = _get_nested(row, f"{prefix}.{field}")
        if value is not None:
            return value
    return _get_nested(row, field)


def _noise_stat(row: Mapping[str, Any], field: str) -> int:
    stats = row.get("observation_noise_stats")
    if isinstance(stats, Mapping):
        return int(_number(stats.get(field)))
    return 0


def _actor_count(row: Mapping[str, Any]) -> int | None:
    for field in (
        "observed_actor_count",
        "actor_count",
        "pedestrian_count",
        "observation.pedestrians.count",
        "pedestrians.count",
    ):
        value = _get_nested(row, field)
        if value is not None:
            return int(_number(value))
    return None


def _noise_profile(row: Mapping[str, Any]) -> str | None:
    spec = row.get("observation_noise")
    if isinstance(spec, Mapping) and spec.get("profile"):
        return str(spec["profile"])
    profile = row.get("observation_noise_profile")
    return str(profile) if profile else None


def _selected_action(row: Mapping[str, Any]) -> Any:
    for field in ("selected_action", "action", "actions.selected", "planner_action"):
        value = _get_nested(row, field)
        if value is not None:
            return value
    return None


def _first_present(first_row: Mapping[str, Any], second_row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        for row in (first_row, second_row):
            value = _get_nested(row, key)
            if value is not None:
                return value
    return None


def _get_nested(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0
