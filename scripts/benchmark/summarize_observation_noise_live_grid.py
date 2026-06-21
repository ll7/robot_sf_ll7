#!/usr/bin/env python3
"""Summarize observation-noise live-smoke results across tracked fixtures.

This tool consumes compact summaries produced from native live step-diagnostics
replays. It does not treat fallback, degraded, unavailable, trace-derived-only,
or malformed summaries as successful evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue_3335_observation_noise_live_grid.v1"
DEFAULT_SOURCE_SUMMARIES = (
    Path(
        "docs/context/evidence/issue_2777_live_observation_noise_replay/"
        "issue_3330_seed_amplitude_grid/summary.json"
    ),
    Path(
        "docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/"
        "issue_3320_seed_amplitude_grid/summary.json"
    ),
)
DEFAULT_OUTPUT_JSON = Path(
    "docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/summary.json"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/context/evidence/issue_3335_observation_noise_cross_fixture_2026-06-21/README.md"
)
CLAIM_BOUNDARY = (
    "Diagnostic-only cross-fixture synthesis of tracked native live "
    "step-diagnostics summaries. This checks whether observation-noise "
    "behavior sensitivity appears across more than one fixture surface, but it "
    "is not benchmark-strength, paper-grade, planner-superiority, robustness, "
    "or hardware-calibrated sensor-realism evidence."
)

BEHAVIOR_LABELS = {"non_null_behavior_delta"}
OBSERVATION_ONLY_LABELS = {
    "observation_only_delta",
    "observation_only_scenario_too_weak",
    "null_policy_insensitive",
}


def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, otherwise an empty mapping."""

    return value if isinstance(value, dict) else {}


def _conditions(summary: dict[str, Any]) -> list[Any]:
    """Return summary conditions when they have the expected list shape."""

    value = summary.get("conditions")
    return value if isinstance(value, list) else []


def _has_condition_list(summary: dict[str, Any]) -> bool:
    """Return whether summary conditions have the expected list shape."""

    return isinstance(summary.get("conditions"), list)


def _first_item(value: Any) -> Any | None:
    """Return the first item from a sequence-like summary field."""

    return value[0] if isinstance(value, (list, tuple)) and value else None


def _number(value: Any) -> float | None:
    """Normalize finite scalar values for report calculations."""

    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_summary(path: Path) -> dict[str, Any]:
    """Load one compact live-smoke summary."""

    return json.loads(path.read_text(encoding="utf-8"))


def _is_native_live_summary(summary: dict[str, Any]) -> bool:
    """Return whether a summary has the expected native live trace shape."""

    schema = str(summary.get("schema_version", ""))
    if schema == "issue_2777_observation_noise_live_replay.v1":
        run_config = _mapping(summary.get("run_config"))
        return (
            str(summary.get("artifact_shape")) == "compact_summary_without_raw_traces"
            and str(run_config.get("condition_set")) == "issue_3330_seed_amplitude_grid"
            and _has_condition_list(summary)
        )
    inputs = _mapping(summary.get("inputs"))
    has_trace_inputs = "clean_trace_json" in inputs and "perturbed_trace_json" in inputs
    has_progress = isinstance(summary.get("progress_delta"), dict)
    has_commands = isinstance(summary.get("command_summary"), dict)
    return schema.endswith("observation_noise_live_smoke.v1") and (
        has_trace_inputs and has_progress and has_commands
    )


def _source_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify whether one source summary is usable for this synthesis."""

    if not _is_native_live_summary(summary):
        return {
            "usable": False,
            "status": "failed_closed",
            "reason": "source is not a native live step-diagnostics summary",
        }
    if str(summary.get("schema_version")) == "issue_2777_observation_noise_live_replay.v1":
        classification = _mapping(summary.get("classification"))
        if str(summary.get("status")) != "live_replay":
            return {
                "usable": False,
                "status": "failed_closed",
                "reason": classification.get("rationale") or "source did not complete live replay",
            }
        fixture_contract = _mapping(summary.get("fixture_contract"))
        if fixture_contract.get("satisfied") is not True:
            return {
                "usable": False,
                "status": "failed_closed",
                "reason": fixture_contract.get("blocker") or "fixture contract was not satisfied",
            }
        grid = _mapping(summary.get("grid_interpretation"))
        if str(grid.get("label")) == "unavailable_fail_closed":
            return {
                "usable": False,
                "status": "failed_closed",
                "reason": grid.get("summary") or "grid interpretation failed closed",
            }
        if any(
            _mapping(_mapping(condition).get("behavior_change_summary")).get(
                "command_sequence_changed"
            )
            is True
            or _mapping(_mapping(condition).get("behavior_change_summary")).get(
                "progress_or_risk_changed"
            )
            is True
            for condition in _conditions(summary)
        ):
            return {
                "usable": True,
                "status": "behavior_sensitive_grid",
                "reason": grid.get("summary") or classification.get("rationale"),
            }
        return {
            "usable": True,
            "status": "policy_insensitive_grid",
            "reason": grid.get("summary") or classification.get("rationale"),
        }
    classification = _mapping(summary.get("classification"))
    label = str(classification.get("label", ""))
    if label in BEHAVIOR_LABELS:
        return {
            "usable": True,
            "status": "behavior_sensitive",
            "reason": classification.get("rationale"),
        }
    if label in OBSERVATION_ONLY_LABELS:
        return {"usable": True, "status": label, "reason": classification.get("rationale")}
    return {
        "usable": False,
        "status": "failed_closed",
        "reason": f"unsupported classification label: {label or '<missing>'}",
    }


def _collision_counts(summary: dict[str, Any]) -> dict[str, Any]:
    """Return compact collision counts from progress deltas."""

    if str(summary.get("schema_version")) == "issue_2777_observation_noise_live_replay.v1":
        for condition in _conditions(summary):
            collision = _mapping(_mapping(condition).get("progress_delta")).get(
                "collision_flag_counts"
            )
            collision = _mapping(collision)
            if collision:
                return {
                    "clean": _mapping(collision.get("noop")),
                    "perturbed": _mapping(collision.get("condition")),
                }
        return {"clean": {}, "perturbed": {}}
    collision = _mapping(_mapping(summary.get("progress_delta")).get("collision_flag_counts"))
    return {
        "clean": _mapping(collision.get("clean")),
        "perturbed": _mapping(collision.get("perturbed")),
    }


def _issue_2777_closest_noop(summary: dict[str, Any]) -> float | None:
    """Return the no-op closest distance from an issue #2777 grid summary."""

    conditions = _conditions(summary)
    noop_summary = _mapping(_mapping(conditions[0]).get("progress_summary")) if conditions else {}
    noop_distance = _number(noop_summary.get("closest_robot_ped_distance"))
    if noop_distance is not None:
        return noop_distance
    for condition in conditions:
        closest = _mapping(_mapping(condition).get("progress_delta")).get(
            "closest_robot_ped_distance"
        )
        noop_distance = _number(_mapping(closest).get("noop"))
        if noop_distance is not None:
            return noop_distance
    return None


def _issue_2777_command_changed(summary: dict[str, Any]) -> bool:
    """Return whether any condition changed selected commands."""

    return any(
        _mapping(_mapping(condition).get("behavior_change_summary")).get("command_sequence_changed")
        is True
        for condition in _conditions(summary)
    )


def _issue_2777_closest_delta(summary: dict[str, Any]) -> float | None:
    """Return the largest finite closest-distance delta across conditions."""

    deltas: list[float] = []
    for condition in _conditions(summary):
        closest = _mapping(_mapping(condition).get("progress_delta")).get(
            "closest_robot_ped_distance"
        )
        closest_map = _mapping(closest)
        noop = _number(closest_map.get("noop"))
        condition_value = _number(closest_map.get("condition"))
        if noop is not None and condition_value is not None:
            deltas.append(condition_value - noop)
    if not deltas:
        return None
    return max(deltas, key=abs)


def _source_row(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Build one source row for the cross-fixture report."""

    status = _source_status(summary)
    if str(summary.get("schema_version")) == "issue_2777_observation_noise_live_replay.v1":
        fixture_contract = _mapping(summary.get("fixture_contract"))
        matched_scenario = _mapping(fixture_contract.get("matched_scenario"))
        run_config = _mapping(summary.get("run_config"))
        closest = _issue_2777_closest_noop(summary)
        return {
            "path": path.as_posix(),
            "schema_version": summary.get("schema_version"),
            "source_issue": summary.get("issue"),
            "scenario": matched_scenario.get("name"),
            "scenario_matrix": run_config.get("scenario_matrix")
            or fixture_contract.get("scenario_matrix"),
            "condition_set": run_config.get("condition_set"),
            "same_scenario": True,
            "seed": _first_item(matched_scenario.get("seeds")),
            "same_seed": True,
            "status": status["status"],
            "usable": status["usable"],
            "status_rationale": status["reason"],
            "classification": _mapping(summary.get("classification")),
            "grid_interpretation": _mapping(summary.get("grid_interpretation")),
            "near_field": {
                "threshold_m": 2.0,
                "clean_closest_robot_ped_distance_m": closest,
                "satisfied": closest is not None and closest <= 2.0,
            },
            "command_sequence_changed": _issue_2777_command_changed(summary),
            "closest_distance_delta_m": _issue_2777_closest_delta(summary),
            "collision_counts": _collision_counts(summary),
            "observation_summary_changed": None,
        }
    near_field = _mapping(summary.get("near_field_target"))
    progress = _mapping(summary.get("progress_delta"))
    closest_delta = _mapping(progress.get("closest_robot_ped_distance"))
    return {
        "path": path.as_posix(),
        "schema_version": summary.get("schema_version"),
        "source_issue": summary.get("issue"),
        "scenario": _mapping(summary.get("scenario")).get("clean"),
        "same_scenario": _mapping(summary.get("scenario")).get("same_scenario"),
        "seed": _mapping(summary.get("seed")).get("clean"),
        "same_seed": _mapping(summary.get("seed")).get("same_seed"),
        "status": status["status"],
        "usable": status["usable"],
        "status_rationale": status["reason"],
        "classification": _mapping(summary.get("classification")),
        "near_field": {
            "threshold_m": near_field.get("threshold_m"),
            "clean_closest_robot_ped_distance_m": near_field.get(
                "clean_closest_robot_ped_distance_m"
            ),
            "satisfied": near_field.get("satisfied"),
        },
        "command_sequence_changed": _mapping(summary.get("command_summary")).get(
            "sequence_changed"
        ),
        "closest_distance_delta_m": closest_delta.get("delta"),
        "collision_counts": _collision_counts(summary),
        "observation_summary_changed": _mapping(summary.get("observation_summary")).get("changed"),
    }


def _grid_classification(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Classify the cross-fixture sensitivity pattern conservatively."""

    usable = [row for row in rows if row["usable"]]
    failed_closed = [row for row in rows if not row["usable"]]
    if not usable:
        return {
            "label": "failed_closed_no_native_live_evidence",
            "rationale": "No usable native live step-diagnostics summaries were available.",
        }
    sensitive = [
        row for row in usable if row["status"] in {"behavior_sensitive", "behavior_sensitive_grid"}
    ]
    near_field = [row for row in usable if _mapping(row.get("near_field")).get("satisfied") is True]
    weak = [row for row in usable if row["status"] == "observation_only_scenario_too_weak"]
    if sensitive and failed_closed:
        return {
            "label": "fixture_candidate_failed_closed_after_sensitive_grid",
            "rationale": (
                "The committed #3330 near-field grid is behavior-sensitive, but the attempted "
                "second matrix failed closed under the near-field guardrail. Treat this as a "
                "useful negative external-validity check, not robustness evidence."
            ),
        }
    if sensitive and len(sensitive) == len(usable):
        return {
            "label": "sensitivity_persists_across_sources",
            "rationale": "All usable native live summaries reported non-null behavior deltas.",
        }
    if sensitive and near_field and weak:
        return {
            "label": "fixture_specific_near_field_sensitivity",
            "rationale": (
                "At least one near-field native live fixture was behavior-sensitive, while "
                "at least one source changed observations only and was explicitly too weak."
            ),
        }
    if sensitive:
        return {
            "label": "mixed_or_profile_specific_sensitivity",
            "rationale": (
                "Some, but not all, usable native live summaries reported non-null behavior deltas."
            ),
        }
    return {
        "label": "no_behavior_sensitivity_observed",
        "rationale": "Usable native live summaries did not report non-null behavior deltas.",
    }


def build_report(source_paths: list[Path], *, command: str) -> dict[str, Any]:
    """Build the compact cross-fixture report."""

    rows = [_source_row(path, _load_summary(path)) for path in source_paths]
    usable = [row for row in rows if row["usable"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3335,
        "claim_boundary": CLAIM_BOUNDARY,
        "reproducibility": {
            "command": command,
            "source_summaries": [path.as_posix() for path in source_paths],
        },
        "source_count": len(rows),
        "usable_native_live_source_count": len(usable),
        "failed_closed_source_count": len(rows) - len(usable),
        "sources": rows,
        "classification": _grid_classification(rows),
    }


def _markdown(report: dict[str, Any]) -> str:
    """Render the report as compact Markdown."""

    classification = _mapping(report.get("classification"))
    lines = [
        "# Issue #3335 Observation-Noise Cross-Fixture Live Grid",
        "",
        "## Claim Boundary",
        "",
        str(report["claim_boundary"]),
        "",
        "## Reproducibility",
        "",
        f"- Command: `{_mapping(report['reproducibility']).get('command')}`",
        f"- Source summaries: `{report['source_count']}`",
        f"- Usable native live sources: `{report['usable_native_live_source_count']}`",
        f"- Failed-closed sources: `{report['failed_closed_source_count']}`",
        "",
        "## Classification",
        "",
        f"- Label: `{classification.get('label')}`",
        f"- Rationale: {classification.get('rationale')}",
        "",
        "## Source Rows",
        "",
        "| Source | Scenario | Seed | Near-field | Status | Command changed | Closest delta |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for row in report["sources"]:
        near_field = _mapping(row.get("near_field"))
        lines.append(
            f"| `{row['path']}` | `{row.get('scenario')}` | `{row.get('seed')}` | "
            f"`{near_field.get('satisfied')}` | `{row['status']}` | "
            f"`{row.get('command_sequence_changed')}` | "
            f"`{row.get('closest_distance_delta_m')}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a synthesis of tracked compact native live summaries, not a new full benchmark.",
            "- Fallback, degraded, unavailable, or malformed sources fail closed.",
            "- Fixture/profile-specific results remain diagnostic-only and do not support sensor-realism claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        dest="source_summaries",
        type=Path,
        action="append",
        default=None,
        help="Tracked compact native live-smoke summary JSON. May be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    source_paths = list(args.source_summaries or DEFAULT_SOURCE_SUMMARIES)
    command = (
        "uv run python scripts/benchmark/summarize_observation_noise_live_grid.py "
        + " ".join(f"--source-summary {path.as_posix()}" for path in source_paths)
        + f" --output-json {args.output_json.as_posix()} --output-md {args.output_md.as_posix()}"
    )
    report = build_report(source_paths, command=command)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "classification": report["classification"],
                "output_json": args.output_json.as_posix(),
                "output_md": args.output_md.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
