#!/usr/bin/env python3
"""Compare same-seed clean and perturbed observation-noise live smoke rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue_3201_observation_noise_live_smoke.v1"
DEFAULT_METRICS = (
    "success",
    "min_distance",
    "min_clearance",
    "path_length",
    "social_proxemic_intrusion_steps",
    "social_proxemic_intrusion_frac",
    "time_to_yield_s",
    "robot_yield_distance_m",
    "pedestrian_path_deviation_proxy_m",
)
TRACE_PROGRESS_FIELDS = (
    "steps_observed",
    "initial_goal_distance",
    "final_goal_distance",
    "best_goal_distance",
    "net_goal_progress",
    "best_goal_progress",
    "progress_step_count",
    "regression_step_count",
    "stagnant_step_count",
    "longest_stagnant_run",
    "closest_robot_ped_distance",
    "closest_robot_ped_step",
    "collision_flag_counts",
)


def _load_single_row(path: Path) -> dict[str, Any]:
    """Load exactly one JSONL row."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL row, found {len(rows)}")
    return rows[0]


def _durable_input_ref(path: Path) -> str:
    """Return a report-safe input reference that does not depend on ignored artifacts."""
    parts = path.as_posix().split("/")
    if "output" in parts:
        tail = "/".join(parts[parts.index("output") + 1 :])
        return f"worktree-local ignored artifact summarized in this report ({tail})"
    return path.as_posix()


def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` as a mapping, treating explicit JSON null as empty."""
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """Normalize scalar numeric values for delta comparison."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_delta(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    """Return per-metric values and deltas for known scalar metrics."""
    clean_metrics = _mapping(clean.get("metrics"))
    perturbed_metrics = _mapping(perturbed.get("metrics"))
    deltas: dict[str, Any] = {}
    for metric in DEFAULT_METRICS:
        clean_value = clean_metrics.get(metric)
        perturbed_value = perturbed_metrics.get(metric)
        clean_number = _number(clean_value)
        perturbed_number = _number(perturbed_value)
        delta = (
            perturbed_number - clean_number
            if clean_number is not None and perturbed_number is not None
            else None
        )
        deltas[metric] = {
            "clean": clean_value,
            "perturbed": perturbed_value,
            "delta": delta,
        }
    return deltas


def _status_delta(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    """Compare top-level episode status/outcome fields."""
    fields = ("status", "termination_reason", "steps", "horizon", "outcome")
    return {
        field: {
            "clean": clean.get(field),
            "perturbed": perturbed.get(field),
            "changed": clean.get(field) != perturbed.get(field),
        }
        for field in fields
    }


def _noise_stats_delta(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    """Compare observation-noise counters."""
    clean_stats = _mapping(clean.get("observation_noise_stats"))
    perturbed_stats = _mapping(perturbed.get("observation_noise_stats"))
    keys = sorted(set(clean_stats) | set(perturbed_stats))
    return {
        key: {
            "clean": clean_stats.get(key),
            "perturbed": perturbed_stats.get(key),
            "delta": (
                perturbed_stats.get(key, 0) - clean_stats.get(key, 0)
                if isinstance(clean_stats.get(key, 0), int)
                and isinstance(perturbed_stats.get(key, 0), int)
                else None
            ),
        }
        for key in keys
    }


def _has_metric_delta(metric_deltas: dict[str, Any], *, epsilon: float = 1e-9) -> bool:
    """Return whether any comparable metric changed."""
    return any(
        item["delta"] is not None and abs(float(item["delta"])) > epsilon
        for item in metric_deltas.values()
    )


def _has_status_delta(status_deltas: dict[str, Any]) -> bool:
    """Return whether any compared top-level status field changed."""
    return any(item["changed"] for item in status_deltas.values())


def _has_noise_delta(noise_deltas: dict[str, Any]) -> bool:
    """Return whether any observation-noise counter changed."""
    return any(item["delta"] is not None and item["delta"] != 0 for item in noise_deltas.values())


def _classification(
    *,
    metric_deltas: dict[str, Any],
    status_deltas: dict[str, Any],
    noise_deltas: dict[str, Any],
) -> dict[str, str]:
    """Classify the same-seed smoke result."""
    if _has_metric_delta(metric_deltas) or _has_status_delta(status_deltas):
        return {
            "label": "non_null_behavior_delta",
            "rationale": (
                "Same-seed clean and perturbed rows differ on at least one "
                "episode metric or top-level outcome/status field."
            ),
        }
    if _has_noise_delta(noise_deltas):
        return {
            "label": "observation_only_delta",
            "rationale": (
                "Observation-noise counters changed, but compared behavior "
                "metrics and top-level outcomes were identical."
            ),
        }
    return {
        "label": "null_policy_insensitive",
        "rationale": (
            "No compared metric, status, or observation-noise counter changed; "
            "the smoke remains policy-insensitive."
        ),
    }


def build_report(clean_path: Path, perturbed_path: Path) -> dict[str, Any]:
    """Build the compact comparison report."""
    clean = _load_single_row(clean_path)
    perturbed = _load_single_row(perturbed_path)
    metric_deltas = _metric_delta(clean, perturbed)
    status_deltas = _status_delta(clean, perturbed)
    noise_deltas = _noise_stats_delta(clean, perturbed)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3201,
        "claim_boundary": (
            "Diagnostic same-seed local smoke only. This retests the issue #2749 "
            "observation-noise null with a pedestrian-dominated fixture; it is "
            "not benchmark-strength or hardware-calibrated sensor evidence."
        ),
        "inputs": {
            "clean_jsonl": _durable_input_ref(clean_path),
            "perturbed_jsonl": _durable_input_ref(perturbed_path),
        },
        "scenario": {
            "clean": clean.get("scenario_id"),
            "perturbed": perturbed.get("scenario_id"),
            "same_scenario": clean.get("scenario_id") == perturbed.get("scenario_id"),
        },
        "seed": {
            "clean": clean.get("seed"),
            "perturbed": perturbed.get("seed"),
            "same_seed": clean.get("seed") == perturbed.get("seed"),
        },
        "observation_noise": {
            "clean": clean.get("observation_noise"),
            "perturbed": perturbed.get("observation_noise"),
            "stats_delta": noise_deltas,
        },
        "status_delta": status_deltas,
        "metric_delta": metric_deltas,
        "classification": _classification(
            metric_deltas=metric_deltas,
            status_deltas=status_deltas,
            noise_deltas=noise_deltas,
        ),
    }


def _load_trace(path: Path) -> dict[str, Any]:
    """Load a policy-search step diagnostics trace."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("steps"), list):
        raise ValueError(f"{path} is not a step diagnostics trace")
    return payload


def _trace_progress_delta(clean: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, Any]:
    """Compare high-level progress summaries from step diagnostics."""
    clean_summary = _mapping(clean.get("progress_summary"))
    perturbed_summary = _mapping(perturbed.get("progress_summary"))
    deltas: dict[str, Any] = {}
    for field in TRACE_PROGRESS_FIELDS:
        clean_value = clean_summary.get(field)
        perturbed_value = perturbed_summary.get(field)
        clean_number = _number(clean_value)
        perturbed_number = _number(perturbed_value)
        delta = (
            perturbed_number - clean_number
            if clean_number is not None and perturbed_number is not None
            else None
        )
        deltas[field] = {
            "clean": clean_value,
            "perturbed": perturbed_value,
            "delta": delta,
            "changed": clean_value != perturbed_value,
        }
    return deltas


def _trace_command_summary(trace: dict[str, Any]) -> list[Any]:
    """Return the selected policy-command sequence."""
    return [_mapping(row).get("policy_command") for row in trace.get("steps", [])]


def _trace_observation_summary(trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize observation perturbation effects across step rows."""
    missed_total = 0
    occluded_total = 0
    observed_counts: list[int] = []
    profiles: set[str] = set()
    evidence_classes: set[str] = set()
    for row in trace.get("steps", []):
        meta = _mapping(_mapping(row).get("observation_perturbation"))
        missed_total += int(meta.get("missed_actor_count", 0) or 0)
        occluded_total += int(meta.get("occluded_actor_count", 0) or 0)
        observed_counts.append(int(meta.get("observed_actor_count", 0) or 0))
        profiles.add(str(meta.get("noise_profile", "")))
        evidence_classes.add(str(meta.get("evidence_class", "")))
    return {
        "missed_actor_observations_total": missed_total,
        "occluded_actor_observations_total": occluded_total,
        "min_observed_actor_count": min(observed_counts) if observed_counts else None,
        "max_observed_actor_count": max(observed_counts) if observed_counts else None,
        "noise_profiles": sorted(profile for profile in profiles if profile),
        "evidence_classes": sorted(value for value in evidence_classes if value),
    }


def _trace_classification(
    *,
    progress_delta: dict[str, Any],
    command_changed: bool,
    observation_changed: bool,
    closest_robot_ped_distance: float | None,
) -> dict[str, str]:
    """Classify paired step-diagnostics traces."""
    progress_changed = any(item["changed"] for item in progress_delta.values())
    if command_changed or progress_changed:
        return {
            "label": "non_null_behavior_delta",
            "rationale": (
                "Same-seed clean and perturbed live traces differ in selected "
                "commands or progress/risk summary fields."
            ),
        }
    if observation_changed:
        if closest_robot_ped_distance is not None and closest_robot_ped_distance > 2.0:
            return {
                "label": "observation_only_scenario_too_weak",
                "rationale": (
                    "Perturbation changed planner-input pedestrian observations, "
                    "but commands and progress/risk summaries were identical. "
                    f"The closest live robot-pedestrian distance was "
                    f"{closest_robot_ped_distance:.2f} m, above the 2 m near-field target."
                ),
            }
        return {
            "label": "observation_only_delta",
            "rationale": (
                "Perturbation changed planner-input pedestrian observations, "
                "but commands and progress/risk summaries were identical."
            ),
        }
    return {
        "label": "null_policy_insensitive",
        "rationale": "No compared observation, command, or progress/risk field changed.",
    }


def build_trace_report(clean_trace_path: Path, perturbed_trace_path: Path) -> dict[str, Any]:
    """Build a compact report from paired step-diagnostics traces."""
    clean = _load_trace(clean_trace_path)
    perturbed = _load_trace(perturbed_trace_path)
    progress_delta = _trace_progress_delta(clean, perturbed)
    clean_commands = _trace_command_summary(clean)
    perturbed_commands = _trace_command_summary(perturbed)
    clean_observation = _trace_observation_summary(clean)
    perturbed_observation = _trace_observation_summary(perturbed)
    observation_changed = clean_observation != perturbed_observation
    closest = _number(_mapping(clean.get("progress_summary")).get("closest_robot_ped_distance"))
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 3201,
        "claim_boundary": (
            "Diagnostic same-seed local smoke only. This retests the issue #2749 "
            "observation-noise null with a pedestrian-dominated candidate surface; "
            "it is not benchmark-strength or hardware-calibrated sensor evidence."
        ),
        "inputs": {
            "clean_trace_json": _durable_input_ref(clean_trace_path),
            "perturbed_trace_json": _durable_input_ref(perturbed_trace_path),
        },
        "scenario": {
            "clean": clean.get("scenario_id"),
            "perturbed": perturbed.get("scenario_id"),
            "same_scenario": clean.get("scenario_id") == perturbed.get("scenario_id"),
        },
        "seed": {
            "clean": clean.get("seed"),
            "perturbed": perturbed.get("seed"),
            "same_seed": clean.get("seed") == perturbed.get("seed"),
        },
        "observation_perturbation_config": {
            "clean": clean.get("observation_perturbation_config"),
            "perturbed": perturbed.get("observation_perturbation_config"),
        },
        "observation_summary": {
            "clean": clean_observation,
            "perturbed": perturbed_observation,
            "changed": observation_changed,
        },
        "near_field_target": {
            "threshold_m": 2.0,
            "clean_closest_robot_ped_distance_m": closest,
            "satisfied": closest is not None and closest <= 2.0,
        },
        "command_summary": {
            "clean_first": clean_commands[0] if clean_commands else None,
            "perturbed_first": perturbed_commands[0] if perturbed_commands else None,
            "clean_last": clean_commands[-1] if clean_commands else None,
            "perturbed_last": perturbed_commands[-1] if perturbed_commands else None,
            "sequence_changed": clean_commands != perturbed_commands,
        },
        "progress_delta": progress_delta,
        "classification": _trace_classification(
            progress_delta=progress_delta,
            command_changed=clean_commands != perturbed_commands,
            observation_changed=observation_changed,
            closest_robot_ped_distance=closest,
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    """Render a short Markdown summary."""
    classification = report["classification"]
    lines = [
        "# Issue #3201 Observation-Noise Live Smoke",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Inputs",
        "",
        f"- Clean JSONL: `{report['inputs']['clean_jsonl']}`",
        f"- Perturbed JSONL: `{report['inputs']['perturbed_jsonl']}`",
        f"- Same scenario: `{report['scenario']['same_scenario']}`",
        f"- Same seed: `{report['seed']['same_seed']}`",
        "",
        "## Classification",
        "",
        f"- Label: `{classification['label']}`",
        f"- Rationale: {classification['rationale']}",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Clean | Perturbed | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in report["metric_delta"].items():
        lines.append(
            f"| `{metric}` | `{values['clean']}` | `{values['perturbed']}` | `{values['delta']}` |"
        )
    lines.extend(["", "## Observation-Noise Counter Deltas", ""])
    lines.extend(["| Counter | Clean | Perturbed | Delta |", "|---|---:|---:|---:|"])
    for key, values in report["observation_noise"]["stats_delta"].items():
        lines.append(
            f"| `{key}` | `{values['clean']}` | `{values['perturbed']}` | `{values['delta']}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- One scenario, one seed, one horizon; diagnostic smoke only.",
            "- Uses non-calibrated observation noise and makes no hardware sensor claim.",
        ]
    )
    return "\n".join(lines)


def _trace_markdown(report: dict[str, Any]) -> str:
    """Render Markdown for paired step-diagnostics traces."""
    classification = report["classification"]
    near_field = _mapping(report.get("near_field_target"))
    clean_closest_distance = near_field.get("clean_closest_robot_ped_distance_m")
    near_field_caveat = (
        f"- Clean trace closest robot-pedestrian distance: "
        f"`{clean_closest_distance}` m "
        f"(target <= `{near_field.get('threshold_m')}` m, "
        f"satisfied: `{near_field.get('satisfied')}`)."
        if clean_closest_distance is not None
        else "- Clean trace near-field target metadata was unavailable."
    )
    lines = [
        "# Issue #3201 Observation-Noise Live Smoke",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Inputs",
        "",
        f"- Clean trace: `{report['inputs']['clean_trace_json']}`",
        f"- Perturbed trace: `{report['inputs']['perturbed_trace_json']}`",
        f"- Same scenario: `{report['scenario']['same_scenario']}`",
        f"- Same seed: `{report['seed']['same_seed']}`",
        "",
        "## Classification",
        "",
        f"- Label: `{classification['label']}`",
        f"- Rationale: {classification['rationale']}",
        "",
        "## Near-Field Target",
        "",
        near_field_caveat,
        "",
        "## Command Summary",
        "",
        f"- Sequence changed: `{report['command_summary']['sequence_changed']}`",
        f"- Clean first/last: `{report['command_summary']['clean_first']}` / "
        f"`{report['command_summary']['clean_last']}`",
        f"- Perturbed first/last: `{report['command_summary']['perturbed_first']}` / "
        f"`{report['command_summary']['perturbed_last']}`",
        "",
        "## Observation Summary",
        "",
        f"- Changed: `{report['observation_summary']['changed']}`",
        f"- Clean: `{report['observation_summary']['clean']}`",
        f"- Perturbed: `{report['observation_summary']['perturbed']}`",
        "",
        "## Progress Deltas",
        "",
        "| Field | Clean | Perturbed | Delta | Changed |",
        "|---|---:|---:|---:|---|",
    ]
    for field, values in report["progress_delta"].items():
        lines.append(
            f"| `{field}` | `{values['clean']}` | `{values['perturbed']}` | "
            f"`{values['delta']}` | `{values['changed']}` |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- One scenario, one seed, one horizon; diagnostic smoke only.",
            near_field_caveat,
            "- Uses non-calibrated observation perturbations and makes no hardware sensor claim.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--clean-jsonl", type=Path)
    input_group.add_argument("--clean-trace-json", type=Path)
    parser.add_argument("--perturbed-jsonl", type=Path)
    parser.add_argument("--perturbed-trace-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.clean_trace_json is not None:
        if args.perturbed_trace_json is None:
            parser.error("--perturbed-trace-json is required with --clean-trace-json")
        report = build_trace_report(args.clean_trace_json, args.perturbed_trace_json)
        markdown = _trace_markdown(report)
    else:
        if args.perturbed_jsonl is None:
            parser.error("--perturbed-jsonl is required with --clean-jsonl")
        report = build_report(args.clean_jsonl, args.perturbed_jsonl)
        markdown = _markdown(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
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
