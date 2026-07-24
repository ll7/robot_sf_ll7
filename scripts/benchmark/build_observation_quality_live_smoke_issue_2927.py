#!/usr/bin/env python3
"""Build issue #2927 observation-quality live-smoke evidence.

This report wraps an existing same-seed live step-diagnostics summary and adds
bounded ``observation_quality.v1`` metadata plus explicit false-negative and
false-positive safety-effect interpretation.  It is diagnostic smoke evidence,
not benchmark-strength or hardware-calibrated sensor evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from robot_sf.benchmark.observation_quality import ObservationQuality

SCHEMA_VERSION = "issue_2927_observation_quality_live_smoke.v1"
OBSERVATION_QUALITY_SCHEMA_VERSION = "observation_quality.v1"
DEFAULT_SOURCE_SUMMARY = Path(
    "docs/context/evidence/issue_3233_near_field_observation_noise/summary.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "docs/context/evidence/issue_2927_observation_quality_live_smoke/summary.json"
)
DEFAULT_OUTPUT_MD = Path(
    "docs/context/evidence/issue_2927_observation_quality_live_smoke/README.md"
)
CLAIM_BOUNDARY = (
    "Smoke/diagnostic live step-diagnostics evidence only. The report attaches "
    "bounded simulator observation-quality metadata to an existing same-seed "
    "near-field live replay summary and reports safety effects from that smoke. "
    "It is not paper-grade, benchmark-strength planner superiority evidence, "
    "or hardware-calibrated sensor realism."
)


def _mapping(value: Any) -> dict[str, Any]:
    """Return a mapping or an empty mapping for null/malformed values."""

    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """Return a finite float for scalar values when available."""

    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quality_from_config(config: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Convert live perturbation config into the observation-quality field group."""

    missed_probability = float(config.get("missed_detection_probability", 0.0) or 0.0)
    delay_steps = int(config.get("delay_steps", 0) or 0)
    quality = ObservationQuality(
        visibility=["live_step_diagnostics_simulated_pedestrians"],
        occlusion=(
            ["distance_occlusion_proxy"]
            if config.get("occlusion_distance_m") is not None
            else ["not_modeled_in_this_smoke"]
        ),
        latency_s=0.0 if delay_steps <= 0 else float(delay_steps) * 0.1,
        dropout_probability=missed_probability,
        range_limit_m=None,
        angular_noise_std_rad=0.0,
        false_negative_rate=missed_probability,
        false_positive_rate=0.0,
        notes=(
            f"{role} diagnostic simulator observation-quality metadata only; "
            "not hardware-calibrated sensor realism."
        ),
    )
    return {
        "schema_version": OBSERVATION_QUALITY_SCHEMA_VERSION,
        "fields": quality.to_dict(),
    }


def _behavior_changed(source: dict[str, Any]) -> bool:
    """Return whether the source live smoke observed planner-behavior differences."""

    if _mapping(source.get("command_summary")).get("sequence_changed") is True:
        return True
    return any(
        _mapping(delta).get("changed") is True
        for delta in _mapping(source.get("progress_delta")).values()
    )


def _collision_delta(source: dict[str, Any]) -> dict[str, Any]:
    """Return compact collision-count changes from the source progress delta."""

    collision = _mapping(_mapping(source.get("progress_delta")).get("collision_flag_counts"))
    clean = _mapping(collision.get("clean"))
    perturbed = _mapping(collision.get("perturbed"))
    keys = sorted(set(clean) | set(perturbed))
    return {
        key: {
            "clean": clean.get(key, 0),
            "perturbed": perturbed.get(key, 0),
            "delta": int(perturbed.get(key, 0) or 0) - int(clean.get(key, 0) or 0),
        }
        for key in keys
    }


def _safety_effects(source: dict[str, Any]) -> dict[str, Any]:
    """Summarize false-negative and false-positive effects without overclaiming."""

    observation = _mapping(source.get("observation_summary"))
    clean = _mapping(observation.get("clean"))
    perturbed = _mapping(observation.get("perturbed"))
    missed_delta = int(perturbed.get("missed_actor_observations_total", 0) or 0) - int(
        clean.get("missed_actor_observations_total", 0) or 0
    )
    occluded_delta = int(perturbed.get("occluded_actor_observations_total", 0) or 0) - int(
        clean.get("occluded_actor_observations_total", 0) or 0
    )
    behavior_changed = _behavior_changed(source)
    if missed_delta > 0 or occluded_delta > 0:
        false_negative_effect = (
            "non_null_behavior_delta_with_false_negative_perturbation"
            if behavior_changed
            else "observation_only_false_negative_perturbation"
        )
        false_negative_rationale = (
            "The perturbed live row removed or occluded pedestrian observations. "
            "Behavior/progress fields changed in the same-seed comparison."
            if behavior_changed
            else "The perturbed live row changed pedestrian observations, but compared "
            "behavior/progress fields stayed unchanged."
        )
    else:
        false_negative_effect = "none_observed"
        false_negative_rationale = (
            "No added missed or occluded pedestrian observations were recorded."
        )

    return {
        "false_negative": {
            "effect": false_negative_effect,
            "missed_actor_observations_delta": missed_delta,
            "occluded_actor_observations_delta": occluded_delta,
            "collision_delta": _collision_delta(source),
            "rationale": false_negative_rationale,
        },
        "false_positive": {
            "effect": "not_available_excluded",
            "false_positive_actor_rows": 0,
            "rationale": (
                "The source live smoke did not inject false-positive actors. "
                "False-positive safety effects are explicitly excluded rather than "
                "treated as successful evidence."
            ),
        },
    }


def _require_bool(name: str, value: Any) -> bool:
    """Return a boolean evidence flag without truthiness coercion."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def build_report(source: dict[str, Any]) -> dict[str, Any]:
    """Build the issue #2927 report from a live-smoke source summary."""

    configs = _mapping(source.get("observation_perturbation_config"))
    clean_config = _mapping(configs.get("clean"))
    perturbed_config = _mapping(configs.get("perturbed"))
    near_field = _mapping(source.get("near_field_target"))
    closest = _number(near_field.get("clean_closest_robot_ped_distance_m"))
    near_field_satisfied = _require_bool(
        "near_field_target.satisfied",
        near_field.get("satisfied", False),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2927,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_live_smoke": {
            "summary_path": DEFAULT_SOURCE_SUMMARY.as_posix(),
            "schema_version": source.get("schema_version"),
            "issue": source.get("issue"),
            "classification": source.get("classification"),
        },
        "scenario": source.get("scenario"),
        "seed": source.get("seed"),
        "execution_boundary": {
            "mode": "live_step_diagnostics_summary",
            "evidence_status": "smoke evidence",
            "near_field_threshold_m": near_field.get("threshold_m", 2.0),
            "clean_closest_robot_ped_distance_m": closest,
            "near_field_satisfied": near_field_satisfied,
            "fallback_rows": [],
            "degraded_rows": [],
            "not_available_rows": [
                {
                    "row": "false_positive_actor_injection",
                    "reason": "not modeled by the source live-smoke perturbation",
                    "classification": "explicitly_excluded",
                }
            ],
        },
        "observation_quality": {
            "clean": _quality_from_config(clean_config, role="clean"),
            "perturbed": _quality_from_config(perturbed_config, role="perturbed"),
        },
        "observation_summary": source.get("observation_summary"),
        "command_summary": source.get("command_summary"),
        "progress_delta": source.get("progress_delta"),
        "safety_effects": _safety_effects(source),
        "classification": source.get("classification"),
    }


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report."""

    classification = _mapping(report.get("classification"))
    boundary = _mapping(report.get("execution_boundary"))
    effects = _mapping(report.get("safety_effects"))
    false_negative = _mapping(effects.get("false_negative"))
    false_positive = _mapping(effects.get("false_positive"))
    quality = _mapping(report.get("observation_quality"))
    perturbed_quality = _mapping(_mapping(quality.get("perturbed")).get("fields"))
    lines = [
        "# Issue #2927 Observation-Quality Live Smoke",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Source",
        "",
        f"- Source summary: `{report['source_live_smoke']['summary_path']}`",
        f"- Source issue: `#{report['source_live_smoke']['issue']}`",
        f"- Source classification: `{classification.get('label')}`",
        "",
        "## Execution Boundary",
        "",
        f"- Evidence status: `{boundary.get('evidence_status')}`",
        f"- Near-field satisfied: `{boundary.get('near_field_satisfied')}`",
        f"- Clean closest robot-pedestrian distance: "
        f"`{boundary.get('clean_closest_robot_ped_distance_m')}` m",
        "- Fallback/degraded rows: none in the source summary.",
        "- Not-available rows: false-positive actor injection is explicitly excluded.",
        "",
        "## Observation Quality",
        "",
        f"- Schema: `{_mapping(quality.get('perturbed')).get('schema_version')}`",
        f"- Perturbed false-negative rate: `{perturbed_quality.get('false_negative_rate')}`",
        f"- Perturbed false-positive rate: `{perturbed_quality.get('false_positive_rate')}`",
        f"- Perturbed angular noise std: `{perturbed_quality.get('angular_noise_std_rad')}`",
        f"- Perturbed range limit: `{perturbed_quality.get('range_limit_m')}`",
        "",
        "## Safety Effects",
        "",
        f"- False-negative effect: `{false_negative.get('effect')}`",
        f"- False-negative rationale: {false_negative.get('rationale')}",
        f"- False-positive effect: `{false_positive.get('effect')}`",
        f"- False-positive rationale: {false_positive.get('rationale')}",
        "",
        "## Caveats",
        "",
        "- One scenario, one seed, one live step-diagnostics summary.",
        "- Uses non-calibrated simulator observation perturbations only.",
        "- This report does not claim planner superiority or paper-grade evidence.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary-json", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    source = json.loads(args.source_summary_json.read_text(encoding="utf-8"))
    report = build_report(source)
    report["source_live_smoke"]["summary_path"] = args.source_summary_json.as_posix()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "issue": report["issue"],
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
