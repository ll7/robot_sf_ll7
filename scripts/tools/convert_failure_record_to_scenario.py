#!/usr/bin/env python3
"""Convert failure records to scenario hypotheses for robot_sf_ll7.

This tool reads a structured failure record YAML and generates a scenario
hypothesis YAML using deterministic templates. Generated scenarios are
marked as draft and require manual review before any evidence claims.

Usage:
    uv run python scripts/tools/convert_failure_record_to_scenario.py \\
      --record configs/failure_records/examples/ammv_sidewalk_blocked_path.yaml \\
      --output-yaml output/failure_record_scenarios/example.scenario.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

FAILURE_RECORD_SCHEMA_VERSION = "failure-record.v1"
SCENARIO_MATRIX_SCHEMA_VERSION = "robot_sf.scenario_matrix.v1"
GENERATION_METHOD = "deterministic_template_v1"
DEFAULT_SEEDS = (101, 102, 103)

ENVIRONMENT_TO_TEMPLATE = {
    "event": "event_disruption",
    "sidewalk": "ammv_sidewalk",
    "shared_space": "ammv_shared_space",
    "crossing": "classic_crossing",
    "road_edge": "road_edge",
}

TEMPLATE_TO_MAP = {
    "event_disruption": "../../../maps/svg_maps/event_disruption/event_disruption.svg",
    "ammv_sidewalk": "../../../maps/svg_maps/ammv_sidewalk/ammv_sidewalk.svg",
    "ammv_shared_space": "../../../maps/svg_maps/ammv_shared_space/ammv_shared_space.svg",
    "classic_crossing": "../../../maps/svg_maps/classic_crossing/classic_crossing.svg",
    "road_edge": "../../../maps/svg_maps/road_edge/road_edge.svg",
}

FAILURE_MODE_TO_EXPECTED = {
    "collision": ["collision", "near_miss"],
    "near_miss": ["near_miss"],
    "stuck": ["stuck", "excessive_detour"],
    "blocked_path": ["stuck", "excessive_detour", "near_miss"],
    "unsafe_fallback": ["unsafe_fallback", "stuck"],
    "pedestrian_disruption": ["near_miss", "pedestrian_disruption"],
}

CONTEXTUAL_FACTOR_WARNINGS = {
    "dense_crowd": "High pedestrian density may require manual tuning",
    "temporary_obstacle": "Obstacle representation is approximate",
    "communication_loss": "Communication factors not modeled in simulator",
    "abnormal_hazard": "Hazard behavior is simplified",
    "narrow_clearance": "Clearance margins may need adjustment",
}


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru for CLI output."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


def _validate_failure_record(record: dict[str, Any]) -> list[str]:  # noqa: C901, PLR0912
    """Validate a failure record against the expected schema.

    Returns a list of validation errors; empty list means valid.
    """
    errors = []

    if "schema_version" not in record:
        errors.append("Missing required field: schema_version")
    elif record["schema_version"] != FAILURE_RECORD_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {record['schema_version']!r}; "
            f"expected {FAILURE_RECORD_SCHEMA_VERSION!r}"
        )

    if "failure_record" not in record:
        errors.append("Missing required field: failure_record")
        return errors

    fr = record["failure_record"]
    required_fields = ["id", "source", "date", "environment", "actors",
                       "triggering_condition", "failure_mode",
                       "required_manual_review", "claim_boundary"]

    for field in required_fields:
        if field not in fr:
            errors.append(f"Missing required field in failure_record: {field}")

    if "environment" in fr and fr["environment"] not in ENVIRONMENT_TO_TEMPLATE:
        errors.append(
            f"Unknown environment: {fr['environment']!r}; "
            f"expected one of {list(ENVIRONMENT_TO_TEMPLATE.keys())}"
        )

    if "failure_mode" in fr and fr["failure_mode"] not in FAILURE_MODE_TO_EXPECTED:
        errors.append(
            f"Unknown failure_mode: {fr['failure_mode']!r}; "
            f"expected one of {list(FAILURE_MODE_TO_EXPECTED.keys())}"
        )

    if "required_manual_review" in fr and fr["required_manual_review"] is not True:
        errors.append("required_manual_review must be true")

    if "claim_boundary" in fr:
        if "not evidence" not in fr["claim_boundary"].lower():
            errors.append("claim_boundary must state 'not evidence'")

    if "actors" in fr:
        if not isinstance(fr["actors"], list):
            errors.append("actors must be a list")
        else:
            for i, actor in enumerate(fr["actors"]):
                if not isinstance(actor, dict):
                    errors.append(f"actors[{i}] must be a dict")
                elif "type" not in actor or "count" not in actor:
                    errors.append(f"actors[{i}] missing required fields: type, count")

    return errors


def _generate_assumptions(failure_record: dict[str, Any]) -> list[str]:
    """Generate list of assumptions made during conversion."""
    assumptions = []

    fr = copy.deepcopy(failure_record)
    assumptions.append(f"Environment '{fr.get('environment', 'unknown')}' mapped to template")
    assumptions.append(f"Failure mode '{fr.get('failure_mode', 'unknown')}' used for expected modes")

    actor_count = sum(a.get("count", 0) for a in fr.get("actors", []) if isinstance(a, dict))
    if actor_count > 4:
        assumptions.append(f"Actor count ({actor_count}) truncated to 4 single pedestrians for v1")

    if "contextual_factors" in fr:
        for factor in fr["contextual_factors"]:
            if factor not in CONTEXTUAL_FACTOR_WARNINGS:
                assumptions.append(f"Contextual factor '{factor}' mapped to metadata only")

    return assumptions


def _generate_invalidity_warnings(failure_record: dict[str, Any]) -> list[str]:
    """Generate list of invalidity warnings for the generated scenario."""
    warnings = []

    fr = failure_record
    for factor in fr.get("contextual_factors", []):
        if factor in CONTEXTUAL_FACTOR_WARNINGS:
            warnings.append(CONTEXTUAL_FACTOR_WARNINGS[factor])
        else:
            warnings.append(f"Unmapped contextual factor: {factor}")

    if fr.get("failure_mode") == "collision":
        warnings.append("Collision scenarios require careful safety review before execution")

    return warnings


def _generate_pedestrians(failure_record: dict[str, Any], max_count: int = 4) -> list[dict[str, Any]]:
    """Generate single_pedestrians list from actors."""
    pedestrians = []

    for actor in copy.deepcopy(failure_record.get("actors", [])):
        if not isinstance(actor, dict):
            continue
        if actor.get("type") != "pedestrian":
            continue

        count = min(actor.get("count", 1), max_count - len(pedestrians))
        for i in range(count):
            pedestrians.append({
                "id": f"h{len(pedestrians) + 1}",
                "goal_poi": f"poi_h{len(pedestrians) + 1}_goal",
                "speed_m_s": 1.0,
                "note": f"Pedestrian {len(pedestrians) + 1} from failure record {failure_record.get('id')}",
            })

    return pedestrians


def _build_scenario_payload(failure_record: dict[str, Any]) -> dict[str, Any]:
    """Build the complete scenario YAML payload from a failure record."""
    fr = copy.deepcopy(failure_record)

    env = fr.get("environment", "sidewalk")
    template = ENVIRONMENT_TO_TEMPLATE.get(env, "ammv_sidewalk")
    map_file = TEMPLATE_TO_MAP.get(template, TEMPLATE_TO_MAP["ammv_sidewalk"])

    failure_mode = fr.get("failure_mode", "blocked_path")
    expected_modes = FAILURE_MODE_TO_EXPECTED.get(failure_mode, ["stuck"])

    if "expected_failure_modes" in fr:
        expected_modes = list(set(expected_modes + fr["expected_failure_modes"]))

    pedestrians = _generate_pedestrians(fr)
    assumptions = _generate_assumptions(fr)
    warnings = _generate_invalidity_warnings(fr)

    actor_count = sum(
        a.get("count", 0) for a in fr.get("actors", []) if isinstance(a, dict)
    )

    ped_density = 0.0
    if "dense_crowd" in fr.get("contextual_factors", []):
        ped_density = 0.8
    elif actor_count > 4:
        ped_density = 0.5

    return {
        "schema_version": SCENARIO_MATRIX_SCHEMA_VERSION,
        "scenarios": [
            {
                "name": f"failure_record_{fr.get('id', 'unknown')}",
                "map_file": map_file,
                "simulation_config": {
                    "max_episode_steps": 400,
                    "ped_density": ped_density,
                },
                "single_pedestrians": pedestrians,
                "robot_config": {},
                "metadata": {
                    "generated_from_failure_record": fr.get("id", "unknown"),
                    "generation_method": GENERATION_METHOD,
                    "required_manual_review": True,
                    "claim_boundary": "scenario hypothesis only; not executed evidence",
                    "generated_assumptions": assumptions,
                    "invalidity_warnings": warnings,
                    "expected_failure_modes": expected_modes,
                    "archetype": template,
                    "flow": "bi",
                    "behavior": "failure_record_derived",
                    "authoring": {
                        "status": "draft",
                        "source_issue": "#4760",
                        "generated_by": "scripts/tools/convert_failure_record_to_scenario.py",
                        "benchmark_evidence": False,
                        "promotion_note": (
                            "Not benchmark evidence until separately reviewed, certified, "
                            "and executed through the benchmark workflow."
                        ),
                    },
                },
                "seeds": list(DEFAULT_SEEDS),
            },
        ],
    }


def convert_failure_record(
    input_path: Path,
    output_path: Path | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Convert a failure record to a scenario hypothesis.

    Args:
        input_path: Path to the input failure record YAML.
        output_path: Path for the output scenario YAML (optional).
        verbose: Enable verbose logging.

    Returns:
        The generated scenario payload, or None if validation fails.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If validation fails.
    """
    _configure_logging(verbose)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Reading failure record: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        record = yaml.safe_load(f)

    errors = _validate_failure_record(record)
    if errors:
        logger.error("Validation failed:")
        for err in errors:
            logger.error(f"  - {err}")
        raise ValueError(f"Invalid failure record: {'; '.join(errors)}")

    logger.info("Validation passed")

    payload = _build_scenario_payload(record["failure_record"])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(payload, f, sort_keys=False, width=100, allow_unicode=False)
        logger.info(f"Wrote scenario to: {output_path}")

    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="Path to the input failure record YAML file.",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        required=False,
        help="Path for the output scenario YAML file.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print output to stdout instead of writing to file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = _build_parser().parse_args(argv)

    if not args.output_yaml and not args.stdout:
        print("Error: Must specify --output-yaml or --stdout", file=sys.stderr)
        return 2

    try:
        payload = convert_failure_record(
            args.record,
            args.output_yaml if not args.stdout else None,
            verbose=args.verbose,
        )

        if args.stdout:
            print(yaml.dump(payload, sort_keys=False, width=100, allow_unicode=False))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
