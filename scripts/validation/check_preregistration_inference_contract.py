#!/usr/bin/env python3
"""Shared checker for the preregistration inference contract.

Plain-language summary: issue #5557 requires that every comparative preregistration
declare four items before execution so the branch verdict can no longer flip on
unspecified methodological choices. This checker validates that section is present
and well-formed in a preregistration YAML payload.

The inference contract must contain:
  1. resampling_unit - the bootstrap resampling hierarchy and rationale
  2. inference_population - whether the scenario suite is fixed or sampled
  3. estimand - the quantitative target (paired delta, per-arm interval, etc.)
  4. decision_rule - the exact rule with thresholds for the branch verdict
  5. primary_metrics - the metrics the decision rule applies to
  6. multiplicity_handling - how multiple metrics or contrasts are handled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "InferenceContractError",
    "check_inference_contract",
    "check_yaml_file",
]


class InferenceContractError(ValueError):
    """Raised when the inference contract is missing, incomplete, or invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceContractError(message)


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, dict), f"{key} must be a mapping")
    return value


REQUIRED_RESAMPLING_UNIT_KEYS = ("method", "rationale")
REQUIRED_INFERENCE_POPULATION_KEYS = ("type", "rationale")
REQUIRED_ESTIMAND_KEYS = ("type", "description")
REQUIRED_DECISION_RULE_KEYS = ("rule", "threshold")


def _validate_resampling_unit(section: dict[str, Any]) -> None:
    for key in REQUIRED_RESAMPLING_UNIT_KEYS:
        value = section.get(key)
        _require(
            isinstance(value, str) and len(value.strip()) >= 5,
            f"inference_contract.resampling_unit.{key} must be a non-empty string "
            "(describe method and rationale, e.g. 'scenario-clustered hierarchical "
            "bootstrap: treat scenarios as the outer unit because between-scenario "
            "heterogeneity is large and we want to generalize to unseen scenarios')",
        )
    if "bootstrap_confidence" in section:
        confidence = section["bootstrap_confidence"]
        _require(
            isinstance(confidence, (int, float)) and 0.8 <= confidence <= 0.99,
            f"bootstrap_confidence must be a probability in [0.8, 0.99], got {confidence}",
        )
    if "resampling_order" in section:
        _require(
            isinstance(section["resampling_order"], str) and section["resampling_order"].strip(),
            "resampling_order must be a non-empty string describing hierarchy",
        )


def _validate_inference_population(section: dict[str, Any]) -> None:
    for key in REQUIRED_INFERENCE_POPULATION_KEYS:
        value = section.get(key)
        _require(
            isinstance(value, str) and len(value.strip()) >= 5,
            f"inference_population.{key} must be a non-empty string "
            "('fixed_suite' or 'sampled_population' with rationale)",
        )
    valid_types = {"fixed_suite", "sampled_population"}
    pop_type = str(section.get("type", "")).strip().lower()
    _require(
        pop_type in valid_types,
        f"inference_population.type must be one of {valid_types}, got '{pop_type}'",
    )


def _validate_estimand(section: dict[str, Any]) -> None:
    for key in REQUIRED_ESTIMAND_KEYS:
        value = section.get(key)
        _require(
            isinstance(value, str) and len(value.strip()) >= 5,
            f"inference_contract.estimand.{key} must be a non-empty string "
            "(e.g. 'paired_delta' or 'per_arm_interval' with description)",
        )
    estimand_type = str(section.get("type", "")).strip().lower()
    valid_types = {"paired_delta", "per_arm_interval", "ratio", "paired_delta_and_per_arm"}
    _require(
        estimand_type in valid_types,
        f"estimand.type must be one of {valid_types}, got '{estimand_type}'",
    )


def _validate_decision_rule(section: dict[str, Any]) -> None:
    for key in REQUIRED_DECISION_RULE_KEYS:
        value = section.get(key)
        _require(
            isinstance(value, str) and len(value.strip()) >= 5,
            f"inference_contract.decision_rule.{key} must be a non-empty string "
            "(e.g. 'CI-excludes-zero' or 'p<0.05 two-sided' with threshold details)",
        )


def _validate_primary_metrics(section: dict[str, Any]) -> None:
    metrics = section.get("metrics")
    _require(
        isinstance(metrics, list) and len(metrics) >= 1,
        "inference_contract.primary_metrics.metrics must be a non-empty list of metric names",
    )
    for m in metrics:
        _require(
            isinstance(m, str) and m.strip(),
            f"Each primary metric must be a non-empty string, got {m!r}",
        )
    if "ordered_by_importance" in section:
        _require(
            isinstance(section["ordered_by_importance"], bool),
            "ordered_by_importance must be a boolean",
        )


def _validate_multiplicity_handling(section: dict[str, Any]) -> None:
    for key in ("strategy", "rationale"):
        value = section.get(key)
        _require(
            isinstance(value, str) and len(value.strip()) >= 5,
            f"inference_contract.multiplicity_handling.{key} must be a non-empty string "
            "(e.g. 'holm_bonferroni over 3 preregistered contrasts' with rationale)",
        )
    if "adjustment_method" in section:
        valid_methods = {
            "holm_bonferroni",
            "bonferroni",
            "hochberg",
            "holmdunn",
            "none_single_metric",
            "none_single_contrast",
        }
        method = str(section["adjustment_method"]).strip().lower()
        _require(
            method in valid_methods,
            f"multiplicity_handling.adjustment_method must be one of {valid_methods}, "
            f"got '{method}'",
        )


def check_inference_contract(
    packet: dict[str, Any],
    *,
    section_key: str = "inference_contract",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate that a preregistration contains a complete inference contract.

    Returns a compact summary on success. Raises ``InferenceContractError`` if
    any required field is missing or malformed.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    _ = root  # available for path checks if needed in the future

    contract = _mapping(packet, section_key)

    _validate_resampling_unit(_mapping(contract, "resampling_unit"))
    _validate_inference_population(_mapping(contract, "inference_population"))
    _validate_estimand(_mapping(contract, "estimand"))
    _validate_decision_rule(_mapping(contract, "decision_rule"))
    _validate_primary_metrics(_mapping(contract, "primary_metrics"))
    _validate_multiplicity_handling(_mapping(contract, "multiplicity_handling"))

    return {
        "status": "ok",
        "resampling_method": contract["resampling_unit"]["method"],
        "population_type": contract["inference_population"]["type"],
        "estimand_type": contract["estimand"]["type"],
        "metric_count": len(contract["primary_metrics"]["metrics"]),
        "multiplicity_strategy": contract["multiplicity_handling"]["strategy"],
    }


def check_yaml_file(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load a YAML preregistration and validate its inference contract."""
    if not path.is_file():
        raise FileNotFoundError(f"preregistration config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "preregistration config must be a YAML mapping")
    schema = payload.get("schema_version")
    _require(
        isinstance(schema, str) and schema,
        "preregistration must have a non-empty schema_version",
    )
    return check_inference_contract(payload, repo_root=repo_root)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the shared inference-contract checker."""
    parser = argparse.ArgumentParser(
        description="Validate the inference contract in a preregistration YAML."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the preregistration YAML. If omitted, lists known configs.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output.")
    args = parser.parse_args(argv)

    if args.config is None:
        # Scan for known preregistrations
        root = Path(__file__).resolve().parents[2]
        candidates = (
            list(root.glob("configs/analysis/*_packet.yaml"))
            + list(root.glob("configs/benchmarks/*_preregistration.yaml"))
            + list(root.glob("configs/research/*.yaml"))
        )
        if args.as_json:
            print(json.dumps({"configs": sorted(str(c.relative_to(root)) for c in candidates)}))
        elif not candidates:
            print("No preregistration configs found.")
        else:
            print("Preregistration configs (pass path to validate):")
            for c in sorted(candidates, key=str):
                print(f"  {c.relative_to(root)}")
        return 0

    try:
        result = check_yaml_file(args.config)
    except (OSError, InferenceContractError, yaml.YAMLError) as exc:
        if args.as_json:
            print(json.dumps({"status": "not_ready", "error": str(exc)}))
        else:
            print(f"not_ready: {exc}")
        return 1

    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"ok: {result['population_type']} | {result['resampling_method']} | "
            f"{result['estimand_type']} | {result['metric_count']} metric(s) | "
            f"multiplicity: {result['multiplicity_strategy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
