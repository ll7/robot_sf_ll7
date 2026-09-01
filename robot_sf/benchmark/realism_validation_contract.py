"""Typed, fail-closed preregistration contract for pedestrian realism validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.pedestrian_realism_validation import (
    INTERACTION_CLASSES,
    InteractionSegmentationConfig,
)
from robot_sf.sim.pedestrian_model_variants import (
    SOCIAL_FORCE_DEFAULT,
    SUPPORTED_PEDESTRIAN_MODELS,
)

REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION = "realism_validation_contract.v1"
REALISM_VALIDATION_CONTRACT_CLAIM_BOUNDARY = (
    "preregistered interaction-conditioned held-out pedestrian-model validation contract; "
    "no staged-data, calibration, benchmark-ranking, or paper-facing claim"
)
CONSTANT_VELOCITY_BASELINE = "constant_velocity"
CONTRACT_STATUS_BLOCKED_EXTERNAL = "blocked-external-input"
CONTRACT_STATUS_READY = "ready"
REALISM_METRIC_FAMILIES: tuple[str, ...] = (
    "trajectory_rmse",
    "fundamental_diagram_comparison",
    "lane_formation_comparison",
    "speed_distribution_distance",
    "proxemic_distribution_distance",
)
DEFAULT_EXTERNAL_DATA_GATE = "#6530"


class RealismValidationContractError(ValueError):
    """Raised when the realism validation preregistration is invalid."""


@dataclass(frozen=True, slots=True)
class RealismPromotionRule:
    """Predeclared held-out promotion rule for an evaluation contract."""

    comparator_arm: str
    required_metric_families: tuple[str, ...]
    required_interaction_classes: tuple[str, ...]
    held_out_only: bool
    max_free_walking_regression: float
    min_interaction_improvement: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe promotion rule mapping."""

        return {
            "comparator": self.comparator_arm,
            "required_metric_families": list(self.required_metric_families),
            "required_interaction_classes": list(self.required_interaction_classes),
            "held_out_only": self.held_out_only,
            "max_free_walking_regression": self.max_free_walking_regression,
            "min_interaction_improvement": self.min_interaction_improvement,
        }


@dataclass(frozen=True, slots=True)
class RealismValidationContract:
    """Versioned calibration/held-out contract for the realism harness."""

    schema_version: str
    calibration_scenes: tuple[str, ...]
    held_out_scenes: tuple[str, ...]
    baseline_arms: tuple[str, ...]
    metric_families: tuple[str, ...]
    minimum_event_counts: dict[str, int]
    promotion_rule: RealismPromotionRule
    segmentation: InteractionSegmentationConfig
    external_data_gate: str = DEFAULT_EXTERNAL_DATA_GATE
    status: str = "blocked-external-input"
    claim_boundary: str = REALISM_VALIDATION_CONTRACT_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract mapping."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "external_data_gate": self.external_data_gate,
            "claim_boundary": self.claim_boundary,
            "calibration_scenes": list(self.calibration_scenes),
            "held_out_scenes": list(self.held_out_scenes),
            "baseline_arms": list(self.baseline_arms),
            "metric_families": list(self.metric_families),
            "minimum_event_counts": dict(self.minimum_event_counts),
            "promotion_rule": self.promotion_rule.to_dict(),
            "segmentation": self.segmentation.to_dict(),
        }


def load_realism_validation_contract(path: str | Path) -> RealismValidationContract:
    """Load and validate a YAML realism-validation contract.

    Returns:
        The typed validated contract.
    """

    contract_path = Path(path)
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RealismValidationContractError(
            f"realism validation contract cannot be read: {contract_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RealismValidationContractError(
            f"realism validation contract must be a mapping: {contract_path}"
        )
    try:
        return realism_validation_contract_from_mapping(payload, source=contract_path)
    except RealismValidationContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise RealismValidationContractError(
            f"invalid realism validation contract: {contract_path}: {exc}"
        ) from exc


def validate_realism_validation_contract(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> RealismValidationContract:
    """Validate a mapping and return its typed contract.

    Returns:
        The typed validated contract.
    """

    return realism_validation_contract_from_mapping(payload, source=source)


def realism_validation_contract_from_mapping(  # noqa: C901 - strict contract validation is branch-heavy
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> RealismValidationContract:
    """Build a typed contract from a mapping with strict semantic checks.

    Returns:
        The typed validated contract.
    """

    location = f" in {source}" if source is not None else ""
    if not isinstance(payload, Mapping):
        raise RealismValidationContractError(f"contract must be a mapping{location}")
    if payload.get("schema_version") != REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION:
        raise RealismValidationContractError(
            f"schema_version must be {REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION!r}{location}"
        )

    calibration_scenes = _string_tuple(
        payload.get("calibration_scenes"), "calibration_scenes", location
    )
    held_out_scenes = _string_tuple(payload.get("held_out_scenes"), "held_out_scenes", location)
    overlap = sorted(set(calibration_scenes) & set(held_out_scenes))
    if overlap:
        raise RealismValidationContractError(
            f"calibration_scenes and held_out_scenes overlap: {overlap}{location}"
        )

    metric_families = _string_tuple(payload.get("metric_families"), "metric_families", location)
    unsupported_metrics = sorted(set(metric_families) - set(REALISM_METRIC_FAMILIES))
    if unsupported_metrics:
        raise RealismValidationContractError(
            f"metric_families contain unsupported values: {unsupported_metrics}{location}"
        )

    baseline_arms = _string_tuple(payload.get("baseline_arms"), "baseline_arms", location)
    _validate_baseline_arms(baseline_arms, location=location)

    raw_minimums = payload.get("minimum_event_counts")
    if not isinstance(raw_minimums, Mapping):
        raise RealismValidationContractError(f"minimum_event_counts must be a mapping{location}")
    if set(raw_minimums) != set(INTERACTION_CLASSES):
        missing = sorted(set(INTERACTION_CLASSES) - set(raw_minimums))
        extra = sorted(set(raw_minimums) - set(INTERACTION_CLASSES))
        raise RealismValidationContractError(
            f"minimum_event_counts must name exactly the interaction classes; "
            f"missing={missing}, extra={extra}{location}"
        )
    minimum_event_counts: dict[str, int] = {}
    for label in INTERACTION_CLASSES:
        value = raw_minimums[label]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RealismValidationContractError(
                f"minimum_event_counts[{label!r}] must be a non-negative integer{location}"
            )
        minimum_event_counts[label] = int(value)

    promotion_rule = _parse_promotion_rule(
        payload.get("promotion_rule"),
        baseline_arms=baseline_arms,
        metric_families=metric_families,
        location=location,
    )
    raw_segmentation = payload.get("segmentation")
    try:
        segmentation = InteractionSegmentationConfig.from_mapping(raw_segmentation)
    except (TypeError, ValueError) as exc:
        raise RealismValidationContractError(f"invalid segmentation{location}: {exc}") from exc

    status = _required_string(
        payload.get("status", CONTRACT_STATUS_BLOCKED_EXTERNAL), "status", location
    )
    if status not in {CONTRACT_STATUS_BLOCKED_EXTERNAL, CONTRACT_STATUS_READY}:
        raise RealismValidationContractError(
            f"status must be one of {sorted((CONTRACT_STATUS_BLOCKED_EXTERNAL, CONTRACT_STATUS_READY))}"
            f"{location}"
        )
    external_data_gate = _required_string(
        payload.get("external_data_gate", DEFAULT_EXTERNAL_DATA_GATE),
        "external_data_gate",
        location,
    )
    claim_boundary = _required_string(
        payload.get("claim_boundary", REALISM_VALIDATION_CONTRACT_CLAIM_BOUNDARY),
        "claim_boundary",
        location,
    )
    return RealismValidationContract(
        schema_version=REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION,
        calibration_scenes=calibration_scenes,
        held_out_scenes=held_out_scenes,
        baseline_arms=baseline_arms,
        metric_families=metric_families,
        minimum_event_counts=minimum_event_counts,
        promotion_rule=promotion_rule,
        segmentation=segmentation,
        external_data_gate=external_data_gate,
        status=status,
        claim_boundary=claim_boundary,
    )


def evaluate_interaction_event_counts(
    counts: Mapping[str, int],
    minimum_event_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Classify each interaction denominator against its preregistered floor.

    Returns:
        Overall status and one observed/minimum row per interaction class.
    """

    if set(minimum_event_counts) != set(INTERACTION_CLASSES):
        raise RealismValidationContractError(
            "minimum_event_counts must name exactly the interaction classes"
        )
    rows: dict[str, dict[str, int | str]] = {}
    for label in INTERACTION_CLASSES:
        observed = counts.get(label, 0)
        minimum = minimum_event_counts[label]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise RealismValidationContractError(
                f"counts[{label!r}] must be a non-negative integer"
            )
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise RealismValidationContractError(
                f"minimum_event_counts[{label!r}] must be a non-negative integer"
            )
        rows[label] = {
            "observed": observed,
            "minimum": minimum,
            "status": "sufficient" if observed >= minimum else "insufficient_events",
        }
    return {
        "status": "sufficient"
        if all(row["status"] == "sufficient" for row in rows.values())
        else "insufficient_events",
        "rows": rows,
    }


def _parse_promotion_rule(
    raw_rule: Any,
    *,
    baseline_arms: tuple[str, ...],
    metric_families: tuple[str, ...],
    location: str,
) -> RealismPromotionRule:
    """Validate the structured promotion rule.

    Returns:
        The typed promotion rule.
    """

    if not isinstance(raw_rule, Mapping):
        raise RealismValidationContractError(f"promotion_rule must be a mapping{location}")
    comparator = _required_string(raw_rule.get("comparator"), "promotion_rule.comparator", location)
    if comparator not in baseline_arms:
        raise RealismValidationContractError(
            f"promotion_rule.comparator {comparator!r} is not a baseline arm{location}"
        )
    required_metrics = _string_tuple(
        raw_rule.get("required_metric_families"),
        "promotion_rule.required_metric_families",
        location,
    )
    if not set(required_metrics).issubset(metric_families):
        raise RealismValidationContractError(
            f"promotion_rule.required_metric_families must be declared in metric_families{location}"
        )
    required_classes = _string_tuple(
        raw_rule.get("required_interaction_classes"),
        "promotion_rule.required_interaction_classes",
        location,
    )
    if not set(required_classes).issubset(INTERACTION_CLASSES):
        raise RealismValidationContractError(
            f"promotion_rule.required_interaction_classes contains an unknown class{location}"
        )
    held_out_only = raw_rule.get("held_out_only")
    if held_out_only is not True:
        raise RealismValidationContractError(f"promotion_rule.held_out_only must be true{location}")
    max_free_walking_regression = _non_negative_float(
        raw_rule.get("max_free_walking_regression"),
        "promotion_rule.max_free_walking_regression",
        location,
    )
    min_interaction_improvement = _non_negative_float(
        raw_rule.get("min_interaction_improvement"),
        "promotion_rule.min_interaction_improvement",
        location,
    )
    return RealismPromotionRule(
        comparator_arm=comparator,
        required_metric_families=required_metrics,
        required_interaction_classes=required_classes,
        held_out_only=True,
        max_free_walking_regression=max_free_walking_regression,
        min_interaction_improvement=min_interaction_improvement,
    )


def _validate_baseline_arms(baseline_arms: tuple[str, ...], *, location: str) -> None:
    """Ensure arms resolve to the current pedestrian-model registry."""

    orca_arms = sorted(arm for arm in baseline_arms if "orca" in arm.lower())
    if orca_arms:
        raise RealismValidationContractError(
            f"ORCA is a planner, not a pedestrian-model baseline: {orca_arms}{location}"
        )
    supported = set(SUPPORTED_PEDESTRIAN_MODELS) | {CONSTANT_VELOCITY_BASELINE}
    unsupported = sorted(set(baseline_arms) - supported)
    if unsupported:
        raise RealismValidationContractError(
            f"baseline_arms contain unsupported values: {unsupported}{location}"
        )
    if SOCIAL_FORCE_DEFAULT not in baseline_arms:
        raise RealismValidationContractError(
            f"baseline_arms must include {SOCIAL_FORCE_DEFAULT!r}{location}"
        )
    if CONSTANT_VELOCITY_BASELINE not in baseline_arms:
        raise RealismValidationContractError(
            f"baseline_arms must include {CONSTANT_VELOCITY_BASELINE!r}{location}"
        )


def _string_tuple(raw: Any, name: str, location: str) -> tuple[str, ...]:
    """Normalize a required non-empty sequence of unique strings.

    Returns:
        The normalized unique string tuple.
    """

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise RealismValidationContractError(f"{name} must be a non-empty list{location}")
    values = tuple(_required_string(value, f"{name}[]", location) for value in raw)
    if len(values) != len(set(values)):
        raise RealismValidationContractError(f"{name} must not contain duplicates{location}")
    return values


def _required_string(raw: Any, name: str, location: str) -> str:
    """Validate a concrete non-empty string.

    Returns:
        The stripped string.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise RealismValidationContractError(f"{name} must be a non-empty string{location}")
    return raw.strip()


def _non_negative_float(raw: Any, name: str, location: str) -> float:
    """Validate a finite non-negative scalar.

    Returns:
        The normalized scalar.
    """

    if isinstance(raw, bool):
        raise RealismValidationContractError(f"{name} must be finite and non-negative{location}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RealismValidationContractError(
            f"{name} must be finite and non-negative{location}"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise RealismValidationContractError(f"{name} must be finite and non-negative{location}")
    return value


__all__ = [
    "CONSTANT_VELOCITY_BASELINE",
    "CONTRACT_STATUS_BLOCKED_EXTERNAL",
    "CONTRACT_STATUS_READY",
    "REALISM_METRIC_FAMILIES",
    "REALISM_VALIDATION_CONTRACT_CLAIM_BOUNDARY",
    "REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION",
    "RealismPromotionRule",
    "RealismValidationContract",
    "RealismValidationContractError",
    "evaluate_interaction_event_counts",
    "load_realism_validation_contract",
    "realism_validation_contract_from_mapping",
    "validate_realism_validation_contract",
]
