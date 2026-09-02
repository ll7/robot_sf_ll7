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
class RealismSyntheticClassMixRule:
    """Predeclared recovery rule for the diagnostic synthetic class-mix fixture."""

    expected_event_counts: dict[str, int]
    minimum_per_class_recall: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe synthetic class-mix rule mapping."""

        return {
            "expected_event_counts": dict(self.expected_event_counts),
            "minimum_per_class_recall": self.minimum_per_class_recall,
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
    synthetic_class_mix: RealismSyntheticClassMixRule
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
            "synthetic_class_mix": self.synthetic_class_mix.to_dict(),
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

    synthetic_class_mix = _parse_synthetic_class_mix(
        payload.get("synthetic_class_mix"), location=location
    )
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
        synthetic_class_mix=synthetic_class_mix,
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
    """Classify independent interaction-episode counts against preregistered floors.

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


def evaluate_synthetic_class_mix_recall(
    observed_event_counts: Mapping[str, int],
    rule: RealismSyntheticClassMixRule,
) -> dict[str, Any]:
    """Evaluate recovery of the declared synthetic class mix.

    ``observed_event_counts`` must contain counts of planted events recovered for each
    interaction class. Missing declared classes are treated as zero so a missed class
    fails the rule rather than disappearing from the report. The result is explicitly
    diagnostic-only and does not establish real-data or benchmark evidence.

    Returns:
        Diagnostic acceptance status and one recall row per interaction class.
    """

    if not isinstance(observed_event_counts, Mapping):
        raise RealismValidationContractError("observed_event_counts must be a mapping")
    if not isinstance(rule, RealismSyntheticClassMixRule):
        raise RealismValidationContractError(
            "synthetic class-mix evaluation requires a RealismSyntheticClassMixRule"
        )

    expected_event_counts = _class_count_mapping(
        rule.expected_event_counts,
        "synthetic_class_mix.expected_event_counts",
        "",
        allow_zero=False,
    )
    minimum_recall = _recall_threshold(
        rule.minimum_per_class_recall,
        "synthetic_class_mix.minimum_per_class_recall",
        "",
    )
    unknown_classes = [
        key
        for key in observed_event_counts
        if not isinstance(key, str) or key not in INTERACTION_CLASSES
    ]
    if unknown_classes:
        raise RealismValidationContractError(
            f"observed_event_counts contains unknown interaction classes: {unknown_classes!r}"
        )

    rows: dict[str, dict[str, int | float | str]] = {}
    for label in INTERACTION_CLASSES:
        observed = observed_event_counts.get(label, 0)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise RealismValidationContractError(
                f"observed_event_counts[{label!r}] must be a non-negative integer"
            )
        expected = expected_event_counts[label]
        recall = min(observed / expected, 1.0)
        rows[label] = {
            "observed": observed,
            "expected": expected,
            "recall": recall,
            "minimum_recall": minimum_recall,
            "status": "sufficient" if recall >= minimum_recall else "insufficient_recall",
        }
    return {
        "status": "sufficient"
        if all(row["status"] == "sufficient" for row in rows.values())
        else "insufficient_recall",
        "evidence_status": "diagnostic-only",
        "minimum_per_class_recall": minimum_recall,
        "rows": rows,
    }


def _parse_synthetic_class_mix(
    raw_rule: Any,
    *,
    location: str,
) -> RealismSyntheticClassMixRule:
    """Validate the required synthetic class-mix acceptance rule.

    Returns:
        The typed synthetic class-mix rule.
    """

    if not isinstance(raw_rule, Mapping):
        raise RealismValidationContractError(f"synthetic_class_mix must be a mapping{location}")
    expected_event_counts = _class_count_mapping(
        raw_rule.get("expected_event_counts"),
        "synthetic_class_mix.expected_event_counts",
        location,
        allow_zero=False,
    )
    minimum_per_class_recall = _recall_threshold(
        raw_rule.get("minimum_per_class_recall"),
        "synthetic_class_mix.minimum_per_class_recall",
        location,
    )
    return RealismSyntheticClassMixRule(
        expected_event_counts=expected_event_counts,
        minimum_per_class_recall=minimum_per_class_recall,
    )


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


def _recall_threshold(raw: Any, name: str, location: str) -> float:
    """Validate a finite recall threshold in the useful interval ``(0, 1]``.

    Returns:
        The normalized recall threshold.
    """

    value = _non_negative_float(raw, name, location)
    if value <= 0.0 or value > 1.0:
        raise RealismValidationContractError(
            f"{name} must be greater than 0 and at most 1{location}"
        )
    return value


def _class_count_mapping(
    raw: Any,
    name: str,
    location: str,
    *,
    allow_zero: bool,
) -> dict[str, int]:
    """Validate a complete interaction-class count mapping.

    Returns:
        A normalized count mapping ordered by the interaction-class vocabulary.
    """

    if not isinstance(raw, Mapping):
        raise RealismValidationContractError(f"{name} must be a mapping{location}")
    unknown_keys = [
        key for key in raw if not isinstance(key, str) or key not in INTERACTION_CLASSES
    ]
    missing_keys = [label for label in INTERACTION_CLASSES if label not in raw]
    if unknown_keys or missing_keys:
        raise RealismValidationContractError(
            f"{name} must name exactly the interaction classes; "
            f"missing={missing_keys}, extra={unknown_keys}{location}"
        )
    counts: dict[str, int] = {}
    for label in INTERACTION_CLASSES:
        value = raw[label]
        valid = isinstance(value, int) and not isinstance(value, bool)
        if not valid or value < 0 or (not allow_zero and value == 0):
            expected = "a non-negative integer" if allow_zero else "a positive integer"
            raise RealismValidationContractError(f"{name}[{label!r}] must be {expected}{location}")
        counts[label] = int(value)
    return counts


__all__ = [
    "CONSTANT_VELOCITY_BASELINE",
    "CONTRACT_STATUS_BLOCKED_EXTERNAL",
    "CONTRACT_STATUS_READY",
    "REALISM_METRIC_FAMILIES",
    "REALISM_VALIDATION_CONTRACT_CLAIM_BOUNDARY",
    "REALISM_VALIDATION_CONTRACT_SCHEMA_VERSION",
    "RealismPromotionRule",
    "RealismSyntheticClassMixRule",
    "RealismValidationContract",
    "RealismValidationContractError",
    "evaluate_interaction_event_counts",
    "evaluate_synthetic_class_mix_recall",
    "load_realism_validation_contract",
    "realism_validation_contract_from_mapping",
    "validate_realism_validation_contract",
]
