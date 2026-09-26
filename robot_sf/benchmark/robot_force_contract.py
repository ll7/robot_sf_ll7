"""Shared declarations for recorded robot-attributable force metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from robot_sf.benchmark.robot_force_kernel import (
    ROBOT_FORCE_REFERENCE_RULE,
    robot_force_reference,
)

ROBOT_FORCE_RECORDED_SOURCE = "recorded_robot_pedestrian_social_force"
ROBOT_FORCE_POSTHOC_SOURCE = "posthoc_recomputed"
ROBOT_FORCE_SAMPLE_TIMING = "pre_integration"
ROBOT_FORCE_QUANTITY = "model acceleration, not measured human discomfort"
ROBOT_FORCE_REFERENCE_VALUE_RULE = "robot_force_reference(social_force_config, prf_ped_radius_m)"
PP_EQUIV_STATUS = "experimental_counterfactual"
PP_EQUIV_VELOCITY_RULE = "backward_difference_first_forward"

_ROBOT_FORCE_CONFIG_KEYS = (
    "prf_active",
    "prf_robot_radius_m",
    "prf_ped_radius_m",
    "prf_multiplier",
    "prf_activation_m",
)
_SOCIAL_FORCE_CONFIG_KEYS = (
    "lambda_importance",
    "gamma",
    "n_prime",
    "n",
    "factor",
    "activation_threshold",
)


def declared_force_source_contract(source: str) -> dict[str, str]:
    """Return the anchor-declared provenance requirements for one F source."""
    _validate_selected_source(source)
    contract = {
        "source": ROBOT_FORCE_RECORDED_SOURCE,
        "sample_timing": ROBOT_FORCE_SAMPLE_TIMING,
        "reference_rule": ROBOT_FORCE_REFERENCE_RULE,
        "reference_value_rule": ROBOT_FORCE_REFERENCE_VALUE_RULE,
        "quantity": ROBOT_FORCE_QUANTITY,
    }
    if source == "robot_force_pp_equiv_impulse_total":
        contract.update(_pp_equiv_contract())
    return contract


def validate_robot_force_provenance(
    metrics: Mapping[str, Any], selected_source: str
) -> dict[str, Any]:
    """Require recorded pre-integration force provenance and return its compact declaration.

    Returns:
        A JSON-compatible force producer contract suitable for retaining in scored outputs.
    """
    _validate_selected_source(selected_source)
    metadata = metrics.get("robot_force_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("SNQI-v2 force provenance requires robot_force_metadata")
    _validate_metadata_declarations(metadata)
    robot_config = _validated_robot_force_config(metadata)
    social_config = _validated_social_force_config(metadata)
    reference = _validated_reference(metadata, robot_config, social_config)

    result: dict[str, Any] = {
        "selected_source": selected_source,
        "source": ROBOT_FORCE_RECORDED_SOURCE,
        "sample_timing": ROBOT_FORCE_SAMPLE_TIMING,
        "reference_rule": ROBOT_FORCE_REFERENCE_RULE,
        "reference_m_s2": reference,
        "quantity": ROBOT_FORCE_QUANTITY,
        "robot_force_config": robot_config,
        "social_force_config": social_config,
    }
    if selected_source == "robot_force_pp_equiv_impulse_total":
        result.update(_validated_pp_equiv_metadata(metadata))
    return result


def compact_robot_force_metadata(
    metrics: Mapping[str, Any], selected_source: str
) -> dict[str, Any]:
    """Return the small validated metadata subset needed by downstream scorers.

    Returns:
        The compact metadata object retained with report records.
    """
    provenance = validate_robot_force_provenance(metrics, selected_source)
    raw_metadata = metrics["robot_force_metadata"]
    result = {
        "source": provenance["source"],
        "sample_timing": provenance["sample_timing"],
        "reference_rule": provenance["reference_rule"],
        "reference_m_s2": provenance["reference_m_s2"],
        "quantity": provenance["quantity"],
        **provenance["robot_force_config"],
        "social_force_config": provenance["social_force_config"],
    }
    for key in ("pp_equiv_status", "pp_equiv_velocity_rule"):
        if key in raw_metadata:
            result[key] = raw_metadata[key]
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    """Validate one numeric metadata value without accepting booleans.

    Returns:
        The validated value converted to a builtin float.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"SNQI-v2 force provenance requires finite nonnegative {name}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"SNQI-v2 force provenance requires finite nonnegative {name}")
    return result


def _validate_selected_source(source: str) -> None:
    """Refuse unsupported force field names before any values are read."""
    if source not in {"robot_force_impulse_total", "robot_force_pp_equiv_impulse_total"}:
        raise ValueError(f"unsupported SNQI-v2 force source: {source}")


def _validate_metadata_declarations(metadata: Mapping[str, Any]) -> None:
    """Require the recorded source, timing, rule and physical quantity labels."""
    expected = {
        "source": ROBOT_FORCE_RECORDED_SOURCE,
        "sample_timing": ROBOT_FORCE_SAMPLE_TIMING,
        "reference_rule": ROBOT_FORCE_REFERENCE_RULE,
        "quantity": ROBOT_FORCE_QUANTITY,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "SNQI-v2 force provenance has a missing or unsupported producer declaration"
        )


def _validated_robot_force_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate robot radius, activation and force scaling declarations.

    Returns:
        The validated robot configuration fields used by the force producer.
    """
    active = metadata.get("prf_active")
    if not isinstance(active, bool):
        raise ValueError("SNQI-v2 force provenance requires boolean prf_active")
    result: dict[str, Any] = {"prf_active": active}
    for key in _ROBOT_FORCE_CONFIG_KEYS[1:]:
        value = _finite_nonnegative(metadata.get(key), f"robot_force_metadata.{key}")
        if key in {"prf_robot_radius_m", "prf_ped_radius_m"} and value <= 0:
            raise ValueError(f"SNQI-v2 force provenance requires positive {key}")
        result[key] = value
    return result


def _validated_social_force_config(metadata: Mapping[str, Any]) -> dict[str, float]:
    """Validate the full set of SocialForce parameters used by the reference kernel.

    Returns:
        The validated SocialForce reference configuration.
    """
    raw = metadata.get("social_force_config")
    if not isinstance(raw, Mapping):
        raise ValueError("SNQI-v2 force provenance requires social_force_config")
    return {
        key: _finite_nonnegative(raw.get(key), f"social_force_config.{key}")
        for key in _SOCIAL_FORCE_CONFIG_KEYS
    }


def _validated_reference(
    metadata: Mapping[str, Any], robot_config: Mapping[str, Any], social_config: Mapping[str, Any]
) -> float:
    """Recompute and verify the recorded acceleration reference.

    Returns:
        The validated recorded reference in m/s².
    """
    value = _finite_nonnegative(metadata.get("reference_m_s2"), "reference_m_s2")
    try:
        expected = robot_force_reference(social_config, robot_config["prf_ped_radius_m"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("SNQI-v2 force provenance has an invalid reference configuration") from exc
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("SNQI-v2 force provenance reference disagrees with its configuration")
    return value


def _validated_pp_equiv_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Require the experimental status and velocity rule of the PP-equivalent source.

    Returns:
        The required status and finite-difference velocity rule.
    """
    expected = _pp_equiv_contract()
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("SNQI-v2 force provenance requires the PP-equivalent contract")
    return expected


def _pp_equiv_contract() -> dict[str, str]:
    """Return the selected PP-equivalent producer assumptions."""
    return {
        "pp_equiv_status": PP_EQUIV_STATUS,
        "pp_equiv_velocity_rule": PP_EQUIV_VELOCITY_RULE,
    }
