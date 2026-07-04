"""Recompute Chapter 8 dissertation statistics from explicit source data.

The helpers in this module are intentionally small and data-driven. They do not
load dissertation prose or infer claims; callers must provide the numeric source
values and expected targets in a manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticResult:
    """Computed statistic plus fail-closed status metadata."""

    statistic_id: str
    statistic_kind: str
    status: str
    computed: dict[str, Any]
    expected: dict[str, Any]
    blockers: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return stable JSON-ready representation for packet artifacts."""

        return {
            "id": self.statistic_id,
            "kind": self.statistic_kind,
            "status": self.status,
            "computed": self.computed,
            "expected": self.expected,
            "blockers": list(self.blockers),
        }


def partial_eta_squared(groups: dict[str, list[float]]) -> float:
    """Compute one-way partial eta squared from named treatment groups.

    Returns:
        Partial eta squared value for the supplied group observations.
    """

    clean_groups = _validate_groups(groups)
    values = np.concatenate([np.asarray(group, dtype=float) for group in clean_groups.values()])
    overall_mean = float(np.mean(values))
    ss_effect = 0.0
    ss_error = 0.0
    for group in clean_groups.values():
        group_array = np.asarray(group, dtype=float)
        group_mean = float(np.mean(group_array))
        ss_effect += len(group_array) * (group_mean - overall_mean) ** 2
        ss_error += float(np.sum((group_array - group_mean) ** 2))
    denominator = ss_effect + ss_error
    if denominator <= 0.0:
        raise ValueError("partial_eta_squared denominator is zero")
    return float(ss_effect / denominator)


def spearman_rho(x_values: list[float], y_values: list[float]) -> float:
    """Compute Spearman rank correlation for paired finite samples.

    Returns:
        Spearman rho statistic.
    """

    x_array = _validate_sequence(x_values, name="x_values", min_length=2)
    if not isinstance(y_values, list) or len(x_array) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    y_array = _validate_sequence(y_values, name="y_values", min_length=2)
    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length")
    rho = stats.spearmanr(x_array, y_array).statistic
    if not math.isfinite(float(rho)):
        raise ValueError("spearman_rho is not finite")
    return float(rho)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Compute a deterministic bootstrap confidence interval for the mean.

    Returns:
        Summary with original mean, bootstrap mean, confidence interval, seed, and sample count.
    """

    value_array = _validate_sequence(values, name="values", min_length=2)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(value_array), size=(samples, len(value_array)))
    means = value_array[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    ci = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "mean": float(np.mean(value_array)),
        "bootstrap_mean": float(np.mean(means)),
        "ci": [float(ci[0]), float(ci[1])],
    }


def evaluate_statistic(spec: dict[str, Any]) -> StatisticResult:
    """Evaluate one manifest statistic and return fail-closed status.

    Returns:
        Structured result row for the reproducibility packet.
    """

    statistic_id = _required_str(spec, "id")
    statistic_kind = _required_str(spec, "statistic_kind")
    data = spec.get("data")
    expected = _expected_block(spec)
    blockers: list[str] = []

    if not isinstance(data, dict) or not data:
        return StatisticResult(
            statistic_id=statistic_id,
            statistic_kind=statistic_kind,
            status="blocked_missing_source_data",
            computed={},
            expected=expected,
            blockers=("data block is missing or empty",),
        )

    try:
        computed = _compute_by_kind(statistic_kind, data)
    except (KeyError, TypeError, ValueError) as exc:
        return StatisticResult(
            statistic_id=statistic_id,
            statistic_kind=statistic_kind,
            status="blocked_invalid_source_data",
            computed={},
            expected=expected,
            blockers=(str(exc),),
        )

    status = "recomputed"
    if expected:
        blockers.extend(_expected_blockers(computed, expected))
        status = "matches_expected" if not blockers else "computed_mismatch"
    else:
        status = "computed_expected_value_missing"
        blockers.append("expected value block is missing")

    return StatisticResult(
        statistic_id=statistic_id,
        statistic_kind=statistic_kind,
        status=status,
        computed=computed,
        expected=expected,
        blockers=tuple(blockers),
    )


def _compute_by_kind(statistic_kind: str, data: dict[str, Any]) -> dict[str, Any]:
    if statistic_kind == "partial_eta_squared":
        return {"value": partial_eta_squared(data["groups"])}
    if statistic_kind == "spearman_rho":
        return {"value": spearman_rho(data["x_values"], data["y_values"])}
    if statistic_kind == "bootstrap_mean_ci":
        return bootstrap_mean_ci(
            data["values"],
            samples=int(data["samples"]),
            confidence_level=float(data["confidence_level"]),
            seed=int(data["seed"]),
        )
    raise ValueError(f"unsupported statistic_kind: {statistic_kind}")


def _expected_block(spec: dict[str, Any]) -> dict[str, Any]:
    expected = spec.get("expected")
    return expected if isinstance(expected, dict) else {}


def _expected_blockers(computed: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    tolerance = float(expected.get("tolerance", 1e-9))
    if "value" in expected:
        expected_value = float(expected["value"])
        computed_value = float(computed.get("value", computed.get("mean", math.nan)))
        if not math.isclose(computed_value, expected_value, rel_tol=tolerance, abs_tol=tolerance):
            blockers.append(f"value {computed_value} does not match expected {expected_value}")
    if "ci" in expected:
        computed_ci = computed.get("ci")
        expected_ci = expected["ci"]
        if not isinstance(computed_ci, list) or len(computed_ci) != 2:
            blockers.append("computed ci is missing")
        elif len(expected_ci) != 2:
            blockers.append("expected ci must contain two values")
        else:
            for index, (actual, target) in enumerate(zip(computed_ci, expected_ci, strict=True)):
                if not math.isclose(
                    float(actual), float(target), rel_tol=tolerance, abs_tol=tolerance
                ):
                    blockers.append(f"ci[{index}] {actual} does not match expected {target}")
    if "samples" in expected and computed.get("samples") != expected["samples"]:
        blockers.append(
            f"samples {computed.get('samples')} does not match expected {expected['samples']}"
        )
    return blockers


def _required_str(spec: dict[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"statistic spec requires non-empty string field: {key}")
    return value


def _validate_groups(groups: dict[str, list[float]]) -> dict[str, list[float]]:
    if not isinstance(groups, dict) or len(groups) < 2:
        raise ValueError("groups must contain at least two named groups")
    clean: dict[str, list[float]] = {}
    for name, values in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("group names must be non-empty strings")
        clean[name] = _validate_sequence(values, name=f"groups.{name}", min_length=1).tolist()
    if sum(len(values) for values in clean.values()) <= len(clean):
        raise ValueError("groups need at least one residual degree of freedom")
    return clean


def _validate_sequence(values: list[float], *, name: str, min_length: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) < min_length:
        raise ValueError(f"{name} must contain at least {min_length} values")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
