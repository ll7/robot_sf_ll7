"""Shared finite-check policy tests for diagnostic producers."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from robot_sf.benchmark import (
    counterfactual_pair,
    heterogeneous_population_metrics,
    reactivity_ablation,
    safety_predicates,
)
from robot_sf.benchmark.finite_checks import (
    require_finite_array,
    require_finite_fields,
    require_finite_scalar,
)
from robot_sf.planner import stream_gap_gate_calibration
from robot_sf.representation import uncertainty_source_generalization


def test_require_finite_scalar_rejects_nan_and_inf() -> None:
    """Scalar diagnostics fail closed on NaN/Inf."""
    assert require_finite_scalar("metric", 1.25) == pytest.approx(1.25)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="metric"):
            require_finite_scalar("metric", bad)


def test_require_finite_scalar_rejects_numeric_string() -> None:
    """Diagnostic scalar fields must not coerce numeric strings."""
    with pytest.raises(TypeError, match="metric"):
        require_finite_scalar("metric", "1.25")


def test_require_finite_array_rejects_nan_and_inf() -> None:
    """Array diagnostics fail closed if any element is NaN/Inf."""
    assert require_finite_array("trace", [0.0, 1.0]).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="trace"):
        require_finite_array("trace", np.array([0.0, math.nan]))
    with pytest.raises(ValueError, match="trace"):
        require_finite_array("trace", np.array([0.0, math.inf]))


def test_require_finite_fields_names_offending_field() -> None:
    """Field diagnostics identify the object and field that failed closed."""

    class Row:
        safe = 0.0
        unsafe = math.inf

    with pytest.raises(ValueError, match=r"row\.unsafe"):
        require_finite_fields("row", Row(), ("safe", "unsafe"))


def test_require_finite_fields_rejects_numeric_string_field() -> None:
    """Dataclass-style diagnostic fields also reject numeric strings."""

    class Row:
        metric = "1.25"

    with pytest.raises(TypeError, match=r"row\.metric"):
        require_finite_fields("row", Row(), ("metric",))


def test_stream_gap_classifier_rejects_non_finite_safety_aggregate() -> None:
    """The stream-gap pure decision layer raises instead of classifying NaN input."""
    baseline = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.0},
        unsafe_commit_rate=0.0,
        collision_rate=0.0,
        min_separation_m=1.0,
    )
    setting = stream_gap_gate_calibration.GateSettingResult(
        thresholds={"existence": 0.5},
        unsafe_commit_rate=math.nan,
        collision_rate=0.0,
        min_separation_m=1.0,
    )

    with pytest.raises(ValueError, match="setting.unsafe_commit_rate"):
        stream_gap_gate_calibration.classify_setting_safety(setting, baseline)


@pytest.mark.parametrize(
    "module",
    [
        counterfactual_pair,
        heterogeneous_population_metrics,
        reactivity_ablation,
        safety_predicates,
        stream_gap_gate_calibration,
        uncertainty_source_generalization,
    ],
)
def test_diagnostic_producers_import_shared_finite_policy(module: object) -> None:
    """Recent diagnostic producers use the shared fail-closed finite policy."""
    assert "robot_sf.benchmark.finite_checks" in inspect.getsource(module)
