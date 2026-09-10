"""Characterize the selected bool-accepting finite-float helper family."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

from robot_sf.adversarial.archive import _finite_float as archive_finite_float
from robot_sf.adversarial.warm_start import _finite_float as warm_start_finite_float
from robot_sf.benchmark.collision.collision_scenario_similarity import (
    _finite_float as collision_finite_float,
)
from robot_sf.benchmark.event_ledger import _finite_float as event_ledger_finite_float
from robot_sf.benchmark.map_runner.map_runner_metrics import (
    _finite_float as map_runner_finite_float,
)
from robot_sf.benchmark.scenario.scenario_coverage import _finite_float as scenario_finite_float
from robot_sf.benchmark.seed_distribution_report import (
    _finite_float as seed_distribution_finite_float,
)
from robot_sf.planner.hybrid_orca_sampler import HybridORCASamplerAdapter

FiniteFloat = Callable[[Any], float | None]

_SELECTED_HELPERS: tuple[tuple[str, FiniteFloat], ...] = (
    ("adversarial.archive", archive_finite_float),
    ("adversarial.warm_start", warm_start_finite_float),
    ("benchmark.collision_scenario_similarity", collision_finite_float),
    ("benchmark.event_ledger", event_ledger_finite_float),
    ("benchmark.map_runner_metrics", map_runner_finite_float),
    ("benchmark.scenario_coverage", scenario_finite_float),
    ("benchmark.seed_distribution_report", seed_distribution_finite_float),
    ("planner.hybrid_orca_sampler", HybridORCASamplerAdapter._finite_float),
)


class _OverflowingFloat:
    """Malformed scalar whose conversion raises an intentionally uncaught error."""

    def __float__(self) -> float:
        raise OverflowError("finite-float overflow sentinel")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(True, 1.0, id="true-is-accepted"),
        pytest.param(False, 0.0, id="false-is-accepted"),
        pytest.param(-3, -3.0, id="integer"),
        pytest.param(2.5, 2.5, id="finite-float"),
        pytest.param(" 3.5 ", 3.5, id="numeric-string"),
        pytest.param(float("nan"), None, id="nan"),
        pytest.param(float("inf"), None, id="positive-infinity"),
        pytest.param(float("-inf"), None, id="negative-infinity"),
        pytest.param("nan", None, id="nan-string"),
        pytest.param("inf", None, id="infinity-string"),
        pytest.param("malformed", None, id="malformed-string"),
        pytest.param(object(), None, id="malformed-object"),
        pytest.param([1.0], None, id="malformed-sequence"),
        pytest.param({"value": 1.0}, None, id="malformed-mapping"),
        pytest.param(1 + 2j, None, id="complex-number"),
    ],
)
@pytest.mark.parametrize(("helper_name", "helper"), _SELECTED_HELPERS)
def test_selected_helpers_have_identical_coercion_contract(
    helper_name: str, helper: FiniteFloat, value: Any, expected: float | None
) -> None:
    """Every selected private helper has the same value and missingness behavior."""
    result = helper(value)

    if expected is None:
        assert result is None, helper_name
    else:
        assert type(result) is float, helper_name
        assert result == pytest.approx(expected), helper_name


@pytest.mark.parametrize(("helper_name", "helper"), _SELECTED_HELPERS)
def test_selected_helpers_preserve_uncaught_conversion_exception_and_message(
    helper_name: str, helper: FiniteFloat
) -> None:
    """The family catches only TypeError/ValueError, preserving overflow failures verbatim."""
    with pytest.raises(OverflowError) as caught:
        helper(_OverflowingFloat())

    assert str(caught.value) == "finite-float overflow sentinel", helper_name


@pytest.mark.parametrize(("helper_name", "helper"), _SELECTED_HELPERS)
def test_selected_helpers_preserve_negative_zero(helper_name: str, helper: FiniteFloat) -> None:
    """Every selected helper keeps the sign bit of finite negative zero."""
    result = helper(-0.0)

    assert result == 0.0, helper_name
    assert math.copysign(1.0, result) == -1.0, helper_name
