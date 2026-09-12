"""Characterization tests for the shared stdlib-only finite-float primitive."""

from __future__ import annotations

import math

import pytest

from robot_sf.common.validation import finite_float


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
def test_finite_float_value_and_missingness_contract(value: object, expected: float | None) -> None:
    """The shared primitive preserves bool coercion and fail-closed values."""
    result = finite_float(value)

    if expected is None:
        assert result is None
    else:
        assert type(result) is float
        assert result == pytest.approx(expected)


def test_finite_float_preserves_uncaught_conversion_exception_and_message() -> None:
    """Only ordinary TypeError/ValueError conversion failures are absorbed."""
    with pytest.raises(OverflowError) as caught:
        finite_float(_OverflowingFloat())

    assert str(caught.value) == "finite-float overflow sentinel"


def test_finite_float_preserves_negative_zero() -> None:
    """The coercion keeps the sign bit of a finite negative zero."""
    result = finite_float(-0.0)

    assert result == 0.0
    assert math.copysign(1.0, result) == -1.0
