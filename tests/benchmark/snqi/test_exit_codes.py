"""Compatibility tests that lock the public SNQI exit-code taxonomy.

``robot_sf/benchmark/snqi/exit_codes.py`` exposes numeric exit codes that are a stable
automation boundary consumed by CI pipelines, notebooks, and SNQI wrapper scripts (for
example ``scripts/recompute_snqi_weights.py``, ``scripts/snqi_weight_optimization.py``,
and the ``robot_sf.benchmark.snqi.cli`` entry point). These tests prevent accidental
renumbering, export drift, or taxonomy reinterpretation without an explicit contract
change.

Expected taxonomy (the locked contract documented in the module docstring):

    0  EXIT_SUCCESS               - Execution completed without detected errors.
    1  EXIT_INPUT_ERROR           - File I/O / JSON parse / structural pre-validation failure.
    2  EXIT_VALIDATION_ERROR      - Schema or finiteness validation failure after assembly.
    3  EXIT_RUNTIME_ERROR         - Unexpected runtime exception during processing/optimization.
    4  EXIT_MISSING_METRIC_ERROR  - (Reserved) forthcoming --fail-on-missing-metric flag.
    5  EXIT_OPTIONAL_DEPS_MISSING - Optional dependency missing when explicitly required.

If any assertion below fails, do NOT renumber, add, remove, or reinterpret a code merely
to satisfy the test. Treat the failure as a taxonomy contract change that must be reviewed
and documented before the public boundary is altered.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.snqi import exit_codes as snqi_exit_codes
from robot_sf.benchmark.snqi.exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_MISSING_METRIC_ERROR,
    EXIT_OPTIONAL_DEPS_MISSING,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)

# Locked public taxonomy: public name -> documented integer. Editing this mapping is a
# taxonomy contract change, not a test fix -- update the module docstring in lockstep.
EXPECTED_EXIT_CODE_VALUES: dict[str, int] = {
    "EXIT_SUCCESS": 0,
    "EXIT_INPUT_ERROR": 1,
    "EXIT_VALIDATION_ERROR": 2,
    "EXIT_RUNTIME_ERROR": 3,
    "EXIT_MISSING_METRIC_ERROR": 4,
    "EXIT_OPTIONAL_DEPS_MISSING": 5,
}

EXPECTED_PUBLIC_EXIT_CODE_NAMES = frozenset(EXPECTED_EXIT_CODE_VALUES)

_SORTED_EXPECTED = sorted(EXPECTED_EXIT_CODE_VALUES.items())


@pytest.mark.parametrize(
    ("name", "expected"),
    _SORTED_EXPECTED,
    ids=[name for name, _ in _SORTED_EXPECTED],
)
def test_exit_code_resolves_to_documented_integer(name: str, expected: int) -> None:
    """Each documented public name resolves to its documented integer value."""
    assert getattr(snqi_exit_codes, name) == expected


def test_public_constants_are_importable_with_documented_values() -> None:
    """The six public constants import directly and equal their documented integers.

    A removed or renamed public constant breaks this import at collection time, which is
    the loud failure downstream wrapper scripts would also hit.
    """
    assert EXIT_SUCCESS == 0
    assert EXIT_INPUT_ERROR == 1
    assert EXIT_VALIDATION_ERROR == 2
    assert EXIT_RUNTIME_ERROR == 3
    assert EXIT_MISSING_METRIC_ERROR == 4
    assert EXIT_OPTIONAL_DEPS_MISSING == 5


def test_exit_code_values_are_exactly_zero_through_five() -> None:
    """The documented SUCCESS..OPTIONAL_DEPS_MISSING values are exactly the set {0..5}."""
    actual = {getattr(snqi_exit_codes, name) for name in EXPECTED_EXIT_CODE_VALUES}
    assert actual == {0, 1, 2, 3, 4, 5}


def test_exit_code_values_are_unique() -> None:
    """No two documented exit codes may share an integer value."""
    values = [getattr(snqi_exit_codes, name) for name in EXPECTED_EXIT_CODE_VALUES]
    assert len(set(values)) == len(EXPECTED_EXIT_CODE_VALUES)


def test_exit_code_values_are_contiguous_from_zero() -> None:
    """Sorted documented values form a contiguous 0..N-1 sequence (no gaps, no offset)."""
    values = sorted(getattr(snqi_exit_codes, name) for name in EXPECTED_EXIT_CODE_VALUES)
    assert values == list(range(len(EXPECTED_EXIT_CODE_VALUES)))


def test_all_exposes_exactly_the_six_public_exit_code_names() -> None:
    """``__all__`` exposes exactly the six documented public exit-code names."""
    assert set(snqi_exit_codes.__all__) == EXPECTED_PUBLIC_EXIT_CODE_NAMES


def test_all_has_no_duplicate_entries() -> None:
    """``__all__`` lists each public name exactly once and matches the documented count."""
    assert len(snqi_exit_codes.__all__) == len(set(snqi_exit_codes.__all__))
    assert len(snqi_exit_codes.__all__) == len(EXPECTED_PUBLIC_EXIT_CODE_NAMES)


def test_all_entries_resolve_to_documented_integers() -> None:
    """Every name exported via ``__all__`` resolves to its documented integer value."""
    for name in snqi_exit_codes.__all__:
        assert name in EXPECTED_EXIT_CODE_VALUES, f"unexpected public name in __all__: {name!r}"
        assert getattr(snqi_exit_codes, name) == EXPECTED_EXIT_CODE_VALUES[name]
