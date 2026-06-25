"""TODO docstring. Document this module."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.failure_extractor import _metric, extract_failures, is_failure


def _rec(ep: str, **metrics):
    """TODO docstring. Document this function.

    Args:
        ep: TODO docstring.
        metrics: TODO docstring.
    """
    return {"episode_id": ep, "scenario_id": "s", "seed": 0, "metrics": metrics}


def test_is_failure_cases():
    """TODO docstring. Document this function."""
    assert is_failure(_rec("a", collisions=1))
    assert is_failure(_rec("b", comfort_exposure=0.25))
    assert is_failure(_rec("c", near_misses=2), near_miss_threshold=1)
    assert is_failure(_rec("d", snqi=0.2), snqi_below=0.5)
    assert not is_failure(_rec("e", collisions=0, comfort_exposure=0.0, near_misses=0))


def test_extract_failures_max_count():
    """TODO docstring. Document this function."""
    recs = [
        _rec("e1", collisions=1),
        _rec("e2", comfort_exposure=0.3),
        _rec("e3", near_misses=2),
        _rec("e4", snqi=0.1),
    ]
    out = extract_failures(recs, snqi_below=0.2, max_count=2)
    assert len(out) == 2


@pytest.mark.parametrize("bad_value", [None, "not-a-number", [1, 2], {"x": 1}, "", "nan!"])
def test_metric_coercion_falls_back_on_non_numeric(bad_value: object) -> None:
    """Non-numeric JSON metric values must fall back to the default, not raise.

    The handler was narrowed from a broad ``except Exception`` to
    ``(TypeError, ValueError)`` (issue #3478); this confirms every realistic
    bad-input shape from JSON still resolves to the numeric default so the
    episode is not silently misclassified.
    """
    rec = {"metrics": {"collisions": bad_value}}
    assert _metric(rec, "collisions", 7.0) == 7.0


def test_metric_coercion_preserves_valid_numbers() -> None:
    """Valid numeric and numeric-string metrics must still coerce unchanged."""
    assert _metric({"metrics": {"collisions": 3}}, "collisions") == 3.0
    assert _metric({"metrics": {"collisions": "2.5"}}, "collisions") == 2.5
    assert _metric({"metrics": {}}, "collisions", 1.5) == 1.5


def test_is_failure_ignores_non_numeric_snqi() -> None:
    """A non-numeric snqi must be treated as "criterion not met", not an error."""
    assert not is_failure({"metrics": {"snqi": "unavailable"}}, snqi_below=0.5)
    # A valid snqi below the threshold still triggers failure.
    assert is_failure({"metrics": {"snqi": 0.1}}, snqi_below=0.5)
