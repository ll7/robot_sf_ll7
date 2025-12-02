"""Module test_failure_extractor auto-generated docstring."""

from __future__ import annotations

from robot_sf.benchmark.failure_extractor import extract_failures, is_failure


def _rec(ep: str, **metrics):
    """Rec.

    Args:
        ep: Auto-generated placeholder description.
        metrics: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    return {"episode_id": ep, "scenario_id": "s", "seed": 0, "metrics": metrics}


def test_is_failure_cases():
    """Test is failure cases.

    Returns:
        Any: Auto-generated placeholder description.
    """
    assert is_failure(_rec("a", collisions=1))
    assert is_failure(_rec("b", comfort_exposure=0.25))
    assert is_failure(_rec("c", near_misses=2), near_miss_threshold=1)
    assert is_failure(_rec("d", snqi=0.2), snqi_below=0.5)
    assert not is_failure(_rec("e", collisions=0, comfort_exposure=0.0, near_misses=0))


def test_extract_failures_max_count():
    """Test extract failures max count.

    Returns:
        Any: Auto-generated placeholder description.
    """
    recs = [
        _rec("e1", collisions=1),
        _rec("e2", comfort_exposure=0.3),
        _rec("e3", near_misses=2),
        _rec("e4", snqi=0.1),
    ]
    out = extract_failures(recs, snqi_below=0.2, max_count=2)
    assert len(out) == 2
