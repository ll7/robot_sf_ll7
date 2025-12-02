"""Polish Phase T050: Unit tests for Wilson interval correctness.

We verify a few known (p_hat, n) cases against reference values computed externally.
Tolerance is loose enough to accommodate approximation in _z_from_conf.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.aggregation import _wilson_interval  # type: ignore


def _approx(a: float, b: float, rel: float = 0.05):  # 5% relative tolerance
    """Approx.

    Args:
        a: Auto-generated placeholder description.
        b: Auto-generated placeholder description.
        rel: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    if a == b:
        return True
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= rel


def test_wilson_extremes_zero_events():
    """Test wilson extremes zero events.

    Returns:
        Any: Auto-generated placeholder description.
    """
    low, high = _wilson_interval(0.0, 50, 0.95)
    # Expected high ~ 0.072 (reference from statistical tables)
    assert 0.05 < high < 0.09, f"Unexpected upper bound for zero events: {high}"
    # Allow negligible positive rounding noise
    assert low <= 1e-12, f"Expected near-zero lower bound, got {low}"


def test_wilson_midpoint_symmetry():
    """Test wilson midpoint symmetry.

    Returns:
        Any: Auto-generated placeholder description.
    """
    low, high = _wilson_interval(0.5, 100, 0.95)
    # For p=0.5, n=100, half-width ~0.096 (so interval ~0.404-0.596)
    assert _approx((high - low) / 2, 0.096, rel=0.15)
    assert 0.39 < low < 0.42
    assert 0.58 < high < 0.61
