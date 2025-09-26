"""T024: Performance tests for schema loading (<100ms).

Measures schema loading performance and ensures it stays under 100ms budget.
Tests both cached and uncached loading scenarios.
"""

from __future__ import annotations

import time

from robot_sf.benchmark.schema_loader import load_schema
from robot_sf.benchmark.schema_reference import SchemaReference


def _time_schema_load(iterations: int = 10) -> float:
    """Time schema loading over multiple iterations."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        schema = load_schema("episode.schema.v1.json")
        assert schema is not None  # Ensure schema was loaded
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # Convert to milliseconds

    return sum(times) / len(times)  # Return mean time


def test_schema_loading_performance_under_100ms():
    """Test that schema loading completes in under 100ms."""
    mean_time = _time_schema_load(iterations=10)

    # Hard budget: 100ms per the task requirement
    assert mean_time < 100.0, (
        f"Schema loading too slow: {mean_time:.2f}ms > 100ms budget. "
        "Check for performance regression in schema loading or caching."
    )


def test_schema_loading_performance_reasonable():
    """Test that schema loading is reasonably fast (<50ms typical)."""
    mean_time = _time_schema_load(iterations=10)

    # Soft expectation: should be much faster than budget
    assert mean_time < 50.0, (
        f"Schema loading slower than expected: {mean_time:.2f}ms. "
        "Expected <50ms for typical performance."
    )


def test_schema_caching_effectiveness():
    """Test that schema caching provides performance benefit."""
    # Clear cache to ensure clean state for measurement
    SchemaReference.clear_cache()

    # First load (will populate cache)
    first_time = _time_schema_load(iterations=1)

    # Subsequent loads (should be cached)
    cached_time = _time_schema_load(iterations=5)

    # Cached loads should be significantly faster
    # Allow some variance but expect at least 2x speedup
    epsilon = 1e-6
    speedup_ratio = first_time / max(cached_time, epsilon)

    assert speedup_ratio > 2.0, (
        f"Schema caching not effective: {speedup_ratio:.1f}x speedup "
        f"(first: {first_time:.2f}ms, cached: {cached_time:.2f}ms). "
        "Caching should provide significant performance benefit."
    )
