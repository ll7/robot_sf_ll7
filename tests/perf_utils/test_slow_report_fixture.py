"""Synthetic slow test report validation (T015).

Validates that the reporting utilities classify and include a deliberately
"slow" sample in the top report output, and that guidance suggestions contain
expected optimization keywords. Uses a short sleep to keep runtime negligible.
Respects ROBOT_SF_PERF_RELAX (no behavioral difference here, but demonstrates
policy integration path).
"""

from __future__ import annotations

import os
import time

from .policy import PerformanceBudgetPolicy
from .reporting import SlowTestSample, generate_report


def test_slow_report_includes_sample():  # basic functional check
    """TODO docstring. Document this function."""
    policy = PerformanceBudgetPolicy(
        soft_threshold_seconds=0.05,
        hard_timeout_seconds=1.0,
        report_count=5,
    )
    # Sleep slightly above soft threshold to trigger soft breach classification
    start = time.perf_counter()
    time.sleep(0.08)
    duration = time.perf_counter() - start

    samples = [SlowTestSample(test_identifier="dummy::test_example", duration_seconds=duration)]
    report = generate_report(samples, policy)
    assert report, "Report unexpectedly empty"
    rec = report[0]
    assert rec.test_identifier == "dummy::test_example"
    assert rec.breach_type == "soft"
    # Guidance should contain at least one canonical optimization phrase
    expected_keywords = ["Reduce", "horizon", "matrix"]
    joined = " ".join(rec.guidance).lower()
    for kw in expected_keywords:
        assert kw.lower() in joined, f"Missing guidance keyword: {kw}"

    # If relax mode is active we still generate guidance; no change in logic yet
    if os.environ.get(policy.relax_env_var) == "1":  # pragma: no cover - env dependent
        assert rec.breach_type == "soft"
