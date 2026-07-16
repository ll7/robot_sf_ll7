"""xdist-aware performance envelope tests (issue #5836).

Validates that the soft runtime budget widens under pytest-xdist contention
(PYTEST_XDIST_WORKER set) while the hard boundary stays unscaled, so an
unrelated full-suite worker can no longer fail the deterministic reproducibility
test, and a genuine regression still trips the hard threshold.
"""

from __future__ import annotations

import os

from .policy import XDIST_WORKER_ENV, PerformanceBudgetPolicy


def _policy(xdist: bool) -> PerformanceBudgetPolicy:
    os.environ.pop(XDIST_WORKER_ENV, None)
    if xdist:
        os.environ[XDIST_WORKER_ENV] = "gw0"
    else:
        os.environ.pop(XDIST_WORKER_ENV, None)
    return PerformanceBudgetPolicy()


def test_standalone_soft_threshold_is_base():
    policy = _policy(xdist=False)
    assert policy.is_under_xdist() is False
    assert policy.effective_soft_threshold() == policy.soft_threshold_seconds


def test_xdist_soft_threshold_widened():
    policy = _policy(xdist=True)
    assert policy.is_under_xdist() is True
    widened = policy.effective_soft_threshold()
    assert widened > policy.soft_threshold_seconds
    # Widened soft stays strictly below the hard boundary (advisory, not the wall).
    assert widened < policy.hard_timeout_seconds


def test_xdist_env_empty_does_not_count_as_contention():
    os.environ[XDIST_WORKER_ENV] = "   "
    policy = PerformanceBudgetPolicy()
    try:
        assert policy.is_under_xdist() is False
        assert policy.effective_soft_threshold() == policy.soft_threshold_seconds
    finally:
        os.environ.pop(XDIST_WORKER_ENV, None)


def test_hard_threshold_never_scaled_under_xdist():
    policy = _policy(xdist=True)
    try:
        # A duration between the widened soft envelope and the (unscaled) hard
        # boundary must classify as soft, not hard -- the hard wall holds.
        soft = policy.effective_soft_threshold()
        assert policy.classify(soft + 1.0) == "soft"
        assert policy.classify(policy.hard_timeout_seconds) == "hard"
    finally:
        os.environ.pop(XDIST_WORKER_ENV, None)
