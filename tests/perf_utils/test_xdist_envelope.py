"""xdist-aware performance envelope tests (issue #5836).

Validates that the soft runtime budget widens under pytest-xdist contention
(PYTEST_XDIST_WORKER set) while the hard boundary stays unscaled, so an
unrelated full-suite worker can no longer fail the deterministic reproducibility
test, and a genuine regression still trips the hard threshold.
"""

from __future__ import annotations

from .policy import XDIST_WORKER_ENV, PerformanceBudgetPolicy


def _policy(xdist: bool, monkeypatch) -> PerformanceBudgetPolicy:
    monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)
    if xdist:
        monkeypatch.setenv(XDIST_WORKER_ENV, "gw0")
    return PerformanceBudgetPolicy()


def test_standalone_soft_threshold_is_base(monkeypatch):
    policy = _policy(xdist=False, monkeypatch=monkeypatch)
    assert policy.is_under_xdist() is False
    assert policy.effective_soft_threshold(ci=True) == policy.soft_threshold_seconds
    assert policy.effective_soft_threshold(ci=False) == policy.soft_threshold_seconds / 2.0


def test_xdist_soft_threshold_widened(monkeypatch):
    policy = _policy(xdist=True, monkeypatch=monkeypatch)
    assert policy.is_under_xdist() is True
    for ci in (True, False):
        widened = policy.effective_soft_threshold(ci=ci)
        base = policy.soft_threshold_seconds if ci else policy.soft_threshold_seconds / 2.0
        assert widened > base
        # Widened soft stays strictly below the hard boundary (advisory, not the wall).
        assert widened < policy.hard_timeout_seconds


def test_xdist_env_empty_does_not_count_as_contention(monkeypatch):
    monkeypatch.setenv(XDIST_WORKER_ENV, "   ")
    policy = PerformanceBudgetPolicy()
    assert policy.is_under_xdist() is False
    assert policy.effective_soft_threshold(ci=True) == policy.soft_threshold_seconds
    assert policy.effective_soft_threshold(ci=False) == policy.soft_threshold_seconds / 2.0


def test_hard_threshold_never_scaled_under_xdist(monkeypatch):
    policy = _policy(xdist=True, monkeypatch=monkeypatch)
    # A duration between each widened soft envelope and the (unscaled) hard
    # boundary must classify as soft, not hard -- the hard wall holds.
    for ci in (True, False):
        soft = policy.effective_soft_threshold(ci=ci)
        assert policy.classify(soft + 1.0) == "soft"
        assert policy.classify(policy.hard_timeout_seconds) == "hard"
