"""Regression tests for the narrowed git provenance probe in ``recompute_snqi_weights``."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.snqi import compute


def test_recompute_weights_git_probe_spawn_failure_degrades_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing git binary keeps the documented ``git_sha='unknown'`` fallback (#6690)."""

    def _spawn_failure(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(compute, "run", _spawn_failure)
    weights = compute.recompute_snqi_weights(baseline_stats={}, method="canonical")
    assert weights.git_sha == "unknown"


def test_recompute_weights_git_probe_programmer_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ValueError is outside the narrowed boundary and must surface (#6690).

    Issue #4895: a latent invalid subprocess argument combination raised
    ``ValueError`` that the old broad handler silently swallowed, degrading
    every recorded ``git_sha`` to ``"unknown"``. Programmer errors must no
    longer be converted into silent provenance loss.
    """

    def _programmer_error(*args: object, **kwargs: object) -> str:
        raise ValueError("capture_output and stderr=DEVNULL are invalid together")

    monkeypatch.setattr(compute, "run", _programmer_error)
    with pytest.raises(ValueError, match="invalid"):
        compute.recompute_snqi_weights(baseline_stats={}, method="canonical")
