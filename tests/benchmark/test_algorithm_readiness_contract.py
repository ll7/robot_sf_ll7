"""Contract tests for benchmark algorithm readiness catalog entries."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)


def test_trivial_reference_adapter_requires_explicit_experimental_opt_in() -> None:
    """The starter-template adapter should be runnable only with explicit test opt-in."""
    spec = get_algorithm_readiness("reference_adapter")
    assert spec is not None
    assert spec.canonical_name == "trivial_reference"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True

    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        require_algorithm_allowed(
            algo="reference_adapter",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
        )

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="reference_adapter",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    assert (
        require_algorithm_allowed(
            algo="reference_adapter",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == spec
    )


def test_risk_surface_dwa_requires_explicit_experimental_opt_in() -> None:
    """The deterministic risk-surface planner should stay behind the testing gate."""
    spec = get_algorithm_readiness("risk_surface_dwa_v0")
    assert spec is not None
    assert spec.canonical_name == "risk_surface_dwa"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="risk_surface_dwa",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    assert (
        require_algorithm_allowed(
            algo="risk_surface_dwa",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == spec
    )
