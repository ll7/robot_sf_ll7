"""Contract tests for benchmark algorithm readiness catalog entries."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    paper_baseline_algorithms,
    require_algorithm_allowed,
)


def test_unknown_algorithm_is_not_blocked_by_catalog() -> None:
    """Out-of-catalog algorithms remain the caller's responsibility."""
    assert get_algorithm_readiness("not_in_catalog") is None
    assert (
        require_algorithm_allowed(
            algo="not_in_catalog",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )
        is None
    )


def test_placeholder_algorithm_is_never_allowed() -> None:
    """Placeholder adapters should fail closed before benchmark execution."""
    with pytest.raises(ValueError, match="marked placeholder"):
        require_algorithm_allowed(
            algo="rvo",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )


def test_dwa_requires_explicit_experimental_opt_in() -> None:
    """The implemented DWA baseline stays gated until benchmark promotion evidence exists."""
    spec = get_algorithm_readiness("dwa")
    assert spec is not None
    assert spec.canonical_name == "dwa"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="dwa",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    assert (
        require_algorithm_allowed(
            algo="dwa",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == spec
    )


def test_paper_baseline_profile_allows_only_publication_baselines() -> None:
    """Paper-baseline profile should reject unproven experimental algorithms."""
    assert paper_baseline_algorithms() == ("goal", "social_force", "orca", "ppo")

    with pytest.raises(ValueError, match="paper-grade gate failed"):
        require_algorithm_allowed(
            algo="ppo",
            benchmark_profile="paper-baseline",
            ppo_paper_ready=False,
        )

    spec = require_algorithm_allowed(
        algo="ppo",
        benchmark_profile="paper-baseline",
        ppo_paper_ready=True,
    )
    assert spec is not None
    assert spec.canonical_name == "ppo"

    with pytest.raises(ValueError, match="blocked by profile 'paper-baseline'"):
        require_algorithm_allowed(
            algo="risk_dwa",
            benchmark_profile="paper-baseline",
            ppo_paper_ready=True,
            allow_testing_algorithms=True,
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


def test_actuation_aware_hybrid_rule_alias_stays_experimental() -> None:
    """The AMV actuation-aware local candidate should require exploratory opt-in."""
    spec = get_algorithm_readiness("actuation_aware_hybrid_rule_v0")
    assert spec is not None
    assert spec.canonical_name == "hybrid_rule_local_planner"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="actuation_aware_hybrid_rule_v0",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    assert (
        require_algorithm_allowed(
            algo="actuation_aware_hybrid_rule_v0",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )
        == spec
    )
