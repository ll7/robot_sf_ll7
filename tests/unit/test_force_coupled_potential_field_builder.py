"""Core-lane coverage for the experimental potential-field policy builder."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec


@pytest.mark.parametrize("opt_in", [True, "yes"])
def test_force_coupled_builder_accepts_explicit_boolean_or_text_opt_in(
    opt_in: bool | str,
) -> None:
    """The canonical builder accepts only an explicit truthy testing opt-in."""
    spec = build_registered_adapter_policy_spec(
        "force_coupled_potential_field",
        {"allow_testing_algorithms": opt_in, "max_linear_speed": 0.8},
    )

    assert spec is not None
    assert spec.algo_key == "force_coupled_potential_field"
    assert spec.adapter_name == "ForceCoupledPotentialFieldPlanner"
    assert spec.limitations == "clean_room_experimental_smoke_only_not_benchmark_evidence"


@pytest.mark.parametrize("opt_in", [False, "no"])
def test_force_coupled_builder_rejects_missing_or_false_opt_in(opt_in: bool | str) -> None:
    """A false boolean or text value cannot enter the testing-only builder."""
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        build_registered_adapter_policy_spec(
            "force_coupled_potential_field",
            {"allow_testing_algorithms": opt_in},
        )
