"""Map-runner resolution and readiness-guard tests for the SIPP-lattice planner.

Covers Slice 2 of #5306: the bounded kinodynamic state-time search adapter must
be resolvable through the real map-runner policy path, gated behind the
testing-algorithm readiness guard, and registered as testing-only in metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.algorithm_metadata import (
    _KINEMATICS_PROFILE_BY_CANONICAL,
    canonical_algorithm_name,
)
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec


def _observation() -> dict:
    """Return a minimal SocNav observation for policy smoke evaluation."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=float),
            "heading": np.array([0.0], dtype=float),
            "speed": np.array([0.0], dtype=float),
        },
        "goal": {
            "current": np.array([1.0, 0.0], dtype=float),
            "next": np.array([1.0, 0.0], dtype=float),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=float),
            "velocities": np.zeros((0, 2), dtype=float),
            "count": np.array([0.0], dtype=float),
            "radius": 0.30,
        },
    }


def test_policy_builder_registers_sipp_lattice() -> None:
    """The registered adapter spec resolves the Slice-2 search adapter."""
    spec = build_registered_adapter_policy_spec(
        "sipp_lattice", {"max_linear_speed": 1.0, "allow_testing_algorithms": True}
    )
    assert spec is not None
    assert spec.algo_key == "sipp_lattice"
    assert spec.adapter_name == "SippLatticeSearchPlannerAdapter"
    assert spec.limitations is not None
    assert "testing_only" in spec.limitations


def test_map_runner_resolves_sipp_lattice_policy() -> None:
    """The real map-runner policy path builds a callable SIPP-lattice policy."""
    policy, meta = _build_policy(
        "sipp_lattice",
        {"max_linear_acceleration": 5.0, "allow_testing_algorithms": True},
    )
    assert meta["algorithm"] == "sipp_lattice"
    assert meta["status"] == "ok"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"
    assert meta["planner_kinematics"]["adapter_name"] == "SippLatticeSearchPlannerAdapter"

    command = policy(_observation())
    linear, angular = command
    assert np.isfinite(linear)
    assert np.isfinite(angular)
    assert abs(linear) <= 1.0 + 1e-6


def test_sipp_lattice_is_testing_only_gated() -> None:
    """The planner stays behind the explicit testing-algorithm opt-in guard."""
    spec = get_algorithm_readiness("sipp_lattice")
    assert spec is not None
    assert spec.canonical_name == "sipp_lattice"
    assert spec.tier == "experimental"
    assert spec.requires_explicit_opt_in is True
    config_path = (
        Path(__file__).resolve().parents[2] / "configs" / "algos" / "sipp_lattice_slice2_smoke.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["allow_testing_algorithms"] is True
    assert config["primitive_duration"] == config["time_slot_duration"]
    forecast_slots = config["pedestrian_forecast_horizon_s"] / config["time_slot_duration"]
    assert forecast_slots == int(forecast_slots)

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="sipp_lattice",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    allowed = require_algorithm_allowed(
        algo="sipp_lattice",
        benchmark_profile="experimental",
        ppo_paper_ready=False,
        allow_testing_algorithms=True,
    )
    assert allowed == spec

    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        build_registered_adapter_policy_spec("sipp_lattice", {})


def test_sipp_lattice_blocked_by_baseline_safe_profile() -> None:
    """Experimental tier must be blocked from the baseline-safe profile."""
    with pytest.raises(ValueError, match="baseline-safe"):
        require_algorithm_allowed(
            algo="sipp_lattice",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
            allow_testing_algorithms=True,
        )


def test_sipp_lattice_aliases_resolve_to_canonical() -> None:
    """Registered aliases collapse to the canonical planner key."""
    for alias in ("sipp_lattice", "sipp_kinodynamic", "kinodynamic_sipp"):
        assert canonical_algorithm_name(alias) == "sipp_lattice"


def test_sipp_lattice_metadata_marks_testing_only_adapter() -> None:
    """Kinematics metadata flags the planner as a testing-only adapter."""
    profile = _KINEMATICS_PROFILE_BY_CANONICAL["sipp_lattice"]
    assert profile["default_execution_mode"] == "adapter"
    assert profile["default_adapter_name"] == "SippLatticeSearchPlannerAdapter"
    assert profile["testing_only_adapter"] is True
    assert profile["projection_documented"] is True
