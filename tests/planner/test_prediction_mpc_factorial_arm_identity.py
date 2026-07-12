"""Preflight identity checks for the issue #5355 prediction-MPC factorial (Refs #5372).

Claim boundary: static config parity check, no execution. Asserts that the four
canonical arm configs
(``configs/algos/prediction_mpc_factorial_A{0,1}_B{0,1}.yaml``) share the
observation contract, kinematics, and runtime budget, and differ **only** in the
two preregistered factor flags plus the mechanically-implied soft pedestrian
weight — the parity condition from
``docs/context/issue_5355_factorial_preregistration.md`` §2 and §7.

Factor encoding in the config files:
- Factor A (pedestrian prediction): ``predictor_backend`` (``none`` -> off,
  ``constant_velocity`` -> on).
- Factor B (hard pedestrian constraints): ``hard_pedestrian_constraints_enabled``
  with the implied soft ``pedestrian_clearance_weight`` (0.0 when hard-on,
  ``W_soft = 4.5`` when hard-off).
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.planner.prediction_mpc import (
    PredictionMPCConfig,
    build_prediction_mpc_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALGO_DIR = _REPO_ROOT / "configs" / "algos"
_ARM_IDS = ("A0_B0", "A0_B1", "A1_B0", "A1_B1")

# The only fields permitted to differ across arms: the two factor flags plus the
# mechanically-implied soft-weight bookkeeping (prereg §1.3, §7).
_FACTOR_FIELDS = frozenset(
    {"predictor_backend", "hard_pedestrian_constraints_enabled", "pedestrian_clearance_weight"}
)

# Preregistered 2x2 truth table (prereg §1.3, "Arm table").
_TRUTH_TABLE: dict[str, dict[str, Any]] = {
    "A0_B0": {"predictor_backend": "none", "hard": False, "weight": 4.5},
    "A1_B0": {"predictor_backend": "constant_velocity", "hard": False, "weight": 4.5},
    "A0_B1": {"predictor_backend": "none", "hard": True, "weight": 0.0},
    "A1_B1": {"predictor_backend": "constant_velocity", "hard": True, "weight": 0.0},
}


def _raw_arm(arm_id: str) -> dict[str, Any]:
    """Parse a canonical arm YAML into a raw mapping."""
    path = _ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} did not parse to a mapping"
    return payload


@pytest.fixture(scope="module")
def raw_arms() -> dict[str, dict[str, Any]]:
    """Return the four parsed arm YAML mappings."""
    return {arm_id: _raw_arm(arm_id) for arm_id in _ARM_IDS}


@pytest.fixture(scope="module")
def built_arms() -> dict[str, PredictionMPCConfig]:
    """Return the four arm configs as built through the production builder."""
    return {arm_id: build_prediction_mpc_config(_raw_arm(arm_id)) for arm_id in _ARM_IDS}


def test_all_four_arm_configs_exist() -> None:
    """The four canonical arm config files are present."""
    for arm_id in _ARM_IDS:
        path = _ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml"
        assert path.is_file(), f"missing canonical arm config: {path}"


def test_arm_configs_share_key_sets(raw_arms: dict[str, dict[str, Any]]) -> None:
    """All four arms declare an identical set of config keys."""
    key_sets = {arm_id: frozenset(cfg) for arm_id, cfg in raw_arms.items()}
    reference = key_sets["A0_B0"]
    for arm_id, keys in key_sets.items():
        assert keys == reference, f"{arm_id} key set differs: {keys ^ reference}"


def test_arm_configs_differ_only_in_factor_fields(raw_arms: dict[str, dict[str, Any]]) -> None:
    """Every non-factor field is identical across all four arms.

    This covers the observation contract, kinematics, and runtime budget in one
    pass: nothing outside the two factor flags and implied soft weight may vary.
    """
    shared_keys = set(raw_arms["A0_B0"]) - _FACTOR_FIELDS
    differing: dict[str, set[Any]] = {}
    for key in shared_keys:
        values = {str(raw_arms[arm_id][key]) for arm_id in _ARM_IDS}
        if len(values) > 1:
            differing[key] = values
    assert not differing, f"non-factor fields must be identical across arms: {differing}"

    # And the factor fields are exactly the set of fields that actually differ.
    actually_differ = {
        key
        for key in raw_arms["A0_B0"]
        if len({str(raw_arms[arm_id][key]) for arm_id in _ARM_IDS}) > 1
    }
    assert actually_differ == set(_FACTOR_FIELDS), (
        f"arms must differ in exactly the factor fields, got {actually_differ}"
    )


def test_shared_kinematics_and_runtime_budget(raw_arms: dict[str, dict[str, Any]]) -> None:
    """Kinematics and per-decision runtime budget match across arms (prereg §2.2, §2.3)."""
    kinematics = ("max_linear_speed", "max_angular_speed", "horizon_steps", "rollout_dt")
    runtime_budget = ("solver_max_iterations", "solver_ftol", "warm_start", "fallback_to_stop")
    for key in (*kinematics, *runtime_budget):
        values = {raw_arms[arm_id][key] for arm_id in _ARM_IDS}
        assert len(values) == 1, f"{key} must be shared across arms, got {values}"


def test_observation_contract_shared_by_construction(raw_arms: dict[str, dict[str, Any]]) -> None:
    """No arm overrides observation access; parity holds by construction (prereg §2.1).

    All arms are the same ``PredictionMPCPlannerAdapter`` consuming the SocNav
    structured observation, so the observation contract cannot vary per arm. This
    guards against a future arm sneaking in an observation-shaping key.
    """
    observation_shaping_keys = {
        "observation_mode",
        "obs_mode",
        "observation_space",
        "adapter",
        "policy_adapter",
        "socnav_fields",
    }
    for arm_id, cfg in raw_arms.items():
        leaked = observation_shaping_keys & set(cfg)
        assert not leaked, f"{arm_id} must not shape the observation contract: {leaked}"


def test_arm_truth_table_matches_preregistration(
    built_arms: dict[str, PredictionMPCConfig],
) -> None:
    """The built configs realize the preregistered 2x2 truth table (prereg §1.3)."""
    for arm_id, expected in _TRUTH_TABLE.items():
        cfg = built_arms[arm_id]
        assert cfg.predictor_backend == expected["predictor_backend"], arm_id
        assert cfg.hard_pedestrian_constraints_enabled is expected["hard"], arm_id
        assert cfg.pedestrian_clearance_weight == expected["weight"], arm_id


def test_built_configs_differ_only_in_factor_fields(
    built_arms: dict[str, PredictionMPCConfig],
) -> None:
    """After building, the four dataclasses differ only in the factor fields.

    Ties the YAML-level parity check to the object the planner actually consumes.
    """
    differing: dict[str, set[Any]] = {}
    for field in fields(PredictionMPCConfig):
        values = {getattr(built_arms[arm_id], field.name) for arm_id in _ARM_IDS}
        if len(values) > 1:
            differing[field.name] = values
    assert set(differing) <= _FACTOR_FIELDS, (
        f"built configs may differ only in factor fields, got {set(differing)}"
    )


def test_uncertainty_envelope_disabled_on_every_arm(
    built_arms: dict[str, PredictionMPCConfig],
) -> None:
    """The opt-in uncertainty envelope is off on all four arms (prereg §7)."""
    for arm_id, cfg in built_arms.items():
        assert cfg.pedestrian_uncertainty_envelope_enabled is False, arm_id
        assert cfg.pedestrian_uncertainty_alpha_mps == 0.0, arm_id
