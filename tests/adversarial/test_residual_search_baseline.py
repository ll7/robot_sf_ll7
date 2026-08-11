"""Tests for the deterministic grid-search residual search baseline (issue #6911).

Covers:
- Config validation and deterministic repeated output
- Action grid construction and objective proxy correctness
- Fail-closed behavior on invalid/non-finite input
- Bound-preserving integration with BoundedResidualAdversary
- Diagnostic record emission and JSON serialization
- Deterministic reproducibility across independent runs

Capability-only slice: no benchmark, planner-ranking, safety, or paper-facing
claim.
"""

from __future__ import annotations

import json
import math
from math import isfinite
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.ped_npc.residual_adversary import (
    EPSILON,
    BoundedResidualAdversary,
    ResidualAdversaryConfig,
    ResidualAdversaryObservation,
)
from robot_sf.ped_npc.residual_search_baseline import (
    ALGORITHM_NAME,
    GridSearchResidualConfig,
    GridSearchResidualPolicy,
    _build_action_grid,
    _objective_proxy,
)

ROBOT_POSE = ((0.0, 0.0), 0.0)
SEARCH_BASELINE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "adversarial"
    / "issue_4360_residual_search_baseline.yaml"
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_grid_search_config_defaults() -> None:
    """Default config values are finite and deterministic."""
    config = GridSearchResidualConfig()
    assert config.num_directions == 8
    assert config.num_magnitudes == 3
    assert config.seed == 42
    assert config.max_macro_budget == 1000


def test_grid_search_config_rejects_invalid_values() -> None:
    """Non-positive or non-finite config values must fail closed."""
    with pytest.raises(ValueError, match="num_directions"):
        GridSearchResidualConfig(num_directions=0)
    with pytest.raises(ValueError, match="num_magnitudes"):
        GridSearchResidualConfig(num_magnitudes=-1)
    with pytest.raises(TypeError, match="seed"):
        GridSearchResidualConfig(seed="not-an-int")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_macro_budget"):
        GridSearchResidualConfig(max_macro_budget=0)


# ---------------------------------------------------------------------------
# Action grid construction
# ---------------------------------------------------------------------------


def test_action_grid_shape() -> None:
    """Grid has 1 + num_directions * num_magnitudes rows."""
    grid = _build_action_grid(4, 2, 1.0)
    assert grid.shape == (1 + 4 * 2, 2)


def test_action_grid_first_row_is_zero() -> None:
    """Row 0 is always the zero-residual baseline."""
    grid = _build_action_grid(8, 3, 1.5)
    np.testing.assert_allclose(grid[0], [0.0, 0.0])


def test_action_grid_magnitudes_are_bounded() -> None:
    """All grid magnitudes are within [0, max_accel]."""
    max_accel = 1.5
    grid = _build_action_grid(8, 3, max_accel)
    norms = np.linalg.norm(grid, axis=1)
    assert np.all(norms <= max_accel + 1e-9)
    assert np.all(norms >= -1e-9)


def test_action_grid_is_deterministic() -> None:
    """Two calls with the same args produce identical grids."""
    g1 = _build_action_grid(8, 3, 1.0)
    g2 = _build_action_grid(8, 3, 1.0)
    np.testing.assert_array_equal(g1, g2)


def test_single_magnitude_grid_still_includes_zero_baseline() -> None:
    """A one-level magnitude grid follows the documented zero-to-max range."""
    grid = _build_action_grid(4, 1, 1.0)
    assert np.allclose(grid, 0.0)


# ---------------------------------------------------------------------------
# Objective proxy
# ---------------------------------------------------------------------------


def test_objective_proxy_prefers_approaching_candidate() -> None:
    """An accelerating-toward-robot candidate scores better than zero."""
    ped_pos = np.array([5.0, 0.0])
    ped_vel = np.array([0.0, 0.0])
    robot_pos = np.array([0.0, 0.0])
    accel_toward = np.array([-1.0, 0.0])
    accel_away = np.array([1.0, 0.0])
    score_toward = _objective_proxy(accel_toward, ped_pos, ped_vel, robot_pos, 1.0, 0.5, 0.1)
    score_away = _objective_proxy(accel_away, ped_pos, ped_vel, robot_pos, 1.0, 0.5, 0.1)
    assert score_toward < score_away


def test_objective_proxy_zero_accel_matches_baseline() -> None:
    """Zero acceleration is a valid candidate with finite score."""
    score = _objective_proxy(
        np.zeros(2), np.array([5.0, 0.0]), np.zeros(2), np.zeros(2), 1.0, 0.5, 0.1
    )
    assert isfinite(score)


def test_objective_proxy_is_deterministic() -> None:
    """Same inputs always produce the same score."""
    args = (
        np.array([0.5, 0.0]),
        np.array([3.0, 1.0]),
        np.array([0.1, 0.0]),
        np.array([0.0, 0.0]),
        1.0,
        0.5,
        0.1,
    )
    s1 = _objective_proxy(*args)
    s2 = _objective_proxy(*args)
    assert s1 == pytest.approx(s2)


# ---------------------------------------------------------------------------
# Policy: deterministic repeated output
# ---------------------------------------------------------------------------


def _build_observation(
    positions: np.ndarray,
    velocities: np.ndarray,
    max_speeds: np.ndarray,
    target_mask: np.ndarray,
    robot_pose: tuple = ROBOT_POSE,
    macro_idx: int = 0,
) -> ResidualAdversaryObservation:
    """Construct a residual adversary observation from arrays."""
    return ResidualAdversaryObservation(
        positions=positions,
        velocities=velocities,
        max_speeds=max_speeds,
        target_ped_mask=target_mask,
        robot_pose=robot_pose,
        sim_time_s=0.0,
        step_index=0,
        macro_action_index=macro_idx,
    )


def test_policy_deterministic_across_independent_runs() -> None:
    """Two independent policy instances with the same config produce identical proposals."""
    config = GridSearchResidualConfig(seed=42, num_directions=8, num_magnitudes=3)
    policy_a = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    policy_b = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    positions = np.array([[3.0, 1.0], [2.0, 4.0]])
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]])
    max_speeds = np.array([1.5, 1.2])
    target_mask = np.array([True, True])
    obs = _build_observation(positions, velocities, max_speeds, target_mask)
    proposal_a = policy_a.propose_residual(obs)
    proposal_b = policy_b.propose_residual(obs)
    np.testing.assert_allclose(proposal_a, proposal_b)


def test_policy_returns_zero_for_non_targeted_peds() -> None:
    """Non-targeted pedestrians receive zero residual."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    positions = np.array([[3.0, 0.0], [2.0, 0.0]])
    velocities = np.zeros((2, 2))
    max_speeds = np.array([1.5, 1.5])
    target_mask = np.array([True, False])
    obs = _build_observation(positions, velocities, max_speeds, target_mask)
    proposal = policy.propose_residual(obs)
    np.testing.assert_allclose(proposal[1], [0.0, 0.0])
    assert np.linalg.norm(proposal[0]) > 0.0


def test_policy_returns_zero_for_empty_crowd() -> None:
    """An empty observation returns an empty proposal."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,)), np.array([], dtype=bool)
    )
    proposal = policy.propose_residual(obs)
    assert proposal.shape == (0, 2)


# ---------------------------------------------------------------------------
# Fail-closed on invalid input
# ---------------------------------------------------------------------------


def test_policy_fail_closed_on_non_finite_robot_pose() -> None:
    """Non-finite robot pose must raise."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.array([[1.0, 0.0]]),
        np.zeros((1, 2)),
        np.array([1.5]),
        np.array([True]),
        robot_pose=((0.0, 0.0), float("nan")),
    )
    with pytest.raises(ValueError, match="robot_pose"):
        policy.propose_residual(obs)


def test_policy_fail_closed_on_non_finite_positions() -> None:
    """Non-finite positions must raise."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.array([[float("inf"), 0.0]]),
        np.zeros((1, 2)),
        np.array([1.5]),
        np.array([True]),
    )
    with pytest.raises(ValueError):
        policy.propose_residual(obs)


def test_policy_fail_closed_on_invalid_observation_shapes_and_speeds() -> None:
    """Malformed observation arrays must not enter the search loop."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    with pytest.raises(ValueError, match="velocities"):
        policy.propose_residual(
            _build_observation(
                np.array([[1.0, 0.0]]),
                np.zeros((1, 3)),
                np.array([1.5]),
                np.array([True]),
            )
        )
    with pytest.raises(ValueError, match="max_speeds"):
        policy.propose_residual(
            _build_observation(
                np.array([[1.0, 0.0]]),
                np.zeros((1, 2)),
                np.array([-1.0]),
                np.array([True]),
            )
        )
    with pytest.raises(ValueError, match="target_ped_mask"):
        policy.propose_residual(
            _build_observation(
                np.array([[1.0, 0.0]]),
                np.zeros((1, 2)),
                np.array([1.5]),
                np.array([1], dtype=np.int64),
            )
        )


# ---------------------------------------------------------------------------
# Diagnostic records
# ---------------------------------------------------------------------------


def test_diagnostic_record_emitted_per_macro_action() -> None:
    """Each macro-action boundary produces one diagnostic record."""
    config = GridSearchResidualConfig(seed=42, max_macro_budget=5)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    positions = np.array([[3.0, 0.0]])
    velocities = np.zeros((1, 2))
    max_speeds = np.array([1.5])
    target_mask = np.array([True])
    for i in range(3):
        obs = _build_observation(positions, velocities, max_speeds, target_mask, macro_idx=i)
        policy.propose_residual(obs)
    records = policy.diagnostic_records
    assert len(records) == 3
    for idx, record in enumerate(records):
        assert record["algorithm"] == ALGORITHM_NAME
        assert record["macro_action_index"] == idx
        assert record["seed"] == 42
        assert record["budget"] == 5
        assert record["grid_size"] == 1 + 8 * 3
        assert "accepted" in record
        assert "rejected" in record
        assert "invalid" in record


def test_diagnostic_records_are_deterministic() -> None:
    """Same config and inputs produce identical diagnostic records."""
    config = GridSearchResidualConfig(seed=42)
    policy_a = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    policy_b = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    positions = np.array([[3.0, 1.0]])
    velocities = np.zeros((1, 2))
    max_speeds = np.array([1.5])
    target_mask = np.array([True])
    obs = _build_observation(positions, velocities, max_speeds, target_mask)
    policy_a.propose_residual(obs)
    policy_b.propose_residual(obs)
    assert policy_a.diagnostic_records == policy_b.diagnostic_records


def test_diagnostic_provenance_includes_order_and_bound_settings() -> None:
    """Diagnostic records identify the config, source, action order, and bounds."""
    config = GridSearchResidualConfig(
        seed=42,
        config_id="fixture-search",
        source_revision="abc123",
    )
    policy = GridSearchResidualPolicy(
        config=config,
        max_residual_accel_mps2=1.5,
        dt_s=0.1,
        bound_settings={"max_jerk_mps3": 7.5, "min_separation_m": 0.6},
    )
    obs = _build_observation(
        np.array([[3.0, 1.0]]), np.zeros((1, 2)), np.array([1.5]), np.array([True])
    )
    policy.propose_residual(obs)
    record = policy.diagnostic_records[0]
    assert record["config_id"] == "fixture-search"
    assert record["source_revision"] == "abc123"
    assert record["action_order"] == "zero_baseline_then_seeded_angle_magnitude"
    assert record["candidate_order"][0] == 0
    assert set(record["candidate_order"]) == set(range(record["grid_size"]))
    assert record["bound_settings"] == {
        "dt_s": 0.1,
        "max_jerk_mps3": 7.5,
        "max_residual_accel_mps2": 1.5,
        "min_separation_m": 0.6,
    }


def test_policy_enforces_finite_macro_budget() -> None:
    """After the configured budget, the policy emits zero without searching."""
    config = GridSearchResidualConfig(seed=42, max_macro_budget=1)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.array([[3.0, 0.0]]), np.zeros((1, 2)), np.array([1.5]), np.array([True])
    )
    first = policy.propose_residual(obs)
    assert policy.budget_exhausted is True
    second = policy.propose_residual(obs)
    assert np.linalg.norm(first) > EPSILON
    np.testing.assert_array_equal(second, np.zeros((1, 2)))
    assert policy.budget_exhausted is True
    assert policy.macro_count == 1
    assert len(policy.diagnostic_records) == 1


def test_write_diagnostics_produces_valid_json(tmp_path: Path) -> None:
    """write_diagnostics produces a valid JSON file with expected keys."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.array([[3.0, 0.0]]), np.zeros((1, 2)), np.array([1.5]), np.array([True])
    )
    policy.propose_residual(obs)
    out_file = tmp_path / "diagnostics.json"
    policy.write_diagnostics(out_file)
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["algorithm"] == ALGORITHM_NAME
    assert payload["seed"] == 42
    assert len(payload["records"]) == 1
    assert payload["total_accepted"] + payload["total_rejected"] >= 1


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------


def test_reset_clears_accumulated_state() -> None:
    """After reset, macro_count and diagnostic records are cleared."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    obs = _build_observation(
        np.array([[3.0, 0.0]]), np.zeros((1, 2)), np.array([1.5]), np.array([True])
    )
    policy.propose_residual(obs)
    assert policy.macro_count == 1
    assert len(policy.diagnostic_records) == 1
    policy.reset()
    assert policy.macro_count == 0
    assert len(policy.diagnostic_records) == 0
    assert policy.rejected_count == 0
    assert policy.invalid_count == 0
    assert policy.accepted_count == 0


# ---------------------------------------------------------------------------
# Integration with BoundedResidualAdversary
# ---------------------------------------------------------------------------


def test_grid_search_policy_respects_bounds_through_bounded_adversary() -> None:
    """Proposals routed through BoundedResidualAdversary satisfy all hard bounds."""
    search_config = GridSearchResidualConfig(seed=42, num_directions=8, num_magnitudes=3)
    residual_config = ResidualAdversaryConfig(
        is_active=True,
        max_residual_accel_mps2=1.5,
        max_jerk_mps3=7.5,
        max_speed_delta_mps=0.5,
        max_heading_change_per_macro_rad=math.pi / 4,
        macro_action_dt_s=0.5,
        target_ped_idx=-1,
    )
    policy = GridSearchResidualPolicy(config=search_config, max_residual_accel_mps2=1.5, dt_s=0.1)
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=0.1,
        num_peds=2,
    )
    positions = np.array([[5.0, 0.0], [3.0, 2.0]])
    velocities = np.array([[0.0, 0.0], [0.0, 0.0]])
    max_speeds = np.array([1.5, 1.5])
    for _ in range(15):
        residual = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
        assert np.all(np.isfinite(residual))
        norms = np.linalg.norm(residual, axis=1)
        assert np.all(norms <= residual_config.max_residual_accel_mps2 + 1e-9)


def test_grid_search_policy_produces_nonzero_residual_when_beneficial() -> None:
    """When robot is nearby, the grid search should find a non-zero acceleration."""
    search_config = GridSearchResidualConfig(seed=42, num_directions=8, num_magnitudes=3)
    residual_config = ResidualAdversaryConfig(
        is_active=True,
        max_residual_accel_mps2=1.5,
        max_jerk_mps3=7.5,
        macro_action_dt_s=0.5,
        target_ped_idx=-1,
    )
    policy = GridSearchResidualPolicy(config=search_config, max_residual_accel_mps2=1.5, dt_s=0.1)
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=0.1,
        num_peds=1,
    )
    positions = np.array([[5.0, 0.0]])
    velocities = np.zeros((1, 2))
    max_speeds = np.array([1.5])
    residual = adversary.step_residual(positions, velocities, max_speeds, ROBOT_POSE)
    assert np.any(np.abs(residual) > EPSILON)


# ---------------------------------------------------------------------------
# Deterministic reproducibility across config load
# ---------------------------------------------------------------------------


def test_config_loads_from_yaml() -> None:
    """The checked-in YAML config loads without error."""
    payload = yaml.safe_load(SEARCH_BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    search_cfg = payload["residual_search_baseline"]
    config = GridSearchResidualConfig(
        num_directions=search_cfg["num_directions"],
        num_magnitudes=search_cfg["num_magnitudes"],
        weight_approach=search_cfg["weight_approach"],
        weight_distance=search_cfg["weight_distance"],
        seed=search_cfg["seed"],
        max_macro_budget=search_cfg["max_macro_budget"],
        config_id=search_cfg["config_id"],
        source_revision=search_cfg["source_revision"],
        schema_version=search_cfg["schema_version"],
    )
    assert config.seed == 42
    assert config.num_directions == 8
    assert config.num_magnitudes == 3
    assert config.config_id == "issue_4360_residual_search_baseline"
    assert config.source_revision is None


def test_yaml_config_deterministic_repeated_output() -> None:
    """Loading from YAML and running twice produces identical proposals."""
    payload = yaml.safe_load(SEARCH_BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    search_cfg = payload["residual_search_baseline"]
    config = GridSearchResidualConfig(
        num_directions=search_cfg["num_directions"],
        num_magnitudes=search_cfg["num_magnitudes"],
        seed=search_cfg["seed"],
        max_macro_budget=search_cfg["max_macro_budget"],
    )
    positions = np.array([[3.0, 1.0], [2.0, 4.0]])
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]])
    max_speeds = np.array([1.5, 1.2])
    target_mask = np.array([True, True])

    proposals = []
    for _ in range(2):
        policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
        obs = _build_observation(positions, velocities, max_speeds, target_mask)
        proposals.append(policy.propose_residual(obs))
    np.testing.assert_allclose(proposals[0], proposals[1])


# ---------------------------------------------------------------------------
# Accepted/rejected accounting
# ---------------------------------------------------------------------------


def test_accepted_rejected_counts_are_consistent() -> None:
    """accepted + rejected == number of targeted pedestrians per macro action."""
    config = GridSearchResidualConfig(seed=42)
    policy = GridSearchResidualPolicy(config=config, max_residual_accel_mps2=1.5, dt_s=0.1)
    positions = np.array([[3.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    velocities = np.zeros((3, 2))
    max_speeds = np.array([1.5, 1.5, 1.5])
    target_mask = np.array([True, True, True])
    obs = _build_observation(positions, velocities, max_speeds, target_mask)
    policy.propose_residual(obs)
    rec = policy.diagnostic_records[0]
    assert rec["accepted"] + rec["rejected"] == 3
