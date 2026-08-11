"""Focused unit tests for the deterministic finite-budget grid search (#6911).

These tests cover the diagnostic-only search baseline: config validation,
deterministic output reproducibility, invalid-candidate accounting,
bound-conflicting candidate handling, and bound-preserving integration
through ``BoundedResidualAdversary``.

This is a capability-only slice: it makes no benchmark, metric,
planner-ranking, safety, or paper-facing claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.ped_npc.residual_adversary import (
    BoundedResidualAdversary,
    ResidualAdversaryConfig,
)
from robot_sf.ped_npc.residual_search import (
    FiniteGridSearchPolicy,
    ResidualSearchConfig,
    SearchDiagnosticRecord,
    _build_action_grid,
    _evaluate_candidate,
    compute_config_digest,
)

ROBOT_POSE = ((0.0, 0.0), 0.0)
SEARCH_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "adversarial"
    / "issue_6911_residual_search.yaml"
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_search_config_defaults() -> None:
    """Default search config has the expected algorithm name and budget."""
    config = ResidualSearchConfig()
    assert config.algorithm_name == "finite_grid_search_v1"
    assert config.objective_proxy == "maximize_residual_magnitude"
    assert config.grid_points_per_dim == 3
    assert config.max_candidates == 9
    assert config.seed == 42


def test_search_config_rejects_non_positive_grid_points() -> None:
    """Non-positive grid_points_per_dim must fail closed."""
    with pytest.raises(ValueError):
        ResidualSearchConfig(grid_points_per_dim=0)
    with pytest.raises(ValueError):
        ResidualSearchConfig(grid_points_per_dim=-1)


def test_search_config_rejects_non_int_grid_points() -> None:
    """Grid points must be an int, not a float or bool."""
    with pytest.raises(TypeError):
        ResidualSearchConfig(grid_points_per_dim=3.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ResidualSearchConfig(grid_points_per_dim=True)  # type: ignore[arg-type]


def test_search_config_rejects_non_int_max_candidates() -> None:
    """Max candidates must be an int."""
    with pytest.raises(TypeError):
        ResidualSearchConfig(max_candidates=9.0)  # type: ignore[arg-type]


def test_search_config_rejects_inverted_action_bounds() -> None:
    """Action min must be strictly less than action max."""
    with pytest.raises(ValueError):
        ResidualSearchConfig(action_min_mps2=1.5, action_max_mps2=1.5)
    with pytest.raises(ValueError):
        ResidualSearchConfig(action_min_mps2=2.0, action_max_mps2=1.0)


def test_search_config_rejects_non_finite_action_bounds() -> None:
    """Non-finite action bounds must fail closed."""
    with pytest.raises(ValueError):
        ResidualSearchConfig(action_min_mps2=float("nan"))
    with pytest.raises(ValueError):
        ResidualSearchConfig(action_max_mps2=float("inf"))


def test_search_config_rejects_empty_algorithm_name() -> None:
    """Algorithm name must be a non-empty string."""
    with pytest.raises(ValueError):
        ResidualSearchConfig(algorithm_name="")


def test_search_config_rejects_empty_objective_proxy() -> None:
    """Objective proxy must be a non-empty string."""
    with pytest.raises(ValueError):
        ResidualSearchConfig(objective_proxy="")


def test_search_config_rejects_bool_seed() -> None:
    """Seed must be an int, not a bool."""
    with pytest.raises(TypeError):
        ResidualSearchConfig(seed=True)  # type: ignore[arg-type]


def test_search_config_rejects_string_seed() -> None:
    """Seed must be an int."""
    with pytest.raises(TypeError):
        ResidualSearchConfig(seed="42")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config digest determinism
# ---------------------------------------------------------------------------


def test_config_digest_is_deterministic() -> None:
    """Two identical configs produce the same SHA-256 digest."""
    config_a = ResidualSearchConfig(seed=42, grid_points_per_dim=3)
    config_b = ResidualSearchConfig(seed=42, grid_points_per_dim=3)
    assert compute_config_digest(config_a) == compute_config_digest(config_b)


def test_config_digest_differs_on_changed_field() -> None:
    """Changing any field changes the digest."""
    base = ResidualSearchConfig(seed=42)
    changed = ResidualSearchConfig(seed=99)
    assert compute_config_digest(base) != compute_config_digest(changed)


def test_config_digest_is_hex_16() -> None:
    """The digest is a 16-character hex string."""
    digest = compute_config_digest(ResidualSearchConfig())
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# Action grid
# ---------------------------------------------------------------------------


def test_action_grid_shape() -> None:
    """Grid with G points per dim has G**2 rows and 2 columns."""
    grid = _build_action_grid(-1.0, 1.0, 3)
    assert grid.shape == (9, 2)


def test_action_grid_extrema() -> None:
    """Grid extrema match the specified bounds."""
    grid = _build_action_grid(-2.0, 2.0, 5)
    assert float(np.min(grid)) == pytest.approx(-2.0)
    assert float(np.max(grid)) == pytest.approx(2.0)


def test_action_grid_deterministic() -> None:
    """Repeated calls produce identical arrays."""
    grid_a = _build_action_grid(-1.0, 1.0, 4)
    grid_b = _build_action_grid(-1.0, 1.0, 4)
    np.testing.assert_array_equal(grid_a, grid_b)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def test_evaluate_candidate_valid() -> None:
    """A zero candidate within bounds evaluates as valid."""
    config = ResidualAdversaryConfig(is_active=True)
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]])
    max_speeds = np.array([1.5, 1.2])
    candidate = np.array([0.1, 0.1])
    score, is_valid = _evaluate_candidate(candidate, 0, 2, velocities, max_speeds, config, 0.1, 5)
    assert is_valid is True
    assert score >= 0.0


def test_evaluate_candidate_zero_candidate() -> None:
    """A zero candidate produces zero score."""
    config = ResidualAdversaryConfig(is_active=True)
    velocities = np.array([[0.5, 0.0]])
    max_speeds = np.array([1.5])
    candidate = np.array([0.0, 0.0])
    score, is_valid = _evaluate_candidate(candidate, 0, 1, velocities, max_speeds, config, 0.1, 5)
    assert is_valid is True
    assert score == pytest.approx(0.0)


def test_evaluate_candidate_non_finite_returns_invalid() -> None:
    """A non-finite candidate is counted as invalid, not promoted."""
    config = ResidualAdversaryConfig(is_active=True)
    velocities = np.array([[0.5, 0.0]])
    max_speeds = np.array([1.5])
    candidate = np.array([float("nan"), 0.0])
    score, is_valid = _evaluate_candidate(candidate, 0, 1, velocities, max_speeds, config, 0.1, 5)
    assert is_valid is False
    assert score == 0.0


# ---------------------------------------------------------------------------
# Deterministic output reproducibility
# ---------------------------------------------------------------------------


def _load_search_config() -> tuple[ResidualSearchConfig, ResidualAdversaryConfig]:
    """Load the checked-in issue config pair."""
    payload = yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    algo = payload["algorithm"]
    action = payload["action_bounds"]
    search_config = ResidualSearchConfig(
        algorithm_name=algo["name"],
        objective_proxy=algo["objective_proxy"],
        grid_points_per_dim=algo["grid_points_per_dim"],
        max_candidates=algo["max_candidates"],
        seed=algo["seed"],
        action_min_mps2=action["min_mps2"],
        action_max_mps2=action["max_mps2"],
    )
    residual_config = ResidualAdversaryConfig(**dict(payload["residual_adversary"]))
    return search_config, residual_config


def _run_search_sequence(
    *,
    seed: int,
    num_steps: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    max_speeds: np.ndarray,
    robot_pose: tuple,
) -> list[np.ndarray]:
    """Build a search policy inside a controller and return residuals."""
    search_config = ResidualSearchConfig(seed=seed, grid_points_per_dim=3, max_candidates=9)
    residual_config = ResidualAdversaryConfig(is_active=True, seed=seed)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=positions.shape[0],
    )
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=0.1,
        num_peds=positions.shape[0],
    )
    residuals = []
    for _ in range(num_steps):
        residual = adversary.step_residual(
            positions.copy(), velocities.copy(), max_speeds.copy(), robot_pose
        )
        residuals.append(residual)
    return residuals


def test_seeded_search_is_deterministic_across_independent_runs() -> None:
    """Two independent instances with the same config produce identical sequences."""
    positions = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=float)
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]], dtype=float)
    max_speeds = np.array([1.5, 1.2])
    num_steps = 10

    run_a = _run_search_sequence(
        seed=42,
        num_steps=num_steps,
        positions=positions,
        velocities=velocities,
        max_speeds=max_speeds,
        robot_pose=ROBOT_POSE,
    )
    run_b = _run_search_sequence(
        seed=42,
        num_steps=num_steps,
        positions=positions,
        velocities=velocities,
        max_speeds=max_speeds,
        robot_pose=ROBOT_POSE,
    )

    assert len(run_a) == num_steps
    assert len(run_b) == num_steps
    for step_idx, (res_a, res_b) in enumerate(zip(run_a, run_b, strict=True)):
        np.testing.assert_allclose(
            res_a,
            res_b,
            err_msg=f"residual mismatch at step {step_idx}",
        )


def test_diagnostic_record_is_deterministic() -> None:
    """Two runs with the same config produce byte-equivalent diagnostic JSON."""
    positions = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=float)
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]], dtype=float)
    max_speeds = np.array([1.5, 1.2])

    records = []
    for _ in range(2):
        search_config = ResidualSearchConfig(seed=42, grid_points_per_dim=3, max_candidates=9)
        residual_config = ResidualAdversaryConfig(is_active=True, seed=42)
        policy = FiniteGridSearchPolicy(
            search_config=search_config,
            residual_config=residual_config,
            dt_s=0.1,
            num_peds=2,
        )
        adversary = BoundedResidualAdversary(
            config=residual_config,
            policy=policy,
            dt_s=0.1,
            num_peds=2,
        )
        for _ in range(5):
            adversary.step_residual(
                positions.copy(), velocities.copy(), max_speeds.copy(), ROBOT_POSE
            )
        records.append(policy.last_record.to_json())

    assert records[0] == records[1]


def test_diagnostic_record_schema_version() -> None:
    """The record carries the expected schema version."""
    record = SearchDiagnosticRecord()
    assert record.schema_version == "residual_search_diagnostic.v1"


def test_diagnostic_record_json_has_sorted_keys() -> None:
    """JSON output has alphabetically sorted keys."""
    record = SearchDiagnosticRecord(
        algorithm_name="finite_grid_search_v1",
        seed=42,
        config_digest="abcdef0123456789",
    )
    json_str = record.to_json()
    parsed = json.loads(json_str)
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_diagnostic_record_to_dict_fields() -> None:
    """The dict contains all required diagnostic fields."""
    record = SearchDiagnosticRecord(
        algorithm_name="finite_grid_search_v1",
        objective_proxy="maximize_residual_magnitude",
        config_digest="abc123",
        seed=42,
        source_revision="deadbeef",
        grid_points_per_dim=3,
        action_bounds={"min_mps2": -1.5, "max_mps2": 1.5},
        budget=9,
        num_targeted_peds=1,
        total_evaluated=9,
        accepted=3,
        rejected=5,
        invalid=1,
    )
    d = record.to_dict()
    assert d["algorithm_name"] == "finite_grid_search_v1"
    assert d["objective_proxy"] == "maximize_residual_magnitude"
    assert d["config_digest"] == "abc123"
    assert d["seed"] == 42
    assert d["source_revision"] == "deadbeef"
    assert d["budget"] == 9
    assert d["accepted"] == 3
    assert d["rejected"] == 5
    assert d["invalid"] == 1
    assert d["action_bounds"]["min_mps2"] == -1.5
    assert d["action_bounds"]["max_mps2"] == 1.5


# ---------------------------------------------------------------------------
# Malformed / non-finite / bound-conflicting candidate handling
# ---------------------------------------------------------------------------


def test_search_counts_invalid_candidates() -> None:
    """Invalid candidates are counted, not promoted silently."""
    search_config = ResidualSearchConfig(
        seed=42,
        grid_points_per_dim=3,
        max_candidates=9,
        action_min_mps2=-1.5,
        action_max_mps2=1.5,
    )
    residual_config = ResidualAdversaryConfig(is_active=True)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=1,
    )
    observation = type(
        "Obs",
        (),
        {
            "positions": np.array([[5.0, 5.0]]),
            "velocities": np.array([[0.0, 0.0]]),
            "max_speeds": np.array([1.5]),
            "target_ped_mask": np.array([True]),
            "robot_pose": ROBOT_POSE,
            "sim_time_s": 0.0,
            "step_index": 0,
            "macro_action_index": 0,
        },
    )()
    result = policy.propose_residual(observation)
    record = policy.last_record
    assert record.total_evaluated == record.accepted + record.rejected + record.invalid
    assert result.shape == (1, 2)
    assert np.all(np.isfinite(result))


def test_search_no_targets_returns_zeros() -> None:
    """When no pedestrians are targeted, all outputs are zero and accounting is empty."""
    search_config = ResidualSearchConfig(seed=42)
    residual_config = ResidualAdversaryConfig(is_active=True)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=2,
    )
    observation = type(
        "Obs",
        (),
        {
            "positions": np.array([[1.0, 1.0], [2.0, 2.0]]),
            "velocities": np.array([[0.1, 0.0], [0.0, 0.1]]),
            "max_speeds": np.array([1.5, 1.5]),
            "target_ped_mask": np.array([False, False]),
            "robot_pose": ROBOT_POSE,
            "sim_time_s": 0.0,
            "step_index": 0,
            "macro_action_index": 0,
        },
    )()
    result = policy.propose_residual(observation)
    np.testing.assert_array_equal(result, np.zeros((2, 2)))
    assert policy.last_record.num_targeted_peds == 0
    assert policy.last_record.total_evaluated == 0


# ---------------------------------------------------------------------------
# Bound-preserving integration through BoundedResidualAdversary
# ---------------------------------------------------------------------------


def test_search_residual_respects_acceleration_bound() -> None:
    """The applied residual never exceeds max_residual_accel_mps2 per row."""
    search_config = ResidualSearchConfig(
        seed=42,
        grid_points_per_dim=5,
        max_candidates=25,
        action_min_mps2=-2.0,
        action_max_mps2=2.0,
    )
    residual_config = ResidualAdversaryConfig(
        is_active=True,
        max_residual_accel_mps2=1.0,
    )
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=2,
    )
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=0.1,
        num_peds=2,
    )
    positions = np.array([[3.0, 1.0], [2.0, 4.0]])
    velocities = np.array([[0.5, 0.0], [0.0, 0.3]])
    max_speeds = np.array([1.5, 1.2])

    for _ in range(20):
        residual = adversary.step_residual(
            positions.copy(), velocities.copy(), max_speeds.copy(), ROBOT_POSE
        )
        norms = np.linalg.norm(residual, axis=1)
        assert np.all(norms <= 1.0 + 1e-9), f"acceleration bound violated: {norms}"


def test_search_residual_respects_speed_bound() -> None:
    """Resulting speed never exceeds max_speeds."""
    search_config = ResidualSearchConfig(
        seed=42,
        grid_points_per_dim=3,
        max_candidates=9,
    )
    residual_config = ResidualAdversaryConfig(is_active=True)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=1,
    )
    adversary = BoundedResidualAdversary(
        config=residual_config,
        policy=policy,
        dt_s=0.1,
        num_peds=1,
    )
    positions = np.array([[2.0, 0.0]])
    velocities = np.array([[0.8, 0.0]])
    max_speeds = np.array([1.0])

    for _ in range(10):
        residual = adversary.step_residual(
            positions.copy(), velocities.copy(), max_speeds.copy(), ROBOT_POSE
        )
        resulting_velocity = velocities + residual * 0.1
        resulting_speed = float(np.linalg.norm(resulting_velocity))
        assert resulting_speed <= 1.0 + 0.5 + 1e-9, f"speed bound violated: {resulting_speed}"


def test_search_policy_implements_residual_adversary_policy() -> None:
    """FiniteGridSearchPolicy has the required propose_residual method."""
    search_config = ResidualSearchConfig(seed=42)
    residual_config = ResidualAdversaryConfig(is_active=True)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=1,
    )
    assert callable(getattr(policy, "propose_residual", None))


def test_search_policy_grid_property() -> None:
    """The grid property returns a copy of the internal action grid."""
    search_config = ResidualSearchConfig(grid_points_per_dim=3)
    residual_config = ResidualAdversaryConfig(is_active=True)
    policy = FiniteGridSearchPolicy(
        search_config=search_config,
        residual_config=residual_config,
        dt_s=0.1,
        num_peds=1,
    )
    grid = policy.grid
    assert grid.shape == (9, 2)
    # Mutating the copy should not affect the internal grid
    grid[0, 0] = 999.0
    np.testing.assert_allclose(policy.grid[0, 0], search_config.action_min_mps2)


def test_search_config_from_yaml_round_trip() -> None:
    """The checked-in YAML loads without error and matches the config fields."""
    payload = yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    algo = payload["algorithm"]
    action = payload["action_bounds"]
    search_config = ResidualSearchConfig(
        algorithm_name=algo["name"],
        objective_proxy=algo["objective_proxy"],
        grid_points_per_dim=algo["grid_points_per_dim"],
        max_candidates=algo["max_candidates"],
        seed=algo["seed"],
        action_min_mps2=action["min_mps2"],
        action_max_mps2=action["max_mps2"],
    )
    assert search_config.algorithm_name == "finite_grid_search_v1"
    assert search_config.seed == 42
    assert search_config.grid_points_per_dim == 3
    assert search_config.max_candidates == 9
