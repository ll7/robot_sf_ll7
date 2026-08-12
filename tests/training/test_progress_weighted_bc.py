"""Tests for the progress-weighted expert-action NLL objective (issue #6951).

These tests verify:
* Arm-A equivalence to ordinary mean expert-action NLL (uniform weights).
* Arm-B weighting direction, bounds, and progress derivation.
* Fail-closed behavior for malformed/missing route-progress data.
* Deterministic config and manifest serialization.

Claim boundary: these are unit-level structural tests for the objective module,
not benchmark evidence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.training.progress_weighted_bc import (
    ProgressWeightedBcError,
    ProgressWeightedObjectiveConfig,
    compute_progress_weights,
    load_remaining_route_length_from_npz,
    objective_config_json,
    serialize_objective_config,
    sha256_objective_config,
    weighted_expert_action_nll,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EPISODES = [
    np.array([10.0, 8.0, 6.0, 4.0, 2.0, 0.0], dtype=np.float64),
    np.array([5.0, 4.5, 4.0, 3.0, 1.0, 0.0], dtype=np.float64),
]
_ROUTE_LENGTH_PROVENANCE = {
    "schema_version": "remaining_route_length.v1",
    "alignment": "one_value_per_observation",
    "derived_signal": "remaining_before_minus_after",
    "semantics": "remaining_route_length_meters",
    "units": "m",
    "source": "recorded_route_remaining_length",
}


@pytest.fixture()
def _episodes() -> list[np.ndarray]:
    """Two synthetic episodes of remaining route length."""
    return [ep.copy() for ep in _EPISODES]


# ---------------------------------------------------------------------------
# Arm A: uniform control
# ---------------------------------------------------------------------------


class TestArmAUniform:
    """Arm-A tests: uniform weights, equivalent to unweighted NLL."""

    def test_arm_a_config_creation(self) -> None:
        """Arm-A config must have lambda=0 and the unweighted objective name."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        assert cfg.arm == "A"
        assert cfg.progress_lambda == 0.0
        assert cfg.objective_name == "mean_expert_action_nll"

    def test_arm_a_weights_are_all_ones(self, _episodes: list[np.ndarray]) -> None:
        """Arm-A weights must be exactly 1.0 regardless of progress."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        weights = compute_progress_weights(_episodes, cfg)
        assert len(weights) == 2
        for w in weights:
            np.testing.assert_array_equal(w, np.ones_like(w))

    def test_arm_a_equivalence_to_unweighted_nll(self) -> None:
        """Weighted NLL with Arm-A weights must equal unweighted mean NLL."""
        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal(100)
        weights = np.ones(100, dtype=np.float64)

        unweighted = weighted_expert_action_nll(log_probs)
        weighted = weighted_expert_action_nll(log_probs, weights=weights)
        assert weighted == pytest.approx(unweighted)

    def test_arm_a_rejects_nonzero_lambda(self) -> None:
        """Arm-A config must fail closed when lambda is nonzero."""
        with pytest.raises(ProgressWeightedBcError, match="progress_lambda"):
            ProgressWeightedObjectiveConfig(
                objective_name="mean_expert_action_nll",
                arm="A",
                progress_lambda=0.5,
            )

    def test_arm_a_default_bounds_are_one(self) -> None:
        """Default Arm-A bounds are weight_min=1, weight_max=1 (strict uniform)."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        assert cfg.weight_min == 1.0
        assert cfg.weight_max == 1.0

    def test_arm_a_does_not_require_route_progress(self) -> None:
        """Arm A can construct its uniform weights without route provenance."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        weights = compute_progress_weights(None, cfg, action_step_counts=[2, 3])
        np.testing.assert_array_equal(weights[0], [1.0, 1.0])
        np.testing.assert_array_equal(weights[1], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Arm B: progress-weighted
# ---------------------------------------------------------------------------


class TestArmBProgressWeighted:
    """Arm-B tests: progress-derived bounded weights."""

    def test_arm_b_config_creation(self) -> None:
        """Arm-B config must carry the progress-weighted objective name."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.5,
            progress_normalization_scale=2.0,
        )
        assert cfg.arm == "B"
        assert cfg.progress_lambda == 0.5
        assert cfg.progress_normalization_scale == 2.0
        assert cfg.objective_name == "progress_weighted_expert_action_nll"

    def test_arm_b_weights_respect_bounds(self, _episodes: list[np.ndarray]) -> None:
        """Arm-B weights must be clipped to [weight_min, weight_max]."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=1.0,
            weight_min=0.5,
            weight_max=1.5,
        )
        weights = compute_progress_weights(_episodes, cfg)
        for w in weights:
            assert np.all(w >= 0.5 - 1e-12)
            assert np.all(w <= 1.5 + 1e-12)

    def test_arm_b_positive_progress_increases_weight(self) -> None:
        """Positive progress (route length decreasing) must produce weights > 1."""
        ep = np.array([10.0, 8.0, 6.0, 4.0], dtype=np.float64)
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=2.0,
            weight_min=0.1,
            weight_max=5.0,
        )
        weights = compute_progress_weights([ep], cfg)
        # progress = [2.0, 2.0, 2.0], normalized = [1.0, 1.0, 1.0]
        # raw = 1 + 1.0 * 1.0 = 2.0 for each step
        np.testing.assert_allclose(weights[0], [2.0, 2.0, 2.0])

    def test_arm_b_stalled_samples_not_removed(self) -> None:
        """Stalled samples (progress=0) must produce weight=1.0, not disappear."""
        ep = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=1.0,
            weight_min=0.5,
            weight_max=2.0,
        )
        weights = compute_progress_weights([ep], cfg)
        # progress = [0.0, 0.0, 0.0], raw = 1.0
        np.testing.assert_allclose(weights[0], [1.0, 1.0, 1.0])

    def test_arm_b_regressing_samples_get_lower_weight(self) -> None:
        """Negative progress (regression) must produce weight < 1."""
        ep = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=1.0,
            weight_min=0.1,
            weight_max=5.0,
        )
        weights = compute_progress_weights([ep], cfg)
        # progress = [-2.0, -2.0], normalized = [-2.0, -2.0]
        # raw = 1 + 1.0 * (-2.0) = -1.0, clipped to 0.1
        np.testing.assert_allclose(weights[0], [0.1, 0.1])

    def test_arm_b_lambda_zero_matches_arm_a(self, _episodes: list[np.ndarray]) -> None:
        """Arm B with lambda=0 produces uniform weights (but different objective name)."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.0,
            progress_normalization_scale=1.0,
            weight_min=0.5,
            weight_max=2.0,
        )
        weights = compute_progress_weights(_episodes, cfg)
        for w in weights:
            np.testing.assert_array_equal(w, np.ones_like(w))


# ---------------------------------------------------------------------------
# Fail-closed: malformed / missing data
# ---------------------------------------------------------------------------


class TestMalformedData:
    """Fail-closed behavior for malformed or missing route-progress data."""

    def test_missing_npz_fails_closed(self, tmp_path: Path) -> None:
        """Missing NPZ file must raise ProgressWeightedBcError."""
        with pytest.raises(ProgressWeightedBcError, match="not found"):
            load_remaining_route_length_from_npz(tmp_path / "missing.npz")

    def test_missing_array_key_fails_closed(self, tmp_path: Path) -> None:
        """NPZ without the required remaining_route_length array must fail closed."""
        path = tmp_path / "data.npz"
        np.savez(path, actions=np.zeros((2, 5, 2)))
        with pytest.raises(ProgressWeightedBcError, match="missing required array"):
            load_remaining_route_length_from_npz(path)

    def test_non_finite_values_fail_closed(self, tmp_path: Path) -> None:
        """NaN or Inf in remaining route length must fail closed."""
        path = tmp_path / "data.npz"
        rl = np.array([[10.0, np.nan, 8.0, 6.0]])
        np.savez(path, remaining_route_length=rl, actions=np.zeros((1, 3, 2)))
        with pytest.raises(ProgressWeightedBcError, match="non-finite"):
            load_remaining_route_length_from_npz(path)

    def test_scalar_array_fails_closed(self, tmp_path: Path) -> None:
        """Scalar remaining_route_length must fail closed."""
        path = tmp_path / "data.npz"
        np.savez(path, remaining_route_length=np.array(10.0))
        with pytest.raises(ProgressWeightedBcError, match="scalar"):
            load_remaining_route_length_from_npz(path)

    def test_alignment_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """remaining_route_length steps != actions+1 must fail closed."""
        path = tmp_path / "data.npz"
        # 3 actions -> need 4 remaining values, but we provide 3
        rl = np.array([[10.0, 8.0, 6.0]])
        actions = np.zeros((1, 3, 2))
        np.savez(path, remaining_route_length=rl, actions=actions)
        with pytest.raises(ProgressWeightedBcError, match="expected 4"):
            load_remaining_route_length_from_npz(path)

    def test_zero_steps_episode_fails_closed(self, tmp_path: Path) -> None:
        """An episode with zero remaining-route-length steps must fail closed."""
        path = tmp_path / "data.npz"
        rl = np.empty((1, 0), dtype=np.float64)
        np.savez(path, remaining_route_length=rl, actions=np.zeros((1, 0, 2)))
        with pytest.raises(ProgressWeightedBcError, match="zero steps"):
            load_remaining_route_length_from_npz(path)

    def test_missing_provenance_fails_closed_after_alignment(self, tmp_path: Path) -> None:
        """Aligned route lengths without declared provenance are not admissible."""
        path = tmp_path / "data.npz"
        np.savez(
            path,
            remaining_route_length=np.array([[10.0, 8.0, 6.0]]),
            actions=np.zeros((1, 2, 2)),
        )
        with pytest.raises(ProgressWeightedBcError, match="provenance"):
            load_remaining_route_length_from_npz(path)

    def test_proxy_provenance_fails_closed(self, tmp_path: Path) -> None:
        """Goal/displacement provenance cannot masquerade as route progress."""
        path = tmp_path / "data.npz"
        np.savez(
            path,
            remaining_route_length=np.array([[10.0, 8.0, 6.0]]),
            actions=np.zeros((1, 2, 2)),
            remaining_route_length_metadata={
                **_ROUTE_LENGTH_PROVENANCE,
                "source": "goal_displacement_proxy",
            },
        )
        with pytest.raises(ProgressWeightedBcError, match="source"):
            load_remaining_route_length_from_npz(path)

    def test_progress_computation_fails_on_non_finite(self) -> None:
        """Non-finite remaining route length must fail in weight computation."""
        ep = np.array([10.0, np.inf, 8.0], dtype=np.float64)
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=1.0,
        )
        with pytest.raises(ProgressWeightedBcError, match="non-finite"):
            compute_progress_weights([ep], cfg)

    def test_too_few_values_for_progress_fails_closed(self) -> None:
        """An episode with <2 remaining values cannot compute progress."""
        ep = np.array([10.0], dtype=np.float64)
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=1.0,
            progress_normalization_scale=1.0,
        )
        with pytest.raises(ProgressWeightedBcError, match=">= 2 values"):
            compute_progress_weights([ep], cfg)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Config construction must fail closed on invalid inputs."""

    def test_invalid_arm_label(self) -> None:
        """Unknown arm label must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="arm must be"):
            ProgressWeightedObjectiveConfig(objective_name="test", arm="C")

    def test_arm_b_wrong_objective_name(self) -> None:
        """Arm B with the Arm-A objective name must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="objective_name"):
            ProgressWeightedObjectiveConfig(
                objective_name="mean_expert_action_nll",
                arm="B",
                progress_lambda=0.5,
            )

    def test_negative_normalization_scale(self) -> None:
        """Negative normalization scale must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="normalization_scale"):
            ProgressWeightedObjectiveConfig(
                objective_name="progress_weighted_expert_action_nll",
                arm="B",
                progress_lambda=0.5,
                progress_normalization_scale=-1.0,
            )

    def test_weight_min_not_positive(self) -> None:
        """Non-positive weight_min must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="weight_min"):
            ProgressWeightedObjectiveConfig(
                objective_name="mean_expert_action_nll",
                arm="A",
                weight_min=0.0,
            )

    def test_weight_max_less_than_min(self) -> None:
        """weight_max < weight_min must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="weight_max"):
            ProgressWeightedObjectiveConfig(
                objective_name="mean_expert_action_nll",
                arm="A",
                weight_min=2.0,
                weight_max=1.0,
            )

    def test_disallowed_remaining_route_length_key(self) -> None:
        """Non-standard remaining_route_length_key must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="remaining_route_length_key"):
            ProgressWeightedObjectiveConfig(
                objective_name="mean_expert_action_nll",
                arm="A",
                remaining_route_length_key="custom_key",
            )


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------


class TestDeterministicSerialization:
    """Config and manifest serialization must be deterministic and JSON-safe."""

    def test_arm_a_to_manifest_dict(self) -> None:
        """Arm-A manifest dict is sorted and contains all config fields."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        d = cfg.to_manifest_dict()
        assert isinstance(d, dict)
        assert d["arm"] == "A"
        assert d["progress_lambda"] == 0.0
        assert d["objective_name"] == "mean_expert_action_nll"
        assert list(d.keys()) == sorted(d.keys())

    def test_arm_b_to_manifest_dict(self) -> None:
        """Arm-B manifest dict is sorted and contains all config fields."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.5,
            progress_normalization_scale=2.0,
        )
        d = cfg.to_manifest_dict()
        assert d["arm"] == "B"
        assert d["progress_lambda"] == 0.5
        assert d["progress_normalization_scale"] == 2.0
        assert list(d.keys()) == sorted(d.keys())

    def test_objective_config_json_is_deterministic(self) -> None:
        """Two serializations of the same config produce identical JSON."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.3,
            progress_normalization_scale=1.5,
        )
        json_a = objective_config_json(cfg)
        json_b = objective_config_json(cfg)
        assert json_a == json_b
        # JSON must be valid and round-trippable
        parsed = json.loads(json_a)
        assert isinstance(parsed, dict)

    def test_sha256_is_deterministic(self) -> None:
        """SHA-256 of the same config is always the same."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        h1 = sha256_objective_config(cfg)
        h2 = sha256_objective_config(cfg)
        assert h1 == h2
        assert len(h1) == 64

    def test_different_configs_produce_different_hashes(self) -> None:
        """Different configs must produce different SHA-256 digests."""
        cfg_a = ProgressWeightedObjectiveConfig.arm_a()
        cfg_b = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.5,
            progress_normalization_scale=1.0,
        )
        assert sha256_objective_config(cfg_a) != sha256_objective_config(cfg_b)

    def test_serialize_objective_config_includes_dataset_digest(self) -> None:
        """Serialization with an NPZ path includes the dataset SHA-256."""
        cfg = ProgressWeightedObjectiveConfig.arm_a()
        # Without NPZ: digest is empty
        d_no_npz = serialize_objective_config(cfg)
        assert d_no_npz["dataset_digest"] == ""

    def test_json_output_is_valid_json(self) -> None:
        """objective_config_json must return valid JSON."""
        cfg = ProgressWeightedObjectiveConfig.arm_b(
            progress_lambda=0.2,
            progress_normalization_scale=3.0,
        )
        raw = objective_config_json(cfg)
        parsed = json.loads(raw)
        assert parsed["arm"] == "B"
        assert parsed["progress_lambda"] == 0.2


# ---------------------------------------------------------------------------
# Weighted NLL loss
# ---------------------------------------------------------------------------


class TestWeightedNll:
    """Tests for the weighted expert-action NLL loss function."""

    def test_unweighted_nll_basic(self) -> None:
        """Unweighted NLL is the mean of negative log-probs."""
        log_probs = np.array([-1.0, -2.0, -3.0])
        result = weighted_expert_action_nll(log_probs)
        expected = -np.mean([-1.0, -2.0, -3.0])
        assert result == pytest.approx(expected)

    def test_weighted_nll_with_custom_weights(self) -> None:
        """Weighted NLL divides by total weight."""
        log_probs = np.array([-1.0, -2.0, -3.0])
        weights = np.array([1.0, 2.0, 3.0])
        result = weighted_expert_action_nll(log_probs, weights=weights)
        expected = -(1.0 * -1.0 + 2.0 * -2.0 + 3.0 * -3.0) / 6.0
        assert result == pytest.approx(expected)

    def test_weighted_nll_fails_on_non_finite_log_probs(self) -> None:
        """Non-finite log probs must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="non-finite"):
            weighted_expert_action_nll(np.array([1.0, np.nan]))

    def test_weighted_nll_fails_on_shape_mismatch(self) -> None:
        """Weights with wrong shape must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="shape"):
            weighted_expert_action_nll(np.array([1.0, 2.0]), weights=np.array([1.0]))

    def test_weighted_nll_fails_on_negative_weights(self) -> None:
        """Negative weights must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="non-negative"):
            weighted_expert_action_nll(np.array([1.0, 2.0]), weights=np.array([-1.0, 1.0]))

    def test_weighted_nll_fails_on_zero_total_weight(self) -> None:
        """Zero total weight must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="must be > 0"):
            weighted_expert_action_nll(np.array([1.0, 2.0]), weights=np.array([0.0, 0.0]))

    def test_weighted_nll_fails_on_non_finite_weights(self) -> None:
        """Non-finite weights must fail closed."""
        with pytest.raises(ProgressWeightedBcError, match="non-finite"):
            weighted_expert_action_nll(np.array([1.0, 2.0]), weights=np.array([1.0, np.inf]))


# ---------------------------------------------------------------------------
# Rectangular action alignment
# ---------------------------------------------------------------------------


class TestRectangularAlignment:
    """Alignment validation must handle rectangular (episodes, steps, dim) arrays."""

    def test_rectangular_actions_alignment_succeeds(self, tmp_path: Path) -> None:
        """Rectangular actions array must produce correct per-episode counts."""
        from robot_sf.training.progress_weighted_bc import _count_action_episodes

        # 3 episodes, 5 steps each, 2-dim actions
        actions = np.zeros((3, 5, 2), dtype=np.float32)
        counts = _count_action_episodes(actions)
        assert counts == [5, 5, 5]

    def test_alignment_with_rectangular_actions(self, tmp_path: Path) -> None:
        """Remaining route length must align with rectangular actions in NPZ."""
        path = tmp_path / "data.npz"
        # 2 episodes, 4 actions each -> need 5 remaining values per episode
        rl = np.array(
            [
                [10.0, 8.0, 6.0, 4.0, 2.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )
        actions = np.zeros((2, 4, 2), dtype=np.float32)
        np.savez(
            path,
            remaining_route_length=rl,
            actions=actions,
            remaining_route_length_metadata=_ROUTE_LENGTH_PROVENANCE,
        )
        result = load_remaining_route_length_from_npz(path)
        assert len(result["remaining_route_length"]) == 2
        assert result["remaining_route_length"][0].shape == (5,)
        assert result["remaining_route_length"][1].shape == (5,)

    def test_ragged_actions_alignment_succeeds(self, tmp_path: Path) -> None:
        """Ragged (object) actions array must produce correct per-episode counts."""
        from robot_sf.training.progress_weighted_bc import _count_action_episodes

        actions = np.empty(2, dtype=object)
        actions[0] = np.zeros((3, 2), dtype=np.float32)
        actions[1] = np.zeros((5, 2), dtype=np.float32)
        counts = _count_action_episodes(actions)
        assert counts == [3, 5]


# ---------------------------------------------------------------------------
# ProgressWeightedBCTrainer
# ---------------------------------------------------------------------------


class TestProgressWeightedBCTrainer:
    """Standalone weighted BC trainer must apply per-step weights."""

    def test_trainer_construction(self) -> None:
        """Trainer must accept valid config without error."""
        import gymnasium as gym
        from torch import nn

        from robot_sf.training.progress_weighted_bc import (
            ProgressWeightedBCTrainer,
            ProgressWeightedObjectiveConfig,
        )

        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Simple 2-layer policy
        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._net = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 2))
                self._log_std = nn.Parameter(torch.zeros(2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self._net(x)

            def get_distribution(self, obs: torch.Tensor) -> Any:
                from torch.distributions import Normal

                mean = self.forward(obs)
                return Normal(mean, self._log_std.exp())

            def parameters(self):  # type: ignore[override]
                return self._net.parameters()

        import torch

        policy = _Policy()
        cfg = ProgressWeightedObjectiveConfig.arm_a()

        # Create a minimal trajectory-like object
        class _Traj:
            obs = np.zeros((5, 4), dtype=np.float32)
            acts = np.zeros((4, 2), dtype=np.float32)

        trainer = ProgressWeightedBCTrainer(
            observation_space=obs_space,
            action_space=act_space,
            demonstrations=[_Traj()],
            policy=policy,
            config=cfg,
            batch_size=2,
            rng=np.random.default_rng(42),
        )
        assert trainer._n_updates == 0

    def test_trainer_performs_updates(self) -> None:
        """Trainer must perform the expected number of gradient steps."""
        import gymnasium as gym
        from torch import nn

        from robot_sf.training.progress_weighted_bc import (
            ProgressWeightedBCTrainer,
            ProgressWeightedObjectiveConfig,
        )

        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        class _Policy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._net = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 2))
                self._log_std = nn.Parameter(torch.zeros(2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self._net(x)

            def get_distribution(self, obs: torch.Tensor) -> Any:
                from torch.distributions import Normal

                mean = self.forward(obs)
                return Normal(mean, self._log_std.exp())

            def parameters(self):  # type: ignore[override]
                return self._net.parameters()

        import torch

        policy = _Policy()
        cfg = ProgressWeightedObjectiveConfig.arm_a()

        class _Traj:
            obs = np.zeros((10, 4), dtype=np.float32)
            acts = np.zeros((9, 2), dtype=np.float32)

        trainer = ProgressWeightedBCTrainer(
            observation_space=obs_space,
            action_space=act_space,
            demonstrations=[_Traj()],
            policy=policy,
            config=cfg,
            batch_size=4,
            rng=np.random.default_rng(42),
        )
        trainer.train(n_epochs=2)
        # 9 samples, batch_size=4 -> 3 batches per epoch -> 6 total updates
        assert trainer._n_updates == 6
