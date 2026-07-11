"""Contract tests for issue #5307's GMM chance-constrained MPC planner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.planner.chance_constrained_mpc import (
    ChanceConstrainedMPCConfig,
    ChanceConstrainedMPCPlannerAdapter,
    GaussianMixturePedestrianForecast,
    build_chance_constrained_mpc_adapter,
    build_chance_constrained_mpc_config,
)


class _FixedGmmPredictor:
    """Small injected provider used to exercise the public predictor boundary."""

    def __init__(self, forecast: GaussianMixturePedestrianForecast) -> None:
        self.forecast = forecast
        self.calls: list[tuple[int, float]] = []

    def predict(
        self,
        observation: dict,
        *,
        horizon_steps: int,
        dt: float,
    ) -> GaussianMixturePedestrianForecast:
        """Return the fixed contract-valid forecast and record the request."""

        del observation
        self.calls.append((horizon_steps, dt))
        return self.forecast


def _forecast(
    *, horizon_steps: int = 2, collision_weight: float = 0.2
) -> GaussianMixturePedestrianForecast:
    """Build one pedestrian with one colliding and one safe Gaussian mode."""

    means = np.zeros((1, 2, horizon_steps, 2), dtype=float)
    means[:, 1, :, 0] = 4.0
    covariances = np.tile(np.eye(2, dtype=float) * 0.05, (1, 2, horizon_steps, 1, 1))
    return GaussianMixturePedestrianForecast(
        means_world=means,
        covariances_world=covariances,
        mode_weights=np.asarray([[collision_weight, 1.0 - collision_weight]], dtype=float),
        dt=0.25,
        source="test_gmm",
    )


def _context() -> SimpleNamespace:
    """Build the minimum NMPC rollout context for direct constraint tests."""

    return SimpleNamespace(
        robot_pos=np.asarray([0.0, 0.0]),
        heading=0.0,
        current_speed=0.0,
        goal=np.asarray([3.0, 0.0]),
        ped_positions=np.zeros((0, 2)),
        ped_velocities=np.zeros((0, 2)),
        robot_radius=0.25,
        ped_radius=0.25,
        observation={},
        speed_cap=0.9,
    )


def _observation() -> dict:
    """Build the compact SocNav observation required by one real planner call."""

    return {
        "robot": {
            "position": np.asarray([0.0, 0.0]),
            "heading": np.asarray([0.0]),
            "speed": np.asarray([0.0]),
            "radius": np.asarray([0.25]),
        },
        "goal": {"current": np.asarray([3.0, 0.0]), "next": np.asarray([3.0, 0.0])},
        "pedestrians": {
            "positions": np.asarray([[0.0, 0.0]]),
            "velocities": np.asarray([[0.0, 0.0]]),
            "count": np.asarray([1.0]),
            "radius": np.asarray([0.25]),
        },
    }


def test_forecast_rejects_non_normalized_mode_weights() -> None:
    """A malformed mixture cannot silently change the requested risk level."""

    with pytest.raises(ValueError, match="sum to one"):
        GaussianMixturePedestrianForecast(
            means_world=np.zeros((1, 1, 1, 2)),
            covariances_world=np.tile(np.eye(2), (1, 1, 1, 1, 1)),
            mode_weights=np.asarray([[0.5]]),
            dt=0.25,
            source="invalid",
        )


@pytest.mark.parametrize("source", [None, "", "   "])
def test_forecast_rejects_missing_or_non_string_source(source: object) -> None:
    """A required provenance source must not stringify a missing value as valid."""

    with pytest.raises(ValueError, match="source must be a non-empty string"):
        GaussianMixturePedestrianForecast(
            means_world=np.zeros((1, 1, 1, 2)),
            covariances_world=np.tile(np.eye(2), (1, 1, 1, 1, 1)),
            mode_weights=np.asarray([[1.0]]),
            dt=0.25,
            source=source,  # type: ignore[arg-type]
        )


def test_marginal_constraint_rejects_collision_probability_above_alpha() -> None:
    """Per-timestep GMM collision risk must stay below the configured alpha."""

    forecast = _forecast(collision_weight=0.2)
    planner = ChanceConstrainedMPCPlannerAdapter(
        ChanceConstrainedMPCConfig(
            horizon_steps=2,
            max_collision_risk=0.1,
            radial_quadrature_order=10,
            angular_quadrature_order=24,
        ),
        predictor=_FixedGmmPredictor(forecast),
    )

    slack = planner._chance_constraint_values(np.zeros(4), context=_context(), forecast=forecast)

    assert slack.shape == (2,)
    assert np.all(slack < 0.0)


def test_joint_horizon_constraint_uses_conservative_union_bound() -> None:
    """Joint mode rejects two individually acceptable risks whose union bound exceeds alpha."""

    forecast = _forecast(collision_weight=0.1)
    planner = ChanceConstrainedMPCPlannerAdapter(
        ChanceConstrainedMPCConfig(
            horizon_steps=2,
            chance_constraint_formulation="joint_horizon",
            max_collision_risk=0.15,
            radial_quadrature_order=10,
            angular_quadrature_order=24,
        ),
        predictor=_FixedGmmPredictor(forecast),
    )

    risks = planner._marginal_collision_risks(np.zeros(4), context=_context(), forecast=forecast)
    slack = planner._chance_constraint_values(np.zeros(4), context=_context(), forecast=forecast)

    assert np.all(risks < 0.15)
    assert slack[0] < 0.0


def test_optimizer_uses_injected_predictor_and_records_claim_boundary() -> None:
    """The planner queries the injected GMM provider and exposes non-claim diagnostics."""

    forecast = _forecast()
    predictor = _FixedGmmPredictor(forecast)
    planner = ChanceConstrainedMPCPlannerAdapter(
        ChanceConstrainedMPCConfig(horizon_steps=2), predictor=predictor
    )

    constraints = planner._optimizer_constraints(_context())
    diagnostics = planner.diagnostics()["chance_constraint"]

    assert len(constraints) == 1
    assert predictor.calls == [(2, 0.25)]
    assert diagnostics["forecast_source"] == "test_gmm"
    assert diagnostics["forecast_modes"] == 2
    assert "implementation-only" in diagnostics["claim_boundary"]


def test_planner_executes_a_real_constrained_control_step() -> None:
    """The experimental adapter runs through the repository NMPC path with a GMM input."""

    forecast = _forecast(collision_weight=0.1)
    predictor = _FixedGmmPredictor(forecast)
    planner = ChanceConstrainedMPCPlannerAdapter(
        ChanceConstrainedMPCConfig(
            horizon_steps=2,
            max_collision_risk=0.2,
            solver_max_iterations=20,
            radial_quadrature_order=6,
            angular_quadrature_order=16,
        ),
        predictor=predictor,
    )

    command = planner.plan(_observation())

    assert np.all(np.isfinite(command))
    assert 0.0 <= command[0] <= 0.9
    assert abs(command[1]) <= 1.1
    assert predictor.calls == [(2, 0.25)]


def test_adapter_and_registered_builder_fail_closed_without_predictor() -> None:
    """No unavailable learned predictor may degrade to a deterministic forecast."""

    with pytest.raises(ValueError, match="requires an injected"):
        ChanceConstrainedMPCPlannerAdapter()
    with pytest.raises(ValueError, match="unavailable"):
        build_chance_constrained_mpc_adapter({"predictor_backend": "issue_2844_k_mode_gmm"})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"allow_predictor_fallback": True}, "never permits predictor fallback"),
        ({"pedestrian_uncertainty_envelope_enabled": True}, "does not compose"),
    ],
)
def test_config_rejects_incompatible_matched_arm_shortcuts(
    overrides: dict[str, bool], message: str
) -> None:
    """Chance constraints must not silently reuse a baseline or envelope arm."""

    with pytest.raises(ValueError, match=message):
        ChanceConstrainedMPCConfig(**overrides)


def test_config_parser_and_example_preserve_the_gmm_arm_boundary() -> None:
    """The shipped YAML records the provider dependency without enabling fallback."""

    config_path = Path(__file__).resolve().parents[2] / "configs" / "algos"
    config_path /= "chance_constrained_mpc_gmm.yaml"
    config = build_chance_constrained_mpc_config(yaml.safe_load(config_path.read_text()))

    assert config.predictor_backend == "issue_2844_k_mode_gmm"
    assert config.allow_predictor_fallback is False
    assert config.chance_constraint_formulation == "marginal"
    assert config.max_collision_risk == pytest.approx(0.05)
