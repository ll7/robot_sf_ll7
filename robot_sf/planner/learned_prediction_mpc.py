"""Learned-prediction MPC planner construction helpers."""

from __future__ import annotations

from typing import Any

from robot_sf.planner.learned_short_horizon_predictor import (
    LearnedShortHorizonPedestrianPredictor,
    build_learned_short_horizon_predictor_config,
)
from robot_sf.planner.prediction_mpc import (
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

LEARNED_PREDICTION_MPC_ALIASES = frozenset(
    {
        "learned_prediction_mpc",
        "learned_short_horizon_mpc",
        "model_based_local_planner",
        "learned_prediction_planner",
    }
)


def build_learned_prediction_mpc_adapter(
    algo_config: dict[str, Any] | None,
) -> PredictionMPCPlannerAdapter:
    """Build prediction MPC with a learned short-horizon pedestrian predictor.

    Returns:
        PredictionMPCPlannerAdapter: Adapter wired to the learned predictor backend.
    """

    raw = dict(algo_config or {})
    prediction_config = build_prediction_mpc_config(
        {
            **raw,
            "predictor_backend": "learned_short_horizon",
        }
    )
    predictor_config = build_learned_short_horizon_predictor_config(raw)
    mpc_config = PredictionMPCConfig(
        **{
            **prediction_config.__dict__,
            "horizon_steps": predictor_config.horizon_steps,
            "rollout_dt": predictor_config.rollout_dt,
            "predictor_backend": "learned_short_horizon",
        }
    )
    predictor = LearnedShortHorizonPedestrianPredictor(predictor_config)
    return PredictionMPCPlannerAdapter(config=mpc_config, predictor=predictor)
