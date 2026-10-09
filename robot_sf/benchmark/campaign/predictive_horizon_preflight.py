"""Check runtime predictive forecast windows before scenarios load or sbatch runs.

This is checkpoint/config compatibility proof, not a planning-step or campaign receipt.
PredictionPlannerAdapter bounds adaptive look-ahead to the forecast; MPPI requires its
entire authored sequence horizon. Neither planner's configuration is rewritten here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
)
from robot_sf.models import resolve_model_path
from robot_sf.planner.predictive_mppi import build_predictive_mppi_config

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

_PREDICTIVE_ALGOS = frozenset({"predictive_mppi", "prediction_planner", "gap_prediction"})


def _effective_configs(planner: PlannerSpec) -> list[tuple[str, str, dict[str, Any]]]:
    """Resolve the base and every authored scenario override with the runtime resolver.

    Returns:
        Tuples of scenario selector, resolved algorithm, and effective planner configuration.
    """
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (  # noqa: PLC0415
        _parse_algo_config,
        _resolve_policy_search_candidate_runtime,
    )

    path = str(planner.algo_config_path) if planner.algo_config_path else None
    raw = _parse_algo_config(path)
    selectors = [None, *raw.get("scenario_algo_overrides", {})]
    resolved = []
    for selector in selectors:
        algo, config = _resolve_policy_search_candidate_runtime(
            default_algo=planner.algo,
            algo_config_path=path,
            algo_config=raw,
            scenario={"name": selector} if selector is not None else {},
        )
        resolved.append((str(selector or "base"), algo.strip().lower(), config))
    return resolved


def _check_binding(
    planner: PlannerSpec,
    scenario: str,
    algo: str,
    raw: dict[str, Any],
    *,
    stage: bool,
    registry_path: str | Path | None,
    cache_dir: str | Path | None,
) -> dict[str, Any]:
    """Load the real checkpoint through the runtime loader and verify one resolved binding.

    Returns:
        Arm/config/checkpoint identity and actual required/available forecast step counts.

    Raises:
        CampaignCheckpointPreflightError: When checkpoint loading or horizon compatibility fails.
    """
    from pathlib import Path  # noqa: PLC0415

    from robot_sf.planner.predictive_model import load_predictive_checkpoint  # noqa: PLC0415

    config = build_predictive_mppi_config(raw)
    predictor = config.socnav
    # PredictionPlannerAdapter gives an explicit path priority over the registered ID.
    checkpoint_ref = predictor.predictive_checkpoint_path or predictor.predictive_model_id
    identity = (
        f"arm '{planner.key}' (algo={algo}, scenario={scenario}), "
        f"config='{planner.algo_config_path}', checkpoint='{checkpoint_ref}'"
    )
    try:
        checkpoint = (
            Path(predictor.predictive_checkpoint_path).expanduser()
            if predictor.predictive_checkpoint_path
            else resolve_model_path(
                predictor.predictive_model_id,
                registry_path=registry_path,
                cache_dir=cache_dir,
                allow_download=stage,
            )
        )
        model, _metadata = load_predictive_checkpoint(
            checkpoint,
            map_location="cpu",
            expected_feature_schema_name=predictor.predictive_feature_schema_name,
        )
        # The runtime loader validates state_dict shapes against the saved model config;
        # a planner YAML horizon override cannot change the checkpoint output head.
        forecast_steps = int(model.config.horizon_steps)
    except (KeyError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise CampaignCheckpointPreflightError(
            f"Predictive horizon preflight cannot verify {identity}: {exc}. "
            "Stage the declared checkpoint before submission; no forecast window was admitted.",
            arms=(planner.key,),
        ) from exc

    if algo == "predictive_mppi":
        required_steps = int(config.horizon_steps)
    else:
        # Match PredictionPlannerAdapter._effective_rollout_steps, including the existing
        # forecast-bound adaptive boost. This boost does not alter MPPI's sequence length.
        base = max(1, int(predictor.predictive_horizon_steps))
        boost = (
            max(0, int(predictor.predictive_horizon_boost_steps))
            if predictor.predictive_adaptive_horizon_enabled
            else 0
        )
        required_steps = min(base + boost, forecast_steps)
    if not 1 <= required_steps <= forecast_steps:
        raise CampaignCheckpointPreflightError(
            f"Predictive horizon preflight failed for {identity}: "
            f"required_horizon_steps={required_steps}, forecast_steps={forecast_steps}. "
            "Aborting before campaign startup/submission; use a reviewed compatible binding.",
            arms=(planner.key,),
        )
    return {
        "planner_key": planner.key,
        "algo": algo,
        "scenario": scenario,
        "algo_config_path": str(planner.algo_config_path),
        "checkpoint": str(checkpoint_ref),
        "resolved_path": str(checkpoint),
        "required_horizon_steps": required_steps,
        "forecast_steps": forecast_steps,
        "status": "compatible",
    }


def check_campaign_predictive_horizons_preflight(
    cfg: CampaignConfig,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Verify every enabled learned-predictor binding without loading scenarios or seeds.

    Constant-velocity/probabilistic baseline forecast variants have no learned checkpoint
    window and are outside this checkpoint guard. Other predictor model families retain
    their own checkpoint contracts.

    Returns:
        Compatibility records for every checked base or scenario-specific binding.
    """
    results = []
    for planner in cfg.planners:
        if not planner.enabled:
            continue
        for scenario, algo, raw in _effective_configs(planner):
            variant = str(raw.get("forecast_variant") or "none").strip().lower() or "none"
            if algo not in _PREDICTIVE_ALGOS or variant != "none":
                continue
            results.append(
                _check_binding(
                    planner,
                    scenario,
                    algo,
                    raw,
                    stage=stage,
                    registry_path=registry_path,
                    cache_dir=cache_dir,
                )
            )
    return results
