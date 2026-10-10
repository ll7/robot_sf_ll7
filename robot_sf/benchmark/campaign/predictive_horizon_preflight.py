"""Check effective runtime predictive forecast windows before campaign execution.

This is checkpoint/config compatibility proof, not a planning-step or campaign receipt.
PredictionPlannerAdapter bounds adaptive look-ahead to the forecast; MPPI requires its
entire authored sequence horizon. Neither planner's configuration is rewritten here.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
)
from robot_sf.models import resolve_model_path
from robot_sf.planner.obstacle_features import (
    infer_predictive_feature_schema,
    validate_predictive_feature_schema_metadata,
)
from robot_sf.planner.socnav_base import _SOCNAV_CONFIG_INIT_KEYS, SocNavPlannerConfig

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

_PREDICTIVE_ALGOS = frozenset({"predictive_mppi", "prediction_planner", "gap_prediction"})
_PREDICTIVE_CONFIG_DEFAULT_HORIZON_STEPS = 8
_PREDICTIVE_MODEL_DEFAULT_INPUT_DIM = 4


class _TensorPayloadPlaceholder:
    """Placeholder for tensor payloads while reading torch-save metadata only."""


def _tensor_payload_placeholder(*_args: Any, **_kwargs: Any) -> _TensorPayloadPlaceholder:
    """Return a sentinel for torch tensor rebuild calls in pickle metadata."""
    return _TensorPayloadPlaceholder()


class _PredictiveCheckpointMetadataUnpickler(pickle.Unpickler):
    """Restricted unpickler for the metadata dict stored by ``torch.save`` checkpoints."""

    _ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch._utils", "_rebuild_tensor"): _tensor_payload_placeholder,
        ("torch._utils", "_rebuild_tensor_v2"): _tensor_payload_placeholder,
        ("torch._utils", "_rebuild_tensor_v3"): _tensor_payload_placeholder,
        ("torch._utils", "_rebuild_parameter"): _tensor_payload_placeholder,
    }

    def find_class(self, module: str, name: str) -> Any:
        """Allow only inert helpers needed to deserialize torch-save metadata.

        Returns:
            Callable or placeholder class used while unpickling metadata.
        """
        key = (module, name)
        if key in self._ALLOWED_GLOBALS:
            return self._ALLOWED_GLOBALS[key]
        if module == "torch" and name.endswith("Storage"):
            return _TensorPayloadPlaceholder
        if module == "torch.storage" and name == "_load_from_bytes":
            return _tensor_payload_placeholder
        raise pickle.UnpicklingError(
            f"blocked global while reading checkpoint metadata: {module}.{name}"
        )

    def persistent_load(self, _pid: Any) -> _TensorPayloadPlaceholder:
        """Replace serialized tensor storage references with inert sentinels.

        Returns:
            Placeholder standing in for an uninterpreted tensor storage.
        """
        return _TensorPayloadPlaceholder()


class PredictiveHorizonPreflightError(CampaignCheckpointPreflightError):
    """A predictive refusal with structured binding evidence for the matrix census."""

    def __init__(self, message: str, *, binding: dict[str, Any]) -> None:
        """Retain binding identity and verification status without parsing error prose."""
        super().__init__(message, arms=(binding["planner_key"],))
        self.binding = binding


def _load_torch_save_metadata(path: Path) -> dict[str, Any]:
    """Read the top-level metadata dict from a zip-format torch checkpoint without torch.

    Returns:
        The top-level checkpoint payload with tensor values replaced by placeholders.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Predictive checkpoint not found: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Predictive checkpoint is not a zip-format torch checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        try:
            data_member = next(
                name
                for name in archive.namelist()
                if name == "data.pkl" or name.endswith("/data.pkl")
            )
        except StopIteration as exc:
            raise ValueError(
                f"Predictive checkpoint has no data.pkl metadata member: {path}"
            ) from exc
        with archive.open(data_member) as handle:
            payload = _PredictiveCheckpointMetadataUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise TypeError(f"Predictive checkpoint metadata must be a mapping: {path}")
    return payload


def _checkpoint_forecast_steps(
    path: Path,
    *,
    expected_feature_schema_name: str | None,
) -> int:
    """Return saved forecast horizon from checkpoint metadata without importing torch."""
    payload = _load_torch_save_metadata(path)
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise TypeError(f"Predictive checkpoint config must be a mapping: {path}")
    input_dim = int(config.get("input_dim", _PREDICTIVE_MODEL_DEFAULT_INPUT_DIM))
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, dict):
        feature_schema = infer_predictive_feature_schema(input_dim)
    validate_predictive_feature_schema_metadata(
        feature_schema,
        input_dim=input_dim,
        expected_schema_name=expected_feature_schema_name,
    )
    return int(config.get("horizon_steps", _PREDICTIVE_CONFIG_DEFAULT_HORIZON_STEPS))


def _predictive_planner_requirements(raw: dict[str, Any]) -> tuple[SocNavPlannerConfig, int]:
    """Build only the torch-free planner fields needed for horizon compatibility.

    Returns:
        SocNav predictive config plus the MPPI sequence horizon.
    """
    socnav_allowed = {
        field.name for field in fields(SocNavPlannerConfig)
    } | _SOCNAV_CONFIG_INIT_KEYS
    socnav = SocNavPlannerConfig(
        **{key: value for key, value in raw.items() if key in socnav_allowed}
    )
    return socnav, int(raw.get("horizon_steps", _PREDICTIVE_CONFIG_DEFAULT_HORIZON_STEPS))


def _effective_configs(
    cfg: CampaignConfig, planner: PlannerSpec
) -> list[tuple[list[dict[str, str]], str, dict[str, Any]]]:
    """Resolve every selected scenario with the same loader and resolver as the runtime.

    Expand includes and scenario metadata overrides, then apply the campaign candidate
    selector. Never resolve the campaign seed policy or instantiate a planner. Context
    independent configs need only one check. Equal effective configs share a checkpoint
    check, retaining every covered scenario/family identity in its evidence.

    Returns:
        Covered contexts, resolved algorithm, and effective planner configuration.
    """
    from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415
        _filter_scenario_candidates,
    )
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (  # noqa: PLC0415
        _parse_algo_config,
        _resolve_policy_search_candidate_runtime,
    )
    from robot_sf.benchmark.policy_search_manifest import (  # noqa: PLC0415
        _scenario_id,
        scenario_family,
    )
    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    path = str(planner.algo_config_path) if planner.algo_config_path else None
    raw = _parse_algo_config(path)
    has_context_overrides = any(
        raw.get(key)
        for key in ("scenario_algo_overrides", "scenario_overrides", "family_overrides")
    )
    scenarios = [{}]
    if has_context_overrides:
        scenarios = _filter_scenario_candidates(
            [
                dict(s)
                for s in load_scenarios(
                    cfg.scenario_matrix_path, base_dir=cfg.scenario_matrix_path.parent
                )
            ],
            names=cfg.scenario_candidates.names,
            matrix_path=cfg.scenario_matrix_path,
        )
        if not scenarios:
            raise CampaignCheckpointPreflightError(
                f"Predictive horizon preflight cannot resolve arm '{planner.key}', "
                f"config='{path}': no selected scenario contexts in '{cfg.scenario_matrix_path}'.",
                arms=(planner.key,),
            )
    resolved = {}
    for scenario in scenarios:
        algo, config = _resolve_policy_search_candidate_runtime(
            default_algo=planner.algo,
            algo_config_path=path,
            algo_config=raw,
            scenario=scenario,
        )
        algo = algo.strip().lower()
        context = {
            "scenario": _scenario_id(scenario) if has_context_overrides else "base",
            "family": scenario_family(scenario) if has_context_overrides else "all",
        }
        key = (algo, json.dumps(config, sort_keys=True, default=str))
        if key not in resolved:
            resolved[key] = ([], algo, config)
        resolved[key][0].append(context)
    return list(resolved.values())


def _check_binding(
    planner: PlannerSpec,
    contexts: list[dict[str, str]],
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

    predictor, sequence_horizon_steps = _predictive_planner_requirements(raw)
    # PredictionPlannerAdapter gives an explicit path priority over the registered ID.
    checkpoint_ref = predictor.predictive_checkpoint_path or predictor.predictive_model_id
    scenario, family = contexts[0]["scenario"], contexts[0]["family"]
    binding = {
        "planner_key": planner.key,
        "algo": algo,
        "scenario": scenario,
        "family": family,
        "contexts": contexts,
        "algo_config_path": str(planner.algo_config_path),
        "checkpoint": str(checkpoint_ref),
    }
    identity = (
        f"arm '{planner.key}' (algo={algo}, scenario={scenario}, family={family}), "
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
        binding["resolved_path"] = str(checkpoint)
        forecast_steps = _checkpoint_forecast_steps(
            checkpoint,
            expected_feature_schema_name=predictor.predictive_feature_schema_name,
        )
    except (KeyError, OSError, RuntimeError, ValueError, TypeError) as exc:
        binding.update(
            status="unverified",
            unverified_reason="missing_checkpoint"
            if isinstance(exc, (FileNotFoundError, KeyError))
            else "checkpoint_error",
        )
        raise PredictiveHorizonPreflightError(
            f"Predictive horizon preflight cannot verify {identity}: {exc}. "
            "Stage the declared checkpoint before submission; no forecast window was admitted.",
            binding=binding,
        ) from exc

    if algo == "predictive_mppi":
        required_steps = int(sequence_horizon_steps)
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
    binding.update(required_horizon_steps=required_steps, forecast_steps=forecast_steps)
    if not 1 <= required_steps <= forecast_steps:
        binding["status"] = "incompatible"
        raise PredictiveHorizonPreflightError(
            f"Predictive horizon preflight failed for {identity}: "
            f"required_horizon_steps={required_steps}, forecast_steps={forecast_steps}. "
            "Aborting before campaign startup/submission; use a reviewed compatible binding.",
            binding=binding,
        )
    return {**binding, "status": "compatible"}


def check_campaign_predictive_horizons_preflight(
    cfg: CampaignConfig,
    *,
    stage: bool = False,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Verify every enabled effective learned-predictor binding without executing scenarios.

    Constant-velocity/probabilistic baseline forecast variants have no learned checkpoint
    window and are outside this checkpoint guard. Other predictor model families retain
    their own checkpoint contracts.

    Returns:
        Compatibility records, including all covered scenario/family contexts.
    """
    results = []
    for planner in cfg.planners:
        if not planner.enabled:
            continue
        for contexts, algo, raw in _effective_configs(cfg, planner):
            variant = str(raw.get("forecast_variant") or "none").strip().lower() or "none"
            if algo not in _PREDICTIVE_ALGOS or variant != "none":
                continue
            results.append(
                _check_binding(
                    planner,
                    contexts,
                    algo,
                    raw,
                    stage=stage,
                    registry_path=registry_path,
                    cache_dir=cache_dir,
                )
            )
    return results
