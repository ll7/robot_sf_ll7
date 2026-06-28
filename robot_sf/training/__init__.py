"""Training utilities shared across CLI workflows.

The package initializer keeps re-exports lazy so benchmark and scenario-loader
imports do not pull optional analysis or torch-backed training dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "DEFAULT_RAY_RUNTIME_ENV_EXCLUDES": ("robot_sf.training.runtime_helpers",),
    "DEFAULT_TMP_ROOT": ("robot_sf.training.multi_extractor_paths",),
    "ENV_TMP_OVERRIDE": ("robot_sf.training.multi_extractor_paths",),
    "BehaviouralCloningConfig": ("robot_sf.training.imitation_config",),
    "ConvergenceCriteria": ("robot_sf.training.imitation_config",),
    "EvaluationSchedule": ("robot_sf.training.imitation_config",),
    "ExpertTrainingConfig": ("robot_sf.training.imitation_config",),
    "ExtractorConfigurationProfile": ("robot_sf.training.multi_extractor_models",),
    "ExtractorRunRecord": ("robot_sf.training.multi_extractor_models",),
    "HardwareProfile": ("robot_sf.training.multi_extractor_models",),
    "PPOFineTuneConfig": ("robot_sf.training.imitation_config",),
    "ScenarioSampler": ("robot_sf.training.scenario_sampling",),
    "ScenarioSwitchingEnv": ("robot_sf.training.scenario_sampling",),
    "TrainingRunSummary": ("robot_sf.training.multi_extractor_models",),
    "TrajectoryCollectionConfig": ("robot_sf.training.imitation_config",),
    "analyze_imitation_results": ("robot_sf.training.imitation_analysis",),
    "append_jsonl_record": ("robot_sf.training.runtime_helpers",),
    "collect_hardware_profile": ("robot_sf.training.hardware_probe",),
    "convergence_timestep": ("robot_sf.training.multi_extractor_analysis",),
    "generate_figures": ("robot_sf.training.multi_extractor_analysis",),
    "load_eval_history": ("robot_sf.training.multi_extractor_analysis",),
    "make_extractor_directory": ("robot_sf.training.multi_extractor_paths",),
    "make_run_directory": ("robot_sf.training.multi_extractor_paths",),
    "resolve_base_output_root": ("robot_sf.training.multi_extractor_paths",),
    "resolve_ray_runtime_env": ("robot_sf.training.runtime_helpers",),
    "sample_efficiency_ratio": ("robot_sf.training.multi_extractor_analysis",),
    "scenario_id_from_definition": ("robot_sf.training.scenario_sampling",),
    "summary_paths": ("robot_sf.training.multi_extractor_paths",),
    "summarize_metric": ("robot_sf.training.multi_extractor_analysis",),
    "write_summary_artifacts": ("robot_sf.training.multi_extractor_summary",),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve compatibility re-exports without eager optional imports.

    Returns:
        Any: Exported symbol loaded from its owner module.
    """

    try:
        (module_name,) = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
