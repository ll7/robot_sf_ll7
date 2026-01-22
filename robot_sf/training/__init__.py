"""Training utilities shared across CLI workflows."""

from .hardware_probe import collect_hardware_profile
from .imitation_analysis import analyze_imitation_results
from .imitation_config import (
    BehaviouralCloningConfig,
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
    PPOFineTuneConfig,
    TrajectoryCollectionConfig,
)
from .multi_extractor_analysis import (
    convergence_timestep,
    generate_figures,
    load_eval_history,
    sample_efficiency_ratio,
    summarize_metric,
)
from .multi_extractor_models import (
    ExtractorConfigurationProfile,
    ExtractorRunRecord,
    HardwareProfile,
    TrainingRunSummary,
)
from .multi_extractor_paths import (
    DEFAULT_TMP_ROOT,
    ENV_TMP_OVERRIDE,
    make_extractor_directory,
    make_run_directory,
    resolve_base_output_root,
    summary_paths,
)
from .multi_extractor_summary import write_summary_artifacts
from .scenario_sampling import (
    ScenarioSampler,
    ScenarioSwitchingEnv,
    scenario_id_from_definition,
)

__all__ = [
    "DEFAULT_TMP_ROOT",
    "ENV_TMP_OVERRIDE",
    "BehaviouralCloningConfig",
    "ConvergenceCriteria",
    "EvaluationSchedule",
    "ExpertTrainingConfig",
    "ExtractorConfigurationProfile",
    "ExtractorRunRecord",
    "HardwareProfile",
    "PPOFineTuneConfig",
    "ScenarioSampler",
    "ScenarioSwitchingEnv",
    "TrainingRunSummary",
    "TrajectoryCollectionConfig",
    "analyze_imitation_results",
    "collect_hardware_profile",
    "convergence_timestep",
    "generate_figures",
    "load_eval_history",
    "make_extractor_directory",
    "make_run_directory",
    "resolve_base_output_root",
    "sample_efficiency_ratio",
    "scenario_id_from_definition",
    "summarize_metric",
    "summary_paths",
    "write_summary_artifacts",
]
