"""Programmable adversarial scenario search helpers."""

from robot_sf.adversarial.archive import curate_failure_archive
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    MultiPedAdversarialConfig,
    MultiPedCandidateSpec,
    Pose2D,
    SearchConfig,
    SearchRunResult,
    SearchSpaceConfig,
)
from robot_sf.adversarial.materialize import (
    materialize_multi_ped_scenario_payload,
    materialize_multi_ped_single_pedestrian_overrides,
)
from robot_sf.adversarial.runtime import (
    build_multi_ped_adversarial_robot_config,
    multi_ped_config_to_single_pedestrian_definitions,
    validate_multi_ped_runtime_plausibility,
)
from robot_sf.adversarial.samplers import (
    CoordinateRefinementSampler,
    OptunaCandidateSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.search import run_adversarial_search
from robot_sf.adversarial.seed_sensitivity import (
    SeedSensitivityReplay,
    SeedSensitivitySummary,
    run_seed_sensitivity,
)

__all__ = [
    "CandidateEvaluation",
    "CandidateSpec",
    "CoordinateRefinementSampler",
    "MultiPedAdversarialConfig",
    "MultiPedCandidateSpec",
    "OptunaCandidateSampler",
    "Pose2D",
    "RandomCandidateSampler",
    "SearchConfig",
    "SearchRunResult",
    "SearchSpaceConfig",
    "SeedSensitivityReplay",
    "SeedSensitivitySummary",
    "build_multi_ped_adversarial_robot_config",
    "curate_failure_archive",
    "materialize_multi_ped_scenario_payload",
    "materialize_multi_ped_single_pedestrian_overrides",
    "multi_ped_config_to_single_pedestrian_definitions",
    "run_adversarial_search",
    "run_seed_sensitivity",
    "validate_multi_ped_runtime_plausibility",
]
