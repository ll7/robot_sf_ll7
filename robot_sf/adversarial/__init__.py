"""Programmable adversarial scenario search helpers."""

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
from robot_sf.adversarial.scenario_manifest import (
    AdversarialScenarioManifest,
    GeneratorInfo,
    ManifestCategory,
    SourceLineage,
    ValidationRecord,
    build_manifest,
    compute_control_hash,
    generate_manifests,
    validate_candidate_manifest,
    validate_manifest_payload,
    write_manifest_yaml,
)
from robot_sf.adversarial.search import run_adversarial_search
from robot_sf.adversarial.seed_sensitivity import (
    SeedSensitivityPerturbation,
    SeedSensitivityReplay,
    SeedSensitivitySummary,
    run_seed_sensitivity,
)

__all__ = [
    "AdversarialScenarioManifest",
    "CandidateEvaluation",
    "CandidateSpec",
    "CoordinateRefinementSampler",
    "GeneratorInfo",
    "ManifestCategory",
    "MultiPedAdversarialConfig",
    "MultiPedCandidateSpec",
    "OptunaCandidateSampler",
    "Pose2D",
    "RandomCandidateSampler",
    "SearchConfig",
    "SearchRunResult",
    "SearchSpaceConfig",
    "SeedSensitivityPerturbation",
    "SeedSensitivityReplay",
    "SeedSensitivitySummary",
    "SourceLineage",
    "ValidationRecord",
    "build_manifest",
    "build_multi_ped_adversarial_robot_config",
    "compute_control_hash",
    "generate_manifests",
    "materialize_multi_ped_scenario_payload",
    "materialize_multi_ped_single_pedestrian_overrides",
    "multi_ped_config_to_single_pedestrian_definitions",
    "run_adversarial_search",
    "run_seed_sensitivity",
    "validate_candidate_manifest",
    "validate_manifest_payload",
    "validate_multi_ped_runtime_plausibility",
    "write_manifest_yaml",
]
