"""Programmable adversarial scenario search helpers."""

from typing import Any

from robot_sf.adversarial.batch_certification import (
    ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA,
    BatchCertification,
    BatchCertificationPolicy,
    CandidateCertification,
    certify_candidate_batch,
    certify_records,
)
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
from robot_sf.adversarial.manifest_quality import (
    MANIFEST_QUALITY_SCHEMA_VERSION,
    ManifestsQualitySummary,
    PlannerOutcome,
    PlannerOutcomeSummary,
    load_adversarial_manifest_quality_records,
    summarize_adversarial_manifest_quality,
    summarize_adversarial_manifest_quality_records,
)
from robot_sf.adversarial.materialize import (
    materialize_manifest_route_overrides,
    materialize_manifest_scenario_payload,
    materialize_manifest_single_pedestrian_override,
    materialize_multi_ped_scenario_payload,
    materialize_multi_ped_single_pedestrian_overrides,
)
from robot_sf.adversarial.qd import (
    GridSpec,
    QDArchive,
    QDComparisonReport,
    QDSearchConfig,
    QDSearchResult,
    compare_qd_vs_single_objective,
    default_behavior_descriptor,
    production_qd_evaluator,
    run_map_elites,
    write_qd_archive,
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
from robot_sf.adversarial.search import (
    production_candidate_evaluator,
    run_adversarial_search,
)
from robot_sf.adversarial.seed_sensitivity import (
    SeedSensitivityPerturbation,
    SeedSensitivityReplay,
    SeedSensitivitySummary,
    run_seed_sensitivity,
)


def __getattr__(name: str) -> Any:
    """Load heavyweight adversarial search dependencies only when requested."""

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA",
    "MANIFEST_QUALITY_SCHEMA_VERSION",
    "AdversarialScenarioManifest",
    "BatchCertification",
    "BatchCertificationPolicy",
    "CandidateCertification",
    "CandidateEvaluation",
    "CandidateSpec",
    "CoordinateRefinementSampler",
    "GeneratorInfo",
    "GridSpec",
    "ManifestCategory",
    "ManifestsQualitySummary",
    "MultiPedAdversarialConfig",
    "MultiPedCandidateSpec",
    "OptunaCandidateSampler",
    "PlannerOutcome",
    "PlannerOutcomeSummary",
    "Pose2D",
    "QDArchive",
    "QDComparisonReport",
    "QDSearchConfig",
    "QDSearchResult",
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
    "certify_candidate_batch",
    "certify_records",
    "compare_qd_vs_single_objective",
    "compute_control_hash",
    "default_behavior_descriptor",
    "generate_manifests",
    "load_adversarial_manifest_quality_records",
    "materialize_manifest_route_overrides",
    "materialize_manifest_scenario_payload",
    "materialize_manifest_single_pedestrian_override",
    "materialize_multi_ped_scenario_payload",
    "materialize_multi_ped_single_pedestrian_overrides",
    "multi_ped_config_to_single_pedestrian_definitions",
    "production_candidate_evaluator",
    "production_qd_evaluator",
    "run_adversarial_search",
    "run_map_elites",
    "run_seed_sensitivity",
    "summarize_adversarial_manifest_quality",
    "summarize_adversarial_manifest_quality_records",
    "validate_candidate_manifest",
    "validate_manifest_payload",
    "validate_multi_ped_runtime_plausibility",
    "write_manifest_yaml",
    "write_qd_archive",
]
