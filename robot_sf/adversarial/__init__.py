"""Programmable adversarial scenario search helpers.

Public compatibility exports load on demand so report and tooling submodules can
be imported without eagerly importing optional planner and visualization stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from robot_sf.adversarial._objective_registry import install_builtin_objective_registration

install_builtin_objective_registration()

__all__ = [  # noqa: F822 - names resolve through module-level __getattr__.
    "ADMISSIBILITY_VERDICTS",
    "ADMISSIBLE_FEASIBILITY_UNKNOWN",
    "ADVERSARIAL_CANDIDATE_QUALITY_SCHEMA",
    "BASELINE_NAMES",
    "CHECK_NAMES",
    "EMPIRICALLY_FEASIBLE",
    "FEASIBILITY_FIRST_CLAIM_BOUNDARY",
    "FEASIBILITY_FIRST_EVIDENCE_TIER",
    "FEASIBILITY_FIRST_EXISTING_BASELINE_ID",
    "FEASIBILITY_FIRST_SCHEMA_VERSION",
    "GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY",
    "MANIFEST_QUALITY_SCHEMA_VERSION",
    "PLANNER_SPECIFIC_FAILURE",
    "PREPARATION_SCHEMA_VERSION",
    "SCENARIO_ADMISSIBILITY_SCHEMA",
    "SEARCH_HARNESS_CLAIM_BOUNDARY",
    "SEARCH_HARNESS_SCHEMA_VERSION",
    "STRUCTURALLY_INVALID",
    "AdmissibilityPartition",
    "AdversarialScenarioManifest",
    "BaselinePreparation",
    "BatchCertification",
    "BatchCertificationPolicy",
    "CMaMeEmitter",
    "CandidateCertification",
    "CandidateEvaluation",
    "CandidateOverlayAdapter",
    "CandidatePoint",
    "CandidateSpec",
    "CandidateSpecOverlayAdapter",
    "CoordinateRefinementSampler",
    "CrossVariableConstraint",
    "FeasibilityCandidate",
    "FeasibilityCheck",
    "FeasibilityFirstError",
    "FiniteBounds",
    "FiniteSearchSpaceManifest",
    "GeneratorInfo",
    "GridSpec",
    "HaltonQuasiRandomBaseline",
    "HierarchicalScenarioValue",
    "ImmutableScenarioOverlay",
    "ManifestCategory",
    "ManifestsQualitySummary",
    "MappingOverlayAdapter",
    "MultiPedAdversarialConfig",
    "MultiPedCandidateSpec",
    "ObjectiveComponent",
    "ObjectiveVector",
    "OptunaCandidateSampler",
    "PlannerOutcome",
    "PlannerOutcomeSummary",
    "Pose2D",
    "PreparedCandidate",
    "QDArchive",
    "QDComparisonReport",
    "QDSearchConfig",
    "QDSearchResult",
    "QuasiRandomBaseline",
    "QuasiRandomSearchBaseline",
    "RandomBaseline",
    "RandomCandidateSampler",
    "RandomSearchBaseline",
    "RejectionRecord",
    "RolloutBudget",
    "ScenarioAdmissibilityVerdict",
    "SearchCandidate",
    "SearchConfig",
    "SearchRunResult",
    "SearchSpaceConfig",
    "SearchVariable",
    "SeedPolicy",
    "SeedSensitivityPerturbation",
    "SeedSensitivityReplay",
    "SeedSensitivitySummary",
    "SourceLineage",
    "ValidationRecord",
    "build_baseline",
    "build_comparison_report",
    "build_fixture_candidates",
    "build_manifest",
    "build_multi_ped_adversarial_robot_config",
    "certify_candidate_batch",
    "certify_records",
    "classify_scenario_admissibility",
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
    "partition_candidates_by_admissibility",
    "prepare_baseline",
    "prepare_equal_budget_baselines",
    "production_candidate_evaluator",
    "production_qd_evaluator",
    "rank_feasible_candidates",
    "run_adversarial_search",
    "run_fixture_diagnostic",
    "run_map_elites",
    "run_seed_sensitivity",
    "sample_risk_feedback",
    "sample_seeded_uniform",
    "summarize_adversarial_manifest_quality",
    "summarize_adversarial_manifest_quality_records",
    "validate_candidate_manifest",
    "validate_manifest_payload",
    "validate_multi_ped_runtime_plausibility",
    "validate_report",
    "validate_scenario_admissibility",
    "write_manifest_yaml",
    "write_qd_archive",
]


def __getattr__(name: str) -> Any:
    """Resolve established package-level exports on first access."""
    if name not in __all__ and name != "register_constraints_first_lexicographic_v2":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "register_constraints_first_lexicographic_v2":
        module = import_module("robot_sf.adversarial.objectives_v2")
    elif name in {"production_candidate_evaluator", "run_adversarial_search"}:
        module = import_module("robot_sf.adversarial.search")
    else:
        module = import_module("robot_sf.adversarial._api")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, "register_constraints_first_lexicographic_v2"})
