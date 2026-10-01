"""Contract tests for the bounded S30/H600 14-arm runtime smoke."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready._config_types import PlannerSpec
from robot_sf.benchmark.camera_ready._preflight import _load_campaign_scenarios
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.benchmark.release_parameter_freeze import (
    ARM_SLOTS_0_0_7_TO_0_0_8,
    COMPARISON_IMPLEMENTATION_REPLACED,
    UnfrozenReleaseParametersError,
)
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest
from robot_sf.benchmark.runtime_smoke_admission import RUNTIME_SMOKE_PLANNER_KEYS
from robot_sf.benchmark.spawn_preflight import _release_manifest_inputs, run_manifest_preflight

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
)
FULL_MANIFEST_PATH = REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)
SMOKE_MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
CAMPAIGN_TEMPLATE_PATH = REPO_ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs/benchmarks/snqi_v2/calibration.dev101_102.yaml"
RUNTIME_SMOKE_V03_CONFIG_PATH = REPO_ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_3.yaml"
)
RUNTIME_SMOKE_V03_MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_3.yaml"
)
RUNTIME_SMOKE_V04_CONFIG_PATH = REPO_ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4.yaml"
)
RUNTIME_SMOKE_V04_MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4.yaml"
)
RUNTIME_SMOKE_V05_CONFIG_PATH = REPO_ROOT / (
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
)
RUNTIME_SMOKE_V05_MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
)
PINNED_V04_CONFIG_SHA256 = "698b4bc44455a3fd4485b5b25b0bca16e47ca7d6ceaf62fd918aaba05df177d4"
PINNED_V04_MANIFEST_SHA256 = "aded0ca71e40bdc8f7193282bb8d28420a9b627f93d47a43034703a45d613197"
PINNED_V03_CONFIG_SHA256 = "fbd900243f5a004cc07f7d10c672126f46ec583eb6f108ec7a0e8fce9daa7ad4"
PINNED_V03_MANIFEST_SHA256 = "d6f3047adaacfb8cad2cc12430ee5ce7331f11b0777ac522209fd1e5af019241"
HISTORICAL_V04_TEMPLATE_SHA256 = "f453b7c824fdd47298cbc66dae3afc1fffcd7eedf57ee4bb87cd1c67b4feb1d7"
CAMPAIGN_TEMPLATE_SHA256 = "a5d35d7e250958e84f4d662efa65864d6bf63c6d923a543bab812c9bd10763f5"

EXPECTED_PLANNER_KEYS = [
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
]
# Issue #9751: the 0.0.8 template (and the v0_4 smoke that mirrors it) replaces the
# four hybrid slots with v4-named keys; every other slot keeps its 0.0.7 key.
EXPECTED_0_0_8_PLANNER_KEYS = [slot.key_0_0_8 for slot in ARM_SLOTS_0_0_7_TO_0_0_8]
REPLACED_V4_KEYS = [
    slot.key_0_0_8
    for slot in ARM_SLOTS_0_0_7_TO_0_0_8
    if slot.comparison == COMPARISON_IMPLEMENTATION_REPLACED
]
BLIND_CORNER_HYBRID_CONFIGS = [
    "configs/policy_search/candidates/"
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_s30_h600_release.yaml",
    "configs/policy_search/candidates/"
    "scenario_adaptive_hybrid_orca_v2_collision_guard_s30_h600_release.yaml",
    "configs/policy_search/candidates/"
    "hybrid_rule_v3_fast_progress_static_escape_s30_h600_release.yaml",
    "configs/policy_search/candidates/"
    "hybrid_rule_v3_fast_progress_static_escape_continuous_s30_h600_release.yaml",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _algo_config(root: Path, path: str) -> dict[str, Any]:
    return _load_yaml(root / path)


def _resolved_arm_identity(
    planner: PlannerSpec, scenario: dict[str, Any], *, root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Resolve runnable inputs or record an unfrozen placeholder's refusal."""
    spec = asdict(planner)
    algo_path = planner.algo_config_path
    spec["algo_config_path"] = str(algo_path.relative_to(root)) if algo_path else None
    manifest = _algo_config(root, spec["algo_config_path"]) if algo_path else {}

    def load_base(path: object) -> dict[str, Any]:
        return _algo_config(root, str(path)) if path else {}

    if manifest.get("release_parameter_freeze", {}).get("status") == "unfrozen":
        with pytest.raises(
            UnfrozenReleaseParametersError, match="release parameters are not frozen"
        ):
            resolve_candidate_manifest_runtime(
                default_algo=planner.algo,
                manifest=manifest,
                scenario=scenario,
                load_config=load_base,
            )
        return {
            "planner": spec,
            "blocked_manifest": manifest,
            "algo_config_sha256": _sha256(algo_path),
        }

    algo, effective = resolve_candidate_manifest_runtime(
        default_algo=planner.algo,
        manifest=manifest,
        scenario=scenario,
        load_config=load_base,
    )
    return {
        "planner": spec,
        "effective_algo": algo,
        "effective_config": effective,
        "algo_config_sha256": _sha256(algo_path) if algo_path else None,
    }


def test_resolved_arm_identity_detects_inherited_inputs_behind_the_same_key(tmp_path: Path) -> None:
    """A shared arm key cannot conceal changed model, kernel or scenario overrides."""
    (tmp_path / "base-a.yaml").write_text(
        "checkpoint: models/a.pt\nkernel_version: legacy\nspeed: 1\n", encoding="utf-8"
    )
    (tmp_path / "base-b.yaml").write_text(
        "checkpoint: models/b.pt\nkernel_version: corrected_v2\nspeed: 1\n",
        encoding="utf-8",
    )
    (tmp_path / "candidate-a.yaml").write_text(
        "base_config_path: base-a.yaml\nscenario_overrides:\n  corner:\n    speed: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "candidate-b.yaml").write_text(
        "base_config_path: base-b.yaml\nscenario_overrides:\n  corner:\n    speed: 3\n",
        encoding="utf-8",
    )
    planner_a = PlannerSpec(
        key="shared-key",
        algo="hybrid_rule_local_planner",
        algo_config_path=tmp_path / "candidate-a.yaml",
    )
    planner_b = replace(planner_a, algo_config_path=tmp_path / "candidate-b.yaml")
    actual = _resolved_arm_identity(planner_a, {"name": "corner"}, root=tmp_path)
    expected = _resolved_arm_identity(planner_b, {"name": "corner"}, root=tmp_path)
    assert actual["planner"]["key"] == expected["planner"]["key"]
    assert {
        "effective_config.checkpoint",
        "effective_config.kernel_version",
        "effective_config.speed",
    } <= _diff_paths(actual, expected)


def _diff_paths(source: Any, target: Any, prefix: str = "") -> set[str]:
    """Return leaf paths that differ between two nested campaign configs."""
    if isinstance(source, dict) and isinstance(target, dict):
        differences: set[str] = set()
        for key in source.keys() | target.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in source or key not in target:
                differences.add(path)
            else:
                differences.update(_diff_paths(source[key], target[key], path))
        return differences
    return set() if source == target else {prefix}


def test_runtime_smoke_is_one_scenario_one_seed_at_h600() -> None:
    """The smoke keeps production horizon/kinematics while bounding cardinality."""
    config = _load_yaml(SMOKE_CONFIG_PATH)
    cfg = load_campaign_config(SMOKE_CONFIG_PATH)
    scenarios = _load_campaign_scenarios(cfg)

    assert config["derived_from"]["config"] == (
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
    )
    assert config["derived_from"]["config_sha256"] == _sha256(SOURCE_CONFIG_PATH)
    source = _load_yaml(SOURCE_CONFIG_PATH)
    assert config["comparability_mapping"] == source["comparability_mapping"]
    assert config["route_clearance_certifications"] == source["route_clearance_certifications"]
    assert cfg.horizon == 600
    assert cfg.dt == 0.1
    assert cfg.workers == 32
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.resume is False
    assert cfg.stop_on_failure is True
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "francis2023_blind_corner"
    assert list(scenarios[0]["seeds"]) == [111]


def test_runtime_smoke_preserves_all_fourteen_source_arms_without_fallback() -> None:
    """Every source arm is present, and prerequisite gaps fail closed in smoke mode."""
    source = _load_yaml(SOURCE_CONFIG_PATH)
    smoke = _load_yaml(SMOKE_CONFIG_PATH)
    comparability = _load_yaml(REPO_ROOT / source["comparability_mapping"])
    source_planners = {row["key"]: row for row in source["planners"]}
    smoke_planners = {row["key"]: row for row in smoke["planners"]}

    assert list(smoke_planners) == EXPECTED_PLANNER_KEYS
    assert list(source_planners) == EXPECTED_PLANNER_KEYS
    assert list(RUNTIME_SMOKE_PLANNER_KEYS) == EXPECTED_PLANNER_KEYS
    assert len(smoke_planners) == 14
    for key in EXPECTED_PLANNER_KEYS:
        source_row = dict(source_planners[key])
        smoke_row = dict(smoke_planners[key])
        assert smoke_row == source_row
        if smoke_row.get("algo_config"):
            assert (REPO_ROOT / smoke_row["algo_config"]).is_file()
        assert key in comparability["planner_key_mapping"]
        assert smoke_row.get("socnav_missing_prereq_policy") != "fallback"
        assert source_row.get("socnav_missing_prereq_policy", "fail-fast") == "fail-fast"


def test_release_learned_ppo_arms_use_parallel_safe_cpu_profiles() -> None:
    """Forked release workers must not initialize one CUDA policy per episode process."""
    for campaign_path in (SOURCE_CONFIG_PATH, SMOKE_CONFIG_PATH):
        campaign = _load_yaml(campaign_path)
        planners = {row["key"]: row for row in campaign["planners"]}

        ppo_path = REPO_ROOT / planners["ppo"]["algo_config"]
        ppo = _load_yaml(ppo_path)
        assert ppo["model_id"] == (
            "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
        )
        assert ppo["device"] == "cpu"
        assert ppo["predictive_foresight_device"] == "cpu"
        assert ppo["fallback_to_goal"] is False

        guarded_path = REPO_ROOT / planners["guarded_ppo"]["algo_config"]
        guarded = _load_yaml(guarded_path)
        assert guarded["model_id"] == ("ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200")
        assert guarded["device"] == "cpu"
        assert guarded["predictive_foresight_device"] == "cpu"
        assert guarded["fallback_to_goal"] is True


def test_blind_corner_hybrid_arms_yield_before_the_release_smoke_conflict() -> None:
    """The four hybrid arms avoid algorithm substitution while yielding before the conflict."""
    for relative_path in BLIND_CORNER_HYBRID_CONFIGS:
        candidate = _load_yaml(REPO_ROOT / relative_path)
        override = candidate["scenario_overrides"]["francis2023_blind_corner"]
        continuous_static_clearance = override.get(
            "continuous_static_clearance_enabled",
            candidate.get("params", {}).get("continuous_static_clearance_enabled", False),
        )

        assert "stop_distance_human" not in override
        assert continuous_static_clearance is True
        assert override["static_hard_safety_margin"] == 0.0
        assert override["slow_distance_human"] == 3.5
        assert override["moderate_distance_human"] == 4.5
        assert "francis2023_blind_corner" not in candidate.get("scenario_algo_overrides", {})

    continuous = _load_yaml(REPO_ROOT / BLIND_CORNER_HYBRID_CONFIGS[-1])
    assert (
        continuous["scenario_overrides"]["francis2023_blind_corner"]["hard_safety_margin"] == 0.05
    )


def test_full_and_smoke_release_configs_are_strict_and_use_new_identity() -> None:
    """Canonical benchmark-data execution cannot use fallback or predecessor metadata."""
    source = _load_yaml(SOURCE_CONFIG_PATH)
    smoke = _load_yaml(SMOKE_CONFIG_PATH)
    full_manifest = _load_yaml(FULL_MANIFEST_PATH)

    for payload in (source, smoke):
        assert payload["checkpoint_provenance_enforcement"] == "error"
        assert all(
            row.get("socnav_missing_prereq_policy", "fail-fast") == "fail-fast"
            for row in payload["planners"]
        )

    assert source["release_tag"] == full_manifest["release_tag"]
    assert source["doi"] == full_manifest["provenance"]["doi"]
    serialized = yaml.safe_dump(source)
    assert "0.0.3.post1" not in serialized
    assert "19482025" not in serialized
    assert "19563812" not in serialized


def test_release_protocol_rejects_relaxed_or_predecessor_execution_contract() -> None:
    """Manifest validation blocks fallback, audit-only checkpoints, and old identity drift."""
    manifest = load_release_manifest(FULL_MANIFEST_PATH)
    cfg = load_campaign_config(SOURCE_CONFIG_PATH)
    planners = tuple(
        replace(
            planner,
            socnav_missing_prereq_policy=("fallback" if planner.key == "orca" else "fail-fast"),
        )
        for planner in cfg.planners
    )
    drifted_cfg = replace(
        cfg,
        checkpoint_provenance_enforcement="warn",
        planners=planners,
        release_tag="0.0.3.post1",
        doi="10.5281/zenodo.19482025",
    )

    validation = validate_release_manifest(manifest, campaign_config=drifted_cfg)

    assert validation["status"] == "invalid"
    assert any(
        "checkpoint_provenance_enforcement=error" in problem for problem in validation["problems"]
    )
    assert any(
        "fail-fast missing-prerequisite policy" in problem for problem in validation["problems"]
    )
    assert "campaign config release_tag does not match release manifest" in validation["problems"]
    assert "campaign config doi does not match release manifest" in validation["problems"]


def test_runtime_smoke_claim_boundary_and_fresh_zenodo_requirement() -> None:
    """Smoke metadata separates benchmark-data evidence from software and ranking claims."""
    config = _load_yaml(SMOKE_CONFIG_PATH)
    manifest = _load_yaml(SMOKE_MANIFEST_PATH)

    for payload in (config, manifest):
        assert payload["release_kind"] == "benchmark-data"
        assert payload["claim_boundary"]["evidence_class"] == "runtime-smoke"
        assert payload["claim_boundary"]["snqi"] == "advisory-no-ranking"
        assert payload["claim_boundary"]["software_release"] is False
        assert payload["zenodo"]["concept"] == "fresh-required"
        assert payload["zenodo"]["doi_status"] == "pending-assignment"
        assert payload["zenodo"]["reuse_existing_concept"] is False
        assert payload["artifact_provenance"]["raw_episode_outputs"] == (
            "worktree-local-ignored-cache"
        )
        assert payload["artifact_provenance"]["compact_manifest"] == "tracked-compact-evidence"
        assert payload["artifact_provenance"]["source_commit"] == "record-at-run"
        assert payload["artifact_provenance"]["campaign_root"].startswith("output/")
        serialized = yaml.safe_dump(payload)
        assert "19482025" not in serialized
        assert "19563812" not in serialized


def test_runtime_smoke_manifest_validates_against_config_and_assets() -> None:
    """The manifest remains consumable by the release protocol without publishing."""
    manifest = load_release_manifest(SMOKE_MANIFEST_PATH)
    validation = validate_release_manifest(manifest)

    assert validation == {
        "manifest_path": "configs/benchmarks/releases/"
        "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml",
        "status": "valid",
        "problem_count": 0,
        "problems": [],
    }
    assert manifest.planner_keys == tuple(EXPECTED_PLANNER_KEYS)
    assert manifest.seed_policy["mode"] == "fixed-list"
    assert manifest.seed_policy["seeds"] == [111]
    assert manifest.expected_kinematics_matrix == ("differential_drive",)


def test_runtime_smoke_v0_4_preserves_its_historical_binding_and_v0_3() -> None:
    """The predecessor keeps its original source pin and v4 fail-closed roster."""
    template = _load_yaml(CAMPAIGN_TEMPLATE_PATH)
    smoke = _load_yaml(RUNTIME_SMOKE_V04_CONFIG_PATH)
    cfg = load_campaign_config(RUNTIME_SMOKE_V04_CONFIG_PATH)
    scenarios = _load_campaign_scenarios(cfg)

    assert _sha256(RUNTIME_SMOKE_V04_CONFIG_PATH) == PINNED_V04_CONFIG_SHA256
    assert _sha256(RUNTIME_SMOKE_V04_MANIFEST_PATH) == PINNED_V04_MANIFEST_SHA256
    assert smoke["derived_from"] == {
        "config": "configs/benchmarks/"
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
        "config_sha256": HISTORICAL_V04_TEMPLATE_SHA256,
    }
    assert [row["key"] for row in smoke["planners"]] == EXPECTED_0_0_8_PLANNER_KEYS
    # The canonical runtime-smoke admission roster stays the 0.0.7 (v0_2) roster;
    # the 0.0.8 keys are its slot-for-slot successors.
    assert list(RUNTIME_SMOKE_PLANNER_KEYS) == EXPECTED_PLANNER_KEYS
    assert [slot.key_0_0_7 for slot in ARM_SLOTS_0_0_7_TO_0_0_8] == EXPECTED_PLANNER_KEYS
    assert len(smoke["planners"]) == 14
    for row in smoke["planners"]:
        if row.get("algo_config"):
            assert (REPO_ROOT / row["algo_config"]).is_file()

    assert smoke["release_kind"] == "benchmark-data"
    assert smoke["release_status"] == "runtime-smoke-only"
    assert smoke["claim_boundary"] == {
        "evidence_class": "runtime-smoke",
        "benchmark_data_release": True,
        "software_release": False,
        "snqi": "advisory-no-ranking",
        "full_benchmark_evidence": False,
    }
    assert smoke["paper_interpretation_profile"] == "runtime-smoke-advisory-no-ranking"
    assert smoke["name"] == "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4"
    assert smoke["release_tag"] == "paper-matrix-v2-h600-s30-runtime-smoke-v0_4"
    assert smoke["planners"][2]["algo_config"] == (
        "configs/algos/social_force_resolution_independent_v2.yaml"
    )
    # Issues #9727/#9746 (#9668 ruling): the socnav_sampling key binds bounded_v2
    # only through this explicit versioned config.
    assert smoke["planners"][5]["key"] == "socnav_sampling"
    assert smoke["planners"][5]["algo_config"] == "configs/algos/socnav_sampling_bounded_v2.yaml"
    assert (
        _load_yaml(REPO_ROOT / smoke["planners"][5]["algo_config"])["socnav_sampling_version"]
        == "bounded_v2"
    )

    expected_config_differences = {
        "artifact_provenance",
        "bootstrap_samples",
        "claim_boundary",
        "comparability_mapping",
        "derived_from",
        "doi",
        "export_publication_bundle",
        "name",
        "overwrite_publication_bundle",
        "paper_interpretation_profile",
        "release_kind",
        "release_status",
        "release_tag",
        "resume",
        "scenario_matrix",
        "seed_policy.mode",
        "seed_policy.seed_set",
        "seed_policy.seeds",
        "snqi_contract.calibration_trials",
        "zenodo",
    }
    # The current template advances the simulator kernel and Social Force selector;
    # v0_4 retains its historical bytes and source pin.
    assert _diff_paths(template, smoke) == expected_config_differences | {"planners"}
    assert smoke["record_forces"] is True
    assert smoke["checkpoint_provenance_enforcement"] == "error"
    assert smoke["resume"] is False
    assert smoke["bootstrap_samples"] == 100 < template["bootstrap_samples"]
    assert smoke["snqi_contract"]["calibration_trials"] == 600
    assert smoke["export_publication_bundle"] is False
    assert smoke["overwrite_publication_bundle"] is False

    assert cfg.horizon == 600
    assert cfg.dt == 0.1
    assert cfg.workers == 32
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.resume is False
    assert cfg.stop_on_failure is True
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "francis2023_blind_corner"
    assert list(scenarios[0]["seeds"]) == [111]

    assert _sha256(RUNTIME_SMOKE_V03_CONFIG_PATH) == PINNED_V03_CONFIG_SHA256
    assert _sha256(RUNTIME_SMOKE_V03_MANIFEST_PATH) == PINNED_V03_MANIFEST_SHA256


def _assert_versioned_kernel_and_v4_freeze(
    profiles: list[dict[str, Any]], scenario_sets: list[dict[str, dict[str, Any]]]
) -> None:
    """Every candidate profile selects wrapped_v2 and the same frozen v4 slots."""
    for payload, profile_scenarios in zip(profiles, scenario_sets, strict=True):
        assert [row["key"] for row in payload["planners"]] == EXPECTED_0_0_8_PLANNER_KEYS
        social_force = next(row for row in payload["planners"] if row["key"] == "social_force")
        assert social_force["algo_config"] == ("configs/algos/social_force_release_v0_0_8.yaml")
        assert (
            _algo_config(REPO_ROOT, social_force["algo_config"])["social_force_kernel_version"]
            == "wrapped_v2"
        )
        assert all(
            scenario["simulation_config"]["social_force_kernel_version"] == "wrapped_v2"
            for scenario in profile_scenarios.values()
        )
        for row in payload["planners"]:
            if row["key"] in REPLACED_V4_KEYS:
                assert (
                    _algo_config(REPO_ROOT, row["algo_config"])["release_parameter_freeze"][
                        "status"
                    ]
                    == "frozen"
                )


def test_calibration_smoke_and_template_match_inputs_and_frozen_v4_slots() -> None:
    """Runnable inputs and all four frozen v4 slots resolve identically (#9850)."""
    assert _sha256(CAMPAIGN_TEMPLATE_PATH) == CAMPAIGN_TEMPLATE_SHA256
    paths = (CALIBRATION_CONFIG_PATH, RUNTIME_SMOKE_V05_CONFIG_PATH, CAMPAIGN_TEMPLATE_PATH)
    raw = [_load_yaml(path) for path in paths]
    configs = [load_campaign_config(path) for path in paths]
    scenarios = [
        {row["name"]: row for row in _load_campaign_scenarios(config)} for config in configs
    ]
    calibration, smoke, template = raw
    assert calibration["planners"] == smoke["planners"] == template["planners"]
    _assert_versioned_kernel_and_v4_freeze(raw, scenarios)
    allowed_calibration_differences = {
        "arm_isolation",  # execution resource policy
        "export_publication_bundle",  # publication identity
        "name",  # publication identity
        "paper_facing",  # publication identity
        "seed_policy.mode",
        "seed_policy.seed_set",
        "seed_policy.seeds",
        "workers",  # execution resource policy
    }
    assert _diff_paths(calibration, template) - {"planners"} <= (allowed_calibration_differences)
    allowed_smoke_differences = {
        "artifact_provenance",  # publication identity and artifact custody
        "bootstrap_samples",  # bounded runtime resources
        "claim_boundary",  # publication identity
        "derived_from",  # publication identity
        "doi",  # publication identity
        "export_publication_bundle",  # publication identity
        "name",  # publication identity
        "overwrite_publication_bundle",  # publication identity
        "paper_interpretation_profile",  # publication identity
        "release_kind",  # publication identity
        "release_status",  # publication identity
        "release_tag",  # publication identity
        "resume",  # runtime resource policy
        "scenario_matrix",  # scenario subset
        "seed_policy.mode",
        "seed_policy.seed_set",
        "seed_policy.seeds",
        "snqi_contract.calibration_trials",  # bounded runtime resources
        "zenodo",  # publication identity
    }
    assert _diff_paths(smoke, template) == allowed_smoke_differences
    assert set(scenarios[0]) == set(scenarios[2])
    assert set(scenarios[1]) <= set(scenarios[2])
    assert len(scenarios[0]) == len(scenarios[2]) == 48
    assert len(scenarios[1]) == 1
    assert len(configs[0].planners) == len(configs[1].planners) == len(configs[2].planners) == 14
    assert {101, 102}.isdisjoint(range(111, 141))
    assert {103}.isdisjoint({101, 102} | set(range(111, 141)) | set(range(1001, 1031)))
    assert {seed for row in scenarios[0].values() for seed in row["seeds"]} == {101, 102}
    assert {seed for row in scenarios[1].values() for seed in row["seeds"]} == {103}
    assert {seed for row in scenarios[2].values() for seed in row["seeds"]} == set(range(111, 141))

    mismatches: list[str] = []
    for index, label in ((0, "calibration"), (1, "smoke")):
        left, right = configs[index], configs[2]
        for field in ("horizon", "dt", "kinematics_matrix"):
            if getattr(left, field) != getattr(right, field):
                mismatches.append(
                    f"{label}.{field}: {getattr(left, field)!r} != {getattr(right, field)!r}"
                )
        for name, scenario in scenarios[index].items():
            reference = scenarios[2][name]
            left_scenario = dict(scenario)
            right_scenario = dict(reference)
            left_scenario.pop("seeds", None)
            right_scenario.pop("seeds", None)
            if left_scenario != right_scenario:
                mismatches.append(
                    f"{label}.{name}.scenario: {_diff_paths(left_scenario, right_scenario)}"
                )
            map_file = scenario.get("map_file")
            if map_file:
                map_path = REPO_ROOT / map_file
                assert map_path.is_file(), f"{label}.{name}.map_file missing: {map_file}"
                assert _sha256(map_path) == _sha256(REPO_ROOT / reference["map_file"])
            for planner, template_planner in zip(left.planners, right.planners, strict=True):
                actual = _resolved_arm_identity(planner, scenario)
                expected = _resolved_arm_identity(template_planner, reference)
                if actual != expected:
                    mismatches.append(
                        f"{label}.{name}.{planner.key}: {sorted(_diff_paths(actual, expected))}"
                    )
    assert not mismatches, "Resolved profile drift:\n" + "\n".join(mismatches[:30])


def test_runtime_smoke_v0_4_manifest_is_source_bound_and_refused_until_v4_freeze() -> None:
    """The historical v0_4 manifest remains bound to its blocked placeholder inputs."""
    template = _load_yaml(CAMPAIGN_TEMPLATE_PATH)
    manifest_payload = _load_yaml(RUNTIME_SMOKE_V04_MANIFEST_PATH)
    manifest = load_release_manifest(RUNTIME_SMOKE_V04_MANIFEST_PATH)
    validation = validate_release_manifest(manifest)

    # Issue #9751: the four v4 slots bind unfrozen placeholders until #9748, so the
    # smoke manifest is refused with exactly those four blockers and nothing else.
    assert validation["manifest_path"] == (
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4.yaml"
    )
    assert validation["status"] == "invalid"
    assert validation["problem_count"] == len(REPLACED_V4_KEYS) == 4
    for key in REPLACED_V4_KEYS:
        assert any(
            problem.startswith(f"planner {key}: release parameters are not frozen")
            and "ll7/robot_sf_ll7#9748" in problem
            for problem in validation["problems"]
        ), key
    assert manifest_payload["canonical_campaign_config"] == (
        "../paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4.yaml"
    )
    assert manifest_payload["campaign_config_sha256"] == _sha256(RUNTIME_SMOKE_V04_CONFIG_PATH)
    assert manifest_payload["derived_from"] == {
        "config": "../paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
        "config_sha256": HISTORICAL_V04_TEMPLATE_SHA256,
    }
    assert manifest_payload["planners"]["keys"] == EXPECTED_0_0_8_PLANNER_KEYS
    assert manifest_payload["planners"]["groups"] == {
        row["key"]: row["planner_group"] for row in template["planners"]
    }
    assert manifest_payload["claim_boundary"] == {
        "evidence_class": "runtime-smoke",
        "snqi": "advisory-no-ranking",
        "benchmark_data_release": True,
        "software_release": False,
        "full_benchmark_evidence": False,
    }
    assert manifest_payload["release_status"] == "runtime-smoke-only"
    assert manifest.expected_paper_interpretation_profile == ("runtime-smoke-advisory-no-ranking")
    assert manifest.planner_keys == tuple(EXPECTED_0_0_8_PLANNER_KEYS)
    assert manifest.seed_policy["mode"] == "fixed-list"
    assert manifest.seed_policy["seeds"] == [111]
    assert manifest.expected_kinematics_matrix == ("differential_drive",)


def test_runtime_smoke_v0_5_advances_wrapped_kernel_and_preserves_v0_4() -> None:
    """The successor selects the new kernel while v0_4 remains byte-pinned."""
    assert _sha256(RUNTIME_SMOKE_V04_CONFIG_PATH) == PINNED_V04_CONFIG_SHA256
    assert _sha256(RUNTIME_SMOKE_V04_MANIFEST_PATH) == PINNED_V04_MANIFEST_SHA256

    predecessor = _load_yaml(RUNTIME_SMOKE_V04_CONFIG_PATH)
    successor = _load_yaml(RUNTIME_SMOKE_V05_CONFIG_PATH)
    assert _diff_paths(predecessor, successor) == {
        "comparability_mapping",
        "derived_from.config_sha256",
        "name",
        "planners",
        "release_tag",
        "scenario_matrix",
        "seed_policy.seeds",
    }
    assert successor["name"] == "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5"
    assert successor["release_tag"] == "paper-matrix-v2-h600-s30-runtime-smoke-v0_5"
    assert successor["seed_policy"]["seeds"] == [103]

    old_manifest = _load_yaml(RUNTIME_SMOKE_V04_MANIFEST_PATH)
    new_manifest = _load_yaml(RUNTIME_SMOKE_V05_MANIFEST_PATH)
    changed_group_keys = set(old_manifest["planners"]["groups"]) ^ set(
        new_manifest["planners"]["groups"]
    )
    assert _diff_paths(old_manifest, new_manifest) == {
        "release_id",
        "release_tag",
        "canonical_campaign_config",
        "campaign_config_sha256",
        "artifact_provenance.command",
        "derived_from.config_sha256",
        "scenario.matrix_path",
        "scenario.matrix_sha256",
        "seed_policy.seeds",
    } | {f"planners.groups.{key}" for key in changed_group_keys}
    assert new_manifest["release_id"] == successor["name"]
    assert new_manifest["release_tag"] == successor["release_tag"]
    assert new_manifest["seed_policy"]["seeds"] == [103]
    assert new_manifest["campaign_config_sha256"] == _sha256(RUNTIME_SMOKE_V05_CONFIG_PATH)
    assert new_manifest["derived_from"]["config_sha256"] == CAMPAIGN_TEMPLATE_SHA256
    assert new_manifest["scenario"]["matrix_sha256"] == _sha256(
        REPO_ROOT / successor["scenario_matrix"]
    )
    assert new_manifest["planners"]["keys"] == EXPECTED_0_0_8_PLANNER_KEYS

    manifest = load_release_manifest(RUNTIME_SMOKE_V05_MANIFEST_PATH)
    validation = validate_release_manifest(manifest)
    assert validation["manifest_path"] == (
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
    )
    assert validation["status"] == "valid"
    assert validation["problem_count"] == 0


def test_runtime_smoke_v0_5_spawn_preflight_resolves_only_seed_103() -> None:
    """The release preflight must admit the fixed-list smoke seed before execution."""
    manifest = load_release_manifest(RUNTIME_SMOKE_V05_MANIFEST_PATH)
    _, scenarios, seeds = _release_manifest_inputs(manifest)
    assert len(scenarios) == 1
    assert seeds == (103,)
    report = run_manifest_preflight(manifest, workers=1)
    assert report["status"] == "valid", report.get("input_error")
    assert report["seed_count"] == 1
    assert report["blocked_cell_count"] == 0
