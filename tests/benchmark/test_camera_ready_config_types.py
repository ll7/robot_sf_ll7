"""Unit tests for camera-ready config dataclasses and constants.

Covers ``robot_sf/benchmark/camera_ready/_config_types.py`` (issue #6078): the pure frozen
configuration dataclasses and module-level constants that form the package-local owner for the
#3385 camera-ready decomposition. Tests assert the declared defaults, frozen immutability,
absence of shared mutable default state, frozen-dataclass equality semantics, and the public
package re-export surface.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready import (
    DEFAULT_SEED_SETS_PATH,
    AmvProfileConfig,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    SnqiContractConfig,
    TuningSpec,
)
from robot_sf.benchmark.camera_ready import _config_types as config_types

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_default_seed_sets_path_is_configs_relative() -> None:
    """DEFAULT_SEED_SETS_PATH points at the canonical seed-sets config under configs/."""
    assert DEFAULT_SEED_SETS_PATH == Path("configs/benchmarks/seed_sets_v1.yaml")
    # The package re-export must be the exact same object, not a copy.
    assert DEFAULT_SEED_SETS_PATH is config_types.DEFAULT_SEED_SETS_PATH


def test_amv_dimensions_are_the_four_paper_axes() -> None:
    """_AMV_DIMENSIONS enumerates the four AMV coverage axes in their canonical order."""
    assert config_types._AMV_DIMENSIONS == ("use_case", "context", "speed_regime", "maneuver_type")


def test_tuning_source_constants_and_vocabulary() -> None:
    """The three tuning-source provenance labels and their closed vocabulary tuple."""
    assert config_types.TUNING_SOURCE_DECLARED == "declared"
    assert config_types.TUNING_SOURCE_BACKFILLED == "backfilled"
    assert config_types.TUNING_SOURCE_UNKNOWN == "unknown"
    assert config_types._TUNING_SOURCES == ("declared", "backfilled", "unknown")


def test_enforcement_vocabulary_tuples() -> None:
    """The tuning-effort and checkpoint-provenance enforcement mode vocabularies."""
    assert config_types._TUNING_EFFORT_ENFORCEMENT == ("off", "warn", "error")
    assert config_types._CHECKPOINT_PROVENANCE_ENFORCEMENT == ("off", "error")


# ---------------------------------------------------------------------------
# AmvProfileConfig
# ---------------------------------------------------------------------------


def test_amv_profile_config_defaults() -> None:
    """AmvProfileConfig defaults to the v1 paper profile with empty required dimensions."""
    profile = AmvProfileConfig()

    assert profile.name == "amv-paper-v1"
    assert profile.contract_version == "1"
    assert profile.coverage_enforcement == "warn"
    # Every AMV axis is present but empty by default.
    assert set(profile.required_dimensions) == set(config_types._AMV_DIMENSIONS)
    assert all(value == () for value in profile.required_dimensions.values())


def test_amv_profile_config_required_dimensions_can_be_overridden() -> None:
    """required_dimensions accepts per-axis value tuples via the default factory."""
    profile = AmvProfileConfig(
        required_dimensions={"use_case": ("doorway", "crossing")},
    )

    assert profile.required_dimensions["use_case"] == ("doorway", "crossing")


# ---------------------------------------------------------------------------
# SeedPolicy
# ---------------------------------------------------------------------------


def test_seed_policy_defaults() -> None:
    """SeedPolicy defaults to scenario-default seeding against the canonical seed-set file."""
    policy = SeedPolicy()

    assert policy.mode == "scenario-default"
    assert policy.seed_set is None
    assert policy.seeds == ()
    assert policy.seed_sets_path == DEFAULT_SEED_SETS_PATH


# ---------------------------------------------------------------------------
# ScenarioCandidateSelection
# ---------------------------------------------------------------------------


def test_scenario_candidate_selection_defaults() -> None:
    """ScenarioCandidateSelection defaults to an empty, unnamed selection."""
    selection = ScenarioCandidateSelection()

    assert selection.names == ()
    assert selection.selection_name is None


# ---------------------------------------------------------------------------
# TuningSpec
# ---------------------------------------------------------------------------


def test_tuning_spec_defaults_to_unknown_source() -> None:
    """A bare TuningSpec records an unknown source and otherwise blank provenance."""
    spec = TuningSpec()

    assert spec.parameters_touched == ()
    assert spec.tuning_scenario_ids == ()
    assert spec.eval_set_disjoint is None
    assert spec.budget_runs is None
    assert spec.budget_hours is None
    assert spec.tuned_by is None
    assert spec.tuned_at_utc is None
    assert spec.source == config_types.TUNING_SOURCE_UNKNOWN


# ---------------------------------------------------------------------------
# PlannerSpec
# ---------------------------------------------------------------------------


def test_planner_spec_required_fields_and_defaults() -> None:
    """PlannerSpec only requires key/algo; every other field has an experimental default."""
    spec = PlannerSpec(key="orca", algo="orca")

    assert spec.key == "orca"
    assert spec.algo == "orca"
    assert spec.human_model_variant is None
    assert spec.human_model_source is None
    assert spec.benchmark_profile == "baseline-safe"
    assert spec.algo_config_path is None
    assert spec.socnav_missing_prereq_policy == "fail-fast"
    assert spec.availability_gate is None
    assert spec.fail_closed_reason is None
    assert spec.adapter_impact_eval is False
    assert spec.observation_mode is None
    assert spec.workers_override is None
    assert spec.horizon_override is None
    assert spec.dt_override is None
    assert spec.enabled is True
    assert spec.planner_group == "experimental"
    assert spec.planner_group_explicit is False
    assert spec.tuning is None


def test_planner_spec_carries_declared_tuning_block() -> None:
    """PlannerSpec accepts an author-declared TuningSpec with a declared source."""
    spec = PlannerSpec(
        key="cadrl",
        algo="cadrl",
        tuning=TuningSpec(
            parameters_touched=("lr",),
            eval_set_disjoint=True,
            source=config_types.TUNING_SOURCE_DECLARED,
        ),
    )

    assert spec.tuning is not None
    assert spec.tuning.parameters_touched == ("lr",)
    assert spec.tuning.eval_set_disjoint is True
    assert spec.tuning.source == "declared"


# ---------------------------------------------------------------------------
# SnqiContractConfig
# ---------------------------------------------------------------------------


def test_snqi_contract_config_defaults() -> None:
    """SnqiContractConfig defaults to enabled, warn-enforcement with documented thresholds."""
    contract = SnqiContractConfig()

    assert contract.enabled is True
    assert contract.enforcement == "warn"
    assert contract.rank_alignment_warn_threshold == 0.5
    assert contract.rank_alignment_fail_threshold == 0.3
    assert contract.outcome_separation_warn_threshold == 0.05
    assert contract.outcome_separation_fail_threshold == 0.0
    assert contract.max_component_dominance_warn_threshold == 0.24
    assert contract.max_component_dominance_fail_threshold == 0.27
    assert contract.calibration_seed == 123
    assert contract.calibration_trials == 3000


# ---------------------------------------------------------------------------
# CampaignConfig
# ---------------------------------------------------------------------------


def _minimal_campaign() -> CampaignConfig:
    """Build a CampaignConfig exercising only the required fields."""
    return CampaignConfig(
        name="smoke-campaign",
        scenario_matrix_path=Path("configs/benchmarks/matrix.yaml"),
        planners=(PlannerSpec(key="sf", algo="sf"),),
    )


def test_campaign_config_required_fields_and_defaults() -> None:
    """CampaignConfig requires name/scenario_matrix_path/planners and defaults enforcement off."""
    cfg = _minimal_campaign()

    assert cfg.name == "smoke-campaign"
    assert cfg.scenario_matrix_path == Path("configs/benchmarks/matrix.yaml")
    assert len(cfg.planners) == 1
    assert cfg.planners[0].key == "sf"
    # Backwards-compatible enforcement defaults.
    assert cfg.tuning_effort_enforcement == "off"
    assert cfg.checkpoint_provenance_enforcement == "off"
    assert cfg.arm_isolation == "in_process"


def test_campaign_config_provides_composite_subconfig_defaults() -> None:
    """CampaignConfig fills in nested subconfig defaults without caller intervention."""
    cfg = _minimal_campaign()

    assert isinstance(cfg.amv_profile, AmvProfileConfig)
    assert cfg.amv_profile == AmvProfileConfig()
    assert isinstance(cfg.seed_policy, SeedPolicy)
    assert cfg.seed_policy == SeedPolicy()
    assert isinstance(cfg.scenario_candidates, ScenarioCandidateSelection)
    assert isinstance(cfg.snqi_contract, SnqiContractConfig)
    assert cfg.snqi_contract == SnqiContractConfig()
    assert cfg.synthetic_actuation_profile is None
    assert cfg.latency_stress_profile is None


def test_campaign_config_requires_name_and_matrix() -> None:
    """CampaignConfig must be constructed with the three required fields."""
    with pytest.raises(TypeError):
        CampaignConfig(  # type: ignore[call-arg]
            scenario_matrix_path=Path("configs/benchmarks/matrix.yaml"),
            planners=(),
        )


def test_campaign_config_mutable_defaults_are_fresh_per_instance() -> None:
    """Factory-backed mutable defaults must not be shared across CampaignConfig instances."""
    first = _minimal_campaign()
    second = _minimal_campaign()

    # field(default_factory=...) guarantees independent mutable containers.
    assert first.amv_profile is not second.amv_profile
    assert first.scenario_candidates is not second.scenario_candidates
    assert first.scenario_amv_overrides is not second.scenario_amv_overrides
    assert first.snqi_contract is not second.snqi_contract
    # Mutating one instance's default must not bleed into the other.
    first.scenario_amv_overrides["s1"] = {"use_case": "doorway"}
    assert second.scenario_amv_overrides == {}


def test_campaign_config_seed_policy_default_is_a_frozen_shared_instance() -> None:
    """seed_policy uses a shared default instance; that is safe because SeedPolicy is frozen."""
    first = _minimal_campaign()
    second = _minimal_campaign()

    # This documents the real declaration (default=SeedPolicy(), not a factory). Sharing is safe
    # because SeedPolicy is frozen and therefore immutable; mutating it must raise.
    assert first.seed_policy is second.seed_policy
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.seed_policy.mode = "fixed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Frozen / equality semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [
        AmvProfileConfig(),
        SeedPolicy(),
        ScenarioCandidateSelection(),
        TuningSpec(),
        PlannerSpec(key="k", algo="a"),
        SnqiContractConfig(),
    ],
)
def test_config_dataclasses_are_frozen(instance: object) -> None:
    """Every config dataclass is frozen: assigning to a field must raise."""
    field_names = [f.name for f in dataclasses.fields(instance)]
    assert field_names, "expected at least one field on the dataclass"
    first_field = field_names[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, first_field, None)


def test_frozen_dataclasses_compare_by_value() -> None:
    """Equal frozen config dataclasses with equal fields compare equal and distinct hash."""
    left = PlannerSpec(key="k", algo="a", tuning=TuningSpec(source="declared"))
    right = PlannerSpec(key="k", algo="a", tuning=TuningSpec(source="declared"))

    assert left == right
    assert hash(left) == hash(right)
    assert left != PlannerSpec(key="k", algo="b")


# ---------------------------------------------------------------------------
# Public package re-export surface
# ---------------------------------------------------------------------------


def test_public_symbols_are_re_exported_from_package() -> None:
    """The package __init__ re-exports each config type and the shared seed-sets path."""
    import robot_sf.benchmark.camera_ready as pkg

    expected = {
        "DEFAULT_SEED_SETS_PATH",
        "AmvProfileConfig",
        "CampaignConfig",
        "PlannerSpec",
        "ScenarioCandidateSelection",
        "SeedPolicy",
        "SnqiContractConfig",
        "TuningSpec",
        "load_campaign_config",
    }
    assert expected.issubset(set(pkg.__all__))
    for name in expected:
        assert hasattr(pkg, name), f"package is missing re-exported symbol {name!r}"
