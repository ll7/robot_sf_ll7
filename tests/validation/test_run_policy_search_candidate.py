"""Tests for the policy-search candidate runner helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.run_policy_search_candidate import (
    _annotate_initial_overlap_exclusions,
    _baseline_deltas,
    _deep_merge,
    _effective_candidate_config_for_scenario,
    _effective_candidate_runtime_for_scenario,
    _format_optional_float,
    _format_signed_optional_float,
    _load_stage_scenarios,
    _prepare_scenarios_for_inline_run,
    decide_stage_status,
    load_candidate_definition,
    split_scenarios_by_family,
)


def test_deep_merge_recurses_without_mutating_inputs() -> None:
    """Nested config overrides should merge without mutating the base mapping."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overrides = {"a": {"c": 9}, "e": 4}

    merged = _deep_merge(base, overrides)

    assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_load_candidate_definition_merges_base_config_and_params(
    tmp_path: Path,
) -> None:
    """Candidate definitions should inherit base config and apply params."""
    base_cfg = tmp_path / "base.yaml"
    candidate_cfg = tmp_path / "candidate.yaml"
    registry = tmp_path / "registry.yaml"

    base_cfg.write_text(yaml.safe_dump({"foo": {"a": 1}, "bar": 2}), encoding="utf-8")
    candidate_cfg.write_text(
        yaml.safe_dump(
            {
                "algo": "orca",
                "base_config_path": "base.yaml",
                "params": {"foo": {"a": 7, "b": 8}},
            }
        ),
        encoding="utf-8",
    )
    registry.write_text(
        yaml.safe_dump(
            {
                "candidates": {
                    "cand": {
                        "candidate_config_path": "candidate.yaml",
                        "promotion_gate": "tier_b",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entry, payload, merged, config_path = load_candidate_definition(registry, "cand")

    assert entry["promotion_gate"] == "tier_b"
    assert payload["algo"] == "orca"
    assert merged == {"foo": {"a": 7, "b": 8}, "bar": 2}
    assert config_path == candidate_cfg.resolve()


def test_checked_in_actuation_aware_candidate_uses_synthetic_amv_smoke_stage() -> None:
    """Issue #1807 candidate routing should stay diagnostic-only and synthetic-slice first."""
    repo_root = Path(__file__).parents[2]

    entry, payload, config, config_path = load_candidate_definition(
        repo_root / "docs/context/policy_search/candidate_registry.yaml",
        "actuation_aware_hybrid_rule_v0",
    )
    funnel = yaml.safe_load(
        (repo_root / "configs/policy_search/funnel.yaml").read_text(encoding="utf-8")
    )

    assert entry["claim_scope"] == "diagnostic_only"
    assert entry["required_stages"] == ["amv_actuation_smoke"]
    assert payload["algo"] == "actuation_aware_hybrid_rule_v0"
    assert (
        config_path
        == (
            repo_root / "configs/policy_search/candidates/actuation_aware_hybrid_rule_v0.yaml"
        ).resolve()
    )
    assert config["planner_variant"] == "actuation_aware_hybrid_rule_v0"
    assert config["actuation_profile_name"] == "amv-actuation-stress-v0"
    assert config["actuation_score_enabled"] is True
    stage = funnel["stages"]["amv_actuation_smoke"]
    assert stage["scenario_matrix"] == "configs/scenarios/sets/classic_cross_trap_subset.yaml"
    assert stage["scenario_filter"] == ["classic_cross_trap_high"]
    assert stage["synthetic_actuation_profile"]["claim_scope"] == "synthetic-only"
    assert stage["fail_closed_initial_overlap_exclusion"] is True
    assert stage["paper_facing"] is False


def test_split_scenarios_by_family_uses_name_when_scenario_id_is_missing() -> None:
    """Scenario names should drive family inference when scenario_id is absent."""
    grouped = split_scenarios_by_family(
        [
            {"name": "classic_bottleneck_medium"},
            {"name": "francis2023_blind_corner"},
            {"name": "planner_sanity_simple"},
        ]
    )

    assert sorted(grouped) == ["classic", "francis2023", "nominal"]


def test_effective_candidate_config_applies_scenario_override_after_family() -> None:
    """Scenario-specific overrides should be narrower than family overrides."""
    candidate_payload = {
        "family_overrides": {
            "francis2023": {
                "very_slow_speed": 0.6,
                "static_clearance_escape_max_speed": 0.6,
            }
        },
        "scenario_overrides": {
            "francis2023_perpendicular_traffic": {
                "very_slow_speed": 0.7,
            }
        },
    }
    base_config = {
        "very_slow_speed": 0.15,
        "static_clearance_escape_max_speed": 0.3,
    }
    scenario = {"name": "francis2023_perpendicular_traffic"}

    effective = _effective_candidate_config_for_scenario(candidate_payload, base_config, scenario)

    assert effective["very_slow_speed"] == 0.7
    assert effective["static_clearance_escape_max_speed"] == 0.6
    assert base_config["very_slow_speed"] == 0.15


def test_effective_candidate_runtime_applies_scenario_algo_override(tmp_path: Path) -> None:
    """Scenario-specific algo overrides should replace default algo and config."""
    base_cfg = tmp_path / "orca.yaml"
    base_cfg.write_text(yaml.safe_dump({"max_linear_speed": 0.8, "foo": 1}), encoding="utf-8")
    candidate_payload = {
        "scenario_algo_overrides": {
            "francis2023_leave_group": {
                "algo": "orca",
                "base_config_path": "orca.yaml",
                "params": {"max_linear_speed": 1.15},
            }
        }
    }
    base_config = {"max_linear_speed": 3.0}

    algo, effective = _effective_candidate_runtime_for_scenario(
        candidate_payload,
        base_config,
        {"name": "francis2023_leave_group"},
        default_algo="hybrid_rule_local_planner",
        config_anchor=tmp_path,
    )

    assert algo == "orca"
    assert effective == {"max_linear_speed": 1.15, "foo": 1}
    assert base_config == {"max_linear_speed": 3.0}


def test_prepare_scenarios_for_inline_run_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    """Inline scenario execution should resolve paths relative to the matrix."""
    scenario_root = tmp_path / "configs" / "scenarios"
    scenario_root.mkdir(parents=True)
    map_path = tmp_path / "maps" / "planner_sanity_open.svg"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("<svg />", encoding="utf-8")

    prepared = _prepare_scenarios_for_inline_run(
        [
            {
                "name": "planner_sanity_simple",
                "map_file": "../../maps/planner_sanity_open.svg",
            }
        ],
        scenario_root=scenario_root,
    )

    assert Path(prepared[0]["map_file"]) == map_path.resolve()


def test_load_stage_scenarios_applies_inline_seed_list(tmp_path: Path) -> None:
    """Stage-level seed_list should override scenario defaults for inline runs."""
    matrix = tmp_path / "matrix.yaml"
    map_path = tmp_path / "maps" / "open.svg"
    map_path.parent.mkdir()
    map_path.write_text("<svg />", encoding="utf-8")
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "planner_sanity_simple",
                        "map_file": "maps/open.svg",
                        "seeds": [101, 102, 103],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scenarios = _load_stage_scenarios(matrix, seed_manifest=None, seed_list=[111])

    assert isinstance(scenarios, list)
    assert scenarios[0]["seeds"] == [111]
    assert Path(scenarios[0]["map_file"]) == map_path.resolve()


def test_load_stage_scenarios_applies_scenario_filter(tmp_path: Path) -> None:
    """Stage-level scenario_filter should narrow inline smoke runs."""
    matrix = tmp_path / "matrix.yaml"
    map_path = tmp_path / "maps" / "open.svg"
    map_path.parent.mkdir()
    map_path.write_text("<svg />", encoding="utf-8")
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {"name": "classic_cross_trap_low", "map_file": "maps/open.svg"},
                    {"name": "classic_cross_trap_high", "map_file": "maps/open.svg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    scenarios = _load_stage_scenarios(
        matrix,
        seed_manifest=None,
        seed_list=[111],
        scenario_filter=["classic_cross_trap_high"],
    )

    assert isinstance(scenarios, list)
    assert [scenario["name"] for scenario in scenarios] == ["classic_cross_trap_high"]
    assert scenarios[0]["seeds"] == [111]


def test_load_stage_scenarios_preserves_path_without_seed_override(tmp_path: Path) -> None:
    """Stages without inline seed overrides can keep the matrix path fast path."""
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("scenarios: []\n", encoding="utf-8")

    assert _load_stage_scenarios(matrix, seed_manifest=None, seed_list=None) == matrix


def test_decide_stage_status_enforces_nominal_gate() -> None:
    """Nominal stages should enforce configured success and collision gates."""
    stage_cfg = {"gate": {"min_success_rate": 0.8, "max_collision_rate": 0.02}}

    assert (
        decide_stage_status(
            "nominal_sanity",
            stage_cfg,
            {"success_rate": 0.85, "collision_rate": 0.01},
        )
        == "pass"
    )
    assert (
        decide_stage_status(
            "nominal_sanity",
            stage_cfg,
            {"success_rate": 0.60, "collision_rate": 0.01},
        )
        == "revise"
    )


def test_decide_stage_status_treats_named_smoke_stages_as_smoke() -> None:
    """Diagnostic smoke stages should pass only when they run at least one episode."""
    assert decide_stage_status("amv_actuation_smoke", {}, {"episodes": 1}) == "pass"
    assert decide_stage_status("amv_actuation_smoke", {}, {"episodes": 0}) == "revise"
    assert (
        decide_stage_status(
            "amv_actuation_smoke",
            {},
            {"episodes": 1, "scenario_exclusions": {"count": 1}},
        )
        == "excluded"
    )


def test_annotate_initial_overlap_exclusions_requires_first_step_evidence() -> None:
    """Initial-overlap rows need explicit evidence before adjusted reporting excludes them."""
    rows = [
        {
            "scenario_id": "classic_cross_trap_high",
            "seed": 111,
            "steps": 1,
            "termination_reason": "collision",
            "metrics": {
                "avg_speed": 0.0,
                "min_clearance": -0.3882961911,
                "ped_collision_count": 1,
            },
            "algorithm_metadata": {
                "planner_runtime": {
                    "last_decision": {
                        "planner_mode": "EMERGENCY_STOP",
                        "selected_source": "all_candidates_rejected",
                        "nearest_pedestrian_distance": 1.0117035253,
                        "rejection_counts": {"dynamic_collision": 67},
                        "rejected_examples": [
                            {
                                "reason": "dynamic_collision",
                                "command": [0.0, 0.0],
                                "min_dynamic_clearance": 0.9159010834,
                                "collision_radius": 1.450000006,
                                "time": 0.2,
                            }
                        ],
                    }
                }
            },
        }
    ]

    annotated = _annotate_initial_overlap_exclusions(rows)

    assert annotated[0]["scenario_exclusion"] == {
        "status": "impossible",
        "reason": "initial_robot_pedestrian_overlap",
        "evidence": [
            "first_step_collision_with_zero_progress",
            "min_clearance_m=-0.3883",
            "nearest_pedestrian_distance_m=1.0117",
            "candidate_collision_radius_m=1.4500",
            "all_first_step_candidates_rejected_for_dynamic_collision",
        ],
    }
    assert "scenario_exclusion" not in rows[0]


def test_format_optional_float_keeps_present_values() -> None:
    """Optional report fields should not hide valid values when a sibling field is missing."""
    assert _format_optional_float(None) == "n/a"
    assert _format_optional_float("bad") == "n/a"
    assert _format_optional_float(1.23456) == "1.2346"


def test_baseline_deltas_do_not_mix_near_miss_counts_with_rates(tmp_path: Path) -> None:
    """Count-style baseline near-miss metrics must not be subtracted from rate summaries."""
    baselines = tmp_path / "baselines.yaml"
    baselines.write_text(
        yaml.safe_dump(
            {
                "baselines": {
                    "orca": {
                        "success_rate": 0.2,
                        "collision_rate": 0.1,
                        "near_misses_mean": 4.3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    deltas = _baseline_deltas(
        {"success_rate": 0.3, "collision_rate": 0.05, "near_miss_rate": 0.4},
        baselines,
    )

    assert deltas["orca"]["success_rate"] == pytest.approx(0.1)
    assert deltas["orca"]["collision_rate"] == pytest.approx(-0.05)
    assert "near_miss_rate" not in deltas["orca"]
    assert _format_signed_optional_float(deltas["orca"].get("near_miss_rate")) == "n/a"
