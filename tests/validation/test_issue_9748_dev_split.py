"""Validation tests for the issue #9748 development split."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validation/check_issue_9748_dev_split.py"
SPEC = importlib.util.spec_from_file_location("issue_9748_dev_split_checker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_checked_in_development_split_passes_canonical_validation() -> None:
    summary = CHECKER.validate()

    assert summary["ok"] is True
    assert summary["seed_range"] == [1001, 1030]
    assert summary["seed_count"] == 30
    assert summary["scenario_count"] == 4
    assert summary["release_scenario_overlap"] == []
    assert summary["planner_keys"] == sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)


def test_checked_in_rows_pin_author_values() -> None:
    rows = CHECKER._load_scenario_rows(
        ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
        label="development scenario matrix",
    )
    by_id = {CHECKER._scenario_id(row, label="test"): row for row in rows}

    assert (
        by_id["issue_9748_dev_classic_doorway_medium"]["simulation_config"]["ped_density"] == 0.065
    )
    assert (
        by_id["issue_9748_dev_classic_doorway_medium"]["simulation_config"][
            "route_spawn_jitter_frac"
        ]
        == 0.30
    )
    assert (
        by_id["issue_9748_dev_classic_group_crossing_medium"]["simulation_config"]["ped_density"]
        == 0.10
    )
    assert (
        by_id["issue_9748_dev_francis2023_perpendicular_traffic"]["simulation_config"][
            "ped_density"
        ]
        == 0.12
    )
    assert (
        by_id["issue_9748_dev_francis2023_crowd_navigation"]["simulation_config"]["ped_density"]
        == 0.10
    )
    for row in rows:
        assert row["seeds"] == list(range(1001, 1031))


@pytest.mark.parametrize("planner_key", sorted(CHECKER.EXPECTED_PLANNER_CONFIGS))
def test_release_scenario_overrides_are_not_selected_for_dev_ids(planner_key: str) -> None:
    config_path = CHECKER.EXPECTED_PLANNER_CONFIGS[planner_key]
    manifest = CHECKER._load_mapping(config_path, label=f"planner config {planner_key}")
    dev_rows = CHECKER._load_scenario_rows(
        ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
        label="development scenario matrix",
    )
    unmatched = {"name": "issue_9748_dev_unmatched_control"}

    assert set(manifest.get("scenario_overrides", {})).isdisjoint(CHECKER.EXPECTED_SCENARIO_IDS)
    for scenario in dev_rows:
        _, dev_config = resolve_candidate_manifest_runtime(
            default_algo="hybrid_rule_local_planner",
            manifest=manifest,
            scenario=scenario,
            load_config=lambda _: {},
        )
        _, control_config = resolve_candidate_manifest_runtime(
            default_algo="hybrid_rule_local_planner",
            manifest=manifest,
            scenario=unmatched,
            load_config=lambda _: {},
        )
        assert dev_config == control_config


def test_scenario_validator_rejects_changed_source_map_geometry() -> None:
    rows = CHECKER._load_scenario_rows(
        ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
        label="development scenario matrix",
    )
    mutated = copy.deepcopy(rows)
    doorway = next(row for row in mutated if row["name"] == "issue_9748_dev_classic_doorway_medium")
    doorway["map_file"] = "../../../maps/svg_maps/classic_group_crossing.svg"

    with pytest.raises(CHECKER.ValidationError, match="approved development overrides"):
        CHECKER._validate_scenario_parameters(mutated)


def test_scenario_validator_rejects_unapproved_source_setting_change() -> None:
    rows = CHECKER._load_scenario_rows(
        ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
        label="development scenario matrix",
    )
    mutated = copy.deepcopy(rows)
    doorway = next(row for row in mutated if row["name"] == "issue_9748_dev_classic_doorway_medium")
    doorway["simulation_config"]["max_episode_steps"] += 1

    with pytest.raises(CHECKER.ValidationError, match="approved development overrides"):
        CHECKER._validate_scenario_parameters(mutated)


def test_scenario_validator_rejects_duplicate_development_ids() -> None:
    rows = CHECKER._load_scenario_rows(
        ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
        label="development scenario matrix",
    )
    mutated = copy.deepcopy(rows)
    mutated[1]["name"] = mutated[0]["name"]

    with pytest.raises(CHECKER.ValidationError, match="duplicate scenario IDs"):
        CHECKER._validate_scenario_parameters(mutated)


def _write_log(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "tuning-log.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_tuning_log_rejects_release_seed_in_typed_field(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        {
            "schema_version": CHECKER.TUNING_LOG_SCHEMA,
            "entries": [{"candidate": "v4", "seeds": [1001, 111]}],
            "notes": "The held-out release range 111–140 is never tuning evidence.",
        },
    )

    with pytest.raises(CHECKER.ValidationError, match="held-out release seeds"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_rejects_release_scenario_id(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        {
            "schema_version": CHECKER.TUNING_LOG_SCHEMA,
            "entries": [
                {
                    "candidate": "v4",
                    "seeds": [1001],
                    "scenario_ids": ["classic_doorway_medium"],
                }
            ],
        },
    )

    with pytest.raises(CHECKER.ValidationError, match="scenario IDs outside"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_rejects_seeds_only_entry(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        {
            "schema_version": CHECKER.TUNING_LOG_SCHEMA,
            "entries": [{"candidate": "v4", "seeds": [1001]}],
        },
    )

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*scenario_id"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_requires_string_scalar_or_list_scenario_ids(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        {
            "schema_version": CHECKER.TUNING_LOG_SCHEMA,
            "entries": [
                {
                    "candidate": "v4",
                    "seeds": [1001],
                    "scenario_ids": "issue_9748_dev_classic_doorway_medium",
                }
            ],
        },
    )

    with pytest.raises(CHECKER.ValidationError, match="list of scenario IDs"):
        CHECKER._validate_tuning_log(path)


def test_campaign_config_rejects_release_seed_or_scenario_admission() -> None:
    payload = CHECKER._load_mapping(
        CHECKER.DEFAULT_CONFIG,
        label="development campaign config",
    )
    payload["tuning_seeds"] = [1001, 111]
    with pytest.raises(CHECKER.ValidationError, match="admits a release seed"):
        CHECKER._validate_config_seed_policy(payload)

    payload = CHECKER._load_mapping(
        CHECKER.DEFAULT_CONFIG,
        label="development campaign config",
    )
    payload["scenario_ids"] = ["classic_doorway_medium"]
    with pytest.raises(CHECKER.ValidationError, match="scenario IDs outside"):
        CHECKER._validate_config_seed_policy(payload)


def test_tuning_log_accepts_dev_fields_and_held_out_prose(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        {
            "schema_version": CHECKER.TUNING_LOG_SCHEMA,
            "entries": [
                {
                    "candidate": "v4",
                    "seed": 1001,
                    "scenario_id": "issue_9748_dev_classic_doorway_medium",
                    "notes": (
                        "Hold out release seeds 111–140 and classic_doorway_medium for evaluation."
                    ),
                },
                {
                    "candidate": "v4-continuous",
                    "seeds": [1002],
                    "scenario_ids": ["issue_9748_dev_francis2023_crowd_navigation"],
                },
            ],
            "rationale": "This prose may mention 111–140 without admitting those seeds.",
        },
    )

    summary = CHECKER._validate_tuning_log(path)
    assert summary["release_seed_overlap"] == []
    assert summary["release_scenario_overlap"] == []
    assert summary["typed_scenario_count"] == 2
