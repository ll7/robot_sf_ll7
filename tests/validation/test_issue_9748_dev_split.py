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


def _frozen_provenance() -> dict:
    scenario_manifest = ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    return {
        "source_commit": CHECKER._current_source_commit(),
        "campaign_config_sha256": CHECKER._sha256(
            CHECKER.DEFAULT_CONFIG, label="test campaign config"
        ),
        "scenario_manifest_sha256": CHECKER._sha256(
            scenario_manifest, label="test scenario manifest"
        ),
        "candidate_configs": {
            key: {
                "path": CHECKER._repo_relative_path(path, label=f"test candidate {key}"),
                "sha256": CHECKER._sha256(path, label=f"test candidate {key}"),
            }
            for key, path in CHECKER.EXPECTED_PLANNER_CONFIGS.items()
        },
    }


def _valid_log_payload() -> dict:
    provenance = _frozen_provenance()
    candidate = sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)[0]
    return {
        "schema_version": CHECKER.TUNING_LOG_SCHEMA,
        "provenance": provenance,
        "entries": [
            {
                "candidate": candidate,
                "candidate_config_sha256": provenance["candidate_configs"][candidate]["sha256"],
                "seeds": [1001],
                "scenario_ids": ["issue_9748_dev_classic_doorway_medium"],
            }
        ],
    }


def test_tuning_log_rejects_release_seed_in_typed_field(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["seeds"] = [1001, 111]
    payload["notes"] = "The held-out release range 111–140 is never tuning evidence."
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="held-out release seeds"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_rejects_release_scenario_id(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["scenario_ids"] = ["classic_doorway_medium"]
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="scenario IDs outside"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_rejects_seeds_only_entry(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    payload["entries"][0].pop("scenario_ids")
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*scenario_id"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_requires_seed_on_each_entry(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    payload["entries"][0].pop("seeds")
    payload["seed"] = 1001
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*typed seed field"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_rejects_mixed_seeded_and_unseeded_entries(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    first = copy.deepcopy(payload["entries"][0])
    second = copy.deepcopy(first)
    second.pop("seeds")
    payload["entries"] = [first, second]
    payload["seed"] = 1001
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="entry 1.*typed seed field"):
        CHECKER._validate_tuning_log(path)


@pytest.mark.parametrize("malformed_seed", [True, "1001", [1001, "1002"]])
def test_tuning_log_rejects_malformed_entry_seed(tmp_path: Path, malformed_seed: object) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["seeds"] = malformed_seed
    payload["seed"] = 1002
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*typed integer seeds"):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_requires_string_scalar_or_list_scenario_ids(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["scenario_ids"] = "issue_9748_dev_classic_doorway_medium"
    path = _write_log(tmp_path, payload)

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


@pytest.mark.parametrize("mutation", ["wrong", "missing"])
def test_campaign_config_rejects_wrong_or_missing_planner_algo(mutation: str) -> None:
    payload = CHECKER._load_mapping(
        CHECKER.DEFAULT_CONFIG,
        label="development campaign config",
    )
    if mutation == "wrong":
        payload["planners"][0]["algo"] = "goal"
    else:
        payload["planners"][0].pop("algo")

    with pytest.raises(
        CHECKER.ValidationError, match="must declare algo: hybrid_rule_local_planner"
    ):
        CHECKER._validate_planner_specs(payload, config_path=CHECKER.DEFAULT_CONFIG)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "missing frozen provenance fields"),
        ("wrong_source_commit", "must match the current checked-out source commit"),
        ("wrong_campaign_hash", "campaign_config_sha256 does not match"),
        ("wrong_scenario_hash", "scenario_manifest_sha256 does not match"),
        ("wrong_candidate_hash", "candidate config hash"),
        ("wrong_candidate_path", "provenance candidate .* must name"),
        ("unknown_candidate", "unknown=.*candidate"),
    ],
)
def test_tuning_log_requires_frozen_provenance_and_known_candidates(
    tmp_path: Path, mutation: str, match: str
) -> None:
    payload = _valid_log_payload()
    if mutation == "missing":
        del payload["provenance"]["scenario_manifest_sha256"]
    elif mutation == "wrong_source_commit":
        payload["provenance"]["source_commit"] = "0" * 40
    elif mutation == "wrong_campaign_hash":
        payload["provenance"]["campaign_config_sha256"] = "0" * 64
    elif mutation == "wrong_scenario_hash":
        payload["provenance"]["scenario_manifest_sha256"] = "0" * 64
    elif mutation == "wrong_candidate_hash":
        payload["provenance"]["candidate_configs"][sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)[0]][
            "sha256"
        ] = "0" * 64
    elif mutation == "wrong_candidate_path":
        payload["provenance"]["candidate_configs"][sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)[0]][
            "path"
        ] = "configs/policy_search/candidates/other.yaml"
    else:
        payload["provenance"]["candidate_configs"]["unknown_candidate"] = {
            "path": "configs/policy_search/candidates/unknown.yaml",
            "sha256": "0" * 64,
        }
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match=match):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_accepts_valid_frozen_provenance(tmp_path: Path) -> None:
    path = _write_log(tmp_path, _valid_log_payload())

    summary = CHECKER._validate_tuning_log(path)

    assert summary["provenance"]["source_commit"] == CHECKER._current_source_commit()
    assert set(summary["provenance"]["candidate_configs"]) == set(CHECKER.EXPECTED_PLANNER_CONFIGS)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("entry_candidate", "must name an approved candidate"),
        ("entry_hash", "unknown or mismatched candidate config hash"),
    ],
)
def test_tuning_log_rejects_unknown_entry_candidate_or_hash(
    tmp_path: Path, mutation: str, match: str
) -> None:
    payload = _valid_log_payload()
    if mutation == "entry_candidate":
        payload["entries"][0]["candidate"] = "unknown_candidate"
    else:
        payload["entries"][0]["candidate_config_sha256"] = "0" * 64
    path = _write_log(tmp_path, payload)

    with pytest.raises(CHECKER.ValidationError, match=match):
        CHECKER._validate_tuning_log(path)


def test_tuning_log_accepts_dev_fields_and_held_out_prose(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    provenance = payload["provenance"]
    first_candidate, second_candidate = sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)
    payload["entries"] = [
        {
            "candidate": first_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][first_candidate]["sha256"],
            "seed": 1001,
            "scenario_id": "issue_9748_dev_classic_doorway_medium",
            "notes": "Hold out release seeds 111–140 and classic_doorway_medium for evaluation.",
        },
        {
            "candidate": second_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][second_candidate]["sha256"],
            "seeds": [1002],
            "scenario_ids": ["issue_9748_dev_francis2023_crowd_navigation"],
        },
    ]
    payload["rationale"] = "This prose may mention 111–140 without admitting those seeds."
    path = _write_log(tmp_path, payload)

    summary = CHECKER._validate_tuning_log(path)
    assert summary["release_seed_overlap"] == []
    assert summary["release_scenario_overlap"] == []
    assert summary["typed_scenario_count"] == 2


def test_tuning_log_accepts_independent_seed_bindings(tmp_path: Path) -> None:
    payload = _valid_log_payload()
    provenance = payload["provenance"]
    first_candidate, second_candidate = sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)
    payload["entries"] = [
        {
            "candidate": first_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][first_candidate]["sha256"],
            "seed": 1001,
            "scenario_id": "issue_9748_dev_classic_doorway_medium",
        },
        {
            "candidate": second_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][second_candidate]["sha256"],
            "seeds": [1002, 1003],
            "scenario_ids": ["issue_9748_dev_francis2023_crowd_navigation"],
        },
    ]
    path = _write_log(tmp_path, payload)

    summary = CHECKER._validate_tuning_log(path)

    assert summary["typed_seed_count"] == 3
