"""Validation tests for the issue #9748 development split."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import asdict
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
    assert summary["release_seed_overlap"] == []
    assert summary["planner_keys"] == sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)
    assert summary["release_bindings"]["release_scenario_matrix_sha256"] == (
        CHECKER.EXPECTED_RELEASE_MATRIX_SHA256
    )
    assert summary["release_bindings"]["release_seed_schedule_sha256"] == (
        CHECKER.EXPECTED_RELEASE_SEED_SCHEDULE_SHA256
    )


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


def test_release_matrix_seed_leakage_is_rejected_even_with_distinct_scenario_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = tmp_path / "release-fixture.yaml"
    fixture.write_text("fixture: true\n", encoding="utf-8")
    monkeypatch.setattr(
        CHECKER,
        "_load_scenario_rows",
        lambda _path, *, label: [{"name": "unrelated_release_scenario", "seeds": [1001]}],
    )

    with pytest.raises(CHECKER.ValidationError, match="development seeds overlap"):
        CHECKER._validate_release_disjointness(release_matrix_path=fixture)


def test_planner_spec_rejects_an_unapproved_algorithm_for_an_approved_key() -> None:
    key = next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS))

    with pytest.raises(CHECKER.ValidationError, match="approved algorithm"):
        CHECKER._validate_planner_spec(
            {"key": key, "algo": "ppo"},
            config_path=CHECKER.DEFAULT_CONFIG,
            seen=set(),
        )


def _write_log(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "tuning-log.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_trial(candidate: str, *, artifact_root: Path, trial_id: str = "trial-001") -> dict:
    summary = CHECKER.validate()
    effective_config = {"candidate": candidate, "parameters": {"tuning_override": 0.0}}
    artifact_bytes = f"fixture run artifact for {trial_id}\n".encode()
    artifact_path = artifact_root / f"{trial_id}.tar"
    artifact_path.write_bytes(artifact_bytes)
    return {
        "trial_id": trial_id,
        "candidate": candidate,
        "base_candidate_config_sha256": summary["candidate_config_sha256"][candidate],
        "effective_candidate_config": effective_config,
        "effective_candidate_config_sha256": CHECKER._canonical_sha256(effective_config),
        "run_artifact_ref": artifact_path.as_uri(),
        "run_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "seed": 1001,
        "scenario_id": "issue_9748_dev_classic_doorway_medium",
    }


def _valid_tuning_log_payload(entries: list[dict], *, source_commit: str | None = None) -> dict:
    summary = CHECKER.validate()
    if source_commit is None:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    return {
        "schema_version": CHECKER.TUNING_LOG_SCHEMA,
        "source_commit": source_commit,
        "input_bindings": summary["tuning_log_bindings"],
        "entries": entries,
    }


def _validate_log(path: Path) -> dict:
    summary = CHECKER.validate()
    return CHECKER._validate_tuning_log(path, expected_bindings=summary["tuning_log_bindings"])


def test_tuning_log_rejects_release_seed_in_typed_field(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [
                {
                    **_valid_trial(
                        next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path
                    ),
                    "seeds": [1001, 111],
                }
            ]
        ),
    )

    with pytest.raises(CHECKER.ValidationError, match="held-out release seeds"):
        _validate_log(path)


def test_tuning_log_rejects_release_scenario_id(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [
                {
                    **_valid_trial(
                        next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path
                    ),
                    "scenario_ids": ["classic_doorway_medium"],
                }
            ]
        ),
    )

    with pytest.raises(CHECKER.ValidationError, match="scenario IDs outside"):
        _validate_log(path)


def test_tuning_log_rejects_seeds_only_entry(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [
                {
                    key: value
                    for key, value in _valid_trial(
                        next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path
                    ).items()
                    if key != "scenario_id"
                }
            ]
        ),
    )

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*scenario_id"):
        _validate_log(path)


def test_tuning_log_requires_string_scalar_or_list_scenario_ids(tmp_path: Path) -> None:
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [
                {
                    **_valid_trial(
                        next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path
                    ),
                    "scenario_ids": "issue_9748_dev_classic_doorway_medium",
                }
            ]
        ),
    )

    with pytest.raises(CHECKER.ValidationError, match="list of scenario IDs"):
        _validate_log(path)


@pytest.mark.parametrize("candidate", [None, "ppo"])
def test_tuning_log_rejects_missing_or_unapproved_candidate(
    tmp_path: Path, candidate: str | None
) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    if candidate is None:
        trial.pop("candidate")
    else:
        trial["candidate"] = candidate
    path = _write_log(tmp_path, _valid_tuning_log_payload([trial]))

    with pytest.raises(CHECKER.ValidationError, match="unapproved candidate"):
        _validate_log(path)


def test_tuning_log_binds_effective_candidate_config_hash(tmp_path: Path) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    trial["effective_candidate_config_sha256"] = "0" * 64
    path = _write_log(tmp_path, _valid_tuning_log_payload([trial]))

    with pytest.raises(CHECKER.ValidationError, match="config checksum mismatch"):
        _validate_log(path)


def test_tuning_log_requires_an_existing_local_artifact_with_matching_checksum(
    tmp_path: Path,
) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    artifact_path = Path(trial["run_artifact_ref"].removeprefix("file://"))
    artifact_path.unlink()
    path = _write_log(tmp_path, _valid_tuning_log_payload([trial]))

    with pytest.raises(CHECKER.ValidationError, match="does not identify an existing artifact"):
        _validate_log(path)


def test_tuning_log_rejects_unverifiable_remote_artifact_reference(tmp_path: Path) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    trial["run_artifact_ref"] = "wandb://issue9748/trial-001"
    path = _write_log(tmp_path, _valid_tuning_log_payload([trial]))

    with pytest.raises(CHECKER.ValidationError, match="local file:// URI"):
        _validate_log(path)


def test_tuning_log_rejects_source_commit_that_is_not_in_git(tmp_path: Path) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    path = _write_log(tmp_path, _valid_tuning_log_payload([trial], source_commit="0" * 40))

    with pytest.raises(CHECKER.ValidationError, match="source_commit is not present"):
        _validate_log(path)


def test_tuning_log_source_commit_must_contain_frozen_inputs(tmp_path: Path) -> None:
    trial = _valid_trial(next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path)
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [trial], source_commit="4665cd13fc205761a4edb256a04d23d37ffbc235"
        ),
    )

    with pytest.raises(CHECKER.ValidationError, match="does not contain frozen development"):
        _validate_log(path)


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


def test_development_campaign_metadata_survives_loading_and_direct_run_is_gated() -> None:
    from robot_sf.benchmark.camera_ready._config import _parse_development_execution_metadata
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload
    from robot_sf.benchmark.camera_ready.campaign import run_campaign

    campaign_config, _rows = CHECKER._load_canonical_campaign_rows(CHECKER.DEFAULT_CONFIG)
    assert campaign_config.development_only is True
    assert campaign_config.not_release_evidence is True
    assert campaign_config.evidence_class == "development_tuning_only"
    assert campaign_config.claim_boundary
    assert "development_only_input" not in asdict(campaign_config)
    hash_payload = _config_hash_payload(campaign_config)
    assert hash_payload["development_only"] is True
    assert hash_payload["not_release_evidence"] is True
    assert hash_payload["evidence_class"] == "development_tuning_only"
    assert hash_payload["claim_boundary"] == campaign_config.claim_boundary

    assert _parse_development_execution_metadata({"claim_boundary": "general diagnostic"}) == (
        False,
        False,
        None,
        None,
    )
    with pytest.raises(ValueError, match="requires development_only"):
        _parse_development_execution_metadata(
            {"not_release_evidence": True, "evidence_class": "development_tuning_only"}
        )

    with pytest.raises(PermissionError, match="requires explicit authorization"):
        run_campaign(campaign_config)


def test_tuning_log_accepts_dev_fields_and_held_out_prose(tmp_path: Path) -> None:
    second_trial = _valid_trial(
        list(CHECKER.EXPECTED_PLANNER_CONFIGS)[1], artifact_root=tmp_path, trial_id="trial-002"
    )
    second_trial.pop("scenario_id")
    second_trial["seeds"] = [1002]
    second_trial["scenario_ids"] = ["issue_9748_dev_francis2023_crowd_navigation"]
    path = _write_log(
        tmp_path,
        _valid_tuning_log_payload(
            [
                {
                    **_valid_trial(
                        next(iter(CHECKER.EXPECTED_PLANNER_CONFIGS)), artifact_root=tmp_path
                    ),
                    "notes": (
                        "Hold out release seeds 111–140 and classic_doorway_medium for evaluation."
                    ),
                },
                second_trial,
            ]
        ),
    )

    summary = _validate_log(path)
    assert summary["release_seed_overlap"] == []
    assert summary["release_scenario_overlap"] == []
    assert summary["typed_scenario_count"] == 2
