"""Validation tests for the issue #9748 development split."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from dataclasses import replace
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
        assert row["simulation_config"]["goal_completion_policy"] == "goal_zone_entry_v1"
        assert row["simulation_config"]["social_force_kernel_version"] == "wrapped_v2"


def test_development_rows_match_resolved_release_candidate_except_approved_changes() -> None:
    development_campaign, development_rows = CHECKER._load_canonical_campaign_rows(
        CHECKER.DEFAULT_CONFIG
    )
    release_campaign, release_rows = CHECKER._load_release_candidate_rows(
        CHECKER.DEFAULT_RELEASE_MATRIX
    )

    CHECKER._validate_release_parity(
        development_campaign=development_campaign,
        development_rows=development_rows,
        release_campaign=release_campaign,
        release_rows=release_rows,
    )


@pytest.mark.parametrize("policy", ["goal_completion_policy", "social_force_kernel_version"])
def test_release_policy_drift_blocks_development_parity(policy: str) -> None:
    development_campaign, development_rows = CHECKER._load_canonical_campaign_rows(
        CHECKER.DEFAULT_CONFIG
    )
    release_campaign, release_rows = CHECKER._load_release_candidate_rows(
        CHECKER.DEFAULT_RELEASE_MATRIX
    )
    changed_release_rows = copy.deepcopy(release_rows)
    source = next(row for row in changed_release_rows if row["name"] == "classic_doorway_medium")
    source["simulation_config"][policy] = "changed_release_policy"

    with pytest.raises(CHECKER.ValidationError, match="differs from the 0.0.8 candidate"):
        CHECKER._validate_release_parity(
            development_campaign=development_campaign,
            development_rows=development_rows,
            release_campaign=release_campaign,
            release_rows=changed_release_rows,
        )


def test_observation_mode_drift_blocks_development_parity() -> None:
    development_campaign, development_rows = CHECKER._load_canonical_campaign_rows(
        CHECKER.DEFAULT_CONFIG
    )
    release_campaign, release_rows = CHECKER._load_release_candidate_rows(
        CHECKER.DEFAULT_RELEASE_MATRIX
    )

    with pytest.raises(CHECKER.ValidationError, match="observation_mode"):
        CHECKER._validate_release_parity(
            development_campaign=development_campaign,
            development_rows=development_rows,
            release_campaign=replace(release_campaign, observation_mode="lidar"),
            release_rows=release_rows,
        )


def test_tuning_input_hash_closure_binds_release_candidate_matrix_chain() -> None:
    paths = set(
        CHECKER._tuning_input_paths(
            config_path=CHECKER.DEFAULT_CONFIG,
            scenario_matrix_path=(
                ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
            ),
            candidate_paths=CHECKER.EXPECTED_PLANNER_CONFIGS,
        )
    )

    assert ROOT / CHECKER.RELEASE_CONFIG_RELATIVE_PATH in paths
    assert CHECKER.DEFAULT_RELEASE_MATRIX in paths
    assert ROOT / "configs/scenarios/classic_interactions_francis2023.yaml" in paths


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


def _write_committed_log(
    tmp_path: Path,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    """Create a frozen-input Git repo and record one log commit after its freeze."""
    source_root = CHECKER.ROOT
    scenario_manifest = (
        source_root / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    )
    tracked_inputs = CHECKER._tuning_input_paths(
        config_path=CHECKER.DEFAULT_CONFIG,
        scenario_matrix_path=scenario_manifest,
        candidate_paths=CHECKER.EXPECTED_PLANNER_CONFIGS,
    )
    repository = tmp_path / "repo"
    for source in tracked_inputs:
        destination = repository / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Issue 9748 test")
    git("config", "user.email", "issue-9748-test@example.invalid")
    relative_inputs = [source.relative_to(source_root).as_posix() for source in tracked_inputs]
    git("add", "--", *relative_inputs)
    git("commit", "--quiet", "-m", "Freeze issue 9748 tuning inputs")
    frozen_source = git("rev-parse", "HEAD")

    config_path = repository / CHECKER.DEFAULT_CONFIG.relative_to(source_root)
    candidate_paths = {
        key: repository / path.relative_to(source_root)
        for key, path in CHECKER.EXPECTED_PLANNER_CONFIGS.items()
    }
    monkeypatch.setattr(CHECKER, "ROOT", repository)
    monkeypatch.setattr(CHECKER, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(CHECKER, "EXPECTED_PLANNER_CONFIGS", candidate_paths)

    payload["provenance"]["source_commit"] = frozen_source
    path = repository / "evidence/tuning-log.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    git("add", "--", "evidence/tuning-log.json")
    git("commit", "--quiet", "-m", "Record issue 9748 tuning log")
    return path, frozen_source


def _validate_committed_log(path: Path) -> dict:
    scenario_manifest = (
        CHECKER.ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    )
    return CHECKER._validate_tuning_log(
        path,
        config_path=CHECKER.DEFAULT_CONFIG,
        scenario_matrix_path=scenario_manifest,
    )


def _frozen_provenance() -> dict:
    scenario_manifest = ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    return {
        # Synthetic repository fixtures replace this with their own frozen commit.
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


def test_tuning_log_rejects_release_seed_in_typed_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["seeds"] = [1001, 111]
    payload["notes"] = "The held-out release range 111–140 is never tuning evidence."
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="held-out release seeds"):
        _validate_committed_log(path)


def test_tuning_log_rejects_release_scenario_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["scenario_ids"] = ["classic_doorway_medium"]
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="scenario IDs outside"):
        _validate_committed_log(path)


def test_tuning_log_rejects_seeds_only_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0].pop("scenario_ids")
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*scenario_id"):
        _validate_committed_log(path)


def test_tuning_log_requires_seed_on_each_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0].pop("seeds")
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*non-empty seeds list"):
        _validate_committed_log(path)


def test_tuning_log_rejects_mixed_seeded_and_unseeded_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    first = copy.deepcopy(payload["entries"][0])
    second = copy.deepcopy(first)
    second.pop("seeds")
    payload["entries"] = [first, second]
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="entry 1.*non-empty seeds list"):
        _validate_committed_log(path)


@pytest.mark.parametrize(
    "location",
    ["episode_seed", "run_seed", "nested_seed", "top_level_episode_seed"],
)
def test_tuning_log_rejects_seed_aliases_outside_entry_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    payload = _valid_log_payload()
    if location == "nested_seed":
        payload["entries"][0]["metadata"] = {"trial": {"episode_seed": 111}}
    elif location == "top_level_episode_seed":
        payload["episode_seed"] = 111
    else:
        payload["entries"][0][location] = 111
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="unsupported seed field"):
        _validate_committed_log(path)


def test_tuning_log_requires_scenario_identity_on_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    scenario_id = payload["entries"][0].pop("scenario_ids")[0]
    payload["entries"][0]["metadata"] = {"scenario_id": scenario_id}
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*scenario_id"):
        _validate_committed_log(path)


def test_tuning_log_rejects_scenario_identity_alias_even_with_valid_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["episode_scenario_id"] = "classic_doorway_medium"
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(
        CHECKER.ValidationError,
        match="unsupported scenario field.*episode_scenario_id",
    ):
        _validate_committed_log(path)


@pytest.mark.parametrize("malformed_seed", [True, "1001", [1001, "1002"]])
def test_tuning_log_rejects_malformed_entry_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_seed: object
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["seeds"] = malformed_seed
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="entry 0.*typed integer seeds"):
        _validate_committed_log(path)


def test_tuning_log_requires_string_scalar_or_list_scenario_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    payload["entries"][0]["scenario_ids"] = "issue_9748_dev_classic_doorway_medium"
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match="list of scenario IDs"):
        _validate_committed_log(path)


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
        ("wrong_source_commit", "must name a valid commit in the current repository"),
        ("wrong_campaign_hash", "campaign_config_sha256 does not match"),
        ("wrong_scenario_hash", "scenario_manifest_sha256 does not match"),
        ("wrong_candidate_hash", "candidate config hash"),
        ("wrong_candidate_path", "provenance candidate .* must name"),
        ("unknown_candidate", "unknown=.*candidate"),
    ],
)
def test_tuning_log_requires_frozen_provenance_and_known_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    payload = _valid_log_payload()
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)
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
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CHECKER.ValidationError, match=match):
        _validate_committed_log(path)


def test_tuning_log_accepts_valid_frozen_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, frozen_source = _write_committed_log(tmp_path, _valid_log_payload(), monkeypatch)

    summary = _validate_committed_log(path)

    assert summary["provenance"]["source_commit"] == frozen_source
    assert summary["provenance"]["log_commit"] != frozen_source
    assert set(summary["provenance"]["candidate_configs"]) == set(CHECKER.EXPECTED_PLANNER_CONFIGS)


def test_tuning_log_requires_a_tracked_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed_path, _ = _write_committed_log(tmp_path, _valid_log_payload(), monkeypatch)
    path = tmp_path / "outside-repository-log.json"
    path.write_bytes(committed_path.read_bytes())

    with pytest.raises(CHECKER.ValidationError, match="structured tuning log must be inside"):
        _validate_committed_log(path)


def test_tuning_log_validates_after_log_is_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    scenario_manifest = (
        CHECKER.ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    )
    tracked_inputs = CHECKER._tuning_input_paths(
        config_path=CHECKER.DEFAULT_CONFIG,
        scenario_matrix_path=scenario_manifest,
        candidate_paths=CHECKER.EXPECTED_PLANNER_CONFIGS,
    )
    for source in tracked_inputs:
        destination = repository / source.relative_to(CHECKER.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Issue 9748 test")
    git("config", "user.email", "issue-9748-test@example.invalid")
    relative_inputs = [source.relative_to(CHECKER.ROOT).as_posix() for source in tracked_inputs]
    git("add", "--", *relative_inputs)
    git("commit", "--quiet", "-m", "Freeze issue 9748 tuning inputs")
    frozen_source = git("rev-parse", "HEAD")

    payload = _valid_log_payload()
    payload["provenance"]["source_commit"] = frozen_source
    log_path = repository / "tuning-log.json"
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    git("add", "--", "tuning-log.json")
    git("commit", "--quiet", "-m", "Record issue 9748 tuning log")
    committed_head = git("rev-parse", "HEAD")
    assert git("merge-base", "--is-ancestor", frozen_source, committed_head) == ""

    frozen_config = repository / CHECKER.DEFAULT_CONFIG.relative_to(CHECKER.ROOT)
    frozen_scenario_manifest = repository / scenario_manifest.relative_to(CHECKER.ROOT)
    frozen_candidates = {
        key: repository / path.relative_to(CHECKER.ROOT)
        for key, path in CHECKER.EXPECTED_PLANNER_CONFIGS.items()
    }
    monkeypatch.setattr(CHECKER, "ROOT", repository)
    monkeypatch.setattr(CHECKER, "DEFAULT_CONFIG", frozen_config)
    monkeypatch.setattr(CHECKER, "EXPECTED_PLANNER_CONFIGS", frozen_candidates)

    summary = CHECKER._validate_tuning_log(
        log_path, config_path=frozen_config, scenario_matrix_path=frozen_scenario_manifest
    )

    assert summary["provenance"]["source_commit"] == frozen_source
    assert summary["provenance"]["source_commit"] != committed_head
    assert (
        summary["provenance"]["campaign_config_sha256"]
        == hashlib.sha256(frozen_config.read_bytes()).hexdigest()
    )
    assert summary["provenance"]["log_commit"] == committed_head

    committed_log_bytes = log_path.read_bytes()
    log_path.write_bytes(committed_log_bytes + b"\n")
    with pytest.raises(CHECKER.ValidationError, match="bytes differ from the tracked file"):
        CHECKER._validate_tuning_log(
            log_path, config_path=frozen_config, scenario_matrix_path=frozen_scenario_manifest
        )
    log_path.write_bytes(committed_log_bytes)

    self_anchored_payload = copy.deepcopy(payload)
    self_anchored_payload["provenance"]["source_commit"] = committed_head
    self_anchored_log = repository / "self-anchored-log.json"
    self_anchored_log.write_text(json.dumps(self_anchored_payload), encoding="utf-8")
    with pytest.raises(CHECKER.ValidationError, match="strict ancestor"):
        CHECKER._validate_tuning_log(
            self_anchored_log,
            config_path=frozen_config,
            scenario_matrix_path=frozen_scenario_manifest,
        )


def test_tuning_log_rejects_source_commit_sibling_to_log_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = CHECKER.ROOT
    scenario_manifest = (
        source_root / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    )
    tracked_inputs = CHECKER._tuning_input_paths(
        config_path=CHECKER.DEFAULT_CONFIG,
        scenario_matrix_path=scenario_manifest,
        candidate_paths=CHECKER.EXPECTED_PLANNER_CONFIGS,
    )
    repository = tmp_path / "repo"
    for source in tracked_inputs:
        destination = repository / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Issue 9748 test")
    git("config", "user.email", "issue-9748-test@example.invalid")
    relative_inputs = [source.relative_to(source_root).as_posix() for source in tracked_inputs]
    git("add", "--", *relative_inputs)
    git("commit", "--quiet", "-m", "Base issue 9748 inputs")
    base_commit = git("rev-parse", "HEAD")
    git("branch", "-M", "base")

    git("checkout", "-b", "frozen-source")
    (repository / "freeze-marker.txt").write_text("source branch\n", encoding="utf-8")
    git("add", "--", "freeze-marker.txt")
    git("commit", "--quiet", "-m", "Create frozen source branch")
    frozen_source = git("rev-parse", "HEAD")

    git("checkout", "base")
    git("checkout", "-b", "log-record")
    payload = _valid_log_payload()
    payload["provenance"]["source_commit"] = frozen_source
    log_path = repository / "evidence/tuning-log.json"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    git("add", "--", "evidence/tuning-log.json")
    git("commit", "--quiet", "-m", "Record log on sibling branch")
    log_commit = git("rev-parse", "HEAD")
    assert git("merge-base", "--is-ancestor", base_commit, frozen_source) == ""
    assert git("merge-base", "--is-ancestor", base_commit, log_commit) == ""

    git("merge", "--quiet", "--no-ff", "frozen-source", "-m", "Join source and log branches")

    frozen_config = repository / CHECKER.DEFAULT_CONFIG.relative_to(source_root)
    frozen_scenario_manifest = repository / scenario_manifest.relative_to(source_root)
    frozen_candidates = {
        key: repository / path.relative_to(source_root)
        for key, path in CHECKER.EXPECTED_PLANNER_CONFIGS.items()
    }
    monkeypatch.setattr(CHECKER, "ROOT", repository)
    monkeypatch.setattr(CHECKER, "DEFAULT_CONFIG", frozen_config)
    monkeypatch.setattr(CHECKER, "EXPECTED_PLANNER_CONFIGS", frozen_candidates)

    with pytest.raises(CHECKER.ValidationError, match="commit recording.*must descend"):
        CHECKER._validate_tuning_log(
            log_path,
            config_path=frozen_config,
            scenario_matrix_path=frozen_scenario_manifest,
        )


@pytest.mark.parametrize(
    "dependency",
    [
        CHECKER.RELEASE_CONFIG_RELATIVE_PATH,
        "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_kernel_wrapped_v2.yaml",
        "configs/scenarios/archetypes/classic_doorway.yaml",
        "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
        "maps/svg_maps/classic_doorway.svg",
        "robot_sf/benchmark/camera_ready/_config.py",
        "robot_sf/benchmark/camera_ready/_config_types.py",
        "robot_sf/benchmark/camera_ready/_util.py",
        "robot_sf/training/scenario_loader.py",
        "robot_sf/benchmark/map_runner_policies/map_runner_policy_resolution.py",
        "robot_sf/benchmark/policy_search_manifest.py",
    ],
)
def test_tuning_log_rejects_changed_transitive_inputs_after_source_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    repository = tmp_path / "repo"
    scenario_manifest = (
        CHECKER.ROOT / "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml"
    )
    tracked_inputs = CHECKER._tuning_input_paths(
        config_path=CHECKER.DEFAULT_CONFIG,
        scenario_matrix_path=scenario_manifest,
        candidate_paths=CHECKER.EXPECTED_PLANNER_CONFIGS,
    )
    for source in tracked_inputs:
        destination = repository / source.relative_to(CHECKER.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Issue 9748 test")
    git("config", "user.email", "issue-9748-test@example.invalid")
    relative_inputs = [source.relative_to(CHECKER.ROOT).as_posix() for source in tracked_inputs]
    git("add", "--", *relative_inputs)
    git("commit", "--quiet", "-m", "Freeze issue 9748 tuning inputs")
    frozen_source = git("rev-parse", "HEAD")

    payload = _valid_log_payload()
    payload["provenance"]["source_commit"] = frozen_source
    log_path = repository / "tuning-log.json"
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    git("add", "--", "tuning-log.json")
    git("commit", "--quiet", "-m", "Record issue 9748 tuning log")

    dependency_path = repository / dependency
    dependency_path.write_bytes(dependency_path.read_bytes() + b"\n# post-freeze mutation\n")

    frozen_config = repository / CHECKER.DEFAULT_CONFIG.relative_to(CHECKER.ROOT)
    frozen_scenario_manifest = repository / scenario_manifest.relative_to(CHECKER.ROOT)
    frozen_candidates = {
        key: repository / path.relative_to(CHECKER.ROOT)
        for key, path in CHECKER.EXPECTED_PLANNER_CONFIGS.items()
    }
    monkeypatch.setattr(CHECKER, "ROOT", repository)
    monkeypatch.setattr(CHECKER, "DEFAULT_CONFIG", frozen_config)
    monkeypatch.setattr(CHECKER, "EXPECTED_PLANNER_CONFIGS", frozen_candidates)

    with pytest.raises(CHECKER.ValidationError, match="transitive tuning input.*frozen source"):
        CHECKER._validate_tuning_log(
            log_path, config_path=frozen_config, scenario_matrix_path=frozen_scenario_manifest
        )


@pytest.mark.parametrize(
    "dependency,match",
    [
        (
            "configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml",
            "campaign_config_sha256 does not match the current HEAD",
        ),
        (
            "configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml",
            "scenario_manifest_sha256 does not match the current HEAD",
        ),
        (
            "configs/policy_search/candidates/"
            "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml",
            "candidate config hash.*does not match the current HEAD",
        ),
        (
            "robot_sf/benchmark/policy_search_manifest.py",
            "transitive tuning input.*frozen source",
        ),
    ],
)
def test_tuning_log_rejects_head_change_hidden_by_old_working_tree_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    match: str,
) -> None:
    payload = _valid_log_payload()
    log_path, _ = _write_committed_log(tmp_path, payload, monkeypatch)
    dependency_path = CHECKER.ROOT / dependency
    frozen_bytes = dependency_path.read_bytes()
    dependency_path.write_bytes(frozen_bytes + b"\n# committed post-freeze mutation\n")
    subprocess.run(
        ["git", "add", "--", dependency],
        cwd=CHECKER.ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Change governed input after log"],
        cwd=CHECKER.ROOT,
        check=True,
        capture_output=True,
    )
    dependency_path.write_bytes(frozen_bytes)

    head_bytes = subprocess.run(
        ["git", "show", f"HEAD:{dependency}"],
        cwd=CHECKER.ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert head_bytes != frozen_bytes
    assert dependency_path.read_bytes() == frozen_bytes
    with pytest.raises(CHECKER.ValidationError, match=match):
        _validate_committed_log(log_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("entry_candidate", "must name an approved candidate"),
        ("entry_hash", "unknown or mismatched candidate config hash"),
    ],
)
def test_tuning_log_rejects_unknown_entry_candidate_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str
) -> None:
    payload = _valid_log_payload()
    if mutation == "entry_candidate":
        payload["entries"][0]["candidate"] = "unknown_candidate"
    else:
        payload["entries"][0]["candidate_config_sha256"] = "0" * 64
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    with pytest.raises(CHECKER.ValidationError, match=match):
        _validate_committed_log(path)


def test_tuning_log_accepts_dev_fields_and_held_out_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    provenance = payload["provenance"]
    first_candidate, second_candidate = sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)
    payload["entries"] = [
        {
            "candidate": first_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][first_candidate]["sha256"],
            "seeds": [1001],
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
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    summary = _validate_committed_log(path)
    assert summary["release_seed_overlap"] == []
    assert summary["release_scenario_overlap"] == []
    assert summary["typed_scenario_count"] == 2


def test_tuning_log_accepts_independent_seed_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _valid_log_payload()
    provenance = payload["provenance"]
    first_candidate, second_candidate = sorted(CHECKER.EXPECTED_PLANNER_CONFIGS)
    payload["entries"] = [
        {
            "candidate": first_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][first_candidate]["sha256"],
            "seeds": [1001],
            "scenario_id": "issue_9748_dev_classic_doorway_medium",
        },
        {
            "candidate": second_candidate,
            "candidate_config_sha256": provenance["candidate_configs"][second_candidate]["sha256"],
            "seeds": [1002, 1003],
            "scenario_ids": ["issue_9748_dev_francis2023_crowd_navigation"],
        },
    ]
    path, _ = _write_committed_log(tmp_path, payload, monkeypatch)

    summary = _validate_committed_log(path)

    assert summary["typed_seed_count"] == 3
