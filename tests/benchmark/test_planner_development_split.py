"""Input-integrity regressions for the author-approved #9748 development split."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from scripts.validation import check_planner_tuning_split as guard

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def split_root(tmp_path: Path) -> Path:
    """Copy only the authored inputs; no simulator or release results are needed."""
    for path in (guard.SPLIT_FILE, guard.FREEZE_FILE, guard.SCENARIO_FILE):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    return tmp_path


def write_json(root: Path, name: str, value: object) -> Path:
    """Write a structured fixture under the temporary repository root."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_author_decision_and_frozen_hashes(split_root: Path) -> None:
    """Exactly four parameter variants retain the unmodified base simulation fields."""
    rows = guard.validate_freeze(split_root, check_maps=False)
    actual = {row["metadata"]["source_scenario"]: row for row in rows}
    expected = {
        "classic_doorway_medium": {
            "max_episode_steps": 500, "ped_density": 0.065,
            "route_spawn_distribution": "spread", "route_spawn_jitter_frac": 0.30,
            "max_peds_per_group": 1,
        },
        "classic_group_crossing_medium": {
            "max_episode_steps": 500, "ped_density": 0.10, "groups": 0.5,
        },
        "francis2023_perpendicular_traffic": {"max_episode_steps": 400, "ped_density": 0.12},
        "francis2023_crowd_navigation": {"max_episode_steps": 400, "ped_density": 0.10},
    }
    assert set(actual) == set(expected)
    for source, settings in expected.items():
        row = actual[source]
        assert row["simulation_config"] == settings
        assert row["name"] == f"dev_v1__{source}"
        assert row["seeds"] == list(range(1001, 1031))
        assert row["robot_config"] == {}
        assert "metrics" not in row["metadata"]["plausibility"]
        assert row["metadata"]["plausibility"]["status"] == "unverified"
    guard.validate_tuning_config(split_root / guard.SPLIT_FILE, split_root)


@pytest.mark.parametrize("seed", range(111, 141))
@pytest.mark.parametrize("shape", ["scalar", "list", "nested", "command"])
def test_each_release_seed_is_rejected(tmp_path: Path, seed: int, shape: str) -> None:
    """Exercise every held-out release seed, not just the two endpoints."""
    values = {
        "scalar": {"seed": seed},
        "list": {"seeds": [1001, seed]},
        "nested": {"trial": {"random_seed": str(seed)}},
        "command": {"command": f"runner --seeds 1001,{seed}"},
    }
    path = write_json(tmp_path, "candidate.json", values[shape])
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.audit_file(path, tmp_path)


@pytest.mark.parametrize("value", ["111-140", "111–140", "110:141", "[111, 140]"])
def test_seed_range_leakage(value: str) -> None:
    """Reject ranges containing held-out seeds even when endpoints are outside them."""
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.check_seeds(value, "fixture")


@pytest.mark.parametrize("value", [[], True, None, "range(1001, 1031)", 1000, 1031, 1001.5])
def test_ambiguous_or_outside_seed_selection_is_not_accepted(value: object) -> None:
    """Unsupported expressions and implicit/default schedules must fail closed."""
    with pytest.raises(ValueError):
        guard.check_seeds(value, "fixture")


def test_named_set_resolves_only_selected_schedule(tmp_path: Path) -> None:
    """Shared registries may contain release seeds, but the selected set may not."""
    write_json(tmp_path, "sets.json", {"development": [1001, 1002], "evaluation": [111, 140]})
    policy = {"mode": "seed-set", "seed_set": "development", "seed_sets_path": "sets.json"}
    path = write_json(tmp_path, "config.json", {"seed_policy": policy})
    guard.audit_file(path, tmp_path)
    policy["seed_set"] = "evaluation"
    write_json(tmp_path, "config.json", {"seed_policy": policy})
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.audit_file(path, tmp_path)


@pytest.mark.parametrize(
    "reference",
    ["includes", "extends", "algo_config", "base_config_path", "config_path", "config"],
)
def test_referenced_config_cannot_hide_release_seeds(tmp_path: Path, reference: str) -> None:
    """Follow common include, inheritance and planner-config references transitively."""
    write_json(tmp_path, "child.json", {"nested": {"seed": 120}})
    path = write_json(tmp_path, "parent.json", {reference: "child.json"})
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.audit_file(path, tmp_path)


@pytest.mark.parametrize(
    "command", ["runner --seed-set paper_eval_s30", "runner --seed_file x.json"]
)
def test_unresolved_command_seed_indirection_fails(tmp_path: Path, command: str) -> None:
    """Opaque command-line seed registries are not an accepted substitute for provenance."""
    path = write_json(tmp_path, "trial.json", {"command": command})
    with pytest.raises(ValueError, match="indirection"):
        guard.audit_file(path, tmp_path)


def test_duplicate_keys_are_not_silently_overwritten(tmp_path: Path) -> None:
    """A safe last declaration must not erase a forbidden earlier declaration."""
    path = tmp_path / "duplicate.yaml"
    path.write_text("seeds: [111]\nseeds: [1001]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key"):
        guard.audit_file(path, tmp_path)


def test_cyclic_missing_and_ambiguous_references_fail(tmp_path: Path) -> None:
    """Never turn unresolved provenance into a successful audit."""
    path = write_json(tmp_path, "cycle.json", {"includes": ["cycle.json"]})
    with pytest.raises(ValueError, match="Cyclic"):
        guard.audit_file(path, tmp_path)
    path = write_json(tmp_path, "missing.json", {"includes": ["missing-child.yaml"]})
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        guard.audit_file(path, tmp_path)
    write_json(tmp_path, "child.json", {"seed": 1001})
    write_json(tmp_path, "nested/child.json", {"seed": 111})
    path = write_json(tmp_path, "nested/parent.json", {"includes": ["child.json"]})
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        guard.audit_file(path, tmp_path)


def test_definition_and_config_drift_fail(split_root: Path) -> None:
    """Byte-level freezing catches unreviewed definition/config edits."""
    path = split_root / guard.SCENARIO_FILE
    path.write_text(path.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        guard.validate_freeze(split_root, check_maps=False)
    shutil.copyfile(ROOT / guard.SCENARIO_FILE, path)
    path = split_root / guard.SPLIT_FILE
    path.write_text(path.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        guard.validate_freeze(split_root, check_maps=False)


def test_individual_row_hash_survives_aggregate_rehash(split_root: Path) -> None:
    """Updating only an aggregate digest cannot hide a changed development row."""
    path = split_root / guard.SCENARIO_FILE
    data = guard.read_data(path)
    data["scenarios"][0]["simulation_config"]["max_peds_per_group"] = 2
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    freeze = guard.read_data(split_root / guard.FREEZE_FILE)
    freeze["scenario_file_sha256"] = guard.sha256(path)
    write_json(split_root, guard.FREEZE_FILE.as_posix(), freeze)
    with pytest.raises(ValueError, match="Frozen variant hash mismatch"):
        guard.validate_freeze(split_root, check_maps=False)


def test_map_byte_pin_is_enforced(split_root: Path) -> None:
    """Small byte fixtures exercise the hash check; they are not simulator geometry."""
    freeze = guard.read_data(split_root / guard.FREEZE_FILE)
    for index, record in enumerate(freeze["variants"]):
        path = split_root / record["map_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"map-byte-fixture-{index}".encode()
        path.write_bytes(content)
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        record["map_git_blob_sha1"] = hashlib.sha1(blob, usedforsecurity=False).hexdigest()
    write_json(split_root, guard.FREEZE_FILE.as_posix(), freeze)
    guard.validate_freeze(split_root)
    (split_root / freeze["variants"][0]["map_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen map bytes changed"):
        guard.validate_freeze(split_root)


def valid_log(root: Path) -> dict:
    """Build one fully attributable development-only trial ledger."""
    config = write_json(root, "trial_config.json", {"gain": 0.5})
    return {
        "schema_version": "planner-tuning-log.v1", "split": guard.SPLIT,
        "scenario_file_sha256": guard.sha256(root / guard.SCENARIO_FILE),
        "trials": [{
            "trial_id": "trial-001", "seeds": [1001, 1030],
            "scenario_ids": list(guard.EXPECTED), "config_path": "trial_config.json",
            "config_sha256": guard.sha256(config), "changes": "Unit-test trial; not a tuning run.",
        }],
    }


def test_empty_registered_log_and_valid_trial(split_root: Path) -> None:
    """The initial empty log claims no tuning; later trials require concrete provenance."""
    source = ROOT / guard.TUNING_DIR / "tuning_log_v1.json"
    guard.validate_log(source, ROOT)
    path = write_json(split_root, "log.json", valid_log(split_root))
    guard.validate_log(path, split_root)


@pytest.mark.parametrize(
    "mutation", ["seed", "scenario", "digest", "missing", "duplicate", "command"]
)
def test_invalid_tuning_log_fails(split_root: Path, mutation: str) -> None:
    """Reject seed/scenario leaks, stale configs, incomplete trials and textual leaks."""
    data = valid_log(split_root)
    trial = data["trials"][0]
    if mutation == "seed":
        trial["seeds"] = [111]
    elif mutation == "scenario":
        trial["scenario_ids"] = ["classic_doorway_medium"]
    elif mutation == "digest":
        trial["config_sha256"] = "0" * 64
    elif mutation == "missing":
        del trial["changes"]
    elif mutation == "duplicate":
        data["trials"].append(trial.copy())
    else:
        trial["command"] = "runner --seed 140"
    path = write_json(split_root, "log.json", data)
    with pytest.raises(ValueError):
        guard.validate_log(path, split_root)


def test_jsonl_trial_is_bound_and_seed_checked(split_root: Path) -> None:
    """Line-oriented logs must carry the same split and config provenance."""
    data = valid_log(split_root)
    trial = {**data["trials"][0], "split": guard.SPLIT,
             "scenario_file_sha256": data["scenario_file_sha256"]}
    path = split_root / "log.jsonl"
    path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
    guard.validate_log(path, split_root)
    trial["seeds"] = [140]
    path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.validate_log(path, split_root)


def test_implicit_seed_policy_and_wrong_matrix_are_rejected(split_root: Path) -> None:
    """Launch configs must not inherit runner defaults or select a release matrix."""
    path = write_json(split_root, "config.json", {"planners": []})
    with pytest.raises(ValueError, match="must declare"):
        guard.validate_tuning_config(path, split_root)
    write_json(split_root, "release.json", {"scenarios": [{"name": "release"}]})
    write_json(split_root, "config.json", {
        "scenario_matrix": "release.json", "seed_policy": {"mode": "fixed-list", "seeds": [1001]},
    })
    with pytest.raises(ValueError, match="frozen development scenarios"):
        guard.validate_tuning_config(path, split_root)


@pytest.mark.parametrize("row", [
    {"name": next(iter(guard.EXPECTED))},
    {"name": "renamed", "metadata": {"development_only": True}},
    {"name": "renamed", "metadata": {"split": guard.SPLIT}},
])
def test_release_admission_rejects_development_rows(row: dict) -> None:
    """Distinct names and the development markers all prevent release admission."""
    with pytest.raises(ValueError, match="admitted to release matrix"):
        guard.check_release_rows([row], "fixture")
    guard.check_release_rows([{"name": "classic_doorway_medium"}], "fixture")


def test_repository_integration() -> None:
    """Run the complete acceptance command with actual map bytes and canonical loading."""
    assert guard.main(["--root", str(ROOT)]) == 0


def test_planner_overrides_are_not_scenario_definition_overrides(tmp_path: Path) -> None:
    """Real candidate manifests may tune planner parameters by scenario name."""
    write_json(tmp_path, "base.json", {"gain": 0.5})
    path = write_json(tmp_path, "candidate.json", {
        "algo": "hybrid_rule_local_planner", "base_config_path": "base.json",
        "params": {"gain": 0.5},
        "scenario_overrides": {"dev_v1__classic_doorway_medium": {"gain": 0.6}},
    })
    guard.audit_file(path, tmp_path)
    path = write_json(tmp_path, "scenario.json", {
        "scenario_overrides": {"simulation_config": {"ped_density": 0.9}},
    })
    with pytest.raises(ValueError, match="would change the frozen split"):
        guard.audit_file(path, tmp_path)


def test_seed_reference_in_comment_is_not_discarded(tmp_path: Path) -> None:
    """The acceptance guard covers textual seed references in structured artifacts."""
    path = tmp_path / "config.yaml"
    path.write_text("seeds: [1001]\n# Previous tuning used seed 111\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Release seeds forbidden"):
        guard.audit_file(path, tmp_path)


def test_unsupported_log_format_fails_closed(split_root: Path) -> None:
    """Unstructured console output is not accepted as the tuning provenance ledger."""
    path = split_root / "console.log"
    path.write_text("seed 111\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported tuning artifact format"):
        guard.validate_log(path, split_root)
