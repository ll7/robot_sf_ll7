"""Tests for adversarial failure archive curation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.archive import (
    curate_failure_archive,
    failure_archive_feature_rows,
    failure_archive_index,
)
from robot_sf.adversarial.disjoint_evaluation import assess_archive_readiness
from scripts.tools.curate_adversarial_failure_archive import main as archive_cli_main


def _candidate(start_x: float, *, seed: int) -> dict:
    """Build a compact candidate manifest payload."""
    return {
        "start": {"x": start_x, "y": 2.0, "theta": 0.0},
        "goal": {"x": 5.0, "y": 2.0, "theta": 0.0},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": seed,
    }


def _manifest(tmp_path: Path) -> Path:
    """Write a synthetic adversarial search manifest."""
    payload = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "policy": "goal",
            "scenario_template": "configs/scenarios/templates/crossing_ttc.yaml",
            "search_space": {
                "variables": {
                    "start_x": {"min": 0.0, "max": 4.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 9},
                }
            },
        },
        "candidates": [
            {
                "candidate": _candidate(0.25, seed=7),
                "objective_value": 9.0,
                "bundle_path": "output/adversarial/run/candidate_0000",
                "scenario_yaml_path": "output/adversarial/run/candidate_0000/scenario.yaml",
                "certification_status": {
                    "schema_version": "scenario_cert.v1",
                    "status": "passed",
                    "reason": "fixture certified",
                    "details": {},
                },
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "collision",
                    "reasons": ["collision"],
                    "details": {
                        "termination_reason": "collision",
                        "outcome": {"collision": True, "route_complete": False},
                    },
                },
            },
            {
                "candidate": _candidate(1.75, seed=8),
                "objective_value": 7.0,
                "bundle_path": "output/adversarial/run/candidate_0001",
                "scenario_yaml_path": "output/adversarial/run/candidate_0001/scenario.yaml",
                "candidate_certification": {
                    "schema_version": "scenario_cert.v1",
                    "status": "passed",
                    "reason": "legacy fixture certified",
                    "details": {},
                },
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "collision",
                    "reasons": ["collision duplicate"],
                    "details": {
                        "termination_reason": "collision",
                        "outcome": {"collision": True, "route_complete": False},
                    },
                },
            },
            {
                "candidate": _candidate(2.0, seed=9),
                "objective_value": 3.0,
                "bundle_path": "output/adversarial/run/candidate_0002",
                "scenario_yaml_path": "output/adversarial/run/candidate_0002/scenario.yaml",
                "certification_status": {
                    "schema_version": "scenario_cert.v1",
                    "status": "passed",
                    "reason": "fixture certified",
                    "details": {},
                },
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "timeout",
                    "reasons": ["timeout"],
                    "details": {
                        "termination_reason": "timeout",
                        "outcome": {"timeout": True, "route_complete": False},
                    },
                },
            },
            {
                "candidate": _candidate(2.0, seed=9),
                "objective_value": -1.0,
                "bundle_path": "output/adversarial/run/candidate_0003",
                "scenario_yaml_path": "output/adversarial/run/candidate_0003/scenario.yaml",
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "success",
                    "reasons": ["success"],
                    "details": {
                        "termination_reason": "success",
                        "outcome": {"route_complete": True},
                    },
                },
            },
            {
                "candidate": _candidate(3.0, seed=9),
                "objective_value": None,
                "bundle_path": "output/adversarial/run/candidate_0004",
                "scenario_yaml_path": "output/adversarial/run/candidate_0004/scenario.yaml",
                "failure_attribution": {
                    "status": "not_evaluated",
                    "primary_failure": "invalid_candidate",
                    "reasons": ["certification failed"],
                    "details": {},
                },
            },
            {
                "candidate": _candidate(1.0, seed=10),
                "objective_value": 99.0,
                "bundle_path": "output/adversarial/run/candidate_0005",
                "scenario_yaml_path": "output/adversarial/run/candidate_0005/scenario.yaml",
                "failure_attribution": {
                    "status": "evaluation_failed",
                    "primary_failure": "simulation_error",
                    "reasons": ["spawn collision or unreachable goal"],
                    "details": {
                        "termination_reason": "simulation_error",
                    },
                },
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_curate_failure_archive_groups_and_selects_representatives(tmp_path: Path) -> None:
    """Archive curation should group duplicate mechanisms and mark minimal representatives."""
    manifest_path = _manifest(tmp_path)
    output_path = tmp_path / "archive.json"

    archive = curate_failure_archive([manifest_path], output_path=output_path)

    assert output_path.exists()
    assert archive["schema_version"] == "adversarial_failure_archive.v1"
    assert archive["summary"] == {
        "source_manifest_count": 1,
        "source_candidate_count": 6,
        "archived_failure_count": 3,
        "cluster_count": 2,
    }
    assert [cluster["mechanism"]["primary_failure"] for cluster in archive["clusters"]] == [
        "collision",
        "timeout",
    ]
    collision_cluster = archive["clusters"][0]
    assert collision_cluster["member_count"] == 2
    assert collision_cluster["representative_archive_id"] == "failure_0001"
    representative = next(
        entry
        for entry in archive["entries"]
        if entry["archive_id"] == collision_cluster["representative_archive_id"]
    )
    assert representative["source_candidate_index"] == 1
    assert representative["candidate_certification"]["status"] == "passed"
    assert representative["replay_command"].startswith("uv run robot_sf_bench run")


def test_curated_certified_archive_can_satisfy_readiness_gate(tmp_path: Path) -> None:
    """Curation preserves certification metadata needed by archive readiness."""
    archive = curate_failure_archive([_manifest(tmp_path)], output_path=tmp_path / "archive.json")
    archive["null_test_manifest"] = {
        "required_tests": [
            "shuffled_outcome_label_permutation",
            "ranking_permutation",
        ],
        "n_permutations": 100,
    }

    report = assess_archive_readiness(archive)

    assert report.ready is True
    assert report.entries_missing_certification_status == 0
    assert report.entries_not_certified == 0


def test_curate_failure_archive_is_deterministic(tmp_path: Path) -> None:
    """Repeated curation of the same manifest should produce identical stable payloads."""
    manifest_path = _manifest(tmp_path)

    left = curate_failure_archive([manifest_path], output_path=tmp_path / "left.json")
    right = curate_failure_archive([manifest_path], output_path=tmp_path / "right.json")

    left.pop("created_at")
    right.pop("created_at")
    assert left == right


def test_simulation_error_candidates_excluded_from_archive(tmp_path: Path) -> None:
    """simulation_error must not enter archive entries, clusters, or representatives."""
    manifest_path = _manifest(tmp_path)
    output_path = tmp_path / "archive.json"

    archive = curate_failure_archive([manifest_path], output_path=output_path)

    primary_failures_in_entries = {
        entry["failure_attribution"]["primary_failure"] for entry in archive["entries"]
    }
    cluster_mechanisms = {
        cluster["mechanism"]["primary_failure"] for cluster in archive["clusters"]
    }
    representative_ids = {cluster["representative_archive_id"] for cluster in archive["clusters"]}
    representative_failures = {
        entry["failure_attribution"]["primary_failure"]
        for entry in archive["entries"]
        if entry["archive_id"] in representative_ids
    }

    assert "simulation_error" not in primary_failures_in_entries, (
        "simulation_error candidate should not appear in archive entries"
    )
    assert "simulation_error" not in cluster_mechanisms, (
        "simulation_error candidate should not appear in cluster mechanisms"
    )
    assert "simulation_error" not in representative_failures, (
        "simulation_error candidate should not appear in cluster representatives"
    )
    assert archive["summary"]["source_candidate_count"] == 6, (
        "source_candidate_count must still count simulation_error for budget auditing"
    )


def test_failure_archive_feature_rows_are_deterministic_export_slice(tmp_path: Path) -> None:
    """Archive feature rows expose stable scalar metadata for proposal fixtures."""
    manifest_path = _manifest(tmp_path)
    archive = curate_failure_archive([manifest_path], output_path=tmp_path / "archive.json")

    rows = failure_archive_feature_rows(archive)

    assert [row["archive_id"] for row in rows] == [
        "failure_0000",
        "failure_0001",
        "failure_0002",
    ]
    assert rows[0] == {
        "archive_id": "failure_0000",
        "cluster_key": (
            '{"policy":"goal","primary_failure":"collision",'
            '"scenario_template":"configs/scenarios/templates/crossing_ttc.yaml",'
            '"termination_reason":"collision"}'
        ),
        "source_manifest": manifest_path.as_posix(),
        "source_candidate_index": 0,
        "bundle_path": "output/adversarial/run/candidate_0000",
        "scenario_yaml_path": "output/adversarial/run/candidate_0000/scenario.yaml",
        "start_x": 0.25,
        "start_y": 2.0,
        "goal_x": 5.0,
        "goal_y": 2.0,
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": 7.0,
        "objective_value": 9.0,
        "primary_failure": "collision",
        "termination_reason": "collision",
        "normalized_perturbation": 0.4375,
        "replay_command": (
            "uv run robot_sf_bench run --matrix "
            "output/adversarial/run/candidate_0000/scenario.yaml "
            "--out output/adversarial/run/candidate_0000/episode_records_replay.jsonl "
            "--algo goal --no-video"
        ),
    }


def test_failure_archive_index_groups_export_rows(tmp_path: Path) -> None:
    """Archive index supports lookup by id, mechanism, failure, and seed."""
    archive = curate_failure_archive([_manifest(tmp_path)], output_path=tmp_path / "archive.json")

    index = failure_archive_index(archive)

    assert index["schema_version"] == "adversarial_failure_archive_index.v1"
    assert index["row_count"] == 3
    assert sorted(index["by_archive_id"]) == ["failure_0000", "failure_0001", "failure_0002"]
    assert index["by_primary_failure"] == {
        "collision": ["failure_0000", "failure_0001"],
        "timeout": ["failure_0002"],
    }
    assert index["by_scenario_seed"] == {
        "7": ["failure_0000"],
        "8": ["failure_0001"],
        "9": ["failure_0002"],
    }


def test_failure_archive_feature_rows_reject_malformed_archives() -> None:
    """Export helpers fail closed for non-archive payloads."""
    with pytest.raises(ValueError, match="Unsupported failure archive schema"):
        failure_archive_feature_rows({"schema_version": "unexpected", "entries": []})

    with pytest.raises(ValueError, match="entries must be a list"):
        failure_archive_index({"schema_version": "adversarial_failure_archive.v1", "entries": {}})


def test_curate_failure_archive_cli_writes_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI wrapper should write the archive and print a compact summary."""
    manifest_path = _manifest(tmp_path)
    output_path = tmp_path / "cli_archive.json"

    assert archive_cli_main([manifest_path.as_posix(), "--out", output_path.as_posix()]) == 0

    captured = json.loads(capsys.readouterr().out)
    assert captured["path"] == output_path.as_posix()
    assert captured["summary"]["archived_failure_count"] == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["cluster_count"] == 2
