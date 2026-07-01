"""Tests oracle-imitation collection-readiness decision manifests (issue #1496)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import yaml

from scripts.validation.validate_oracle_imitation_launch_packet import main as validate_cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _write_packet(tmp_path: Path, packet: dict[str, object]) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _valid_packet(tmp_path: Path) -> dict[str, object]:
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"status": "dry-run"}\n', encoding="utf-8")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    return {
        "schema_version": "oracle-imitation-launch-packet.v1",
        "dataset_id": "demo_oracle_imitation",
        "source_candidate": "hybrid_rule_v3_static_margin0_waypoint2",
        "source_candidate_config": (
            "configs/policy_search/candidates/hybrid_rule_v3_static_margin0_waypoint2.yaml"
        ),
        "source_report": (
            "docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md"
        ),
        "split_contract": "docs/context/policy_search/contracts/oracle_imitation_dataset_split.md",
        "scenario_source": "configs/policy_search/nominal_sanity_matrix.yaml",
        "scenario_ids": ["planner_sanity_simple", "classic_crossing_low"],
        "seed_set_refs": {
            "manifest": "configs/benchmarks/seed_sets_v1.yaml",
            "validation": "dev",
            "evaluation": "eval",
            "train_excludes": "paper_eval_s20",
        },
        "seeds_by_split": {
            "train": [201, 202],
            "validation": [101, 102, 103],
            "evaluation": [111, 112, 113],
        },
        "episode_ids_by_split": {
            "train": ["train__planner_sanity_simple__seed201"],
            "validation": ["validation__planner_sanity_simple__seed101"],
            "evaluation": ["evaluation__planner_sanity_simple__seed111"],
        },
        "hard_slice_assignment": [
            {
                "slice_id": "nominal_slice",
                "split": "evaluation",
                "predeclared_for_evaluation": True,
            }
        ],
        "relabeling_policy": None,
        "exclusion_rules": ["exclude eval seeds from train"],
        "provenance": "unit-test fixture",
        "created_at": "2026-05-24T08:55:00+00:00",
        "generating_commit": "e14e2f8bc2058d9f0e071219629915dd5b5dd5a8",
        "artifact_paths": {
            "dry_run_fixture": str(fixture),
            "future_dataset_npz_uri": "wandb-artifact://robot-sf/demo:pending",
        },
        "checksums": {str(fixture): digest},
        "collection_roots": {
            "log_root": "wandb-artifact://robot-sf/demo-logs:pending",
            "dataset_output_root": "wandb-artifact://robot-sf/demo-traces:pending",
            "manifest_destination": "wandb-artifact://robot-sf/demo-manifest:pending",
        },
    }


def test_cli_writes_collection_decision_manifest_for_issue_packet(tmp_path: Path) -> None:
    """The CLI writes a read-only collection-readiness decision manifest."""
    output = tmp_path / "decision" / "collection.json"

    exit_code = validate_cli_main(
        [
            "--config",
            "configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml",
            "--output",
            str(output),
            "--json",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema"] == "oracle-imitation-collection-readiness-decision.v1"
    assert payload["issue"] == 1496
    assert payload["status"] == "ready"
    assert payload["report"]["training_ready"] is False
    assert payload["forbidden_actions_confirmed"] == {
        "data_collection": False,
        "compute_submit": False,
        "training": False,
    }


def test_cli_records_collection_manifest_destination_blocker(tmp_path: Path) -> None:
    """Missing output-manifest destination becomes a structured blocker."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    del broken["collection_roots"]["manifest_destination"]
    config = _write_packet(tmp_path, broken)
    output = tmp_path / "collection-decision.json"

    exit_code = validate_cli_main(["--config", str(config), "--output", str(output), "--json"])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["report"] is None
    assert payload["blockers"] == [
        "collection_roots.manifest_destination must be a non-empty string"
    ]


def test_cli_records_missing_provenance_blocker(tmp_path: Path) -> None:
    """Missing provenance field becomes a structured blocker without collecting data."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    del broken["generating_commit"]
    config = _write_packet(tmp_path, broken)
    output = tmp_path / "collection-decision.json"

    exit_code = validate_cli_main(["--config", str(config), "--output", str(output), "--json"])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["generating_commit must be a 40-character git SHA"]
