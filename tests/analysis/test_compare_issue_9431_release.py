"""Focused tests for the issue #9431 release comparator."""

from __future__ import annotations

import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
import yaml

import scripts.analysis.compare_issue_9431_release as release_diff
from robot_sf.training.scenario_loader import load_scenarios
from scripts.analysis.compare_issue_9431_release import (
    EXPECTED_ARM_KEYS,
    EXPECTED_PREDECESSOR_SCENARIO_MANIFEST,
    EXPECTED_PREDECESSOR_SCENARIO_MANIFEST_SHA256,
    EXPECTED_SCENARIO_IDS,
    EXPECTED_SEEDS,
    EXPECTED_SUCCESSOR_SCENARIO_MANIFEST,
    EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256,
    _bundle_campaign_id,
    _execution_audit,
    _markdown,
    _validate_matrix,
    _validate_successor_bundle_root,
)


def test_execution_audit_admits_mixed_mode_and_outcome_failures() -> None:
    row = {
        "status": "collision",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "mixed"},
            "fallback_used": False,
            "degraded": False,
            "unavailable": False,
        },
    }

    assert _execution_audit(row) == []


def test_execution_audit_rejects_fallback_and_runtime_error() -> None:
    row = {
        "status": "runtime_error",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "fallback_used": True,
        },
    }

    assert _execution_audit(row) == [
        "algorithm_metadata.fallback_used=true",
        "status=runtime_error",
    ]


@pytest.mark.parametrize(
    ("runtime_metadata", "expected_marker"),
    [
        ({"status": "degraded"}, "status=degraded"),
        ({"planner_runtime": {"readiness_status": "fallback"}}, "readiness_status=fallback"),
        ({"planner_runtime": {"fallback_used": True}}, "fallback_used=true"),
        ({"planner_diagnostics": {"fallback_count": 1}}, "fallback_count=1"),
    ],
)
def test_execution_audit_rejects_nested_runtime_fallback_or_degradation(
    runtime_metadata: dict[str, object], expected_marker: str
) -> None:
    row = {
        "status": "completed",
        "algorithm_metadata": {
            **runtime_metadata,
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "config": {"fallback_to_goal": False},
            "planner_kinematics": {"execution_mode": "native"},
        },
    }

    assert expected_marker in " ".join(_execution_audit(row, expected_algorithm="guarded_ppo"))


def test_execution_audit_admits_guarded_ppo_config_and_native_safe_intervention() -> None:
    row = {
        "status": "completed",
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "config": {"fallback_to_goal": False},
            "planner_kinematics": {"execution_mode": "native"},
            "guard_stats": {"fallback_safe": 1},
            "shield_stats": {"decision_counts": {"fallback_safe": 0}},
        },
    }

    assert _execution_audit(row, expected_algorithm="guarded_ppo") == []


def test_guarded_ppo_safe_intervention_requires_validated_arm_identity() -> None:
    row = {
        "status": "completed",
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "config": {"fallback_to_goal": False},
            "planner_kinematics": {"execution_mode": "native"},
            "guard_stats": {"fallback_safe": 1},
        },
    }

    issues = _execution_audit(row, expected_algorithm="ppo")

    assert "algorithm_metadata.guard_stats.fallback_safe=1" in issues


def test_execution_audit_rejects_malformed_guarded_ppo_safe_counter() -> None:
    row = {
        "status": "completed",
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "config": {"fallback_to_goal": False},
            "planner_kinematics": {"execution_mode": "native"},
            "guard_stats": {"fallback_safe": "1"},
        },
    }

    issues = _execution_audit(row, expected_algorithm="guarded_ppo")

    assert "algorithm_metadata.guard_stats.fallback_safe=invalid" in issues


def test_markdown_records_exact_source_range_and_execution_modes() -> None:
    report = {
        "predecessor": {
            "archive_sha256": "old-archive",
            "source_commit": "old-source",
        },
        "successor": {
            "source_commit": "new-source",
            "bundle_sha256": "new-bundle",
        },
        "paired_rows": 0,
        "changed_episode_count": 0,
        "execution_audit": {"admitted": True},
        "arms": {},
        "seed_block_bootstrap": {},
        "matrix_identity": {
            "planner_arms": sorted(EXPECTED_ARM_KEYS),
            "scenario_ids": sorted(EXPECTED_SCENARIO_IDS),
            "seeds": sorted(EXPECTED_SEEDS),
            "total_episodes": len(EXPECTED_ARM_KEYS)
            * len(EXPECTED_SCENARIO_IDS)
            * len(EXPECTED_SEEDS),
            "scenario_manifests": {
                "predecessor": {"manifest": "old.yaml", "sha256": "old-sha"},
                "successor": {"manifest": "new.yaml", "sha256": "new-sha"},
            },
            "scenario_definitions_identical": False,
        },
    }

    rendered = _markdown(report)

    assert "`old-source..new-source`" in rendered
    assert "native, adapter, or mixed mode" in rendered
    assert "descriptive multi-change release comparison" in rendered
    assert "not a single-change causal ablation" in rendered
    assert "scenario definitions differ between releases" in rendered
    assert "old.yaml" in rendered
    assert "new.yaml" in rendered


def test_markdown_keeps_missing_goal_adjacent_labels_explicitly_unavailable() -> None:
    report = {
        "predecessor": {"archive_sha256": "old-archive", "source_commit": "old-source"},
        "successor": {
            "source_commit": "new-source",
            "bundle_sha256": "new-bundle",
        },
        "paired_rows": 1,
        "changed_episode_count": 0,
        "execution_audit": {"admitted": True},
        "arms": {
            "goal": {
                "paired_rows": 1,
                "predecessor_outcomes": {"success": 1},
                "successor_outcomes": {"success": 1},
                "predecessor_goal_adjacent_timeout": {"unavailable": 1},
                "successor_goal_adjacent_timeout": {"unavailable": 1},
            }
        },
        "seed_block_bootstrap": {},
        "matrix_identity": {
            "planner_arms": sorted(EXPECTED_ARM_KEYS),
            "scenario_ids": sorted(EXPECTED_SCENARIO_IDS),
            "seeds": sorted(EXPECTED_SEEDS),
            "total_episodes": len(EXPECTED_ARM_KEYS)
            * len(EXPECTED_SCENARIO_IDS)
            * len(EXPECTED_SEEDS),
            "scenario_manifests": {
                "predecessor": {"manifest": "old.yaml", "sha256": "old-sha"},
                "successor": {"manifest": "new.yaml", "sha256": "new-sha"},
            },
            "scenario_definitions_identical": False,
        },
    }

    rendered = _markdown(report)

    assert "old `unavailable=1`; new `unavailable=1`" in rendered
    assert "{'unavailable': 1}" not in rendered


def test_successor_root_must_match_pinned_bundle_episode_bytes(tmp_path) -> None:
    payload = b'{"scenario_id":"s1","seed":1}\n'
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo("release/payload/runs/goal__differential_drive/episodes.jsonl")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))

    root = tmp_path / "root"
    run_dir = root / "runs/goal__differential_drive"
    run_dir.mkdir(parents=True)
    episode_file = run_dir / "episodes.jsonl"
    episode_file.write_bytes(payload)

    _validate_successor_bundle_root(bundle, root)

    episode_file.write_bytes(b'{"scenario_id":"other","seed":1}\n')
    with pytest.raises(ValueError, match="episode bytes do not match"):
        _validate_successor_bundle_root(bundle, root)


def test_bundle_campaign_id_comes_from_pinned_campaign_manifest(tmp_path) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    manifest_bytes = b'{"campaign_id":"issue9431_release_campaign"}'
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo("publication/payload/campaign_manifest.json")
        member.size = len(manifest_bytes)
        archive.addfile(member, BytesIO(manifest_bytes))

    assert _bundle_campaign_id(bundle) == "issue9431_release_campaign"


def test_bundle_scenario_identity_comes_from_resolved_release_manifest(tmp_path) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    manifest_bytes = (
        b'{"scenario":{"matrix_path":"configs/scenarios/successor.yaml",'
        b'"matrix_sha256":"'
        b'03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"}}'
    )
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo("release/payload/release/release_manifest.resolved.json")
        member.size = len(manifest_bytes)
        archive.addfile(member, BytesIO(manifest_bytes))

    assert release_diff._bundle_scenario_identity(bundle) == {
        "manifest": "configs/scenarios/successor.yaml",
        "sha256": "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c",
    }


def test_matrix_validation_rejects_equal_sized_noncanonical_scenario_seed_set() -> None:
    expected_scenarios = {"scenario-a", "scenario-b"}
    expected_seeds = {22, 23}
    actual = {
        ("scenario-a", 22): {},
        ("scenario-a", 23): {},
        ("scenario-b", 22): {},
        ("unexpected-scenario", 24): {},
    }

    with pytest.raises(ValueError, match="canonical scenario/seed matrix"):
        _validate_matrix(
            {"goal": actual},
            {"goal": actual},
            expected_arm_keys={"goal"},
            expected_scenario_ids=expected_scenarios,
            expected_seeds=expected_seeds,
        )


def test_frozen_matrix_matches_the_versioned_issue_9431_sources() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    predecessor_scenario_manifest = repo_root / EXPECTED_PREDECESSOR_SCENARIO_MANIFEST
    successor_scenario_manifest = repo_root / EXPECTED_SUCCESSOR_SCENARIO_MANIFEST
    campaign_template = (
        repo_root
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
    )
    seed_sets = repo_root / "configs/benchmarks/seed_sets_v1.yaml"

    assert (
        sha256(predecessor_scenario_manifest.read_bytes()).hexdigest()
        == EXPECTED_PREDECESSOR_SCENARIO_MANIFEST_SHA256
    )
    assert (
        sha256(successor_scenario_manifest.read_bytes()).hexdigest()
        == EXPECTED_SUCCESSOR_SCENARIO_MANIFEST_SHA256
    )
    predecessor_ids = {str(row["name"]) for row in load_scenarios(predecessor_scenario_manifest)}
    successor_ids = {str(row["name"]) for row in load_scenarios(successor_scenario_manifest)}
    resolved_template = yaml.safe_load(campaign_template.read_text(encoding="utf-8"))
    resolved_seed_sets = yaml.safe_load(seed_sets.read_text(encoding="utf-8"))

    assert predecessor_ids == EXPECTED_SCENARIO_IDS
    assert successor_ids == EXPECTED_SCENARIO_IDS
    assert resolved_template["scenario_matrix"] == EXPECTED_SUCCESSOR_SCENARIO_MANIFEST
    assert {row["key"] for row in resolved_template["planners"]} == EXPECTED_ARM_KEYS
    assert set(resolved_seed_sets[resolved_template["seed_policy"]["seed_set"]]) == EXPECTED_SEEDS
