"""Tests for the issue #5273 deterministic launch-packet preparation."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts.benchmark.prepare_issue_5273_s20_launch_packet import build_packet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(
    tmp_path: Path, *, planner_keys: tuple[str, ...] = ("goal", "social_force"), fallback: bool = False
) -> Path:
    source = tmp_path / "source.yaml"
    source.write_text("name: source\n", encoding="utf-8")
    manifest_dir = tmp_path / "split"
    manifest_dir.mkdir()
    children: list[dict[str, Any]] = []
    for planner_key in planner_keys:
        filename = f"arm_{planner_key}.yaml"
        planner = {"key": planner_key}
        if fallback and planner_key == "social_force":
            planner["missing_prereq_policy"] = "fallback"
        config = {
            "planners": [planner],
            "split_provenance": {
                "source_config": "source.yaml",
                "source_sha256": _sha256(source),
                "split_mode": "per_planner",
                "arm_key": planner_key,
                "arm_total": len(planner_keys),
            },
        }
        child_path = manifest_dir / filename
        child_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        children.append(
            {
                "filename": filename,
                "planner_keys": [planner_key],
                "sha256": _sha256(child_path),
            }
        )
    manifest = {
        "children": children,
        "source_config": "source.yaml",
        "source_sha256": _sha256(source),
        "split_mode": "per_planner",
    }
    manifest_path = manifest_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _preflight_runner(_root: Path, _config_path: Path, planner_key: str) -> dict[str, Any]:
    scenarios = ["scenario_a", "scenario_b"]
    seeds = [111, 112]
    identities = [
        {
            "planner_key": planner_key,
            "scenario_id": scenario,
            "seed": seed,
            "kinematics": "differential_drive",
        }
        for scenario in scenarios
        for seed in seeds
    ]
    return {
        "status": "passed",
        "command": f"preflight {planner_key}",
        "validate_config": {
            "config_sha256": _sha256(_config_path),
            "scenario_matrix": "configs/scenarios/test.yaml",
            "scenario_candidates": {"resolved": scenarios},
            "seed_policy": {
                "seed_set": "test_s2",
                "resolved_seeds": seeds,
            },
        },
        "preview_scenarios": {
            "truncated": False,
            "scenarios": [{"name": scenario} for scenario in scenarios],
        },
        "matrix_summary": {
            "rows": [
                {
                    "planner_key": planner_key,
                    "scenario_count": len(scenarios),
                    "resolved_seeds": seeds,
                    "repeats": len(seeds),
                    "kinematics": "differential_drive",
                    "scenario_horizons_path": "configs/policy_search/scenario_horizons_h500.yaml",
                }
            ]
        },
        "expected_identity_fixture": identities,
    }


def test_packet_is_deterministic_and_never_submit_ready(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path)

    first = build_packet(tmp_path, manifest, preflight_runner=_preflight_runner)
    second = build_packet(tmp_path, manifest, preflight_runner=_preflight_runner)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "prepared_not_submitted"
    assert first["submission_allowed"] is False
    assert first["production_execution_performed"] is False
    assert first["arm_count"] == 2
    assert first["expected_row_count"] == 8
    assert first["expected_row_count_complete"] is True
    assert all(arm["execution"]["status"] == "planned_not_executed" for arm in first["arms"])


def test_fallback_setting_is_explicitly_excluded(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path, fallback=True)

    packet = build_packet(tmp_path, manifest, preflight_runner=_preflight_runner)

    assert packet["status"] == "blocked"
    assert any(item["code"] == "fallback_enabled" for item in packet["blockers"])
    fallback_arm = next(arm for arm in packet["arms"] if arm["planner_key"] == "social_force")
    assert fallback_arm["aggregation"]["native_aggregation_eligible"] is False
    assert fallback_arm["aggregation"]["evidence_classification"] == "excluded"


def test_mutated_child_hash_fails_closed(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path, planner_keys=("goal",))
    child = tmp_path / "split" / "arm_goal.yaml"
    child.write_text(child.read_text(encoding="utf-8") + "# mutated\n", encoding="utf-8")

    packet = build_packet(tmp_path, manifest, preflight_runner=_preflight_runner)

    assert packet["status"] == "blocked"
    assert any(item["code"] == "child_hash_mismatch" for item in packet["blockers"])
    assert packet["submission_allowed"] is False


def test_preflight_failure_is_recorded_without_execution(tmp_path: Path) -> None:
    manifest = _write_fixture(tmp_path, planner_keys=("orca",))

    def blocked_runner(_root: Path, _config_path: Path, _planner_key: str) -> dict[str, Any]:
        return {
            "status": "blocked",
            "command": "preflight orca",
            "error_type": "OrcaRvo2PreflightError",
            "error": "rvo2 is not importable",
        }

    packet = build_packet(tmp_path, manifest, preflight_runner=blocked_runner)

    assert packet["status"] == "blocked"
    assert packet["expected_row_count_complete"] is False
    assert any(item["code"] == "canonical_preflight_failed" for item in packet["blockers"])
    assert packet["arms"][0]["execution"]["production_execution_performed"] is False
