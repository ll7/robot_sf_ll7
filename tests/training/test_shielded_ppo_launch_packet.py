"""Tests for shielded-PPO repair launch-packet validation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from robot_sf.training.shielded_ppo_launch_packet import (
    ShieldedPPOLaunchPacketError,
    validate_launch_packet,
)
from scripts.validation.validate_shielded_ppo_launch_packet import main as validate_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_packet(tmp_path: Path, packet: dict[str, object]) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _valid_packet(tmp_path: Path) -> dict[str, object]:
    fixture = tmp_path / "freeze.json"
    fixture.write_text('{"fixture": true}\n', encoding="utf-8")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    return {
        "schema_version": "shielded-ppo-repair-launch-packet.v1",
        "campaign_id": "demo_shielded_ppo_repair",
        "generating_commit": "e14e2f8bc2058d9f0e071219629915dd5b5dd5a8",
        "slurm_handoff": "docs/context/policy_search/SLURM/002_shielded_ppo_repair_campaign.md",
        "repair_hypothesis": {
            "id": "collision_penalty_only_v1",
            "statement": "single repair delta",
            "enabled_deltas": [
                {
                    "type": "reward_weight",
                    "parameter": "env_factory_kwargs.reward_kwargs.weights.collision",
                    "value": -20.0,
                }
            ],
        },
        "training_starting_points": {
            "base_ppo_training_config": "configs/training/ppo/"
            "expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml",
            "guarded_candidate_config": "configs/policy_search/candidates/risk_guarded_ppo_v1.yaml",
            "guarded_algo_config": "configs/algos/guarded_ppo_relaxed_v2.yaml",
            "checksums": {
                "configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml": (
                    "91f9d9a7c0bcb54f3628beef9140181a64277ab8bdf9b12dcdc37f0251a7d437"
                ),
                "configs/policy_search/candidates/risk_guarded_ppo_v1.yaml": (
                    "69030a865be80be5e3ba98db447551c3707f966de8ac5f216e66044d9ea50735"
                ),
            },
        },
        "runtime_guard": {
            "active_in_all_evaluations": True,
            "evaluation_candidate_config": "configs/policy_search/candidates/risk_guarded_ppo_v1.yaml",
            "required_diagnostics": [
                "guard_veto_count",
                "guard_fallback_count",
                "raw_ppo_action",
                "guarded_action",
                "fallback_action_source",
            ],
        },
        "comparison_references": {
            "ppo_baseline": _reference(fixture, digest, "ppo_issue791_best_v1"),
            "risk_guarded_ppo_v1": _reference(fixture, digest, "risk_guarded_ppo_v1"),
        },
        "stop_gates": {
            "smoke": {
                "min_success_rate": 1.0,
                "max_collision_rate": 0.0,
                "max_guard_fallback_rate": 0.6,
            },
            "nominal_sanity": {
                "min_success_rate": 0.2778,
                "max_collision_rate": 0.0556,
                "max_guard_fallback_rate": 0.5,
                "allows_stress_escalation": True,
            },
        },
        "execution_boundary": {
            "full_training_in_this_issue": False,
            "submit_slurm_from_this_issue": False,
            "local_preflight_command": "uv run python scripts/validation/"
            "validate_shielded_ppo_launch_packet.py --config packet.yaml --json",
            "slurm_command_shape": "submit follow-up only",
        },
    }


def _reference(fixture: Path, digest: str, candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "config": "configs/policy_search/candidates/risk_guarded_ppo_v1.yaml",
        "scenario_slice": "nominal_sanity",
        "seeds": [111, 112, 113],
        "source_report": "docs/context/policy_search/reports/"
        "2026-04-29_risk_guarded_ppo_v1_nominal_sanity.md",
        "summary_artifacts": [str(fixture), f"wandb-artifact://robot-sf/{candidate_id}:pending"],
        "checksums": {str(fixture): digest},
    }


def test_issue_1396_launch_packet_validates() -> None:
    """The checked-in #1396 launch packet should pass local preflight."""
    report = validate_launch_packet(
        _REPO_ROOT / "configs/training/shielded_ppo_issue_1396_launch_packet.yaml",
        repo_root=_REPO_ROOT,
    )

    assert report["status"] == "valid"
    assert report["campaign_id"] == "issue_1396_shielded_ppo_repair_v1"
    assert report["repair_hypothesis"]["id"] == "collision_penalty_only_v1"


def test_validate_launch_packet_rejects_multiple_repair_deltas(tmp_path: Path) -> None:
    """The pre-SLURM packet must keep exactly one repair hypothesis."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["repair_hypothesis"]["enabled_deltas"].append(
        {"type": "curriculum", "parameter": "scenario_sampling", "value": "easy_first"}
    )

    with pytest.raises(ShieldedPPOLaunchPacketError, match="exactly one delta"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_disabled_guard(tmp_path: Path) -> None:
    """Runtime guard must remain active in all evaluations."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["runtime_guard"]["active_in_all_evaluations"] = False

    with pytest.raises(ShieldedPPOLaunchPacketError, match="active_in_all_evaluations"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_missing_nominal_gate(tmp_path: Path) -> None:
    """Nominal sanity must gate stress/full-matrix escalation."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["stop_gates"].pop("nominal_sanity")

    with pytest.raises(ShieldedPPOLaunchPacketError, match="stop_gates missing"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_cli_reports_json() -> None:
    """The CLI should expose a machine-readable valid report."""
    exit_code = validate_cli_main(
        [
            "--config",
            str(_REPO_ROOT / "configs/training/shielded_ppo_issue_1396_launch_packet.yaml"),
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
        ]
    )

    assert exit_code == 0
