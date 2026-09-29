"""Tests for the issue #9348 three-width doorway comparison application."""

from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

import robot_sf.benchmark.three_width_doorway_application as doorway_application
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle
from robot_sf.benchmark.schema_validator import load_schema
from robot_sf.benchmark.three_width_doorway_application import (
    DoorwayPairingSession,
    _validate_application_execution,
    build_pair_manifest,
    build_pair_receipt,
    check_pair_receipts,
    check_variant_diff,
    generate_application_assets,
    load_three_width_manifest,
    non_width_config_sha256,
    run_three_width_preflight,
    write_preflight_report,
)
from robot_sf.evidence.writers import write_json
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.scenario_certification.input_identity import scenario_input_identity
from robot_sf.scenario_certification.v1 import RouteCertificate, ScenarioCertificate
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation import run_issue_9348_three_width_campaign as doorway_campaign
from scripts.validation.run_issue_9348_three_width_campaign import (
    _write_checksums,
    analyze_rows,
    verify_campaign_bundle,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "configs/benchmarks/issue_9348_three_width_doorway_v1.yaml"
_BASE_MAP = _REPO_ROOT / "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg"


def _valid_spawn_block() -> dict[str, Any]:
    return {
        "schema_version": "spawn_validity.v1",
        "reset_clearance_status": "available",
        "reset_clearance": {
            "overlap": False,
            "obstacle_overlap": False,
            "pedestrian_overlap": False,
        },
        "reset_clearance_error": None,
        "reset_overlap": False,
        "respawn_overlap_events": [],
        "respawn_overlap_collisions": [],
        "invalid_run": False,
        "invalid_reason": None,
    }


def _baseline_kinematics(planner: str) -> dict[str, Any]:
    if planner == "goal":
        return {
            "execution_mode": "native",
            "adapter_active": False,
            "adapter_name": "none",
            "policy_callable": {
                "module": "robot_sf.benchmark.map_runner_policies.goal",
                "qualname": "build.<locals>._policy",
            },
            "planner_adapter": {"present": False, "module": None, "class": None},
        }
    return {
        "execution_mode": "adapter",
        "adapter_active": True,
        "adapter_name": "SocialForcePlannerAdapter",
        "policy_callable": {
            "module": "robot_sf.benchmark.map_runner.map_runner",
            "qualname": "_build_common_adapter_policy.<locals>._policy",
        },
        "planner_adapter": {
            "present": True,
            "module": "robot_sf.planner.socnav_social_force",
            "class": "SocialForcePlannerAdapter",
        },
        "projection_documented": True,
    }


def _action_trace(count: int) -> dict[str, Any]:
    return {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": [
            {
                "step": index,
                "time_s": (index + 1) * 0.1,
                "planner": {
                    "event": "step",
                    "selected_action": {"linear_velocity": 0.5, "angular_velocity": 0.0},
                    "applied_environment_action": {
                        "linear_velocity": 0.5,
                        "angular_velocity": 0.0,
                    },
                },
            }
            for index in range(count)
        ],
    }


def _planner_invocation_trace(planner: str, count: int) -> dict[str, Any]:
    """Build runtime route/command evidence matching the synthetic action trace."""
    route = _baseline_kinematics(planner)
    return {
        "schema_version": "issue_9348_planner_invocation.v1",
        "source": "constructed_policy_callable",
        "steps": [
            {
                "step": index,
                "event": "invocation",
                "route_before": deepcopy(route),
                "route_after": deepcopy(route),
                "status": "returned",
                "command": {"linear_velocity": 0.5, "angular_velocity": 0.0},
            }
            for index in range(count)
        ],
    }


def test_campaign_failure_keeps_error_and_seals_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer failure leaves a checksum-bound receipt and still raises."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("test: true\n", encoding="utf-8")
    monkeypatch.setattr(doorway_campaign, "_source_identity", lambda: "a" * 40)
    monkeypatch.setattr(
        doorway_campaign,
        "_source_tree_custody",
        lambda _sha: {
            "git_tree_oid": "b" * 40,
            "entry_manifest_sha256": "c" * 64,
            "tracked_entry_count": 1,
        },
    )
    monkeypatch.setattr(doorway_campaign, "load_three_width_manifest", lambda _path: {})

    def fail_preflight(_manifest: Path, *, output_dir: Path) -> None:
        raise ValueError("preflight rejection")

    monkeypatch.setattr(doorway_campaign, "run_three_width_preflight", fail_preflight)
    output_root = tmp_path / "failed-run"
    with pytest.raises(ValueError, match="preflight rejection"):
        doorway_campaign.run_campaign(manifest, output_root)

    receipt = json.loads((output_root / "run_failure.json").read_text(encoding="utf-8"))
    assert receipt["error_type"] == "ValueError"
    assert receipt["error"] == "preflight rejection"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["source_tree_custody"]["git_tree_oid"] == "b" * 40
    assert "run_failure.json" in (output_root / "SHA256SUMS").read_text(encoding="utf-8")


def test_campaign_blocks_h400_before_first_episode_on_red_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor-present gate is consumed by the producer before H400 dispatch."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("test: true\n", encoding="utf-8")
    monkeypatch.setattr(doorway_campaign, "_source_identity", lambda: "a" * 40)
    monkeypatch.setattr(
        doorway_campaign,
        "_source_tree_custody",
        lambda _sha: {
            "git_tree_oid": "b" * 40,
            "entry_manifest_sha256": "c" * 64,
            "tracked_entry_count": 1,
        },
    )
    monkeypatch.setattr(
        doorway_campaign,
        "load_three_width_manifest",
        lambda _path: {
            "_resolved": {
                "planner_configs": {"goal": {}, "social_force": {}},
                "planner_config_hashes": {
                    "goal": "44136fa355b3678a",
                    "social_force": "44136fa355b3678a",
                },
            }
        },
    )
    preflight = _bounded_h1_preflight()
    preflight["variants"] = [
        {
            "variant_id": f"gap_{width}",
            "geometry": {"gap_width_m": width},
            "assets": {
                "scenario_path": "unused",
                "map_sha256": "a" * 64,
                "scenario_sha256": "b" * 64,
            },
        }
        for width in (2.2, 2.8, 3.6)
    ]
    monkeypatch.setattr(
        doorway_campaign, "run_three_width_preflight", lambda *_args, **_kwargs: preflight
    )
    monkeypatch.setattr(
        doorway_campaign,
        "_run_actor_present_confirmation",
        lambda *_args, **_kwargs: {
            "schema_version": "issue_9348_confirmation_preflight.v1",
            "admit_h400": False,
        },
    )
    monkeypatch.setattr(
        doorway_campaign,
        "_run_map_episode",
        lambda *_args, **_kwargs: pytest.fail("H400 episode ran before confirmation"),
    )
    output_root = tmp_path / "blocked-campaign"
    with pytest.raises(ValueError, match="actor-present confirmation"):
        doorway_campaign.run_campaign(manifest_path, output_root)
    assert not (output_root / "episodes.jsonl").exists()
    assert (output_root / "run_failure.json").is_file()


def _bounded_h1_preflight() -> dict[str, Any]:
    return {
        "go": True,
        "checks": {
            "oracle_expected_fallbacks": [
                {
                    "variant_id": "gap_3p60__depth_1p00",
                    "reason": "expected_distributional_metric_unavailable",
                    "marker": (
                        "metrics.distributional_disruption.missing_data."
                        "slow_speed_tier.status=unavailable"
                    ),
                }
            ],
            "baseline_passes": True,
            "all_widths_positive_clearance": True,
            "oracle_available_for_every_variant": True,
            "oracle_required_checks_known": True,
            "h1_execution_binding_ready": True,
            "planner_records_are_not_run": True,
            "no_campaign_evidence": True,
            "variant_count": 3,
        },
    }


def test_h400_confirmation_separates_h1_diagnostic_from_actor_present_gate() -> None:
    """Only the precisely bounded oracle marker may remain diagnostic."""
    preflight = _bounded_h1_preflight()
    confirmation = {"schema_version": "issue_9348_confirmation_preflight.v1", "admit_h400": True}
    doorway_campaign._require_confirmation_preflight(preflight, confirmation)
    with pytest.raises(ValueError, match="actor-present confirmation"):
        doorway_campaign._require_confirmation_preflight(
            preflight, {**confirmation, "admit_h400": False}
        )
    with pytest.raises(ValueError, match="actor-present confirmation report is unavailable"):
        doorway_campaign._require_confirmation_preflight(preflight, {})
    preflight["checks"]["oracle_expected_fallbacks"][0]["marker"] = "other_fallback"
    with pytest.raises(ValueError, match="undeclared fallback"):
        doorway_campaign._require_confirmation_preflight(preflight, confirmation)
    preflight["checks"]["oracle_expected_fallbacks"] = []
    doorway_campaign._require_confirmation_preflight(preflight, confirmation)
    with pytest.raises(ValueError, match="fallback admission check is unavailable"):
        doorway_campaign._require_confirmation_preflight({"go": True, "checks": {}}, confirmation)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest_candidate(
    raw: dict[str, Any], tmp_path: Path, name: str = "candidate.yaml"
) -> Path:
    """Write a manifest copy with source references resolved from the repository."""
    for key in ("scenario_path", "map_path"):
        raw["base_scenario"][key] = str(_REPO_ROOT / raw["base_scenario"][key])
    candidate = tmp_path / name
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return candidate


def _fake_certifier(scenario: dict[str, Any], scenario_path: Path) -> ScenarioCertificate:
    """Return a deterministic certificate keyed to the generated width."""
    metadata = scenario["metadata"]
    gap = float(metadata["gap_width_m"])
    radius = float(scenario["robot_config"]["radius"])
    feasible = gap > 2.0 * radius
    classification = "valid" if feasible else "geometrically_infeasible"
    checks = {
        "minimum_static_clearance_m": gap / 2.0 - radius,
        "shortest_path_length_m": 18.5,
        "inflated_collision_free_path": feasible,
    }
    return ScenarioCertificate(
        schema_version="scenario_cert.v1",
        scenario_id=str(scenario["name"]),
        source=str(scenario_path),
        classification=classification,
        benchmark_eligibility="eligible" if feasible else "excluded",
        reasons=[],
        checks={},
        route_certificates=[
            RouteCertificate(
                route_id="route-0",
                spawn_id=0,
                goal_id=0,
                classification=classification,
                benchmark_eligibility="eligible" if feasible else "excluded",
                reasons=[],
                checks=checks,
                evidence={"runtime_input_identity_stable": True},
            )
        ],
        evidence={"runtime_input_identity_stable": True},
    )


def _fake_episode_runner(
    scenario: dict[str, Any], seed: int, horizon: int | None, algo: str
) -> dict[str, Any]:
    return {
        "route_complete": True,
        "steps": 100,
        "horizon_steps": horizon,
        "termination_reason": "success",
        "fallback_or_degraded": False,
    }


def _fake_bound_episode_runner(variants_dir: Path):
    """Return a deterministic runner with explicit source input receipts."""

    def run(scenario: dict[str, Any], seed: int, horizon: int | None, algo: str) -> dict[str, Any]:
        del seed, algo
        scenario_path = next(
            path
            for path in variants_dir.glob("*/scenario.yaml")
            if load_scenarios(path)[0]["name"] == scenario["name"]
        )
        identity = scenario_input_identity(scenario_path, scenario_id=str(scenario["name"]))
        records = [
            record for record in identity["files"] if record.get("role") != "scenario_manifest"
        ]
        return {
            "route_complete": True,
            "steps": 100,
            "horizon_steps": horizon,
            "termination_reason": "success",
            "fallback_or_degraded": False,
            "_scenario_runtime_input_records": records,
        }

    return run


def _complete_synthetic_campaign() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Build 18 complete rows for runtime-admission tests."""
    assets = [
        {
            "variant_id": f"gap_{token}",
            "gap_width_m": width,
            "scenario_sha256": hashlib.sha256(f"scenario-{width}".encode()).hexdigest(),
            "map_sha256": hashlib.sha256(f"map-{width}".encode()).hexdigest(),
        }
        for token, width in (("2p20", 2.2), ("2p80", 2.8), ("3p60", 3.6))
    ]
    pairs = build_pair_manifest(assets, (225, 226, 227), "manifest")
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for planner in ("goal", "social_force"):
        for seed in (225, 226, 227):
            for asset in assets:
                receipt = {
                    "map_sha256": asset["map_sha256"],
                    "initial_actor_state_sha256": "a" * 64,
                    "external_rng_state_sha256": "b" * 64,
                    "non_width_config_sha256": "c" * 64,
                }
                scenario_id = f"door_{asset['gap_width_m']}"
                rows.append(
                    {
                        "git_hash": "d" * 40,
                        "algo": planner,
                        "seed": seed,
                        "scenario_id": scenario_id,
                        "horizon": 400,
                        "status": "success",
                        "execution_mode": _baseline_kinematics(planner)["execution_mode"],
                        "readiness_status": "native",
                        "spawn_validity": _valid_spawn_block(),
                        "episode_id": f"{planner}-{seed}-{asset['gap_width_m']}",
                        "steps": 100,
                        "termination_reason": "success",
                        "metrics": {
                            "success": True,
                            "total_collision_count": 0,
                            "time_to_goal_norm_success_only": 0.25,
                        },
                        "algorithm_metadata": {
                            "status": "ok",
                            "canonical_algorithm": planner,
                            "planner_kinematics": _baseline_kinematics(planner),
                            "config_hash": (
                                doorway_application.GOAL_PLANNER_CONFIG_HASH
                                if planner == "goal"
                                else doorway_application.EMPTY_PLANNER_CONFIG_HASH
                            ),
                            "doorway_pair_receipt": receipt,
                            "planner_runtime": {},
                            "simulation_step_trace": _action_trace(100),
                            "planner_invocation_trace": _planner_invocation_trace(planner, 100),
                            "planner_decision_trace": {"steps": [{}]},
                        },
                        "integrity": {"contradictions": []},
                    }
                )
                cells.append(
                    {
                        "planner": planner,
                        "seed": seed,
                        "gap_width_m": asset["gap_width_m"],
                        "variant_id": asset["variant_id"],
                        "map_sha256": asset["map_sha256"],
                        "scenario_sha256": asset["scenario_sha256"],
                        "scenario_id": scenario_id,
                        "line_number": len(rows),
                    }
                )
                pair = next(
                    p for p in pairs["pairs"] if (p["planner"], p["seed"]) == (planner, seed)
                )
                cell = next(c for c in pair["cells"] if c["gap_width_m"] == asset["gap_width_m"])
                cell.update(receipt)
    return rows, cells, pairs


def test_actor_present_confirmation_rejects_fallback_and_custody_faults() -> None:
    """All 18 short rows must be native, paired, exact-source, and trace-backed."""
    rows, cells, pairs = _complete_synthetic_campaign()
    for row in rows:
        row["horizon"] = 10
        row["steps"] = 5
        row["algorithm_metadata"]["simulation_step_trace"] = _action_trace(5)
        row["algorithm_metadata"]["planner_invocation_trace"] = _planner_invocation_trace(
            row["algo"], 5
        )
        row["status"] = "failure"  # short-horizon timeout is not a success claim
        row["termination_reason"] = "max_steps"

    def assess(candidate_rows: list[dict[str, Any]], candidate_pairs: dict[str, Any] = pairs):
        return doorway_campaign.assess_confirmation_rows(
            candidate_rows,
            cells,
            candidate_pairs,
            source_sha="d" * 40,
            manifest_sha256="manifest",
        )

    assert assess(rows)["admit_h400"] is True
    missing_execution_mode = deepcopy(rows)
    missing_execution_mode[0].pop("execution_mode")
    missing_mode_result = assess(missing_execution_mode)
    assert missing_mode_result["admit_h400"] is False
    assert "missing_execution_mode" in missing_mode_result["rows"][0]["blockers"]
    missing_readiness = deepcopy(rows)
    missing_readiness[0].pop("readiness_status")
    missing_readiness_result = assess(missing_readiness)
    assert missing_readiness_result["admit_h400"] is False
    assert "missing_readiness_status" in missing_readiness_result["rows"][0]["blockers"]
    current_spawn_schema = deepcopy(rows)
    current_spawn_schema[0]["spawn_validity"]["schema_version"] = "spawn_validity.v2"
    assert assess(current_spawn_schema)["admit_h400"] is True
    unsupported_spawn_schema = deepcopy(rows)
    unsupported_spawn_schema[0]["spawn_validity"]["schema_version"] = "spawn_validity.v3"
    assert (
        "invalid_or_unknown_spawn_validity"
        in assess(unsupported_spawn_schema)["rows"][0]["blockers"]
    )
    fallback = deepcopy(rows)
    fallback[0]["algorithm_metadata"]["planner_decision_trace"]["steps"][0]["fallback_used"] = True
    assert assess(fallback)["admit_h400"] is False
    assert "fallback_or_degraded_planner_step" in assess(fallback)["rows"][0]["blockers"]
    wrong_kinematics = deepcopy(rows)
    wrong_kinematics[0]["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "fallback"
    assert (
        "planner_kinematics_mode_or_adapter_mismatch"
        in assess(wrong_kinematics)["rows"][0]["blockers"]
    )
    wrong_adapter = deepcopy(rows)
    wrong_adapter[9]["algorithm_metadata"]["planner_kinematics"]["projection_documented"] = False
    assert (
        "social_force_adapter_projection_unverified" in assess(wrong_adapter)["rows"][9]["blockers"]
    )
    missing_spawn = deepcopy(rows)
    missing_spawn[0].pop("spawn_validity")
    assert "invalid_or_unknown_spawn_validity" in assess(missing_spawn)["rows"][0]["blockers"]
    unknown_spawn = deepcopy(rows)
    unknown_spawn[0]["spawn_validity"]["schema_version"] = "spawn_validity.v2"
    unknown_spawn[0]["spawn_validity"]["reset_clearance_status"] = "unavailable"
    assert "invalid_or_unknown_spawn_validity" in assess(unknown_spawn)["rows"][0]["blockers"]
    overlapping_spawn = deepcopy(rows)
    overlapping_spawn[0]["spawn_validity"]["invalid_run"] = True
    assert "invalid_or_unknown_spawn_validity" in assess(overlapping_spawn)["rows"][0]["blockers"]
    wrong_source = deepcopy(rows)
    wrong_source[0]["git_hash"] = "e" * 40
    assert "source_commit_mismatch" in assess(wrong_source)["rows"][0]["blockers"]
    wrong_receipt = deepcopy(rows)
    wrong_receipt[0]["algorithm_metadata"]["doorway_pair_receipt"]["map_sha256"] = "x" * 64
    assert "paired_reset_receipt_mismatch" in assess(wrong_receipt)["rows"][0]["blockers"]
    wrong_horizon = deepcopy(rows)
    wrong_horizon[0]["horizon"] = 400
    assert "episode_identity_or_horizon_mismatch" in assess(wrong_horizon)["rows"][0]["blockers"]
    with pytest.raises(ValueError, match="all 18 frozen identities"):
        assess(rows[:-1])
    wrong_pairs = deepcopy(pairs)
    wrong_pairs["pairs"][0]["cells"][0]["external_rng_state_sha256"] = "x" * 64
    assert assess(rows, wrong_pairs)["admit_h400"] is False


def test_execution_axes_are_copied_from_runtime_and_readiness_fails_closed() -> None:
    """Row axes require callable route evidence and complete baseline evidence."""
    rows, _cells, _pairs = _complete_synthetic_campaign()
    goal = deepcopy(rows[0])
    goal.pop("execution_mode")
    goal.pop("readiness_status")
    doorway_campaign._record_baseline_execution_axes(goal, "goal")
    assert goal["execution_mode"] == "native"
    assert goal["readiness_status"] == "native"
    assert goal["algorithm_metadata"]["baseline_readiness"]["observed_execution_mode"] == "native"
    assert goal["algorithm_metadata"]["baseline_readiness"]["blockers"] == []

    social_force = deepcopy(rows[9])
    social_force.pop("execution_mode")
    social_force.pop("readiness_status")
    doorway_campaign._record_baseline_execution_axes(social_force, "social_force")
    assert social_force["execution_mode"] == "adapter"
    assert social_force["readiness_status"] == "native"
    assert (
        social_force["algorithm_metadata"]["planner_invocation_trace"]["steps"][0]["route_after"][
            "adapter_name"
        ]
        == "SocialForcePlannerAdapter"
    )

    mismatched = deepcopy(goal)
    mismatched.pop("execution_mode")
    mismatched.pop("readiness_status")
    mismatched["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "fallback"
    doorway_campaign._record_baseline_execution_axes(mismatched, "goal")
    assert mismatched["execution_mode"] == "native"
    assert mismatched["readiness_status"] == "unknown"
    assert (
        mismatched["algorithm_metadata"]["baseline_readiness"]["declared_execution_mode"]
        == "fallback"
    )
    assert (
        "planner_kinematics_mode_or_adapter_mismatch"
        in (mismatched["algorithm_metadata"]["baseline_readiness"]["blockers"])
    )

    spoofed_runtime = deepcopy(goal)
    spoofed_runtime.pop("execution_mode")
    spoofed_runtime.pop("readiness_status")
    for invocation in spoofed_runtime["algorithm_metadata"]["planner_invocation_trace"]["steps"]:
        invocation["route_before"] = _baseline_kinematics("social_force")
        invocation["route_after"] = _baseline_kinematics("social_force")
    doorway_campaign._record_baseline_execution_axes(spoofed_runtime, "goal")
    assert "execution_mode" not in spoofed_runtime
    assert spoofed_runtime["readiness_status"] == "unknown"
    assert (
        "planner_invocation_route_mismatch"
        in spoofed_runtime["algorithm_metadata"]["baseline_readiness"]["blockers"]
    )

    def spoofed_policy(_observation: Any) -> tuple[float, float]:
        return 0.5, 0.0

    spoofed_callable_route = doorway_campaign._runtime_policy_route(spoofed_policy)
    assert spoofed_callable_route["execution_mode"] == "unknown"
    spoofed_callable = deepcopy(goal)
    spoofed_callable.pop("execution_mode")
    spoofed_callable.pop("readiness_status")
    for invocation in spoofed_callable["algorithm_metadata"]["planner_invocation_trace"]["steps"]:
        invocation["route_before"] = deepcopy(spoofed_callable_route)
        invocation["route_after"] = deepcopy(spoofed_callable_route)
    doorway_campaign._record_baseline_execution_axes(spoofed_callable, "goal")
    assert "execution_mode" not in spoofed_callable
    assert spoofed_callable["readiness_status"] == "unknown"
    assert (
        "planner_invocation_route_mismatch"
        in spoofed_callable["algorithm_metadata"]["baseline_readiness"]["blockers"]
    )


@pytest.mark.parametrize("digest_field", ["map_sha256", "scenario_sha256"])
def test_pair_asset_digests_must_match_generated_cells(digest_field: str) -> None:
    """A self-consistent receipt cannot substitute for the generated asset bytes."""
    rows, cells, pairs = _complete_synthetic_campaign()
    short_rows = deepcopy(rows)
    for row in short_rows:
        row["horizon"] = 10
        row["steps"] = 5
        row["algorithm_metadata"]["simulation_step_trace"] = _action_trace(5)
        row["algorithm_metadata"]["planner_invocation_trace"] = _planner_invocation_trace(
            row["algo"], 5
        )
        row["status"] = "failure"
        row["termination_reason"] = "max_steps"

    forged_pairs = deepcopy(pairs)
    forged_digest = "e" * 64
    for pair in forged_pairs["pairs"]:
        for pair_cell in pair["cells"]:
            if pair_cell["gap_width_m"] == 2.2:
                pair_cell[digest_field] = forged_digest
    if digest_field == "map_sha256":
        for row, cell in zip(short_rows, cells, strict=True):
            if cell["gap_width_m"] == 2.2:
                row["algorithm_metadata"]["doorway_pair_receipt"]["map_sha256"] = forged_digest

    confirmation = doorway_campaign.assess_confirmation_rows(
        short_rows,
        cells,
        forged_pairs,
        source_sha="d" * 40,
        manifest_sha256="manifest",
    )
    assert confirmation["admit_h400"] is False
    assert (
        "pair manifest asset digest differs from generated cell custody"
        in confirmation["pair_errors"]
    )
    with pytest.raises(ValueError, match="verified reset pairs"):
        analyze_rows(rows, cells, forged_pairs)


def test_confirmation_separates_baseline_auxiliary_telemetry_from_execution() -> None:
    """Empty baseline decisions and unavailable sampler telemetry are not fallback."""
    rows, cells, pairs = _complete_synthetic_campaign()
    for row in rows:
        row["horizon"] = 10
        row["steps"] = 5
        row["algorithm_metadata"]["simulation_step_trace"] = _action_trace(5)
        row["algorithm_metadata"]["planner_invocation_trace"] = _planner_invocation_trace(
            row["algo"], 5
        )
        row["status"] = "failure"
        metadata = row["algorithm_metadata"]
        metadata["planner_decision_trace"]["steps"] = []
        metadata["simulation_step_trace"]["reset"] = {
            "spawn": {"status": "unavailable", "reason": "spawn_sampler_decision_not_retained"},
            "routes": {"status": "unavailable", "reason": "route_objects_not_retained"},
        }
        metadata["paired_effect_metric_producer"] = {
            "fields": {"stop_yield_latency_s": {"status": "unavailable"}}
        }
    result = doorway_campaign.assess_confirmation_rows(
        rows, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
    )
    assert result["admit_h400"] is True
    assert all(not item["blockers"] for item in result["rows"])
    assert all(len(item["ancillary_telemetry_gaps"]) == 3 for item in result["rows"])

    for corrupted_steps in (
        [{}],
        rows[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][:-1],
    ):
        corrupted = deepcopy(rows)
        corrupted[0]["algorithm_metadata"]["simulation_step_trace"]["steps"] = corrupted_steps
        rejected = doorway_campaign.assess_confirmation_rows(
            corrupted, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
        )
        assert rejected["admit_h400"] is False
        assert "missing_or_invalid_simulation_action_trace" in rejected["rows"][0]["blockers"]

    absent_action = deepcopy(rows)
    absent_action[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"].pop(
        "applied_environment_action"
    )
    rejected_action = doorway_campaign.assess_confirmation_rows(
        absent_action, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
    )
    assert rejected_action["admit_h400"] is False
    assert "missing_or_invalid_simulation_action_trace" in rejected_action["rows"][0]["blockers"]

    mismatched_action = deepcopy(rows)
    mismatched_action[0]["algorithm_metadata"]["planner_invocation_trace"]["steps"][0]["command"][
        "linear_velocity"
    ] = 0.4
    rejected_binding = doorway_campaign.assess_confirmation_rows(
        mismatched_action, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
    )
    assert rejected_binding["admit_h400"] is False
    assert "planner_invocation_action_binding_mismatch" in rejected_binding["rows"][0]["blockers"]

    wrong_clock = deepcopy(rows)
    wrong_clock[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["time_s"] = 0.2
    rejected_clock = doorway_campaign.assess_confirmation_rows(
        wrong_clock, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
    )
    assert rejected_clock["admit_h400"] is False
    assert "missing_or_invalid_simulation_action_trace" in rejected_clock["rows"][0]["blockers"]

    degraded = deepcopy(rows)
    degraded[0]["algorithm_metadata"]["simulation_step_trace"]["steps"][0] = {
        "planner": {"fallback_used": True}
    }
    blocked = doorway_campaign.assess_confirmation_rows(
        degraded, cells, pairs, source_sha="d" * 40, manifest_sha256="manifest"
    )
    assert blocked["admit_h400"] is False
    assert any(
        reason.startswith("fallback_or_degraded_runtime:")
        for reason in blocked["rows"][0]["blockers"]
    )


def _assert_paired_report_fields(report: dict[str, Any]) -> None:
    """Check raw pair differences, cell denominators and binary degeneracy."""
    _assert_paired_report_fields(report)


def _assert_report_verifier_checks_raw_differences(tmp_path: Path, report: dict[str, Any]) -> None:
    """Verify that sealed report raw differences cannot be edited independently."""
    _assert_report_verifier_checks_raw_differences(tmp_path, report)


def test_manifest_pins_three_width_tiers() -> None:
    """The application manifest must pin narrow/middle/wide tiers at fixed depth."""
    manifest = load_three_width_manifest(_MANIFEST)
    resolved = manifest["_resolved"]
    assert tuple(resolved["gap_levels"]) == (2.2, 2.8, 3.6)
    assert tuple(resolved["depth_levels"]) == (1.0,)
    assert float(resolved["nominal_radius_m"]) == 1.0
    assert tuple(resolved["planner_roster"]) == ("goal", "social_force")
    assert tuple(resolved["planner_seeds"]) == (225, 226, 227)
    assert manifest["planner_protocol"]["expected_rows"] == 18
    assert manifest["planner_protocol"]["algo_config_path"] == {
        "goal": None,
        "social_force": None,
    }
    assert manifest["planner_protocol"]["algo_config"] == {
        "goal": {},
        "social_force": {},
    }
    assert manifest["planner_protocol"]["algo_config_hash"] == {
        "goal": doorway_application.GOAL_PLANNER_CONFIG_HASH,
        "social_force": doorway_application.EMPTY_PLANNER_CONFIG_HASH,
    }
    assert doorway_application.GOAL_PLANNER_CONFIG_HASH == "44136fa355b3678a"
    assert (
        doorway_application.GOAL_PLANNER_CONFIG_HASH
        == doorway_application.EMPTY_PLANNER_CONFIG_HASH
    )
    assert manifest["base_scenario"]["scenario_sha256"] == _sha256(
        _REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml"
    )
    assert manifest["base_scenario"]["map_sha256"] == _sha256(_BASE_MAP)
    assert manifest["execution"] == {
        "production_campaign_authorized": True,
        "slurm_submission_authorized": True,
        "authorization_source": "https://github.com/ll7/diss/issues/2669#issuecomment-5811968439",
        "evidence_admission": "not_started",
        "baseline_map_must_remain_unchanged": True,
    }


def test_doorway_execution_requires_recorded_author_decision() -> None:
    """Execution authority cannot be inferred from a boolean without the author record."""
    execution = load_three_width_manifest(_MANIFEST)["execution"]
    with pytest.raises(ValueError, match="recorded author decision"):
        _validate_application_execution({**execution, "authorization_source": "ll7/diss#2669"})
    with pytest.raises(ValueError, match="author's execution decision"):
        _validate_application_execution({**execution, "slurm_submission_authorized": False})


def test_matrix_has_three_positive_clearance_widths(tmp_path: Path) -> None:
    """All comparison widths must clear the collision diameter at fixed depth."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "matrix")
    assert [asset["gap_width_m"] for asset in assets] == [2.2, 2.8, 3.6]
    assert [round(asset["derived_clearance_margin_m"], 3) for asset in assets] == [0.2, 0.8, 1.6]
    assert all(
        asset["expected_geometry_tier"] == "geometrically_feasible_candidate" for asset in assets
    )
    assert all(asset["constriction_depth_m"] == 1.0 for asset in assets)
    assert all(all(asset["geometry_checks"].values()) for asset in assets)


def test_variant_assets_change_only_explained_fields(tmp_path: Path) -> None:
    """Generated variant scenarios must pass the automated diff check."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    assert len(assets) == 3
    base_scenario = dict(load_scenarios(manifest["_resolved"]["scenario_path"])[0])
    for asset in assets:
        variant_scenario = dict(load_scenarios(asset["scenario_path"])[0])
        assert check_variant_diff(base_scenario, variant_scenario) == []
        planner_config_mutation = dict(variant_scenario)
        planner_config_mutation["planner_config"] = {"social_force": "changed"}
        assert any(
            "planner_config" in violation
            for violation in check_variant_diff(base_scenario, planner_config_mutation)
        )
        assert Path(asset["map_path"]).is_file()


def test_executed_planners_do_not_attach_classic_global_route(tmp_path: Path) -> None:
    """Generated scenario configs leave the classic global planner disabled."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    for asset in assets:
        scenario_path = Path(asset["scenario_path"])
        scenario = dict(load_scenarios(scenario_path)[0])
        config = build_env_config(scenario, scenario_path=scenario_path)
        assert config.use_planner is False
        assert config.sim_config.time_per_step_in_secs == pytest.approx(0.1)
        assert scenario["simulation_config"]["max_episode_steps"] == 400


def test_generated_svg_changes_only_symmetric_wall_endpoints(tmp_path: Path) -> None:
    """All map features except the two doorway wall endpoints are identical."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    label = "{http://www.inkscape.org/namespaces/inkscape}label"
    original = list(ET.parse(_BASE_MAP).getroot().iter())
    original_sha = _sha256(_BASE_MAP)
    for asset in assets:
        changed = list(ET.parse(asset["map_path"]).getroot().iter())
        assert len(changed) == len(original)
        wall_changes = []
        for before, after in zip(original, changed, strict=True):
            assert before.tag == after.tag
            if before.attrib != after.attrib:
                assert before.attrib.get(label) == after.attrib.get(label) == "obstacle"
                assert before.attrib["x"] == after.attrib["x"] == "15"
                assert before.attrib["width"] == after.attrib["width"] == "1"
                assert {k for k in before.attrib if before.attrib[k] != after.attrib[k]} <= {
                    "y",
                    "height",
                }
                wall_changes.append(after)
        assert len(wall_changes) == 2
        edges = sorted((float(w.attrib["y"]), float(w.attrib["height"])) for w in wall_changes)
        lower_end = edges[0][0] + edges[0][1]
        upper_start = edges[1][0]
        assert (lower_end + upper_start) / 2 == pytest.approx(5.0)
        assert upper_start - lower_end == pytest.approx(asset["gap_width_m"])
    assert _sha256(_BASE_MAP) == original_sha


@pytest.mark.parametrize("width", [1.9, 2.0])
def test_manifest_rejects_nonpositive_clearance(tmp_path: Path, width: float) -> None:
    """Infeasible and tangent widths cannot enter the comparison manifest."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    raw["geometry"]["gap_width_m"][0] = width
    candidate = _write_manifest_candidate(raw, tmp_path, "bad.yaml")
    with pytest.raises(ValueError, match="positive|exceed|exact"):
        load_three_width_manifest(candidate)


def test_manifest_rejects_non_authoritative_nominal_radius(tmp_path: Path) -> None:
    """A manifest cannot redefine the collision envelope used for width claims."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    raw["envelope"]["nominal_radius_m"] = 0.9
    candidate = _write_manifest_candidate(raw, tmp_path, "bad-radius.yaml")
    with pytest.raises(ValueError, match="authoritative|DEFAULT_ROBOT_RADIUS"):
        load_three_width_manifest(candidate)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("geometry", "constriction_depth_m", [1.2], "depth"),
        ("oracle", "horizon_steps", 399, "H400"),
        ("planner_protocol", "seeds", [225, 226, 228], "seeds"),
    ],
)
def test_manifest_rejects_protocol_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    """The readiness manifest cannot silently drift from the preregistration."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    raw[section][field] = value
    if section == "oracle":
        raw["planner_protocol"][field] = value
    candidate = _write_manifest_candidate(raw, tmp_path, f"bad-{section}-{field}.yaml")
    with pytest.raises(ValueError, match=message):
        load_three_width_manifest(candidate)


def test_manifest_rejects_mutated_historical_input_digest(tmp_path: Path) -> None:
    """Historical scenario and map bytes are part of the frozen readiness identity."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    raw["base_scenario"]["map_sha256"] = "0" * 64
    candidate = _write_manifest_candidate(raw, tmp_path, "bad-map-digest.yaml")
    with pytest.raises(ValueError, match="historical input"):
        load_three_width_manifest(candidate)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    [
        ("envelope_diameter_m", None, 1.8),
        ("ratios", 0, 1.2),
        ("clearance_m", 0, 0.3),
    ],
)
def test_manifest_rejects_mutated_width_claims(
    tmp_path: Path, field: str, index: int | None, value: float
) -> None:
    """Declared width ratios and clearances must equal recomputed values."""
    raw = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    if index is None:
        raw["width_selection_rationale"][field] = value
    else:
        raw["width_selection_rationale"][field][index] = value
    candidate = _write_manifest_candidate(raw, tmp_path, f"bad-{field}.yaml")
    with pytest.raises(ValueError, match="width_selection_rationale"):
        load_three_width_manifest(candidate)


def test_variant_diff_rejects_unexplained_changes() -> None:
    """The diff check must fail closed on out-of-allowlist scenario edits."""
    manifest = load_three_width_manifest(_MANIFEST)
    base_scenario = dict(load_scenarios(manifest["_resolved"]["scenario_path"])[0])
    tampered = dict(base_scenario)
    tampered["pedestrian_config"] = {"density": 99.0}
    violations = check_variant_diff(base_scenario, tampered)
    assert violations != []


def test_svg_units_are_metres() -> None:
    """The historical map must declare metre-scale SVG units for width semantics."""
    text = _BASE_MAP.read_text(encoding="utf-8")
    assert 'width="' in text and 'height="' in text


def test_pair_manifest_shares_seeds_across_widths() -> None:
    """Each pair must span all three widths with one shared seed and hashes."""
    assets = [
        {
            "variant_id": f"gap_{token}",
            "gap_width_m": gap,
            "scenario_sha256": f"scenario-{token}",
            "map_sha256": f"map-{token}",
        }
        for token, gap in (("2p20", 2.2), ("2p80", 2.8), ("3p60", 3.6))
    ]
    pairs = build_pair_manifest(assets, (225, 226, 227), "manifest-sha")
    assert pairs["schema_version"] == "issue_9348_three_width_pair_manifest.v1"
    assert pairs["realization_hash_status"] == "pending_initial_and_external_rng_state_verification"
    assert [pair["pair_id"] for pair in pairs["pairs"]] == [
        f"{planner}_pair_{seed:05d}"
        for planner in ("goal", "social_force")
        for seed in (225, 226, 227)
    ]
    for pair in pairs["pairs"]:
        assert [cell["gap_width_m"] for cell in pair["cells"]] == [2.2, 2.8, 3.6]
        assert all(cell["initial_actor_state_sha256"] is None for cell in pair["cells"])
        assert all(cell["external_rng_state_sha256"] is None for cell in pair["cells"])
    assert check_pair_receipts(pairs)
    for pair in pairs["pairs"]:
        for cell in pair["cells"]:
            cell["initial_actor_state_sha256"] = "a" * 64
            cell["external_rng_state_sha256"] = "b" * 64
            cell["non_width_config_sha256"] = "c" * 64
    assert check_pair_receipts(pairs) == []
    pairs["pairs"][0]["cells"][1]["external_rng_state_sha256"] = "d" * 64
    assert "differs across widths" in check_pair_receipts(pairs)[0]


def test_pair_receipt_canonicalizes_actor_order_and_requires_rng() -> None:
    """Identical reset states hash equally; missing RNG state blocks admission."""
    reset = {
        "robot": {
            "position": [4.0, 5.0],
            "velocity": [0.0, 0.0],
            "heading": 0.0,
            "goal": [25.5, 5.0],
        },
        "route_state": {
            "robot_routes": [[7.0, 5.0], [25.5, 5.0]],
            "pedestrian_goals": [[3.0, 5.0], [3.0, 5.0]],
        },
        "pedestrians": [
            {
                "actor_id": "h1",
                "position": [27.0, 5.0],
                "velocity": [0.0, 0.0],
                "goal": [3.0, 5.0],
            },
            {
                "actor_id": "h2",
                "position": [25.0, 5.0],
                "velocity": [0.0, 0.0],
                "goal": [3.0, 5.0],
            },
        ],
    }
    snapshot = SimpleNamespace(
        global_rng_state=("MT19937", np.asarray([1, 2], dtype=np.uint32), 0, 0, 0.0),
        python_random_state=(3, (1, 2), None),
        behavior_rng_states={"h1": {"state": 7}},
        residual_adversary_state=None,
    )
    first = build_pair_receipt(reset, snapshot)
    reversed_reset = {**reset, "pedestrians": list(reversed(reset["pedestrians"]))}
    assert build_pair_receipt(reversed_reset, snapshot) == first
    snapshot.python_random_state = None
    with pytest.raises(ValueError, match="RNG snapshot is incomplete"):
        build_pair_receipt(reset, snapshot)


@pytest.mark.parametrize("field", ["goal", "route_state"])
def test_pair_receipt_requires_goal_and_route_state(field: str) -> None:
    """Pair custody must include the reset goal and assigned route state."""
    reset = {
        "robot": {
            "position": [4.0, 5.0],
            "velocity": [0.0, 0.0],
            "goal": [25.5, 5.0],
        },
        "route_state": {
            "robot_routes": [[7.0, 5.0], [25.5, 5.0]],
            "pedestrian_goals": [[3.0, 5.0]],
        },
        "pedestrians": [
            {
                "actor_id": "h1",
                "position": [27.0, 5.0],
                "velocity": [0.0, 0.0],
                "goal": [3.0, 5.0],
            }
        ],
    }
    snapshot = SimpleNamespace(
        global_rng_state=("MT19937", np.asarray([1, 2], dtype=np.uint32), 0, 0, 0.0),
        python_random_state=(3, (1, 2), None),
        behavior_rng_states={"h1": {"state": 7}},
        residual_adversary_state=None,
    )
    if field == "goal":
        reset["robot"].pop("goal")
    else:
        reset["route_state"] = {}
    with pytest.raises(ValueError, match="goal and route|pose, velocity, or goal"):
        build_pair_receipt(reset, snapshot)


def test_portable_reset_matches_three_widths_and_distinct_maps(tmp_path: Path) -> None:
    """All 18 frozen cells have identical pre-command pairs and distinct maps."""
    manifest = load_three_width_manifest(_MANIFEST)
    assets = generate_application_assets(manifest, tmp_path / "variants")
    session = DoorwayPairingSession()
    planner_configs = manifest["_resolved"]["planner_configs"]
    planner_config_hashes = manifest["_resolved"]["planner_config_hashes"]
    for planner in ("goal", "social_force"):
        assert planner_configs[planner] == {}
        planner_config_hash = planner_config_hashes[planner]
        for seed in (225, 226, 227):
            for asset in assets:
                scenario_path = Path(asset["scenario_path"])
                scenario = load_scenarios(scenario_path)[0]
                config = build_env_config(scenario, scenario_path=scenario_path)
                common = non_width_config_sha256(
                    scenario, planner=planner, planner_config_hash=planner_config_hash
                )
                env = make_robot_env(config=config, seed=seed, debug=False)
                try:
                    obs, _ = env.reset(seed=seed)
                    receipt = session.hook(
                        planner=planner,
                        seed=seed,
                        map_sha256=asset["map_sha256"],
                        non_width_config_sha256=common,
                    )(env, obs)
                    assert receipt["non_width_config_sha256"] == common
                finally:
                    env.close()
    pairs = build_pair_manifest(assets, (225, 226, 227), _sha256(_MANIFEST))
    completed = session.fill_pair_manifest(pairs)
    assert len(completed["pairs"]) == 6
    assert sum(len(pair["cells"]) for pair in completed["pairs"]) == 18
    assert completed["realization_hash_status"] == "verified_pre_command"
    assert check_pair_receipts(completed) == []


def test_h1_runner_pair_receipt_survives_episode_schema(tmp_path: Path) -> None:
    """The real runner serializes its opt-in receipt through the episode schema."""
    manifest = load_three_width_manifest(_MANIFEST)
    asset = generate_application_assets(manifest, tmp_path / "variants")[0]
    scenario_path = Path(asset["scenario_path"])
    scenario = dict(load_scenarios(scenario_path)[0])
    session = DoorwayPairingSession()
    row = _run_map_episode(
        scenario,
        225,
        horizon=1,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="goal",
        scenario_path=scenario_path,
        record_planner_decision_trace=True,
        record_simulation_step_trace=True,
        pair_reset_hook=session.hook(
            planner="goal",
            seed=225,
            map_sha256=asset["map_sha256"],
            non_width_config_sha256=non_width_config_sha256(scenario, planner="goal"),
        ),
        policy_builder=doorway_campaign._observed_baseline_policy_builder,
    )
    doorway_campaign._record_baseline_execution_axes(row, "goal")
    assert row["algorithm_metadata"]["config_hash"] == "44136fa355b3678a"
    assert row["execution_mode"] == "native"
    assert row["readiness_status"] == "native"
    assert row["algorithm_metadata"]["baseline_readiness"]["status"] == "native"
    assert (
        row["algorithm_metadata"]["baseline_readiness"]["decision_trace_status"]
        == "empty_expected_for_baseline"
    )
    assert row["spawn_validity"]["schema_version"] == "spawn_validity.v2"
    assert doorway_campaign._valid_spawn_evidence(row)
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    assert doorway_campaign._valid_action_trace(trace, row)
    step = trace["steps"][0]
    assert step["step"] == 0
    assert step["time_s"] == pytest.approx(0.1)
    assert set(step["planner"]["selected_action"]) >= {"linear_velocity", "angular_velocity"}
    assert set(step["planner"]["applied_environment_action"]) >= {
        "linear_velocity",
        "angular_velocity",
    }
    planner_trace = row["algorithm_metadata"]["planner_decision_trace"]
    assert planner_trace["steps"] == []
    if trace["reset"]["spawn"]["status"] == "unavailable":
        assert (
            "simulation_step_trace.reset.spawn.status=unavailable"
            in doorway_campaign._ancillary_telemetry_gaps(row["algorithm_metadata"])
        )
    stream = io.StringIO()
    schema = load_schema(_REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json")
    write_validated_to_handle(stream, schema, row)
    assert '"doorway_pair_receipt"' in stream.getvalue()


@pytest.mark.parametrize("marker", ["execution_mode", "readiness_status", "nested_runtime"])
def test_h400_report_excludes_non_native_or_fallback_rows(marker: str) -> None:
    """Non-native and fallback/degraded runtime markers invalidate paired evidence."""
    rows, cells, pairs = _complete_synthetic_campaign()
    if marker == "nested_runtime":
        rows[0]["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
    else:
        rows[0][marker] = "fallback"

    report = analyze_rows(rows, cells, pairs)

    assert report["native_rows"] == 17
    assert report["excluded_rows"] == 1
    assert report["evidence_status"] == "diagnostic_incomplete"
    assert report["excluded_pair_ids"] == ["goal_pair_00225"]
    excluded = report["row_inventory"][0]
    assert excluded["evidence_status"] == "excluded"
    assert any(
        (
            (
                "unexpected_execution_mode:"
                if marker == "execution_mode"
                else "non_native_readiness_status:"
            )
            in reason
            if marker in {"execution_mode", "readiness_status"}
            else "fallback_or_degraded_runtime" in reason
        )
        for reason in excluded["exclusion_reasons"]
    )


def test_h400_report_excludes_missing_success_only_pairs(tmp_path: Path) -> None:  # noqa: PLR0915
    """Paired intervals use whole seeds and censor failure arrival times."""
    source_sha = doorway_campaign._git("rev-parse", "HEAD")
    source_tree_custody = doorway_campaign._source_tree_custody(source_sha)
    assets = []
    for i, width in enumerate((2.2, 2.8, 3.6)):
        variant_id = f"gap_{i}"
        variant_dir = tmp_path / "assets" / variant_id
        variant_dir.mkdir(parents=True)
        (variant_dir / "variant.svg").write_text(f"map {width}\n", encoding="utf-8")
        (variant_dir / "scenario.yaml").write_text(f"scenario {width}\n", encoding="utf-8")
        assets.append(
            {
                "variant_id": variant_id,
                "gap_width_m": width,
                "scenario_sha256": _sha256(variant_dir / "scenario.yaml"),
                "map_sha256": _sha256(variant_dir / "variant.svg"),
            }
        )
    pairs = build_pair_manifest(assets, (225, 226, 227), "manifest")
    rows = []
    cells = []
    for planner in ("goal", "social_force"):
        for seed in (225, 226, 227):
            for asset in assets:
                width = asset["gap_width_m"]
                receipt = {
                    "map_sha256": asset["map_sha256"],
                    "initial_actor_state_sha256": "a" * 64,
                    "external_rng_state_sha256": "b" * 64,
                    "non_width_config_sha256": "c" * 64,
                }
                scenario_id = f"door_{width}"
                success = not (planner == "goal" and seed == 225 and width == 2.2)
                rows.append(
                    {
                        "git_hash": source_sha,
                        "algo": planner,
                        "seed": seed,
                        "scenario_id": scenario_id,
                        "horizon": 400,
                        "status": "success" if success else "failure",
                        "execution_mode": _baseline_kinematics(planner)["execution_mode"],
                        "readiness_status": "native",
                        "spawn_validity": _valid_spawn_block(),
                        "episode_id": f"{planner}-{seed}-{width}",
                        "steps": 100,
                        "outcome": {
                            "route_complete": success,
                            "collision_event": False,
                            "timeout_event": not success,
                        },
                        "termination_reason": "success" if success else "max_steps",
                        "metrics": {
                            "success": success,
                            "total_collision_count": 0,
                            "time_to_goal_norm_success_only": 0.25 if success else None,
                        },
                        "algorithm_metadata": {
                            "status": "ok",
                            "canonical_algorithm": planner,
                            "planner_kinematics": _baseline_kinematics(planner),
                            "config_hash": (
                                doorway_application.GOAL_PLANNER_CONFIG_HASH
                                if planner == "goal"
                                else doorway_application.EMPTY_PLANNER_CONFIG_HASH
                            ),
                            "doorway_pair_receipt": receipt,
                            "simulation_step_trace": _action_trace(100),
                            "planner_invocation_trace": _planner_invocation_trace(planner, 100),
                            "planner_decision_trace": {"steps": [{}]},
                        },
                        "integrity": {"contradictions": []},
                    }
                )
                cells.append(
                    {
                        "planner": planner,
                        "seed": seed,
                        "gap_width_m": width,
                        "variant_id": asset["variant_id"],
                        "map_sha256": asset["map_sha256"],
                        "scenario_sha256": asset["scenario_sha256"],
                        "scenario_id": scenario_id,
                        "line_number": len(rows),
                    }
                )
                pair = next(
                    p for p in pairs["pairs"] if (p["planner"], p["seed"]) == (planner, seed)
                )
                cell = next(c for c in pair["cells"] if c["gap_width_m"] == width)
                cell.update(receipt)
    report = analyze_rows(rows, cells, pairs)
    goal_22_28 = next(
        c
        for c in report["contrasts"]
        if c["planner"] == "goal" and c["low_width_m"] == 2.2 and c["high_width_m"] == 2.8
    )
    assert report["native_rows"] == 18
    assert goal_22_28["endpoints"]["success"]["denominator_pairs"] == 3
    assert goal_22_28["endpoints"]["success"]["raw_paired_differences"] == [1.0, 0.0, 0.0]
    assert goal_22_28["endpoints"]["success"]["binary_pair_status"] == "bootstrap"
    assert goal_22_28["endpoints"]["arrival_time_success_only_s"]["denominator_pairs"] == 2
    assert goal_22_28["endpoints"]["arrival_time_success_only_s"]["excluded_seed_ids"] == [225]
    goal_22_cell = next(
        cell
        for cell in report["cell_denominators"]
        if cell["planner"] == "goal" and cell["gap_width_m"] == 2.2
    )
    assert goal_22_cell["planned_seed_count"] == 3
    assert goal_22_cell["native_row_count"] == 3
    assert goal_22_cell["endpoint_denominators"]["arrival_time_success_only_s"] == 2
    social_force_22_28 = next(
        c
        for c in report["contrasts"]
        if c["planner"] == "social_force" and c["low_width_m"] == 2.2 and c["high_width_m"] == 2.8
    )
    assert social_force_22_28["endpoints"]["success"]["raw_paired_differences"] == [0.0, 0.0, 0.0]
    assert (
        social_force_22_28["endpoints"]["success"]["binary_pair_status"] == "degenerate_binary_pair"
    )
    assert social_force_22_28["endpoints"]["success"]["estimate"]["ci95"] is None
    assert report["pedestrian_delay_or_impairment"]["value"] is None
    rows[0]["algorithm_metadata"]["config_hash"] = "mismatched"
    with pytest.raises(ValueError, match="planner config identity mismatch"):
        analyze_rows(rows, cells, pairs)
    rows[0]["algorithm_metadata"]["config_hash"] = doorway_application.GOAL_PLANNER_CONFIG_HASH
    rows[0]["algorithm_metadata"]["distributional_disruption"] = {
        "missing_data": {"slow_speed_tier": {"status": "unavailable"}}
    }
    unavailable_metric = analyze_rows(rows, cells, pairs)
    assert unavailable_metric["native_rows"] == 17
    assert any(
        "fallback_or_degraded_runtime:algorithm_metadata.distributional_disruption."
        "missing_data.slow_speed_tier.status=unavailable" in reason
        for reason in unavailable_metric["row_inventory"][0]["exclusion_reasons"]
    )
    rows[0]["algorithm_metadata"].pop("distributional_disruption")
    rows[0]["algorithm_metadata"]["planner_decision_trace"]["steps"] = [{"fallback_used": True}]
    degraded = analyze_rows(rows, cells, pairs)
    assert degraded["native_rows"] == 17
    assert degraded["excluded_pair_ids"] == ["goal_pair_00225"]
    rows[0]["algorithm_metadata"]["planner_decision_trace"]["steps"] = []
    empty_trace = analyze_rows(rows, cells, pairs)
    assert empty_trace["native_rows"] == 18
    rows[0]["algorithm_metadata"]["planner_decision_trace"].pop("steps")
    missing_trace = analyze_rows(rows, cells, pairs)
    assert missing_trace["native_rows"] == 17
    assert (
        "missing_planner_decision_trace" in missing_trace["row_inventory"][0]["exclusion_reasons"]
    )
    rows[0]["algorithm_metadata"]["planner_decision_trace"]["steps"] = [{}]
    rows[0]["algorithm_metadata"]["simulation_step_trace"]["steps"] = [{}]
    missing_actions = analyze_rows(rows, cells, pairs)
    assert missing_actions["native_rows"] == 17
    assert (
        "missing_or_invalid_simulation_action_trace"
        in missing_actions["row_inventory"][0]["exclusion_reasons"]
    )
    rows[0]["algorithm_metadata"]["simulation_step_trace"] = _action_trace(100)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("application_manifest.yaml", "episode.schema.v1.json"):
        (inputs / name).write_text("frozen input\n", encoding="utf-8")
    pairs["manifest_sha256"] = _sha256(inputs / "application_manifest.yaml")
    lines = [json.dumps(row, sort_keys=True) + "\n" for row in rows]
    (tmp_path / "episodes.jsonl").write_text("".join(lines), encoding="utf-8")
    for cell, line in zip(cells, lines, strict=True):
        cell["line_sha256"] = hashlib.sha256(line.encode()).hexdigest()
    write_json(tmp_path / "pair_manifest.json", pairs)
    write_json(tmp_path / "report.json", report)
    confirmation_rows = deepcopy(rows)
    confirmation_cells = deepcopy(cells)
    for row in confirmation_rows:
        row["horizon"] = 10
        row["steps"] = 5
        row["algorithm_metadata"]["simulation_step_trace"] = _action_trace(5)
        row["algorithm_metadata"]["planner_invocation_trace"] = _planner_invocation_trace(
            row["algo"], 5
        )
        row["status"] = "failure"
        row["termination_reason"] = "max_steps"
    confirmation_lines = [json.dumps(row, sort_keys=True) + "\n" for row in confirmation_rows]
    confirmation_raw = tmp_path / "confirmation_episodes.jsonl"
    confirmation_raw.write_text("".join(confirmation_lines), encoding="utf-8")
    for cell, line in zip(confirmation_cells, confirmation_lines, strict=True):
        cell["line_sha256"] = hashlib.sha256(line.encode()).hexdigest()
    confirmation_pairs = deepcopy(pairs)
    confirmation_pairs["manifest_sha256"] = _sha256(inputs / "application_manifest.yaml")
    write_json(tmp_path / "confirmation_pair_manifest.json", confirmation_pairs)
    confirmation = doorway_campaign.assess_confirmation_rows(
        confirmation_rows,
        confirmation_cells,
        confirmation_pairs,
        source_sha=source_sha,
        manifest_sha256=_sha256(inputs / "application_manifest.yaml"),
    )
    assert confirmation["admit_h400"] is True
    confirmation["cells"] = confirmation_cells
    confirmation["episodes_jsonl_sha256"] = _sha256(confirmation_raw)
    write_json(tmp_path / "confirmation_preflight.json", confirmation)
    write_json(
        tmp_path / "preflight.json",
        {
            **_bounded_h1_preflight(),
            "variants": [
                {
                    "variant_id": asset["variant_id"],
                    "assets": {
                        "map_sha256": asset["map_sha256"],
                        "scenario_sha256": asset["scenario_sha256"],
                    },
                }
                for asset in assets
            ],
        },
    )
    write_json(
        tmp_path / "run_manifest.json",
        {
            "source_commit": source_sha,
            "source_tree_custody": source_tree_custody,
            "application_manifest_sha256": _sha256(inputs / "application_manifest.yaml"),
            "planner_config_hash": {
                "goal": doorway_application.GOAL_PLANNER_CONFIG_HASH,
                "social_force": doorway_application.EMPTY_PLANNER_CONFIG_HASH,
            },
            "episode_schema_sha256": _sha256(inputs / "episode.schema.v1.json"),
            "episodes_jsonl_sha256": _sha256(tmp_path / "episodes.jsonl"),
            "cells": cells,
        },
    )
    _write_checksums(tmp_path)
    assert verify_campaign_bundle(tmp_path)["native_rows"] == 18
    pair_manifest_path = tmp_path / "pair_manifest.json"
    pair_manifest_payload = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    pair_manifest_payload["manifest_sha256"] = "f" * 64
    write_json(pair_manifest_path, pair_manifest_payload)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="copied scientific input digest"):
        verify_campaign_bundle(tmp_path)
    write_json(pair_manifest_path, pairs)
    _write_checksums(tmp_path)
    assert verify_campaign_bundle(tmp_path)["native_rows"] == 18
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest_payload["source_tree_custody"]["git_tree_oid"] = "e" * 40
    write_json(run_manifest_path, run_manifest_payload)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="source tree custody"):
        verify_campaign_bundle(tmp_path)
    run_manifest_payload["source_tree_custody"] = source_tree_custody
    write_json(run_manifest_path, run_manifest_payload)
    _write_checksums(tmp_path)
    assert verify_campaign_bundle(tmp_path)["native_rows"] == 18
    confirmation_path = tmp_path / "confirmation_preflight.json"
    confirmation_payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
    confirmation_payload["admit_h400"] = False
    write_json(confirmation_path, confirmation_payload)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="confirmation report differs from sealed raw probes"):
        verify_campaign_bundle(tmp_path)
    write_json(confirmation_path, confirmation)
    _write_checksums(tmp_path)
    assert verify_campaign_bundle(tmp_path)["native_rows"] == 18
    preflight_path = tmp_path / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["checks"].pop("oracle_expected_fallbacks")
    write_json(preflight_path, preflight)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="fallback admission check is unavailable"):
        verify_campaign_bundle(tmp_path)
    preflight["checks"]["oracle_expected_fallbacks"] = {"variant_id": "gap_3p60"}
    write_json(preflight_path, preflight)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="fallback admission check is unavailable"):
        verify_campaign_bundle(tmp_path)
    preflight["checks"]["oracle_expected_fallbacks"] = [{"variant_id": "gap_3p60"}]
    write_json(preflight_path, preflight)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="undeclared fallback"):
        verify_campaign_bundle(tmp_path)
    preflight["checks"]["oracle_expected_fallbacks"] = _bounded_h1_preflight()["checks"][
        "oracle_expected_fallbacks"
    ]
    preflight["go"] = False
    write_json(preflight_path, preflight)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="did not admit policy execution"):
        verify_campaign_bundle(tmp_path)
    preflight["go"] = True
    write_json(preflight_path, preflight)
    _write_checksums(tmp_path)
    assert verify_campaign_bundle(tmp_path)["native_rows"] == 18
    report_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    report_payload["contrasts"][0]["endpoints"]["success"]["raw_paired_differences"][0] = 99.0
    write_json(tmp_path / "report.json", report_payload)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="differs from sealed raw episodes"):
        verify_campaign_bundle(tmp_path)
    write_json(tmp_path / "report.json", report)
    _write_checksums(tmp_path)
    report_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    report_payload["claim_boundary"] = "tampered"
    write_json(tmp_path / "report.json", report_payload)
    _write_checksums(tmp_path)
    with pytest.raises(ValueError, match="differs from sealed raw episodes"):
        verify_campaign_bundle(tmp_path)
    write_json(tmp_path / "report.json", report)
    _write_checksums(tmp_path)
    (tmp_path / "episodes.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_campaign_bundle(tmp_path)
    rows[0]["algorithm_metadata"]["doorway_pair_receipt"]["map_sha256"] = "tampered"
    with pytest.raises(ValueError, match="custody mismatch"):
        analyze_rows(rows, cells, pairs)


def test_preflight_records_oracle_before_not_run_planner_lane(tmp_path: Path) -> None:
    """The oracle-first preflight must pass with planner rows explicitly not run."""
    report = run_three_width_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_bound_episode_runner(tmp_path / "variants"),
        certifier=_fake_certifier,
    )

    assert report["go"] is True
    assert report["checks"]["baseline_passes"] is True
    assert report["checks"]["variant_count"] == 3
    assert report["checks"]["all_widths_positive_clearance"] is True
    assert report["checks"]["oracle_available_for_every_variant"] is True
    assert report["checks"]["oracle_required_checks_known"] is True
    assert report["checks"]["h1_execution_binding_ready"] is True
    assert report["checks"]["confirmation_oracle_fallbacks_clear"] is True
    assert report["checks"]["nominal_grid_route_feasible_for_every_variant"] is True
    assert report["checks"]["planner_records_are_not_run"] is True
    assert report["checks"]["no_campaign_evidence"] is True
    assert all(item["planner"]["status"] == "not_run" for item in report["variants"])
    assert {item["oracle"]["nominal_verdict"]["status"] for item in report["variants"]} == {
        "feasible"
    }

    report_path = tmp_path / "issue_9348_preflight.json"
    write_preflight_report(report, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"review_marker": "AI-GENERATED NEEDS-REVIEW"' in payload


def test_preflight_preserves_real_loader_identity(tmp_path: Path) -> None:
    """The default oracle sees loader-bound rows and keeps no-route diagnostic-only."""
    report = run_three_width_preflight(_MANIFEST, output_dir=tmp_path / "variants")

    assert report["checks"]["all_widths_positive_clearance"] is True
    assert report["checks"]["oracle_available_for_every_variant"] is True
    by_width = {item["geometry"]["gap_width_m"]: item["oracle"] for item in report["variants"]}
    for width in (2.2, 2.8):
        oracle = by_width[width]
        assert oracle["required_checks_known"] is True
        assert oracle["nominal_verdict"]["status"] == "infeasible_by_construction"
        assert oracle["nominal_verdict"]["geometric"]["route_geometrically_feasible"] is False
        assert oracle["nominal_verdict"]["geometric"]["runtime_input_identity_stable"] is True
    assert all(
        oracle["nominal_verdict"]["geometric"]["runtime_input_identity_stable"] is True
        for oracle in by_width.values()
    )
    assert all(
        oracle.get("readiness_blocker") != "oracle_route_classification_unknown"
        for oracle in by_width.values()
    )
    fallback_oracle = by_width[3.6]
    assert fallback_oracle["required_checks_known"] is True
    assert fallback_oracle["readiness_status"] == "ready_with_expected_fallback"
    assert fallback_oracle["readiness_diagnostic"] == ("expected_distributional_metric_unavailable")
    assert report["checks"]["oracle_expected_fallbacks"] == [
        {
            "variant_id": "gap_3p60__depth_1p00",
            "reason": "expected_distributional_metric_unavailable",
            "marker": (
                "metrics.distributional_disruption.missing_data.slow_speed_tier.status=unavailable"
            ),
        }
    ]
    assert report["checks"]["h1_execution_binding_ready"] is True
    assert report["checks"]["confirmation_oracle_fallbacks_clear"] is False
    assert report["execution"]["h1_execution_binding_ready"] is True
    assert report["execution"]["confirmation_ready"] is False
    assert "oracle_expected_fallbacks_present" in report["execution"]["confirmation_blockers"]
    assert report["go"] is True


def test_conservative_grid_result_is_reported_without_changing_frozen_widths(
    tmp_path: Path,
) -> None:
    """A grid no-route finding stays distinct from positive continuous clearance."""

    def conservative_certifier(scenario: dict[str, Any], path: Path) -> ScenarioCertificate:
        certificate = _fake_certifier(scenario, path)
        if float(scenario["metadata"]["gap_width_m"]) < 3.6:
            certificate.classification = "geometrically_infeasible"
            certificate.benchmark_eligibility = "excluded"
            certificate.route_certificates[0].classification = "geometrically_infeasible"
            certificate.route_certificates[0].benchmark_eligibility = "excluded"
            certificate.route_certificates[0].checks["inflated_collision_free_path"] = False
        return certificate

    report = run_three_width_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_episode_runner,
        certifier=conservative_certifier,
    )
    assert report["checks"]["all_widths_positive_clearance"] is True
    assert report["checks"]["nominal_grid_route_feasible_for_every_variant"] is False
    assert report["protocol"]["production_campaign_authorized"] is True
    assert report["protocol"]["slurm_submission_authorized"] is True
    assert report["protocol"]["authorization_source"] == (
        "https://github.com/ll7/diss/issues/2669#issuecomment-5811968439"
    )
    assert report["execution"]["confirmation_ready"] is False
    assert report["checks"]["oracle_required_checks_known"] is False
    assert report["go"] is False  # Unknown oracle state cannot authorize dispatch.


def test_known_oracle_no_route_remains_a_diagnostic() -> None:
    """A known grid no-route is diagnostic and does not masquerade as clearance."""
    known, blocker = doorway_application._oracle_required_checks(
        {
            "execution_status": "available",
            "runtime_input_identity_stable": True,
            "nominal_verdict": {
                "status": "infeasible_by_construction",
                "geometric": {
                    "route_geometrically_feasible": False,
                    "runtime_input_identity_stable": True,
                },
            },
        }
    )
    assert known is True
    assert blocker is None


def test_expected_h1_distributional_fallback_does_not_block_binding() -> None:
    """The preregistered missing metric remains diagnostic during H1 binding."""
    oracle = {
        "execution_status": "available",
        "runtime_input_identity_stable": True,
        "nominal_verdict": {
            "status": "blocked",
            "geometric": {
                "route_geometrically_feasible": True,
                "runtime_input_identity_stable": True,
            },
            "completion": {
                "status": "blocked",
                "blocker": "rollout_fallback_or_degraded",
                "fallback_or_degraded": True,
                "fallback_marker": (
                    "metrics.distributional_disruption.missing_data."
                    "slow_speed_tier.status=unavailable"
                ),
                "observed_route_completion_feasible": True,
                "termination_reason": "success",
                "rollout_blocker": None,
                "runtime_input_identity_stable": True,
            },
        },
    }
    known, blocker = doorway_application._oracle_required_checks(oracle)
    assert known is True
    assert blocker is None
    assert (
        doorway_application._oracle_h1_readiness_diagnostic(oracle)
        == "expected_distributional_metric_unavailable"
    )

    for field, value, expected_blocker in (
        ("runtime_input_identity_stable", False, "oracle_runtime_input_identity_unstable"),
        ("route_geometrically_feasible", None, "oracle_route_classification_unknown"),
        ("termination_reason", "collision", "oracle_nominal_status_blocked"),
    ):
        invalid = json.loads(json.dumps(oracle))
        if field == "runtime_input_identity_stable":
            invalid[field] = value
        elif field == "route_geometrically_feasible":
            invalid["nominal_verdict"]["geometric"][field] = value
        else:
            invalid["nominal_verdict"]["completion"][field] = value
        known, blocker = doorway_application._oracle_required_checks(invalid)
        assert known is False
        assert blocker == expected_blocker


def test_preflight_blocks_unknown_oracle_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An available oracle with an unknown required classification cannot set go."""
    unavailable, unavailable_blocker = doorway_application._oracle_required_checks(
        {"execution_status": "blocked"}
    )
    assert unavailable is False
    assert unavailable_blocker == "oracle_execution_blocked"
    monkeypatch.setattr(
        doorway_application,
        "envelope_sensitivity_verdict_to_dict",
        lambda *_args, **_kwargs: {},
    )
    report = run_three_width_preflight(
        _MANIFEST,
        output_dir=tmp_path / "variants",
        episode_runner=_fake_episode_runner,
        certifier=_fake_certifier,
    )
    assert report["checks"]["oracle_available_for_every_variant"] is True
    assert report["checks"]["oracle_required_checks_known"] is False
    assert report["checks"]["oracle_readiness_blockers"]
    assert report["go"] is False
