"""Tests for the conservative scenario-admissibility adapter."""

# evidence-writer-exempt: synthetic scenario YAML is written only under pytest tmp_path to prove
# candidate artifact identity binding; no canonical evidence is published.

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import (
    ADMISSIBLE_FEASIBILITY_UNKNOWN,
    EMPIRICALLY_FEASIBLE,
    GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY,
    PLANNER_SPECIFIC_FAILURE,
    STRUCTURALLY_INVALID,
    partition_candidates_by_admissibility,
    scenario_admissibility,
    validate_scenario_admissibility,
)
from robot_sf.adversarial import (
    classify_scenario_admissibility as _classify_scenario_admissibility,
)
from robot_sf.adversarial.feasibility_first import (
    SCENARIO_FEASIBILITY_CONTRACT_VERSION,
    SCENARIO_FEASIBILITY_PREDICATE_NAMES,
)
from robot_sf.adversarial.scenario_admissibility import (
    _oracle_excludes,
    _producer_selected_map_binding,
    _same_selected_map_identity,
)
from robot_sf.benchmark.map_runner.map_runner_identity import (
    scenario_with_episode_seed_defaults as _scenario_with_episode_seed_defaults,
)
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    manifest_path_for_result_jsonl,
    write_result_provenance_manifest,
)
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback
from robot_sf.scenario_certification.feasibility_oracle import (
    FEASIBILITY_ORACLE_SCHEMA,
    ISSUE_5574_REPORT_SCHEMA,
)
from robot_sf.scenario_certification.input_identity import (
    runtime_input_records_match,
    scenario_input_identity,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    certificate_to_dict,
    certify_scenario_file,
)

_SCENARIO_ARTIFACT = Path(__file__).resolve().parent / "fixtures/issue_9651/case_static.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_ARTIFACT_SHA256 = hashlib.sha256(_SCENARIO_ARTIFACT.read_bytes()).hexdigest()
_SCENARIO_RUNTIME_IDENTITY = scenario_input_identity(_SCENARIO_ARTIFACT, scenario_id="case-static")
_SCENARIO_EFFECTIVE_INPUT_SHA256 = _SCENARIO_RUNTIME_IDENTITY.get("effective_input_sha256")
_SCENARIO_RUNTIME_INPUT_RECORDS = [
    dict(record)
    for record in _SCENARIO_RUNTIME_IDENTITY.get("files", [])
    if record.get("role") != "scenario_manifest"
]
_EXECUTION_FIXTURE_TEMP = tempfile.TemporaryDirectory(prefix="issue-9651-execution-evidence-")
_EXECUTION_FIXTURE_ROOT = Path(_EXECUTION_FIXTURE_TEMP.name)
_EXECUTION_FIXTURE_COUNTER = itertools.count()


def classify_scenario_admissibility(case_id: str, **kwargs: Any) -> Any:
    """Use one explicit artifact and scenario identity for normalized test evidence."""
    kwargs.setdefault("scenario_artifact_path", _SCENARIO_ARTIFACT)
    kwargs.setdefault("scenario_id", "case-static")
    return _classify_scenario_admissibility(case_id, **kwargs)


def _certificate(
    classification: str = "valid",
    *,
    eligibility: str = "eligible",
    scenario_id: str = "case-static",
    pedestrian_count: int = 0,
    route_reason: str | None = None,
    include_producer_identity: bool = True,
) -> dict[str, Any]:
    if route_reason is not None:
        reasons = [route_reason]
    elif classification == "invalid":
        reasons = ["route_requires_at_least_two_waypoints"]
    elif classification == "geometrically_infeasible":
        reasons = ["no_inflated_collision_free_path: empty_path"]
    elif classification == "kinodynamically_infeasible":
        reasons = ["route_turn_radius_below_bicycle_limit: 1.000 < 2.000"]
    else:
        reasons = []
    route_checks: dict[str, Any] = {"dynamic": {"single_pedestrian_count": pedestrian_count}}
    if classification == "invalid":
        route_checks.update(start=None, goal=None, waypoint_count=1)
    if classification == "geometrically_infeasible":
        route_checks["inflated_collision_free_path"] = False
        route_checks["planner"] = {"path_status": "no_path"}
    if classification == "kinodynamically_infeasible":
        route_checks["kinodynamic"] = {
            "robot_model": "BicycleDriveSettings",
            "command_limits_valid": True,
            "minimum_turning_radius_m": 2.0,
            "route_minimum_turn_radius_m": 1.0,
        }
    return {
        "schema_version": CERT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source": _SCENARIO_ARTIFACT.as_posix(),
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "reasons": reasons,
        "checks": {"route_count": 1, "settings": {"robot_radius_m": 0.4}},
        "route_certificates": [
            {
                "route_id": "route-0",
                "spawn_id": 0,
                "goal_id": 0,
                "classification": classification,
                "benchmark_eligibility": eligibility,
                "reasons": reasons,
                "checks": route_checks,
                "evidence": {},
            }
        ],
        "evidence": {
            "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
            "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
            "effective_input_identity_stable": True,
            "runtime_input_identity_stable": True,
        }
        if include_producer_identity
        else {},
    }


def _predicates(case_id: str = "case-static", *, verdict: str = "valid") -> dict[str, Any]:
    return {
        "contract_version": SCENARIO_FEASIBILITY_CONTRACT_VERSION,
        "candidate_id": case_id,
        "predicates": [
            {
                "name": name,
                "verdict": verdict,
                "reason": "fixture evidence",
                "evidence": {"source": "fixture"} if verdict == "valid" else {},
            }
            for name in SCENARIO_FEASIBILITY_PREDICATE_NAMES
        ],
    }


def _oracle(
    *,
    scenario_id: str = "case-static",
    status: str = "feasible",
    geometric: bool = True,
    complete: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": FEASIBILITY_ORACLE_SCHEMA,
        "scenario_id": scenario_id,
        "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
        "envelope_radius_m": 0.4,
        "feasible": status == "feasible",
        "status": status,
        "claim_boundary": "diagnostic_only_not_benchmark_evidence",
        "geometric": {
            "route_geometrically_feasible": geometric,
            "classification": "hard_but_solvable" if geometric else "geometrically_infeasible",
            "benchmark_eligibility": "eligible" if geometric else "excluded",
            "runtime_input_identity_stable": True,
        },
        "completion": {
            "route_completion_feasible": (
                True if complete else None if status == "blocked" else False
            ),
            "min_completion_steps": 20 if complete else None,
            "horizon_steps": 100,
            "completion_horizon_margin_steps": 80 if complete else None,
            "termination_reason": "success" if complete else None,
            "status": ("passed" if complete else "blocked" if status == "blocked" else "failed"),
            "blocker": (
                None
                if complete
                else "rollout_unavailable"
                if status == "blocked"
                else "route_geometrically_infeasible_no_traversal_path"
                if status == "infeasible_by_construction"
                else "rollout_incomplete"
            ),
            "fallback_or_degraded": False,
            "runtime_input_identity_stable": True,
            "fallback_marker": None,
            "observed_route_completion_feasible": True if complete else None,
            "rollout_blocker": None,
        },
        "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
        "source_artifact_identity_stable": True,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "runtime_input_identity_stable": True,
    }


def _oracle_report(oracle: dict[str, Any], *, scenario_id: str = "case-static") -> dict[str, Any]:
    return {
        "schema_version": ISSUE_5574_REPORT_SCHEMA,
        "scenario_ids": [scenario_id],
        "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
        "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
        "source_artifact_identity_stable": True,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "rollout_algo": "goal",
        "cells": [
            {
                "schema_version": "envelope_sensitivity_axis.v1",
                "scenario_id": scenario_id,
                "category": oracle["status"],
                "nominal_envelope_radius_m": oracle["envelope_radius_m"],
                "nominal_verdict": oracle,
                "reduced_verdicts": [],
                "scenario_manifest": _SCENARIO_ARTIFACT.as_posix(),
                "source_artifact_sha256": _SCENARIO_ARTIFACT_SHA256,
                "source_artifact_identity_stable": True,
                "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
                "effective_input_identity_stable": True,
                "runtime_input_identity_stable": True,
                "rollout_algo": "goal",
                "rollout_seed": 19,
                "claim_boundary": "diagnostic_only_not_benchmark_evidence",
            }
        ],
    }


def _execution(  # noqa: PLR0913 - fixture fields model canonical episode and producer dimensions.
    planner_id: str,
    *,
    route_complete: bool,
    case_id: str = "case-static",
    scenario_id: str = "case-static",
    seed: int = 19,
    replay: bool = False,
    include_context: bool = True,
    include_producer_context: bool = True,
    sim_dt: float = 0.1,
    producer_goal_x: float | None = None,
    include_runtime_input_records: bool = True,
    include_selected_map_identity: bool = True,
    selected_map_id: str = "uni_campus_big",
) -> dict[str, Any]:
    planner_aliases = {
        "reference": "goal",
        "target": "orca",
        "replay": "orca",
        "other": "social_force",
    }
    canonical_planner_id = planner_aliases.get(planner_id, planner_id)
    episode_id = "episode-target"
    termination_reason = "success" if route_complete else "max_steps"
    fixture_index = next(_EXECUTION_FIXTURE_COUNTER)
    planner_config_path = _EXECUTION_FIXTURE_ROOT / f"planner-{fixture_index:04d}.yaml"
    planner_config_path.write_text(f"planner: {canonical_planner_id}\n", encoding="utf-8")
    planner_config_sha256 = hashlib.sha256(planner_config_path.read_bytes()).hexdigest()
    planner_checkpoint_sha256 = (
        "not_applicable" if canonical_planner_id in {"goal", "social_force", "orca"} else "d" * 64
    )
    scenario_params = _scenario_with_episode_seed_defaults(
        yaml.safe_load(_SCENARIO_ARTIFACT.read_text(encoding="utf-8")), seed=seed
    )
    if producer_goal_x is not None:
        scenario_params["robot"]["goal"][0] = producer_goal_x
    scenario_params.update(
        {
            "algo": canonical_planner_id,
            "algo_config_hash": _config_hash({"planner": canonical_planner_id}),
            "record_forces": False,
            "observation_mode": "socnav_state",
            "observation_level": "full",
            "record_planner_decision_trace": False,
            "record_simulation_step_trace": False,
            "run_horizon": 100,
            "run_dt": sim_dt,
        }
    )
    config_hash = _config_hash(scenario_params)
    episode_row = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "scenario_params": scenario_params,
        "config_hash": config_hash,
        "seed": seed,
        "horizon": 100,
        "algo": canonical_planner_id,
        "git_hash": _git_hash_fallback(),
        "status": "success" if route_complete else "failure",
        "metrics": {"collisions": 0, "success": int(route_complete)},
        "termination_reason": termination_reason,
        "outcome": {
            "route_complete": route_complete,
            "collision_event": False,
            "timeout_event": not route_complete,
        },
        "integrity": {"contradictions": []},
        "algorithm_metadata": {
            "status": "ok",
            "canonical_algorithm": canonical_planner_id,
            "execution_mode": "native",
        },
    }
    if include_runtime_input_records:
        episode_row["runtime_input_records"] = [
            dict(record) for record in _SCENARIO_RUNTIME_INPUT_RECORDS
        ]
    if include_selected_map_identity:
        map_record = next(
            record
            for record in _SCENARIO_RUNTIME_INPUT_RECORDS
            if record.get("role") in {"map_file", "default_map_pool"}
        )
        episode_row["selected_map_identity"] = {
            "schema_version": "selected_map_identity.v1",
            "status": "available",
            "map_id": selected_map_id,
            "path": map_record["path"],
            "sha256": map_record["sha256"],
            "source_role": map_record["role"],
            "reason": None,
        }
    if include_context:
        # This extension is deliberately non-authoritative: producer-owned run metadata below
        # supplies numerical context, while row extensions must not forge it.
        episode_row["execution_context"] = {
            "horizon_steps": 100,
            "planner_config_sha256": "0" * 64,
            "planner_checkpoint_sha256": planner_checkpoint_sha256,
            "environment_sha256": "1" * 64,
        }
    episode_store_path = _EXECUTION_FIXTURE_ROOT / f"episodes-{fixture_index:04d}.jsonl"
    episode_store_bytes = (json.dumps(episode_row, sort_keys=True) + "\n").encode("utf-8")
    episode_store_path.write_bytes(episode_store_bytes)
    episode_store_sha256 = hashlib.sha256(episode_store_bytes).hexdigest()

    def write_producer_manifest(path: Path, row: dict[str, Any], *, keep_context: bool) -> Path:
        manifest = build_result_provenance_manifest(
            out_path=path,
            episode_records=[row],
            schema_path=_REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
            scenario_path=_SCENARIO_ARTIFACT,
            scenarios=[],
            algo=canonical_planner_id,
            algo_config_path=planner_config_path,
            benchmark_profile="baseline-safe",
            suite_key="issue-9651-synthetic-fixture",
            total_jobs=1,
            written=1,
            horizon=100,
            dt=sim_dt,
            record_forces=False,
            active_observation_mode="socnav_state",
            active_observation_level="full",
        )
        if not keep_context:
            manifest["run"].pop("execution_context", None)
        manifest_path = manifest_path_for_result_jsonl(path)
        write_result_provenance_manifest(manifest_path, manifest)
        return manifest_path

    source_manifest_path = write_producer_manifest(
        episode_store_path, episode_row, keep_context=include_producer_context
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    execution_context_sha256 = (
        source_manifest.get("run", {})
        .get("execution_context", {})
        .get("execution_context_sha256", "e" * 64)
    )
    evidence_ref = episode_store_path.as_posix()
    if planner_id == "replay" or replay:
        replay_sidecar_path = _EXECUTION_FIXTURE_ROOT / f"replay-{fixture_index:04d}.json"
        replay_result_path = (
            _EXECUTION_FIXTURE_ROOT / f"target-replay-result-{fixture_index:04d}.json"
        )
        replay_episode_row = dict(episode_row)
        replay_episode_store_path = (
            _EXECUTION_FIXTURE_ROOT / f"replay-episodes-{fixture_index:04d}.jsonl"
        )
        replay_episode_store_bytes = (json.dumps(replay_episode_row, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        replay_episode_store_path.write_bytes(replay_episode_store_bytes)
        replay_manifest_path = write_producer_manifest(
            replay_episode_store_path,
            replay_episode_row,
            keep_context=include_producer_context,
        )
        replay_manifest_sha256 = hashlib.sha256(replay_manifest_path.read_bytes()).hexdigest()
        replay_result = {
            "schema_version": "target_planner_replay_result.v1",
            "replay_kind": "target_planner",
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "planner_id": canonical_planner_id,
            "source_commit": episode_row["git_hash"],
            "source_episodes_jsonl_sha256": episode_store_sha256,
            "replay_episode_id": replay_episode_row["episode_id"],
            "replay_episodes_jsonl_path": replay_episode_store_path.as_posix(),
            "replay_episodes_jsonl_sha256": hashlib.sha256(replay_episode_store_bytes).hexdigest(),
            "replay_provenance_manifest_path": replay_manifest_path.as_posix(),
            "replay_provenance_manifest_sha256": replay_manifest_sha256,
            "run_status": "ok",
            "fallback_or_degraded": False,
            "route_complete": route_complete,
            "termination_reason": termination_reason,
            "replay_command": "robot-sf-target-planner-replay --fixture",
        }
        replay_result_bytes = (json.dumps(replay_result, sort_keys=True) + "\n").encode()
        replay_result_path.write_bytes(replay_result_bytes)
        replay_sidecar_path.write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "planner_key": canonical_planner_id,
                    "repo_commit": episode_row["git_hash"],
                    "replay_command": "fixture replay producer",
                    "determinism_check_status": "pass",
                    "source_episodes_jsonl_path": episode_store_path.as_posix(),
                    "source_episodes_jsonl_sha256": episode_store_sha256,
                    "target_planner_replay_result_path": replay_result_path.as_posix(),
                    "target_planner_replay_result_sha256": hashlib.sha256(
                        replay_result_bytes
                    ).hexdigest(),
                    "resimulated": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        evidence_ref = replay_sidecar_path.as_posix()
    record = {
        "case_id": case_id,
        "scenario_id": scenario_id,
        "scenario_variant": "original",
        "planner_id": canonical_planner_id,
        "episode_id": episode_id,
        "run_status": "ok",
        "fallback_or_degraded": False,
        "route_complete": route_complete,
        "seed": seed,
        "horizon_steps": 100,
        "scenario_sha256": _SCENARIO_ARTIFACT_SHA256,
        "robot_model_sha256": "b" * 64,
        "simulator_config_sha256": "c" * 64,
        "planner_config_sha256": planner_config_sha256,
        "planner_checkpoint_sha256": planner_checkpoint_sha256,
        "environment_sha256": "e" * 64,
        "source_episodes_jsonl_sha256": episode_store_sha256,
        "effective_input_sha256": _SCENARIO_EFFECTIVE_INPUT_SHA256,
        "effective_input_identity_stable": True,
        "source_commit": episode_row["git_hash"],
        "evidence_ref": evidence_ref,
    }
    record["environment_sha256"] = execution_context_sha256
    if not include_producer_context:
        record["environment_sha256"] = "e" * 64
    if planner_id == "replay" or replay:
        record.update(determinism_check_status="pass", resimulated=True)
    return record


def _referenced_scenario(tmp_path: Path, *, route_override: bool = True) -> Path:
    """Materialize a canonical scenario with separately hashed runtime resources."""
    scenario_path = tmp_path / "scenario.yaml"
    map_path = tmp_path / "map.svg"
    shutil.copyfile(_REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg", map_path)
    scenario: dict[str, Any] = {
        "name": "case-static",
        "map_file": map_path.name,
        "simulation_config": {"max_episode_steps": 200, "ped_density": 0.0},
        "robot_config": {},
        "metadata": {"archetype": "head_on_corridor"},
        "seeds": [19],
    }
    if route_override:
        scenario["route_overrides_file"] = "routes.yaml"
        (tmp_path / "routes.yaml").write_text("routes: []\n", encoding="utf-8")
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False), encoding="utf-8"
    )
    return scenario_path


def test_runtime_input_identity_is_stable_across_checkouts(tmp_path: Path) -> None:
    """Equivalent content closures bind across roots while retaining diagnostic paths."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    checkout_a.mkdir()
    checkout_b.mkdir()
    scenario_a = _referenced_scenario(checkout_a)
    for name in ("scenario.yaml", "map.svg", "routes.yaml"):
        shutil.copyfile(scenario_a.parent / name, checkout_b / name)
    scenario_b = checkout_b / "scenario.yaml"

    identity_a = scenario_input_identity(scenario_a, scenario_id="case-static")
    identity_b = scenario_input_identity(scenario_b, scenario_id="case-static")
    consumed: list[dict[str, str]] = []
    scenario_row = yaml.safe_load(scenario_b.read_text(encoding="utf-8"))["scenarios"][0]
    build_robot_config_from_scenario(
        scenario_row,
        scenario_path=scenario_b,
        runtime_input_records=consumed,
    )

    assert identity_a["effective_input_sha256"] == identity_b["effective_input_sha256"]
    assert identity_a["path"] != identity_b["path"]
    assert runtime_input_records_match(identity_a, consumed, scenario_id="case-static") is True
    assert runtime_input_records_match(identity_b, consumed, scenario_id="case-static") is True


def test_selected_map_binding_uses_content_identity_across_checkouts(tmp_path: Path) -> None:
    """A replay in another checkout binds by stable map identity, not absolute path."""
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    checkout_a.mkdir()
    checkout_b.mkdir()
    scenario_a = checkout_a / "scenario.yaml"
    scenario_a.write_text(
        yaml.safe_dump({"scenarios": [{"name": "case-static"}]}, sort_keys=False),
        encoding="utf-8",
    )

    identity = scenario_input_identity(scenario_a, scenario_id="case-static")
    map_record = next(
        record
        for record in identity["files"]
        if record["role"] == "default_map_pool" and record["map_id"] == "uni_campus_big"
    )
    selected_map_path = checkout_b / "uni_campus_big.svg"
    shutil.copyfile(_REPO_ROOT / "robot_sf/maps/uni_campus_big.svg", selected_map_path)
    status, binding = _producer_selected_map_binding(
        {"_candidate_runtime_input_identity": identity},
        {
            "scenario_id": "case-static",
            "selected_map_identity": {
                "status": "available",
                "map_id": "uni_campus_big",
                "path": str(selected_map_path),
                "sha256": map_record["sha256"],
                "source_role": "default_map_pool",
            },
        },
    )

    assert status == "valid"
    assert binding is not None
    assert binding["path"] == str(selected_map_path)


def _bind_certificate_to_scenario(path: Path) -> dict[str, Any]:
    """Attach raw and runtime-input identities to a schema-valid fixture certificate."""
    identity = scenario_input_identity(path, scenario_id="case-static")
    assert identity["status"] == "available"
    certificate = _certificate()
    certificate["source"] = path.as_posix()
    certificate["evidence"].update(
        source_artifact_sha256=identity["source_artifact_sha256"],
        effective_input_sha256=identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    return certificate


def _bind_oracle_to_scenario(path: Path) -> dict[str, Any]:
    """Attach raw and runtime-input identities to a fixture oracle report."""
    identity = scenario_input_identity(path, scenario_id="case-static")
    assert identity["status"] == "available"
    oracle = _oracle()
    oracle.update(
        scenario_manifest=path.as_posix(),
        source_artifact_sha256=identity["source_artifact_sha256"],
        effective_input_sha256=identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    return oracle


def test_certificate_rejects_only_structural_and_geometric_exclusions() -> None:
    structural = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate("invalid", eligibility="excluded")
    )
    impossible = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("geometrically_infeasible", eligibility="excluded"),
    )
    kinematic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("kinodynamically_infeasible", eligibility="excluded"),
    )
    invalid_geometry = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid", eligibility="excluded", route_reason="start_outside_map_bounds"
        ),
    )

    assert structural.verdict == STRUCTURALLY_INVALID
    assert structural.search_disposition == "reject"
    assert impossible.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert impossible.search_disposition == "reject"
    assert kinematic.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert invalid_geometry.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "scenario_certificate_invalidity_unresolved" in invalid_geometry.reason_codes
    assert impossible.assumptions["scenario_certificate"]["settings"] == {"robot_radius_m": 0.4}


def test_planner_exception_stays_unknown_and_retained() -> None:
    """Legacy v1 planner-error labels do not become proof of geometric impossibility."""

    reason = "no_inflated_collision_free_path: injected planner failure"
    certificate = _certificate(
        "geometrically_infeasible", eligibility="excluded", route_reason=reason
    )
    certificate["route_certificates"][0]["checks"].update(
        inflated_collision_free_path=False,
        planner={"path_status": "error"},
    )
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


def test_legacy_empty_path_without_producer_status_stays_unknown() -> None:
    """Legacy v1 empty-path records do not prove that the planner completed a no-path search."""

    certificate = _certificate(
        "geometrically_infeasible",
        eligibility="excluded",
        include_producer_identity=False,
    )
    route_checks = certificate["route_certificates"][0]["checks"]
    route_checks["planner"] = {"algorithm": "a_star"}

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_source_identity_bound_by_adapter" in verdict.reason_codes
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


def test_legacy_certificate_current_identity_does_not_attest_generation() -> None:
    """Current adapter identity cannot attest a legacy certificate's generation inputs."""

    certificate = _certificate(include_producer_identity=False)
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    binding = verdict.assumptions["scenario_certificate"]["identity_binding"]
    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert binding["source"] == "admissibility_adapter_current_input_identity"
    assert binding["source_artifact_sha256"] == _SCENARIO_ARTIFACT_SHA256
    assert binding["effective_input_sha256"] == _SCENARIO_EFFECTIVE_INPUT_SHA256
    assert binding["producer_fields_present"] is False
    assert binding["effective_input_generation_bound"] is False
    assert "scenario_certificate_effective_input_identity_generation_unbound" in (
        verdict.reason_codes
    )


def test_legacy_certificate_identity_unavailable_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing adapter-owned input identity cannot validate an exclusion certificate."""

    monkeypatch.setattr(
        scenario_admissibility,
        "scenario_input_identity",
        lambda *_args, **_kwargs: {
            "status": "unavailable",
            "source_artifact_sha256": None,
            "effective_input_sha256": None,
            "requires_effective_input_binding": True,
            "files": [],
            "reason_code": "fixture_identity_unavailable",
        },
    )
    certificate = _certificate(
        "geometrically_infeasible",
        eligibility="excluded",
        include_producer_identity=False,
    )

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_adapter_identity_unavailable" in verdict.reason_codes


def test_certificate_exclusion_fails_closed_when_adapter_identity_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source/input change during classification downgrades an exclusion to unknown."""
    identity_before = dict(_SCENARIO_RUNTIME_IDENTITY)
    identity_after = {
        **identity_before,
        "effective_input_sha256": "a" * 64,
    }
    identities = iter((identity_before, identity_after))
    monkeypatch.setattr(
        scenario_admissibility,
        "scenario_input_identity",
        lambda *_args, **_kwargs: next(identities),
    )
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.evidence["scenario_artifact_identity"]["adapter_identity_stable"] is False
    assert verdict.assumptions["scenario_certificate"]["identity_binding"]["stable"] is False
    assert "scenario_certificate_adapter_identity_changed_or_unavailable" in (verdict.reason_codes)


@pytest.mark.parametrize(
    ("reason", "path_status"),
    [
        ("no_inflated_collision_free_path: planner error", "no_path"),
        ("no_inflated_collision_free_path: empty_path", "error"),
    ],
)
def test_geometric_exclusion_requires_completed_no_path_evidence(
    reason: str, path_status: str
) -> None:
    """A reason label and a contradictory planner status cannot exclude a candidate."""

    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    route["reasons"] = [reason]
    route["checks"]["planner"]["path_status"] = path_status
    certificate["reasons"] = [reason]

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    "mutation", ["contradictory_waypoint_count", "wrong_endpoint", "top_reason"]
)
def test_structural_certificate_label_needs_matching_producer_check(mutation: str) -> None:
    certificate = _certificate("invalid", eligibility="excluded")
    route = certificate["route_certificates"][0]
    if mutation == "contradictory_waypoint_count":
        route["checks"].update(waypoint_count=2, start=[0.0, 0.0], goal=[1.0, 1.0])
    elif mutation == "wrong_endpoint":
        route["checks"].update(waypoint_count=2, start=[0.0, 0.0], goal=None)
    else:
        certificate["reasons"] = []

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_producer_shaped_nonfinite_endpoint_supports_structural_exclusion() -> None:
    certificate = _certificate(
        "invalid", eligibility="excluded", route_reason="start_point_not_finite"
    )
    certificate["route_certificates"][0]["checks"].update(
        waypoint_count=2, start=[None, 1.0], goal=[4.0, 5.0]
    )

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == STRUCTURALLY_INVALID
    assert verdict.search_disposition == "reject"


@pytest.mark.parametrize("reason", ["map_pool_empty", "no_applicable_robot_routes"])
def test_empty_route_certificate_requires_exact_supported_reason(reason: str) -> None:
    certificate = _certificate("invalid", eligibility="excluded")
    certificate.update(reasons=[reason], checks={"route_count": 0}, route_certificates=[])

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == STRUCTURALLY_INVALID
    assert verdict.search_disposition == "reject"


@pytest.mark.parametrize(
    "mutation",
    ["label_only", "missing_check", "contradictory_check", "missing_summary_reason"],
)
def test_geometric_certificate_label_needs_matching_producer_evidence(mutation: str) -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    if mutation == "label_only":
        certificate["reasons"] = []
        route["reasons"] = []
        route["checks"].pop("inflated_collision_free_path")
    elif mutation == "missing_check":
        route["checks"].pop("inflated_collision_free_path")
    elif mutation == "contradictory_check":
        route["checks"]["inflated_collision_free_path"] = True
    else:
        certificate["reasons"] = []

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


def test_supported_geometric_collision_check_patterns_can_exclude() -> None:
    swept = _certificate("geometrically_infeasible", eligibility="excluded")
    swept_reason = "planned_path_swept_envelope_clips_obstacle: full_polyline_clearance_m=-0.25"
    swept["reasons"] = [swept_reason]
    swept_route = swept["route_certificates"][0]
    swept_route["reasons"] = [swept_reason]
    swept_route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "swept_envelope": {
                "validated": True,
                "clips_obstacle": True,
                "clearance_m": -0.25,
                "vertex_clearance_m": -0.25,
                "clipped_vertex_count": 1,
                "planned_waypoint_count": 4,
            },
        }
    )

    simulator = _certificate("geometrically_infeasible", eligibility="excluded")
    simulator_reason = "planned_path_simulator_collision: first_collision_sample_index=3"
    simulator["reasons"] = [simulator_reason]
    simulator_route = simulator["route_certificates"][0]
    simulator_route["reasons"] = [simulator_reason]
    simulator_route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "simulator_obstacle_collision": {
                "validated": True,
                "collides_obstacle": True,
                "runtime_component": "ContinuousOccupancy.is_obstacle_collision",
                "obstacle_source": "MapDefinition.obstacles_pysf_runtime_normalized",
                "sample_spacing_m": 0.05,
                "checked_sample_count": 4,
                "first_collision_sample_index": 3,
            },
        }
    )

    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=swept).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=simulator).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )

    swept["route_certificates"][0]["checks"]["swept_envelope"]["clearance_m"] = -0.5
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=swept).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )
    simulator["route_certificates"][0]["checks"]["simulator_obstacle_collision"][
        "first_collision_sample_index"
    ] = 2
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=simulator).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )


def test_kinodynamic_certificate_label_needs_matching_bicycle_check_evidence() -> None:
    certificate = _certificate("kinodynamically_infeasible", eligibility="excluded")
    route = certificate["route_certificates"][0]
    certificate["reasons"] = []
    route["reasons"] = []
    route["checks"].pop("kinodynamic")

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"

    inconsistent = _certificate("kinodynamically_infeasible", eligibility="excluded")
    inconsistent["route_certificates"][0]["checks"]["kinodynamic"][
        "route_minimum_turn_radius_m"
    ] = 3.0
    unresolved = classify_scenario_admissibility("case-static", scenario_certificate=inconsistent)
    assert unresolved.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN


def test_supported_steering_limit_evidence_excludes_but_unknown_robot_type_does_not() -> None:
    steering = _certificate("kinodynamically_infeasible", eligibility="excluded")
    steering_reason = "bicycle_max_steer_non_positive"
    steering["reasons"] = [steering_reason]
    steering_route = steering["route_certificates"][0]
    steering_route["reasons"] = [steering_reason]
    steering_route["checks"]["kinodynamic"] = {
        "robot_model": "BicycleDriveSettings",
        "command_limits_valid": False,
    }

    unsupported = _certificate("kinodynamically_infeasible", eligibility="excluded")
    unsupported_reason = "unsupported_robot_config: CustomDriveSettings"
    unsupported["reasons"] = [unsupported_reason]
    unsupported_route = unsupported["route_certificates"][0]
    unsupported_route["reasons"] = [unsupported_reason]
    unsupported_route["checks"]["kinodynamic"] = {
        "robot_model": "CustomDriveSettings",
        "command_limits_valid": False,
    }

    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=steering).verdict
        == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    )
    assert (
        classify_scenario_admissibility("case-static", scenario_certificate=unsupported).verdict
        == ADMISSIBLE_FEASIBILITY_UNKNOWN
    )


def test_valid_certificate_and_valid_predicates_do_not_establish_feasibility() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        predicate_contract=_predicates(),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "valid_predicates_do_not_prove_solvability" in verdict.reason_codes


def test_certificate_conflicts_and_dynamic_overconstraint_remain_unknown() -> None:
    conflict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("valid", eligibility="excluded"),
        reference_execution=_execution("reference", route_complete=True),
    )
    dynamic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate("dynamically_overconstrained", eligibility="excluded"),
    )

    assert conflict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "conflicting_feasibility_evidence" in conflict.reason_codes
    assert dynamic.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert dynamic.search_disposition == "retain"


def test_unverifiable_invalid_certificate_remains_unknown() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="planned_path_swept_envelope_unverifiable: geometry unavailable",
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_top_level_impossibility_does_not_reject_a_case_with_another_usable_route() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    certificate["route_certificates"].append(
        {
            "route_id": "route-1",
            "spawn_id": 1,
            "goal_id": 0,
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": [],
            "checks": {"dynamic": {"single_pedestrian_count": 0}},
            "evidence": {},
        }
    )
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


def test_declared_route_count_must_match_before_certificate_can_reject() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.assumptions["scenario_certificate"]["route_inventory"] == {
        "declared_count": 2,
        "observed_count": 1,
        "complete": False,
    }
    assert "scenario_certificate_route_coverage_unresolved" in verdict.reason_codes


@pytest.mark.parametrize("duplicate_identity", ["route_id", "spawn_goal_pair"])
def test_duplicate_route_identities_cannot_exclude_or_prove_actor_free_rollout(
    duplicate_identity: str,
) -> None:
    excluded_certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    positive_certificate = _certificate()
    for certificate in (excluded_certificate, positive_certificate):
        existing = certificate["route_certificates"][0]
        duplicate = {
            **existing,
            "route_id": (existing["route_id"] if duplicate_identity == "route_id" else "route-1"),
            "spawn_id": (
                existing["spawn_id"] + 1
                if duplicate_identity == "route_id"
                else existing["spawn_id"]
            ),
            "checks": dict(existing["checks"]),
            "evidence": dict(existing["evidence"]),
            "reasons": list(existing["reasons"]),
        }
        certificate["checks"]["route_count"] = 2
        certificate["route_certificates"].append(duplicate)

    excluded = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=excluded_certificate,
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )
    empirical = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=positive_certificate,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    for verdict in (excluded, empirical):
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
        assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in excluded.reason_codes
    assert "oracle_success_not_bound_to_static_case_and_provenance" in empirical.reason_codes


def test_oracle_exclusion_does_not_override_unresolved_mixed_route_certificate() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["checks"]["route_count"] = 2
    certificate["route_certificates"].append(
        {
            "route_id": "route-1",
            "spawn_id": 1,
            "goal_id": 0,
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": [],
            "checks": {"dynamic": {"single_pedestrian_count": 0}},
            "evidence": {},
        }
    )
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_exclusion_does_not_override_unresolved_invalid_certificate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="unrecognized invalidity reason",
        ),
        feasibility_evidence=_oracle(
            status="infeasible_by_construction", geometric=False, complete=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_exclusion_requires_a_complete_matching_certificate() -> None:
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    no_certificate = classify_scenario_admissibility("case-static", feasibility_evidence=excluded)
    incomplete = _certificate("geometrically_infeasible", eligibility="excluded")
    incomplete["checks"]["route_count"] = 2
    incomplete_certificate = classify_scenario_admissibility(
        "case-static", scenario_certificate=incomplete, feasibility_evidence=excluded
    )
    conflicting_positive = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=excluded,
    )

    for verdict in (no_certificate, incomplete_certificate, conflicting_positive):
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
        assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in no_certificate.reason_codes
    assert (
        "oracle_geometric_exclusion_route_coverage_unresolved"
        in incomplete_certificate.reason_codes
    )
    assert "oracle_geometric_exclusion_certificate_conflict" in conflicting_positive.reason_codes


@pytest.mark.parametrize("fallback_marker", [0, "false", 1, True])
def test_malformed_no_traversal_fallback_marker_cannot_exclude_case(
    fallback_marker: Any,
) -> None:
    """Malformed or affirmative fallback markers cannot exclude a scenario."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["completion"]["fallback_or_degraded"] = fallback_marker

    assert _oracle_excludes(oracle) is False


@pytest.mark.parametrize("fallback_marker", [None, False])
def test_explicit_no_traversal_fallback_marker_can_exclude_case(
    fallback_marker: bool | None,
) -> None:
    """Only explicitly present literal null or false can pass the no-traversal gate."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["completion"]["fallback_or_degraded"] = fallback_marker

    assert _oracle_excludes(oracle) is True


def test_missing_no_traversal_fallback_marker_cannot_exclude_case() -> None:
    """An absent marker is unknown, not equivalent to explicit JSON null."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    del oracle["completion"]["fallback_or_degraded"]

    assert _oracle_excludes(oracle) is False


def test_incomplete_route_inventory_cannot_bind_oracle_success() -> None:
    certificate = _certificate()
    certificate["checks"]["route_count"] = 2
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_success_not_bound_to_static_case_and_provenance" in verdict.reason_codes


@pytest.mark.parametrize("status", ["blocked", "time_truncated"])
def test_blocked_or_truncated_oracle_results_are_unknown(status: str) -> None:
    oracle = _oracle(status=status, complete=False)
    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=oracle
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"oracle_{status}_does_not_prove_impossibility" in verdict.reason_codes


def test_oracle_geometry_exclusion_without_certificate_remains_unknown() -> None:
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=excluded)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_geometric_exclusion_route_coverage_unresolved" in verdict.reason_codes


def test_oracle_geometry_exclusion_is_named_with_complete_certificate() -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    excluded = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=certificate, feasibility_evidence=excluded
    )

    assert verdict.verdict == GEOMETRIC_OR_KINODYNAMIC_IMPOSSIBILITY
    assert verdict.search_disposition == "reject"
    assert verdict.assumptions["feasibility_oracle"]["envelope_radius_m"] == 0.4


def test_oracle_without_canonical_claim_boundary_remains_unknown() -> None:
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle.pop("claim_boundary")
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=oracle)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "feasibility_oracle_claim_boundary_missing_or_unsupported" in verdict.reason_codes


def test_committed_issue_5574_report_maps_excluded_and_unresolved_cells() -> None:
    report = json.loads(
        Path(
            "docs/context/evidence/issue_5574_feasibility_oracle_2026-07-14/verdicts.json"
        ).read_text(encoding="utf-8")
    )
    excluded = classify_scenario_admissibility(
        "francis2023_narrow_doorway",
        scenario_artifact_path=report["scenario_manifest"],
        scenario_id="francis2023_narrow_doorway",
        feasibility_evidence=report,
    )
    unresolved = classify_scenario_admissibility(
        "francis2023_blind_corner",
        scenario_artifact_path=report["scenario_manifest"],
        scenario_id="francis2023_blind_corner",
        feasibility_evidence=report,
    )

    assert excluded.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert excluded.search_disposition == "retain"
    assert "feasibility_oracle_report_producer_source_digest_missing_mismatch_or_unstable" in (
        excluded.reason_codes
    )
    assert excluded.assumptions["feasibility_oracle"] == {}
    assert unresolved.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert unresolved.search_disposition == "retain"


def test_actor_free_oracle_success_requires_a_static_named_report() -> None:
    report = _oracle_report(_oracle())
    static = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )
    dynamic = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(pedestrian_count=1),
        feasibility_evidence=report,
    )
    unbound = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=_oracle()
    )

    assert static.verdict == EMPIRICALLY_FEASIBLE
    assert static.assumptions["feasibility_oracle"]["empirical_scope"] == (
        "named_actor_free_rollout_of_original_static_case"
    )
    assert dynamic.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert unbound.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("status", "blocked"),
        ("blocker", "rollout_was_blocked"),
        ("fallback_or_degraded", True),
        ("observed_route_completion_feasible", False),
        ("fallback_marker", "fallback_used=true"),
        ("rollout_blocker", "hidden-blocker"),
        ("termination_reason", "collision"),
        ("min_completion_steps", 101),
        ("completion_horizon_margin_steps", 81),
    ],
)
def test_contradictory_actor_free_completion_cannot_prove_feasibility(
    mutation: str, value: Any
) -> None:
    """Blocked, fallback, termination, and horizon conflicts retain the candidate as unknown."""
    report = _oracle_report(_oracle())
    completion = report["cells"][0]["nominal_verdict"]["completion"]
    completion[mutation] = value

    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "oracle_success_not_bound_to_static_case_and_provenance" in verdict.reason_codes


def test_contradictory_positive_completion_cannot_be_overridden_by_geometric_exclusion() -> None:
    """A geometrically excluded cell with contradictory positive completion stays unknown."""
    report = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    completion = report["completion"]
    completion.update(
        route_completion_feasible=True,
        min_completion_steps=20,
        completion_horizon_margin_steps=80,
        termination_reason="success",
        status="passed",
        blocker=None,
    )
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=report,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_route_completion_feasible", True),
        ("rollout_blocker", "hidden-blocker"),
        ("fallback_marker", "fallback_used=true"),
    ],
)
def test_geometric_exclusion_rejects_contradictory_raw_completion_fields(
    field: str, value: Any
) -> None:
    """A no-traversal verdict requires the producer's raw rollout fields to be empty."""
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["completion"][field] = value
    verdict = classify_scenario_admissibility("case-static", feasibility_evidence=oracle)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


def test_oracle_report_producer_digest_rejects_stale_same_path_report(tmp_path: Path) -> None:
    """A fresh certificate cannot make a stale report valid after its source path is reused."""
    scenario_path = tmp_path / "reused.yaml"
    source_a = b"scenarios:\n  - name: case-static\n    seeds: [19]\n"
    source_b = b"scenarios:\n  - name: case-static\n    seeds: [20]\n"
    scenario_path.write_bytes(source_a)
    digest_a = hashlib.sha256(source_a).hexdigest()
    digest_b = hashlib.sha256(source_b).hexdigest()
    report = _oracle_report(_oracle())
    report["scenario_manifest"] = scenario_path.as_posix()
    report["source_artifact_sha256"] = digest_a
    report["cells"][0]["scenario_manifest"] = scenario_path.as_posix()
    report["cells"][0]["source_artifact_sha256"] = digest_a
    scenario_path.write_bytes(source_b)
    fresh_certificate = _certificate()
    fresh_certificate["source"] = scenario_path.as_posix()
    fresh_certificate["evidence"]["source_artifact_sha256"] = digest_b

    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=fresh_certificate,
        feasibility_evidence=report,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "feasibility_oracle_report_producer_source_digest_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )


def test_legacy_certificate_output_keeps_external_closure_unknown(tmp_path: Path) -> None:
    """Legacy v1 output stays unknown when closure identity is only adapter-time."""
    scenario_path = tmp_path / "case_static.yaml"
    map_path = _REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg"
    original_bytes = f"""scenarios:
  - name: case-static
    map_file: {map_path.as_posix()}
    simulation_config:
      max_episode_steps: 100
      ped_density: 0.0
    robot_config: {{}}
    metadata:
      archetype: head_on_corridor
    seeds: [19]
""".encode()
    scenario_path.write_bytes(original_bytes)
    certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )
    identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    assert "source_artifact_sha256" not in certificate["evidence"]
    assert "effective_input_sha256" not in certificate["evidence"]
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=certificate,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    binding = verdict.assumptions["scenario_certificate"]["identity_binding"]
    assert binding["source"] == "admissibility_adapter_current_input_identity"
    assert binding["source_artifact_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert binding["effective_input_sha256"] == identity["effective_input_sha256"]
    assert binding["producer_fields_present"] is False
    assert binding["effective_input_generation_bound"] is False
    assert "scenario_certificate_effective_input_identity_generation_unbound" in (
        verdict.reason_codes
    )


def test_legacy_certificate_cannot_reject_after_external_map_changes(tmp_path: Path) -> None:
    """A genuine legacy cert cannot exclude a changed map under the same root manifest."""
    scenario_path = tmp_path / "scenario.yaml"
    map_path = tmp_path / "map.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "case-static",
                        "map_file": map_path.name,
                        "simulation_config": {"max_episode_steps": 100, "ped_density": 0.0},
                        "robot_config": {
                            "type": "bicycle_drive",
                            "radius": 0.4,
                            "wheelbase": 4.0,
                            "max_steer": 0.5,
                        },
                        "seeds": [19],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def map_payload(*, straight_route: bool) -> dict[str, Any]:
        """Return a serialized map with either a tight turn or straight route."""
        if straight_route:
            waypoints = [[2, 2], [7, 6]]
            goal_zone = [[6.8, 5.8], [7.2, 5.8], [7.2, 6.2]]
        else:
            waypoints = [[2, 2], [3, 2], [3, 3]]
            goal_zone = [[2.8, 2.8], [3.2, 2.8], [3.2, 3.2]]
        return {
            "x_margin": [0, 12],
            "y_margin": [0, 8],
            "obstacles": [],
            "robot_spawn_zones": [
                [[1.8, 1.8], [2.2, 1.8], [2.2, 2.2]],
                [[9.8, 6.8], [10.2, 6.8], [10.2, 7.2]],
            ],
            "robot_goal_zones": [
                [[1.8, 1.8], [2.2, 1.8], [2.2, 2.2]],
                goal_zone,
            ],
            "ped_spawn_zones": [],
            "ped_goal_zones": [],
            "ped_crowded_zones": [],
            "robot_routes": [{"spawn_id": 0, "goal_id": 1, "waypoints": waypoints}],
            "ped_routes": [],
        }

    map_path.write_text(yaml.safe_dump(map_payload(straight_route=False)), encoding="utf-8")
    legacy_certificate = certificate_to_dict(
        certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    )
    assert legacy_certificate["classification"] == "kinodynamically_infeasible"
    assert legacy_certificate["benchmark_eligibility"] == "excluded"
    assert "effective_input_sha256" not in legacy_certificate["evidence"]
    root_manifest_bytes = scenario_path.read_bytes()

    map_path.write_text(yaml.safe_dump(map_payload(straight_route=True)), encoding="utf-8")
    from robot_sf.training import scenario_loader

    scenario_loader._load_map_definition.cache_clear()
    current_certificate = certify_scenario_file(scenario_path, scenario_id="case-static")[0]
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=legacy_certificate,
    )

    assert scenario_path.read_bytes() == root_manifest_bytes
    assert current_certificate.classification == "hard_but_solvable"
    assert current_certificate.benchmark_eligibility == "eligible"
    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_effective_input_identity_generation_unbound" in (
        verdict.reason_codes
    )
    assert (
        verdict.assumptions["scenario_certificate"]["identity_binding"][
            "effective_input_generation_bound"
        ]
        is False
    )


def test_present_certificate_producer_digest_must_match_adapter_identity() -> None:
    """Optional future producer digests remain checked when a certificate supplies them."""

    certificate = _certificate()
    certificate["evidence"]["source_artifact_sha256"] = "0" * 64

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "scenario_certificate_producer_source_digest_missing_or_mismatch" in (
        verdict.reason_codes
    )


@pytest.mark.parametrize("referenced_file", ["map.svg", "routes.yaml"])
def test_certificate_and_oracle_reject_stale_runtime_input_bytes(
    tmp_path: Path, referenced_file: str
) -> None:
    """Same manifest bytes cannot preserve evidence after a referenced input changes."""
    scenario_path = _referenced_scenario(tmp_path)
    certificate = _bind_certificate_to_scenario(scenario_path)
    oracle = _bind_oracle_to_scenario(scenario_path)
    original_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    original_manifest_digest = original_identity["source_artifact_sha256"]
    execution = _execution("reference", route_complete=True)
    execution.update(
        scenario_sha256=original_identity["source_artifact_sha256"],
        effective_input_sha256=original_identity["effective_input_sha256"],
        effective_input_identity_stable=True,
    )
    referenced_path = tmp_path / referenced_file
    referenced_path.write_bytes(referenced_path.read_bytes() + b"\n# changed input\n")

    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=scenario_path,
        scenario_id="case-static",
        scenario_certificate=certificate,
        feasibility_evidence=oracle,
        reference_execution=execution,
    )

    assert hashlib.sha256(scenario_path.read_bytes()).hexdigest() == original_manifest_digest
    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_effective_input_identity_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )
    assert "feasibility_oracle_effective_input_identity_missing_mismatch_or_unstable" in (
        verdict.reason_codes
    )
    assert "reference_execution_effective_input_identity_missing_or_mismatch" in (
        verdict.reason_codes
    )


@pytest.mark.parametrize(
    "external_reference",
    (
        "map_id: fixture-map",
        "map_file: map.svg",
        "route_overrides_file: routes.yaml",
        "include: [included.yaml]",
        "includes: [included.yaml]",
        "scenario_files: [included.yaml]",
        "map_search_paths: [maps]",
    ),
)
def test_legacy_single_row_identity_rejects_external_references(
    tmp_path: Path, external_reference: str
) -> None:
    """Incomplete expansion cannot fall back to a root-only identity with references."""
    path = tmp_path / "legacy.yaml"
    path.write_text(f"name: legacy\n{external_reference}\n", encoding="utf-8")

    identity = scenario_input_identity(path, scenario_id="legacy")

    assert identity["status"] == "unavailable"
    assert identity["effective_input_sha256"] is None


def test_adapter_identity_rejects_aba_include_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter-owned manifest identity fails closed when an include changes during parse."""
    scenario_path = tmp_path / "root.yaml"
    included_path = tmp_path / "included.yaml"
    scenario_path.write_text("includes: [included.yaml]\n", encoding="utf-8")
    original_bytes = b"scenarios:\n  - name: case-static\n    marker: restored-A\n    seeds: [19]\n"
    consumed_bytes = b"scenarios:\n  - name: case-static\n    marker: consumed-B\n    seeds: [19]\n"
    included_path.write_bytes(original_bytes)
    original_read_bytes = Path.read_bytes
    swapped = False

    def read_with_aba(path: Path) -> bytes:
        nonlocal swapped
        if path.resolve() == included_path.resolve() and not swapped:
            swapped = True
            included_path.write_bytes(consumed_bytes)
            try:
                return original_read_bytes(path)
            finally:
                included_path.write_bytes(original_bytes)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_with_aba)

    identity = scenario_input_identity(scenario_path, scenario_id="case-static")

    assert included_path.read_bytes() == original_bytes
    assert identity["status"] == "unavailable"
    assert identity["effective_input_sha256"] is None
    assert identity["reason_code"] == "included_scenario_manifest_changed_during_load"


def test_opt_in_map_parser_cache_tracks_exact_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing map bytes at one path parses the new immutable source snapshot."""
    from robot_sf.nav import svg_map_parser
    from robot_sf.training import scenario_loader

    source_path = tmp_path / "map.svg"
    initial_bytes = (_REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg").read_bytes()
    updated_bytes = initial_bytes + b"\n<!-- cache-key-change -->\n"
    source_path.write_bytes(initial_bytes)
    parsed_sources: list[bytes] = []
    original_convert = svg_map_parser.convert_map

    def capture_source(
        path: str, *, geometry_contract: str = "legacy", source_bytes: bytes | None = None
    ) -> Any:
        assert source_bytes is not None
        parsed_sources.append(source_bytes)
        return original_convert(
            path, geometry_contract=geometry_contract, source_bytes=source_bytes
        )

    scenario_loader._load_map_definition_cached.cache_clear()
    monkeypatch.setattr(svg_map_parser, "convert_map", capture_source)
    try:
        first, _ = scenario_loader._load_map_definition_with_digest(str(source_path))
        source_path.write_bytes(updated_bytes)
        second, _ = scenario_loader._load_map_definition_with_digest(str(source_path))
    finally:
        scenario_loader._load_map_definition_cached.cache_clear()

    assert first is not None and second is not None
    assert parsed_sources == [initial_bytes, updated_bytes]
    assert first is not second


def test_legacy_map_loader_keeps_path_parser_without_input_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy scenario loader does not opt into the new parser snapshot path."""
    from robot_sf.nav import svg_map_parser
    from robot_sf.training import scenario_loader

    source_path = _REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg"
    parsed_sources: list[bytes | None] = []
    original_convert = svg_map_parser.convert_map

    def capture_source(
        path: str, *, geometry_contract: str = "legacy", source_bytes: bytes | None = None
    ) -> Any:
        parsed_sources.append(source_bytes)
        return original_convert(
            path, geometry_contract=geometry_contract, source_bytes=source_bytes
        )

    scenario_loader._load_map_definition.cache_clear()
    monkeypatch.setattr(svg_map_parser, "convert_map", capture_source)
    try:
        assert scenario_loader._load_map_definition(str(source_path)) is not None
    finally:
        scenario_loader._load_map_definition.cache_clear()

    assert parsed_sources == [None]


def test_default_map_pool_is_part_of_runtime_input_identity(tmp_path: Path) -> None:
    """An implicit default map pool is included and matches the bytes its loader consumed."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = tmp_path / "default-map.yaml"
    scenario = {"name": "default-map-case", "seeds": [19]}
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False), encoding="utf-8"
    )
    identity = scenario_input_identity(scenario_path, scenario_id="default-map-case")
    consumed: list[dict[str, str]] = []

    build_robot_config_from_scenario(
        scenario,
        scenario_path=scenario_path,
        runtime_input_records=consumed,
    )

    assert identity["status"] == "available"
    assert identity["requires_effective_input_binding"] is True
    assert any(item["role"] == "default_map_pool" for item in identity["files"])
    assert runtime_input_records_match(identity, consumed, scenario_id="default-map-case") is True


def test_runtime_identity_rejects_bytes_consumed_during_aba_map_replacement(
    tmp_path: Path,
) -> None:
    """Parser-consumed map hashes catch replacement even when path bytes are restored."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = _referenced_scenario(tmp_path)
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    original_map_bytes = (tmp_path / "map.svg").read_bytes()
    identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    consumed_variant = original_map_bytes + b"\n<!-- consumed-during-ABA -->\n"
    (tmp_path / "map.svg").write_bytes(consumed_variant)
    consumed: list[dict[str, str]] = []
    try:
        build_robot_config_from_scenario(
            scenario,
            scenario_path=scenario_path,
            runtime_input_records=consumed,
        )
    finally:
        (tmp_path / "map.svg").write_bytes(original_map_bytes)

    assert (
        scenario_input_identity(scenario_path, scenario_id="case-static")["effective_input_sha256"]
        == identity["effective_input_sha256"]
    )
    assert runtime_input_records_match(identity, consumed, scenario_id="case-static") is False


def test_runtime_identity_rejects_route_bytes_parsed_during_aba_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route YAML identity follows the exact byte snapshot passed to its parser."""
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario

    scenario_path = _referenced_scenario(tmp_path)
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    route_path = tmp_path / "routes.yaml"
    original_route_bytes = route_path.read_bytes()
    consumed_variant = original_route_bytes + b"# consumed-during-ABA\n"
    identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    original_read_bytes = Path.read_bytes

    def read_route_during_aba(path: Path) -> bytes:
        if path.resolve() == route_path.resolve():
            route_path.write_bytes(consumed_variant)
            try:
                consumed = original_read_bytes(path)
            finally:
                route_path.write_bytes(original_route_bytes)
            return consumed
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_route_during_aba)
    consumed: list[dict[str, str]] = []
    build_robot_config_from_scenario(
        scenario,
        scenario_path=scenario_path,
        runtime_input_records=consumed,
    )

    assert original_read_bytes(route_path) == original_route_bytes
    assert any(
        item["role"] == "route_overrides_file"
        and item["sha256"] == hashlib.sha256(consumed_variant).hexdigest()
        for item in consumed
    )
    assert runtime_input_records_match(identity, consumed, scenario_id="case-static") is False


def test_map_registry_remap_changes_effective_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal map bytes selected through different suffixes retain parser identity."""
    from robot_sf.training import scenario_loader

    scenario_path = tmp_path / "candidate.yaml"
    registry_path = tmp_path / "registry.yaml"
    map_svg = tmp_path / "map.svg"
    map_yaml = tmp_path / "map.yaml"
    map_bytes = b"same map bytes\n"
    map_svg.write_bytes(map_bytes)
    map_yaml.write_bytes(map_bytes)
    scenario_path.write_text(
        "scenarios:\n  - name: case-static\n    map_id: fixture-map\n    seeds: [19]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", registry_path.as_posix())
    scenario_loader._load_map_registry.cache_clear()
    try:
        registry_path.write_text("maps:\n  fixture-map: map.svg\n", encoding="utf-8")
        svg_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
        registry_path.write_text("maps:\n  fixture-map: map.yaml\n", encoding="utf-8")
        scenario_loader._load_map_registry.cache_clear()
        yaml_identity = scenario_input_identity(scenario_path, scenario_id="case-static")
    finally:
        scenario_loader._load_map_registry.cache_clear()

    assert svg_identity["status"] == yaml_identity["status"] == "available"
    assert svg_identity["effective_input_sha256"] != yaml_identity["effective_input_sha256"]
    svg_map = next(item for item in svg_identity["files"] if item["role"] == "map_file")
    yaml_map = next(item for item in yaml_identity["files"] if item["role"] == "map_file")
    assert svg_map["sha256"] == yaml_map["sha256"]
    assert (svg_map["map_id"], svg_map["parser"], svg_map["path"]) == (
        "fixture-map",
        "svg",
        "map.svg",
    )
    assert (yaml_map["map_id"], yaml_map["parser"], yaml_map["path"]) == (
        "fixture-map",
        "legacy_serialized_map",
        "map.yaml",
    )


def test_actor_free_oracle_success_does_not_resolve_unknown_invalid_certificate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(
            "invalid",
            eligibility="excluded",
            route_reason="unrecognized invalidity reason",
        ),
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_invalidity_unresolved" in verdict.reason_codes


def test_actor_free_oracle_uses_explicit_scenario_binding_for_stable_case_id() -> None:
    verdict = classify_scenario_admissibility(
        "counterexample-0001",
        scenario_id="case-static",
        scenario_certificate=_certificate(),
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.case_id == "counterexample-0001"
    assert verdict.scenario_id == "case-static"


def test_malformed_certificate_cannot_bind_oracle_success() -> None:
    malformed = _certificate()
    malformed.pop("schema_version")
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=malformed,
        feasibility_evidence=_oracle_report(_oracle()),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "scenario_certificate_missing_or_malformed" in verdict.reason_codes


def test_certificate_identity_mismatch_cannot_reject_a_candidate() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_id="case-static",
        scenario_certificate=_certificate("invalid", eligibility="excluded", scenario_id="other"),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_certificate_identity_mismatch" in verdict.reason_codes


def test_rejection_and_execution_evidence_must_share_candidate_artifact_bytes(
    tmp_path: Path,
) -> None:
    other_artifact = tmp_path / "same-id-different-artifact.yaml"
    other_artifact.write_text("id: case-static\nchanged: true\n", encoding="utf-8")
    other_digest = hashlib.sha256(other_artifact.read_bytes()).hexdigest()
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    certificate["source"] = other_artifact.as_posix()
    oracle = _oracle(status="infeasible_by_construction", geometric=False, complete=False)
    oracle["scenario_manifest"] = other_artifact.as_posix()
    execution = _execution("reference", route_complete=True)
    execution["scenario_sha256"] = other_digest

    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=certificate,
        feasibility_evidence=oracle,
        reference_execution=execution,
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.evidence["scenario_artifact_identity"]["sha256"] == (_SCENARIO_ARTIFACT_SHA256)
    assert "scenario_certificate_scenario_artifact_identity_mismatch" in verdict.reason_codes
    assert "feasibility_oracle_scenario_artifact_identity_mismatch" in verdict.reason_codes
    assert "reference_execution_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_oracle_report_and_selected_cell_cannot_disagree_on_artifact(
    tmp_path: Path,
) -> None:
    other_artifact = tmp_path / "other.yaml"
    other_artifact.write_text("id: case-static\nrevision: other\n", encoding="utf-8")
    report = _oracle_report(_oracle())
    report["cells"][0]["scenario_manifest"] = other_artifact.as_posix()

    verdict = classify_scenario_admissibility(
        "case-static", scenario_certificate=_certificate(), feasibility_evidence=report
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "feasibility_oracle_cell_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_missing_canonical_candidate_artifact_cannot_support_a_certificate_exclusion() -> None:
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_id="case-static",
        scenario_certificate=_certificate("invalid", eligibility="excluded"),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "scenario_artifact_identity_missing_or_unavailable" in verdict.reason_codes
    assert "scenario_certificate_scenario_artifact_identity_unavailable" in verdict.reason_codes


def test_execution_scenario_hash_must_match_canonical_candidate_artifact() -> None:
    run = _execution("reference", route_complete=True)
    run["scenario_sha256"] = "f" * 64

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_scenario_artifact_identity_mismatch" in verdict.reason_codes


def test_named_execution_requires_caller_bound_scenario_id() -> None:
    verdict = _classify_scenario_admissibility(
        "case-static",
        scenario_artifact_path=_SCENARIO_ARTIFACT,
        scenario_certificate=_certificate(),
        reference_execution=_execution(
            "reference", route_complete=True, scenario_id="different-scenario"
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert verdict.scenario_id is None
    assert "scenario_certificate_identity_unbound" in verdict.reason_codes
    assert "reference_execution_scenario_identity_unbound" in verdict.reason_codes


def test_named_execution_nonexistent_evidence_ref_stays_unknown(tmp_path: Path) -> None:
    run = _execution("reference", route_complete=True)
    run["evidence_ref"] = (tmp_path / "missing-episodes.jsonl").as_posix()

    verdict = classify_scenario_admissibility(
        "case-static", reference_execution=run, evidence_root=tmp_path
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_evidence_ref_missing_or_unreadable" in verdict.reason_codes
    assert verdict.evidence["execution_artifact_bindings"]["reference"]["status"] == "unavailable"


def test_named_execution_episode_store_digest_must_match_readable_bytes() -> None:
    run = _execution("reference", route_complete=True)
    run["source_episodes_jsonl_sha256"] = "0" * 64

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_episode_store_digest_mismatch" in verdict.reason_codes
    assert verdict.evidence["execution_artifact_bindings"]["reference"]["status"] == "mismatch"


def test_named_execution_episode_identity_must_exist_in_store() -> None:
    run = _execution("reference", route_complete=True)
    run["episode_id"] = "fabricated-episode"

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_episode_identity_missing_or_ambiguous" in verdict.reason_codes


def test_named_execution_normalized_outcome_must_match_episode_row() -> None:
    run = _execution("reference", route_complete=True)
    run["route_complete"] = False

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_episode_outcome_mismatch" in verdict.reason_codes


def test_producer_scenario_matrix_must_match_the_admissibility_artifact() -> None:
    run = _execution("reference", route_complete=True)
    episode_store = Path(run["evidence_ref"])
    manifest_path = manifest_path_for_result_jsonl(episode_store)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["scenario_matrix"]["sha256"] = "0" * 64
    write_result_provenance_manifest(manifest_path, manifest)

    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    assert binding["status"] == "valid"
    assert binding["run_context_binding"]["status"] == "mismatch"
    assert binding["run_context_binding"]["scenario_matrix_binding_status"] == "mismatch"
    assert binding["producer_provenance_binding"]["status"] == "mismatch"
    assert "reference_execution_run_context_mismatch" in verdict.reason_codes
    assert "reference_execution_scenario_case_identity_mismatch" in verdict.reason_codes


def test_relative_execution_evidence_ref_uses_explicit_root(tmp_path: Path) -> None:
    run = _execution("reference", route_complete=True)
    source_path = Path(run["evidence_ref"])
    target_path = tmp_path / "episodes.jsonl"
    target_path.write_bytes(source_path.read_bytes())
    run["evidence_ref"] = target_path.name

    verdict = classify_scenario_admissibility(
        "case-static", reference_execution=run, evidence_root=tmp_path
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    assert binding["episode_store_path"] == target_path.as_posix()
    assert binding["producer_provenance_binding"]["status"] == "unavailable"
    assert "reference_execution_scenario_case_identity_unavailable" in verdict.reason_codes


def test_reference_success_is_empirical_but_target_failure_alone_is_unknown() -> None:
    reference = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
    )
    target_only = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        target_execution=_execution("target", route_complete=False),
    )

    assert reference.verdict == EMPIRICALLY_FEASIBLE
    reference_binding = reference.evidence["execution_artifact_bindings"]["reference"]
    assert reference_binding["status"] == "valid"
    run_context = reference_binding["run_context_binding"]
    assert run_context["selected_scenario_row_binding_status"] == "valid"
    assert run_context["scenario_runtime_input_closure_binding_status"] == "valid"
    assert target_only.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert target_only.target_planner_outcome == "route_incomplete"
    assert "target_failure_alone_does_not_prove_infeasibility" in target_only.reason_codes


def test_self_consistent_producer_episode_for_a_different_candidate_is_unknown() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True, producer_goal_x=1.0),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_scenario_case_identity_mismatch" in verdict.reason_codes
    binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    producer_binding = binding["producer_provenance_binding"]
    row_binding = producer_binding["run_context_binding"]
    assert row_binding["status"] == "mismatch"
    assert row_binding["selected_scenario_row_binding_status"] == "mismatch"
    assert (
        producer_binding["candidate_scenario_case_identity_sha256"]
        != producer_binding["producer_scenario_case_identity_sha256"]
    )


def test_consistent_wrong_case_reference_target_and_replay_do_not_prove_failure() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True, producer_goal_x=1.0),
        target_execution=_execution("target", route_complete=False, producer_goal_x=1.0),
        replay_execution=_execution(
            "target", route_complete=False, replay=True, producer_goal_x=1.0
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_scenario_case_identity_mismatch" in verdict.reason_codes
    assert "target_execution_scenario_case_identity_mismatch" in verdict.reason_codes
    assert "replay_execution_scenario_case_identity_mismatch" in verdict.reason_codes
    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE


def test_execution_with_unbound_producer_resource_closure_remains_unknown() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution(
            "reference", route_complete=True, include_runtime_input_records=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_scenario_runtime_input_closure_unavailable" in (
        verdict.reason_codes
    )
    binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    assert (
        binding["run_context_binding"]["scenario_runtime_input_closure_binding_status"]
        == "unavailable"
    )


def test_execution_with_unbound_selected_map_remains_unknown() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution(
            "reference", route_complete=True, include_selected_map_identity=False
        ),
    )

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_selected_map_identity_unavailable" in verdict.reason_codes


def test_selected_map_mismatch_cannot_support_planner_specific_failure() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution(
            "target", route_complete=False, selected_map_id="different-map"
        ),
        replay_execution=_execution(
            "target", route_complete=False, replay=True, selected_map_id="different-map"
        ),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    assert "target_execution_selected_map_identity_mismatch" in verdict.reason_codes
    assert "replay_execution_selected_map_identity_mismatch" in verdict.reason_codes


def test_same_realized_map_identity_ignores_worktree_local_path() -> None:
    left = {
        "map_id": "map-a",
        "path": "/worktrees/one/maps/map-a.svg",
        "sha256": "a" * 64,
        "source_role": "default_map_pool",
    }
    right = {
        "map_id": "map-a",
        "path": "/worktrees/two/maps/map-a.svg",
        "sha256": "a" * 64,
        "source_role": "default_map_pool",
    }

    assert _same_selected_map_identity(left, right)
    assert not _same_selected_map_identity(left, {**right, "sha256": "b" * 64})


def test_matched_reference_target_failure_needs_reproducing_replay() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("target", route_complete=False, replay=True),
    )
    absent_replay = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
    )
    replay_success = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("target", route_complete=True, replay=True),
    )
    wrong_planner = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("other", route_complete=False, replay=True),
    )
    unmatched = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False, seed=20),
    )

    assert verdict.verdict == PLANNER_SPECIFIC_FAILURE
    assert verdict.search_disposition == "retain"
    assert verdict.evidence["execution_artifact_bindings"]["target"]["status"] == "valid"
    assert verdict.evidence["execution_artifact_bindings"]["replay"]["status"] == "valid"
    target_producer_binding = verdict.evidence["execution_artifact_bindings"]["target"][
        "producer_provenance_binding"
    ]
    reference_producer_binding = verdict.evidence["execution_artifact_bindings"]["reference"][
        "producer_provenance_binding"
    ]
    assert (
        target_producer_binding["row_config_hash"] != reference_producer_binding["row_config_hash"]
    )
    assert (
        target_producer_binding["case_identity_sha256"]
        == reference_producer_binding["case_identity_sha256"]
    )
    assert (
        verdict.evidence["execution_artifact_bindings"]["target"]["producer_provenance_binding"][
            "run_context_binding"
        ]["status"]
        == "valid"
    )
    assert "matched_reference_target_failure_reproduced_by_replay" in verdict.reason_codes
    assert absent_replay.verdict == EMPIRICALLY_FEASIBLE
    assert absent_replay.target_planner_outcome == "route_incomplete"
    assert "planner_specific_failure_replay_missing_or_invalid" in absent_replay.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in absent_replay.reason_codes
    assert replay_success.verdict == EMPIRICALLY_FEASIBLE
    assert replay_success.target_planner_outcome == "route_incomplete"
    assert "planner_specific_failure_replay_source_episode_mismatch" in replay_success.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in replay_success.reason_codes
    assert wrong_planner.verdict == EMPIRICALLY_FEASIBLE
    assert wrong_planner.target_planner_outcome == "route_incomplete"
    assert "planner_specific_failure_replay_wrong_planner" in wrong_planner.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in wrong_planner.reason_codes
    assert unmatched.verdict == EMPIRICALLY_FEASIBLE
    assert "matched_reference_success_target_planner_failure" not in unmatched.reason_codes


def test_replay_result_must_bind_a_separate_target_planner_outcome() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["route_complete"] = True
    result["termination_reason"] = "success"
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "replay_execution_target_planner_result_outcome_mismatch" in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_replay_result_requires_a_distinct_canonical_replay_episode_store() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("replay_episodes_jsonl_path")
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "replay_execution_target_planner_result_schema_invalid" in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_replay_result_cannot_relabel_the_source_episode_as_replay_output() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["replay_episode_id"] = result["episode_id"]
    result["replay_episodes_jsonl_path"] = sidecar["source_episodes_jsonl_path"]
    result["replay_episodes_jsonl_sha256"] = sidecar["source_episodes_jsonl_sha256"]
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "replay_execution_target_planner_result_replay_episode_store_not_distinct" in (
        verdict.reason_codes
    )
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_replay_result_cannot_reuse_the_target_producer_run_id() -> None:
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    output_manifest_path = Path(result["replay_provenance_manifest_path"])
    source_manifest_path = manifest_path_for_result_jsonl(Path(target["evidence_ref"]))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_store_path = Path(result["replay_episodes_jsonl_path"])
    for artifact in source_manifest["raw_artifacts"]:
        if artifact["kind"] == "episodes_jsonl":
            artifact["path"] = output_store_path.as_posix()
            artifact["sha256"] = hashlib.sha256(output_store_path.read_bytes()).hexdigest()
    for row in source_manifest["rows"]:
        row["raw_artifact"] = output_store_path.as_posix()
    write_result_provenance_manifest(output_manifest_path, source_manifest)
    result["replay_provenance_manifest_sha256"] = hashlib.sha256(
        output_manifest_path.read_bytes()
    ).hexdigest()
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "planner_specific_failure_replay_reused_target_producer_run" in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_replay_output_episode_store_digest_must_match_its_bytes() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["replay_episodes_jsonl_sha256"] = "0" * 64
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert (
        "replay_execution_target_planner_result_replay_episode_store_digest_mismatch"
        in verdict.reason_codes
    )
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_episode_horizon_must_match_normalized_execution_context() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    episode_path = Path(target["evidence_ref"])
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["horizon"] = 101
    episode_bytes = (json.dumps(episode, sort_keys=True) + "\n").encode()
    episode_path.write_bytes(episode_bytes)
    target["source_episodes_jsonl_sha256"] = hashlib.sha256(episode_bytes).hexdigest()

    verdict = classify_scenario_admissibility(
        "case-static", reference_execution=reference, target_execution=target
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    assert "target_execution_episode_horizon_mismatch" in verdict.reason_codes


def test_replay_runtime_error_cannot_establish_planner_specific_failure() -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    sidecar_path = Path(replay["evidence_ref"])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_path = Path(sidecar["target_planner_replay_result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["termination_reason"] = "error"
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
    result_path.write_bytes(result_bytes)
    sidecar["target_planner_replay_result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "replay_execution_target_planner_result_schema_invalid" in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


def test_unbound_run_context_cannot_support_planner_specific_attribution() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution(
            "reference", route_complete=True, include_producer_context=False
        ),
        target_execution=_execution("target", route_complete=False),
        replay_execution=_execution("target", route_complete=False, replay=True),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert "reference_execution_run_context_unavailable" in verdict.reason_codes
    assert "matched_reference_target_failure_run_context_unbound" in verdict.reason_codes
    assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes
    reference_binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    assert reference_binding["run_context_binding"]["scenario_matrix_binding_status"] == "valid"
    assert reference_binding["run_context_binding"]["case_identity_binding_status"] == "valid"


def test_missing_producer_manifest_cannot_prove_candidate_case_feasibility() -> None:
    reference = _execution("reference", route_complete=True)
    Path(manifest_path_for_result_jsonl(Path(reference["evidence_ref"]))).unlink()

    verdict = classify_scenario_admissibility("case-static", reference_execution=reference)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_run_context_unavailable" in verdict.reason_codes
    assert "reference_execution_scenario_case_identity_unavailable" in verdict.reason_codes
    binding = verdict.evidence["execution_artifact_bindings"]["reference"]
    assert binding["producer_provenance_binding"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("case_id", "other-case", "replay_execution_identity_mismatch"),
        ("scenario_id", "other-scenario", "replay_execution_identity_mismatch"),
        (
            "scenario_sha256",
            "f" * 64,
            "replay_execution_scenario_artifact_identity_mismatch",
        ),
        (
            "robot_model_sha256",
            "f" * 64,
            "planner_specific_failure_replay_case_mismatch",
        ),
        (
            "simulator_config_sha256",
            "f" * 64,
            "planner_specific_failure_replay_case_mismatch",
        ),
        (
            "environment_sha256",
            "f" * 64,
            "planner_specific_failure_replay_case_mismatch",
        ),
        ("source_commit", "e" * 40, "replay_execution_episode_identity_mismatch"),
        ("seed", 20, "replay_execution_episode_identity_mismatch"),
        (
            "horizon_steps",
            101,
            "replay_execution_episode_horizon_mismatch",
        ),
        (
            "planner_config_sha256",
            "f" * 64,
            "replay_execution_run_context_mismatch",
        ),
        (
            "planner_checkpoint_sha256",
            "f" * 64,
            "replay_execution_run_context_unavailable",
        ),
        (
            "episode_id",
            "unrelated-episode",
            "replay_execution_episode_identity_missing_or_ambiguous",
        ),
        (
            "source_episodes_jsonl_sha256",
            "b" * 64,
            "replay_execution_episode_store_digest_mismatch",
        ),
    ],
)
def test_replay_must_match_target_case_and_execution_bindings(
    field: str, value: Any, reason: str
) -> None:
    replay = _execution("target", route_complete=False, replay=True)
    replay[field] = value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=_execution("reference", route_complete=True),
        target_execution=_execution("target", route_complete=False),
        replay_execution=replay,
    )

    assert verdict.target_planner_outcome == "route_incomplete"
    if reason is None:
        assert verdict.verdict == PLANNER_SPECIFIC_FAILURE
        assert "matched_reference_target_failure_reproduced_by_replay" in verdict.reason_codes
    else:
        assert verdict.verdict == EMPIRICALLY_FEASIBLE
        assert reason in verdict.reason_codes
        assert "planner_specific_failure_attribution_unconfirmed" in verdict.reason_codes


@pytest.mark.parametrize("checked_count", [True, 1.0], ids=["boolean", "float"])
def test_simulator_collision_requires_integer_sample_count(checked_count: Any) -> None:
    certificate = _certificate("geometrically_infeasible", eligibility="excluded")
    reason = "planned_path_simulator_collision: first_collision_sample_index=0"
    certificate["reasons"] = [reason]
    route = certificate["route_certificates"][0]
    route["reasons"] = [reason]
    route["checks"].update(
        {
            "inflated_collision_free_path": False,
            "simulator_obstacle_collision": {
                "validated": True,
                "collides_obstacle": True,
                "runtime_component": "ContinuousOccupancy.is_obstacle_collision",
                "obstacle_source": "MapDefinition.obstacles_pysf_runtime_normalized",
                "sample_spacing_m": 0.05,
                "checked_sample_count": checked_count,
                "first_collision_sample_index": 0,
            },
        }
    )

    verdict = classify_scenario_admissibility("case-static", scenario_certificate=certificate)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"determinism_check_status": "fail"}, "replay_determinism_check_not_passed"),
        ({"resimulated": False}, "replay_did_not_resimulate_source_episode"),
    ],
)
def test_replay_success_requires_resimulation_and_determinism(
    change: dict[str, Any], reason: str
) -> None:
    replay = _execution("replay", route_complete=True)
    replay.update(change)
    verdict = classify_scenario_admissibility("case-static", replay_execution=replay)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert reason in verdict.reason_codes


def test_replay_completion_is_empirical_only_after_validated_resimulation() -> None:
    verdict = classify_scenario_admissibility(
        "case-static",
        scenario_certificate=_certificate(),
        replay_execution=_execution("replay", route_complete=True),
    )

    assert verdict.verdict == EMPIRICALLY_FEASIBLE
    assert verdict.evidence["replay_execution"]["resimulated"] is True


def test_execution_with_invalid_budget_is_not_used_as_feasibility_evidence() -> None:
    run = _execution("reference", route_complete=True, seed=-1)
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_outcome_or_budget_invalid" in verdict.reason_codes


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
@pytest.mark.parametrize(
    "status_value",
    ["false", None],
    ids=["malformed", "null"],
)
def test_fallback_or_degraded_execution_cannot_establish_feasibility_or_planner_failure(
    role: str, status_value: Any
) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    changed = {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role]
    changed["fallback_or_degraded"] = status_value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_fallback_status_missing_or_malformed" in verdict.reason_codes


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_or_degraded", True),
        ("planner_runtime", {"readiness_status": "degraded"}),
        ("planner_runtime", {"fallback_used": True}),
        ("planner_runtime", {"availability_status": "fallback"}),
    ],
    ids=["summary-flag", "nested-degraded", "nested-fallback-flag", "nested-fallback-status"],
)
def test_canonical_fallback_and_degraded_signals_never_support_classification(
    role: str, field: str, value: Any
) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    changed = {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role]
    changed[field] = value
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_fallback_or_degraded" in verdict.reason_codes


@pytest.mark.parametrize("role", ["reference", "target", "replay"])
def test_missing_fallback_status_cannot_establish_outcomes(role: str) -> None:
    reference = _execution("reference", route_complete=True)
    target = _execution("target", route_complete=False)
    replay = _execution("target", route_complete=False, replay=True)
    {
        "reference": reference,
        "target": target,
        "replay": replay,
    }[role].pop("fallback_or_degraded")
    verdict = classify_scenario_admissibility(
        "case-static",
        reference_execution=reference,
        target_execution=target,
        replay_execution=replay,
    )

    assert verdict.verdict != PLANNER_SPECIFIC_FAILURE
    if role == "reference":
        assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert f"{role}_execution_provenance_incomplete" in verdict.reason_codes


def test_execution_with_invalid_digest_is_not_used_as_feasibility_evidence() -> None:
    run = _execution("reference", route_complete=True)
    run["environment_sha256"] = "fixture-hash"
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


def test_not_applicable_checkpoint_provenance_is_limited_to_classical_planners() -> None:
    classical = classify_scenario_admissibility(
        "case-static", reference_execution=_execution("goal", route_complete=True)
    )
    learned = _execution("ppo", route_complete=True)
    learned["planner_checkpoint_sha256"] = "not_applicable"
    rejected = classify_scenario_admissibility("case-static", reference_execution=learned)

    assert classical.verdict == EMPIRICALLY_FEASIBLE
    assert rejected.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in rejected.reason_codes


def test_not_applicable_checkpoint_sentinel_rejects_checkpoint_backed_sicnav() -> None:
    sicnav = _execution("sicnav", route_complete=True)
    sicnav["planner_checkpoint_sha256"] = "not_applicable"
    verdict = classify_scenario_admissibility("case-static", reference_execution=sicnav)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert verdict.search_disposition == "retain"
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


@pytest.mark.parametrize("field", ["planner_config_sha256", "planner_checkpoint_sha256"])
def test_execution_with_missing_planner_provenance_stays_unknown(field: str) -> None:
    run = _execution("reference", route_complete=True)
    run.pop(field)
    verdict = classify_scenario_admissibility("case-static", reference_execution=run)

    assert verdict.verdict == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert "reference_execution_provenance_incomplete" in verdict.reason_codes


def test_partition_rejects_only_explicit_exclusions_and_preserves_unknown() -> None:
    verdicts = [
        classify_scenario_admissibility(
            "invalid", scenario_certificate=_certificate("invalid", eligibility="excluded")
        ),
        classify_scenario_admissibility("unknown", scenario_certificate=_certificate()),
        classify_scenario_admissibility(
            "failure",
            scenario_certificate=_certificate(),
            reference_execution=_execution("reference", route_complete=True, case_id="failure"),
            target_execution=_execution("target", route_complete=False, case_id="failure"),
            replay_execution=_execution(
                "target", route_complete=False, case_id="failure", replay=True
            ),
        ),
    ]
    partition = partition_candidates_by_admissibility(verdicts)

    assert partition.retained_case_ids == ("unknown", "failure")
    assert partition.rejected[0]["case_id"] == "invalid"
    assert partition.by_verdict[ADMISSIBLE_FEASIBILITY_UNKNOWN] == ("unknown",)
    assert partition.by_verdict[PLANNER_SPECIFIC_FAILURE] == ("failure",)
    with pytest.raises(ValueError, match="unique"):
        partition_candidates_by_admissibility([verdicts[1], verdicts[1]])


def test_serialized_verdict_matches_contract_schema() -> None:
    payload = classify_scenario_admissibility("case-static").to_dict()
    validate_scenario_admissibility(payload)

    assert payload["schema_version"] == "scenario_admissibility.v1"
    assert payload["verdict"] == ADMISSIBLE_FEASIBILITY_UNKNOWN
    assert payload["target_planner_outcome"] == "not_evaluated"
    schema_path = Path("robot_sf/benchmark/schemas/scenario_admissibility.v1.json")
    json.loads(schema_path.read_text(encoding="utf-8"))
