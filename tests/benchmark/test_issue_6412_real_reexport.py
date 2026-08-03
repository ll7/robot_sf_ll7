"""Focused tests for the real #6412 88/2 package and resolver boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.candidate_trace_resolution import (
    ISSUE_5756_MAPPING_SCHEMA_VERSION,
    ISSUE_5756_PINNED_PROVENANCE,
    load_episode_mapping,
    load_episode_requests,
    resolve_episode_requests,
    validate_candidate_trace_resolution,
)
from robot_sf.benchmark.issue_6412_real_reexport import (
    FIGURE_QA_SCHEMA_VERSION,
    RealReexportPackageError,
    assemble_real_reexport_package,
    export_compact_evidence,
    finalize_real_reexport_package,
    materialize_resolver_mapping,
    verify_compact_evidence,
    verify_complete_package,
)
from robot_sf.benchmark.trace_reexport_packaging import (
    REAL_REEXPORT_ARMS,
    REAL_REEXPORT_BINDING_SCHEMA,
    REAL_REEXPORT_EXCEPTION_SEEDS,
)
from scripts.analysis import render_worked_example_trace_figures_issue_5756 as render_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _trace(path: Path, *, scenario_id: str, planner: str, seed: int, episode_id: str) -> str:
    payload = json.loads(TRACE_FIXTURE.read_text(encoding="utf-8"))
    payload["source"].update(
        {
            "scenario_id": scenario_id,
            "planner_id": planner,
            "seed": seed,
            "episode_id": episode_id,
        }
    )
    _write_json(path, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_and_mapping_for_explicit_exclusion(tmp_path: Path) -> tuple[Path, Path]:
    accepted = {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": 111,
        "episode_id": "fixture_episode_001",
    }
    excluded = {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": 112,
        "episode_id": "fixture_episode_002",
    }
    request_path = _write_json(
        tmp_path / "requests.json",
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "n_tuples": 2,
            "tuples": [accepted, excluded],
        },
    )
    accepted_trace = tmp_path / "accepted.json"
    excluded_trace = tmp_path / "excluded.json"
    accepted_sha = _trace(
        accepted_trace,
        scenario_id=accepted["scenario_id"],
        planner=accepted["planner"],
        seed=accepted["seed"],
        episode_id=accepted["episode_id"],
    )
    excluded_sha = _trace(
        excluded_trace,
        scenario_id=excluded["scenario_id"],
        planner=excluded["planner"],
        seed=excluded["seed"],
        episode_id=excluded["episode_id"],
    )
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    mapping_path = _write_json(
        tmp_path / "mapping.json",
        {
            "schema_version": ISSUE_5756_MAPPING_SCHEMA_VERSION,
            "n_rows": 2,
            "provenance": {
                **ISSUE_5756_PINNED_PROVENANCE,
                "request_manifest_sha256": request_sha,
            },
            "rows": [
                {
                    **accepted,
                    "planner": accepted["planner"],
                    "release_episode_id": accepted["episode_id"],
                    "expected_release_outcome": "success",
                    "rerun_outcome": "success",
                    "admission_status": "admitted",
                    "trace_artifact_uri": str(accepted_trace),
                    "trace_sha256": accepted_sha,
                },
                {
                    **excluded,
                    "planner": excluded["planner"],
                    "release_episode_id": excluded["episode_id"],
                    "expected_release_outcome": "success",
                    "rerun_outcome": "collision_event",
                    "admission_status": "not_admitted",
                    "exclusion_reason": "outcome_mismatch",
                    "release_outcome": "success",
                    "trace_artifact_uri": str(excluded_trace),
                    "trace_sha256": excluded_sha,
                },
            ],
        },
    )
    return request_path, mapping_path


def test_explicit_outcome_exclusion_is_resolver_visible(tmp_path: Path) -> None:
    """A named mismatch remains visible while the accepted row resolves normally."""
    request_path, mapping_path = _request_and_mapping_for_explicit_exclusion(tmp_path)
    request_manifest = load_episode_requests(
        request_path,
        expected_count=2,
        expected_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )
    mapping = load_episode_mapping(
        mapping_path,
        expected_count=2,
        expected_provenance={
            **ISSUE_5756_PINNED_PROVENANCE,
            "request_manifest_sha256": request_manifest.content_sha256,
        },
    )
    resolution = resolve_episode_requests(request_manifest, mapping)
    assert resolution["summary"] == {
        "n_candidates": 2,
        "n_resolved": 1,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 1,
    }
    excluded = next(row for row in resolution["rows"] if row["seed"] == 112)
    assert excluded["admission_status"] == "not_admitted"
    assert excluded["exclusion_reason"] == "outcome_mismatch"
    assert excluded["release_outcome"] == "success"
    assert excluded["rerun_outcome"] == "collision_event"
    assert validate_candidate_trace_resolution(resolution)["ok"]


def _synthetic_real_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    request_rows: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    arm_receipts: list[dict[str, Any]] = []
    row_index = 0
    for arm in REAL_REEXPORT_ARMS:
        manifest_sha = hashlib.sha256(f"manifest:{arm.key}".encode()).hexdigest()
        episodes_sha = hashlib.sha256(f"episodes:{arm.key}".encode()).hexdigest()
        run_summary_sha = hashlib.sha256(f"summary:{arm.key}".encode()).hexdigest()
        preflight_sha = hashlib.sha256(f"preflight:{arm.key}".encode()).hexdigest()
        arm_receipts.append(
            {
                "arm": arm.key,
                "job_id": arm.job_id,
                "planner": arm.planner,
                "scenario_id": arm.scenario_id,
                "manifest_path": f"external/{arm.key}/campaign_manifest.json",
                "manifest_sha256": manifest_sha,
                "episodes_path": f"external/{arm.key}/episodes.jsonl",
                "episodes_sha256": episodes_sha,
                "n_rows": 30,
            }
        )
        for seed in arm.seeds:
            key = (arm.planner, arm.scenario_id, seed)
            release_episode_id = f"release-{arm.key}-{seed}"
            rerun_episode_id = f"rerun-{arm.key}-{seed}"
            excluded = key in {
                ("ppo", "classic_doorway_medium", seed_value)
                for seed_value in REAL_REEXPORT_EXCEPTION_SEEDS
            }
            release_outcome = "route_complete" if excluded else "success"
            rerun_outcome = "collision_event" if excluded else release_outcome
            request_rows.append(
                {
                    "scenario_id": arm.scenario_id,
                    "planner": arm.planner,
                    "seed": seed,
                    "episode_id": release_episode_id,
                }
            )
            expected_rows.append(
                {
                    "scenario_id": arm.scenario_id,
                    "planner": arm.planner,
                    "seed": seed,
                    "episode_id": release_episode_id,
                    "outcome": {
                        "success": release_outcome == "success",
                        "route_complete": release_outcome == "route_complete",
                        "collision_event": release_outcome == "collision_event",
                        "timeout_event": release_outcome == "timeout_event",
                    },
                }
            )
            trace_path = tmp_path / "normalized" / arm.key / f"seed-{seed}.json"
            trace_sha = _trace(
                trace_path,
                scenario_id=arm.scenario_id,
                planner=arm.planner,
                seed=seed,
                episode_id=rerun_episode_id,
            )
            binding_rows.append(
                {
                    "admission_status": "not_admitted" if excluded else "admitted",
                    "algorithm_config_hash": "a" * 16,
                    "arm": arm.key,
                    "campaign": f"{arm.key}_a307",
                    "config": {
                        "evidence_paths": [
                            {"kind": "campaign_manifest", "sha256": manifest_sha},
                            {"kind": "run_summary.yaml", "sha256": run_summary_sha},
                            {"kind": "validate_config.json", "sha256": preflight_sha},
                        ]
                    },
                    "execution_commit": "a307ef276d701f8d14dead1aa0513f44ee97c0b0",
                    "job_id": arm.job_id,
                    "normalization_policy": "simulation_trace_export.allowlisted_metadata.v1",
                    "normalized_trace_path": str(trace_path),
                    "normalized_trace_sha256": trace_sha,
                    "outcome_status": "outcome_mismatch" if excluded else "outcome_match",
                    "planner": arm.planner,
                    "raw_trace_sha256": trace_sha,
                    "release_outcome": release_outcome,
                    "removed_field_count": 0,
                    "removed_field_counts": {},
                    "removed_field_paths_sha256": "b" * 64,
                    "rerun_outcome": rerun_outcome,
                    "row_config_hash": "c" * 16,
                    "scenario_id": arm.scenario_id,
                    "schema_version": "issue_6411_trace_transformation_receipt.v1",
                    "seed": seed,
                    "semantic_payload_unchanged": True,
                    "source": {
                        "episode_id": rerun_episode_id,
                        "episodes_sha256": episodes_sha,
                        "row_index": row_index,
                    },
                    "status": "complete",
                    "trace_schema_version": "simulation_trace_export.v1",
                }
            )
            row_index += 1
    request_path = _write_json(
        tmp_path / "request.json",
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "n_tuples": 90,
            "tuples": request_rows,
        },
    )
    expected_path = _write_json(
        tmp_path / "expected.json",
        {"schema_version": "synthetic_expected.v1", "rows": expected_rows},
    )
    binding_path = _write_json(
        tmp_path / "binding.json",
        {
            "schema_version": REAL_REEXPORT_BINDING_SCHEMA,
            "status": "complete",
            "execution_commit": "a307ef276d701f8d14dead1aa0513f44ee97c0b0",
            "trace_schema_version": "simulation_trace_export.v1",
            "normalization_policy": "simulation_trace_export.allowlisted_metadata.v1",
            "request_contract": {
                "schema_version": "issue_5446_trace_reexport_list.v1",
                "n_tuples": 90,
                "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            },
            "arms": arm_receipts,
            "rows": binding_rows,
            "summary": {"n_rows": 90, "n_admitted": 88, "n_not_admitted": 2},
            "exception_boundary": [],
            "package_status": "not_created; package assembly belongs to issue #6412",
        },
    )
    return binding_path, request_path, expected_path


def test_real_package_materializes_resolves_and_finalizes_88_2(tmp_path: Path) -> None:
    """The bounded assembler proves package, resolver, and checksum invariants."""
    binding, request, expected = _synthetic_real_inputs(tmp_path)
    package = tmp_path / "package"
    manifest = assemble_real_reexport_package(
        binding_receipt=binding,
        request_manifest=request,
        expected_outcomes=expected,
        output_dir=package,
    )
    assert manifest["status"] == "assembled"
    assert len(list((package / "traces").rglob("*.json"))) == 88
    assert len(list((package / "excluded_traces").rglob("*.json"))) == 2
    assert len(list((package / "exclusions").glob("*.json"))) == 2

    resolver_path = tmp_path / "resolver.json"
    materialized = materialize_resolver_mapping(package, resolver_path)
    request_manifest = load_episode_requests(
        request,
        expected_count=90,
        expected_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
    )
    mapping = load_episode_mapping(
        resolver_path,
        expected_count=90,
        expected_provenance={
            **ISSUE_5756_PINNED_PROVENANCE,
            "request_manifest_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
        },
    )
    resolution = resolve_episode_requests(request_manifest, mapping)
    assert resolution["summary"] == {
        "n_candidates": 90,
        "n_resolved": 88,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 2,
    }
    assert validate_candidate_trace_resolution(resolution)["ok"]
    render_cli._require_complete_resolution(resolution)

    figure_qa = {
        "schema_version": FIGURE_QA_SCHEMA_VERSION,
        "status": "passed",
        "visualization_only": True,
        "n_figures": 2,
        "n_error_defects": 0,
        "figures": [
            {
                "figure": "doorway_ppo_seed113_vs_114.pdf",
                "status": "passed",
                "n_error_defects": 0,
            },
            {
                "figure": "double_bottleneck_goal_vs_ppo_seed118.pdf",
                "status": "passed",
                "n_error_defects": 0,
            },
        ],
    }
    for name in (
        "doorway_ppo_seed113_vs_114.pdf",
        "double_bottleneck_goal_vs_ppo_seed118.pdf",
    ):
        _write_json(package / "figures" / name, {"synthetic": True})
        _write_json(package / "figures" / name.replace(".pdf", ".png"), {"synthetic": True})
    finalize_real_reexport_package(package, resolution=resolution, figure_qa=figure_qa)
    assert verify_complete_package(package)["status"] == "complete"
    assert len(materialized["rows"]) == 90
    compact = tmp_path / "compact-evidence"
    export_manifest = export_compact_evidence(package, compact)
    assert export_manifest["status"] == "complete_compact_export"
    assert verify_compact_evidence(compact)["n_admitted"] == 88
    assert not list(compact.rglob("*trace*.json"))


def test_package_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    """Assembly is atomic and never overwrites a prior package."""
    binding, request, expected = _synthetic_real_inputs(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    try:
        assemble_real_reexport_package(
            binding_receipt=binding,
            request_manifest=request,
            expected_outcomes=expected,
            output_dir=package,
        )
    except RealReexportPackageError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("existing package was overwritten")
