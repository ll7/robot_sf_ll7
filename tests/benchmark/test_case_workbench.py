"""Contract tests for provenance-first case discovery."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.analysis_trace import (
    build_analysis_trace,
    canonical_json,
    trace_artifact_sha256,
    trace_coverage,
)
from robot_sf.benchmark.case_publication_figure import render_publication_figure
from robot_sf.benchmark.case_workbench import (
    _candidate,
    admit_package,
    analyze_cases,
    apply_admission_overlay,
)
from robot_sf.benchmark.parquet_export import (
    derive_episode_metrics,
    export_campaign_result_store_v2,
    is_comparison_compatible,
)
from scripts.tools.trace_viewer import TraceViewerError, load_episode_bundles, prepared_package_dirs


def _trace(seed: int, *, collision: bool = False) -> dict:
    """Build a tiny deterministic analysis trace fixture."""

    return build_analysis_trace(
        steps=[
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.2, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"id": 0, "position": [1.0, 0.0], "velocity": [0.0, 0.0]}],
                "planner": {
                    "amv": {
                        "requested_linear_m_s": 1.0,
                        "requested_angular_rad_s": 0.1,
                        "applied_linear_m_s": 0.8,
                        "applied_angular_rad_s": 0.05,
                    }
                },
            }
        ],
        initial_robot_position=[0.0, 0.0],
        initial_robot_heading=0.0,
        initial_pedestrians=[[1.0, 0.0]],
        initial_robot_velocity=[0.0, 0.0],
        initial_pedestrian_velocities=[[0.0, 0.0]],
        initial_pedestrian_ids=[0],
        dt=0.1,
        horizon=1,
        robot_radius_m=0.25,
        pedestrian_radius_m=0.25,
        scenario={
            "id": "classic_doorway_medium",
            "seed": seed,
            "map_file": "maps/svg_maps/classic_doorway.svg",
        },
        planner="ppo",
        planner_commit="commit",
        config_hash="config-doorway",
        git_hash="commit",
        termination_reason="collision" if collision else "max_steps",
        safety_events=[{"event_type": "collision", "time_s": 0.1}] if collision else [],
    )


def _record(seed: int, *, collision: bool = False) -> dict:
    """Build an episode row with explicit artifact provenance."""

    trace = _trace(seed, collision=collision)
    return {
        "episode_id": f"doorway--{seed}",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "algo": "ppo",
        "status": "collision" if collision else "success",
        "row_status": "native",
        "outcome": {"collision": collision, "success": not collision},
        "provenance": {
            "artifact_uri": f"trace-{seed}.json",
            "artifact_sha256": trace["artifact_sha256"],
        },
        "metrics": {
            "surface_clearance_min": -0.1 if collision else 0.2,
            "progress": 1.0,
            "control_effort": 0.2,
        },
        "algorithm_metadata": {"analysis_trace": trace},
    }


def test_analysis_trace_has_explicit_initial_state_and_coverage() -> None:
    """The opt-in envelope carries stable actors, radii, and provenance."""

    record = _record(113)
    record["algorithm_metadata"]["telemetry"] = {
        "analysis_trace": "all",
        "planner_debug_trace": "none",
    }
    assert trace_coverage(record)["status"] == "complete"
    assert trace_coverage(record)["map_digest"] is True
    steps = record["algorithm_metadata"]["analysis_trace"]["steps"]
    assert steps[0]["time_s"] == 0.0
    assert steps[0]["robot"]["actor_id"] == "robot"
    assert steps[0]["pedestrians"][0]["radius_m"] == 0.25


def test_trace_coverage_fails_closed_for_missing_reset_state_and_map() -> None:
    """Absent reset velocities or map provenance cannot be promoted to complete."""

    trace = _trace(113)
    trace["steps"][0]["robot"]["velocity"] = None
    trace["steps"][0]["pedestrians"][0]["velocity"] = None
    trace["map_digest"] = None
    record = {"algorithm_metadata": {"analysis_trace": trace}}
    coverage = trace_coverage(record)
    assert coverage["status"] == "unavailable"
    assert "finite_states" in coverage["reasons"]
    assert "provenance" in coverage["reasons"]


def test_trace_coverage_rejects_nonmonotonic_time_and_radius_reuse() -> None:
    """Stable actor geometry and absolute time are required for promotion."""

    trace = _trace(113)
    trace["steps"][1]["time_s"] = 0.0
    trace["steps"][1]["pedestrians"][0]["radius_m"] = 0.5
    trace["artifact_sha256"] = trace_artifact_sha256(trace)
    coverage = trace_coverage({"algorithm_metadata": {"analysis_trace": trace}})
    assert coverage["status"] == "unavailable"
    assert "monotonic_time" in coverage["reasons"]
    assert "radii" in coverage["reasons"]


def test_positional_actor_slots_are_not_promoted_as_stable_ids() -> None:
    """Legacy index-only actor labels remain unavailable for evidence use."""

    trace = build_analysis_trace(
        steps=[
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.1, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"id": 0, "position": [1.0, 0.0], "velocity": [0.0, 0.0]}],
                "controls": {
                    "requested": {"linear_m_s": 1.0, "turn_rate_rad_s": 0.0},
                    "applied": {"linear_m_s": 1.0, "turn_rate_rad_s": 0.0},
                },
            }
        ],
        initial_robot_position=[0.0, 0.0],
        initial_robot_heading=0.0,
        initial_pedestrians=[[1.0, 0.0]],
        initial_robot_velocity=[0.0, 0.0],
        initial_pedestrian_velocities=[[0.0, 0.0]],
        dt=0.1,
        horizon=1,
        robot_radius_m=0.25,
        pedestrian_radius_m=0.25,
        scenario={"id": "classic_doorway_medium", "map_digest": "map"},
        planner="ppo",
        planner_commit="planner",
        config_hash="config",
        git_hash="git",
        termination_reason="success",
        safety_events=[],
    )
    coverage = trace_coverage({"algorithm_metadata": {"analysis_trace": trace}})
    assert coverage["status"] == "unavailable"
    assert "stable_actor_ids" in coverage["reasons"]


def test_canonical_outcome_fields_drive_collision_selection() -> None:
    """Release outcome keys are not silently downgraded to legacy aliases."""

    record = _record(113)
    record["outcome"] = {"route_complete": False, "collision_event": True}
    candidate = _candidate(record)
    assert candidate["outcome"]["collision"] is True
    assert candidate["outcome"]["success"] is False


def test_runtime_collision_event_is_canonicalized_at_trace_boundary() -> None:
    """Legacy collision_time and index partners become joinable typed events."""

    trace = build_analysis_trace(
        steps=[],
        initial_robot_position=[0.0, 0.0],
        initial_robot_heading=0.0,
        initial_pedestrians=[[1.0, 0.0]],
        initial_robot_velocity=[0.0, 0.0],
        initial_pedestrian_velocities=[[0.0, 0.0]],
        initial_pedestrian_ids=[0],
        dt=0.1,
        horizon=1,
        robot_radius_m=0.25,
        pedestrian_radius_m=0.25,
        scenario={"id": "classic_doorway_medium", "map_file": "maps/svg_maps/classic_doorway.svg"},
        planner="ppo",
        planner_commit="commit",
        config_hash="config",
        git_hash="commit",
        termination_reason="collision",
        safety_events=[{"collision_time": 0.0, "collision_partner_id": 0}],
    )
    assert trace["events"][0]["event_type"] == "collision"
    assert trace["events"][0]["time_s"] == 0.0
    assert trace["events"][0]["partner_id"] == "pedestrian-0"


def test_case_workbench_is_deterministic_and_excludes_missing_provenance(tmp_path: Path) -> None:
    """Selection is stable and a broken provenance row cannot enter the portfolio."""

    source = tmp_path / "episodes.jsonl"
    rows = [_record(113), _record(114, collision=True), _record(115)]
    rows[-1]["provenance"] = {"artifact_uri": "missing-hash.json"}
    source.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    output = tmp_path / "package"
    proposal = analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=source,
        output=output,
        check_determinism=True,
    )
    assert proposal["schema_version"] == "case-workbench.v1"
    assert proposal["eligible_count"] == 2
    assert any(case["primary_role"] == "seed_sensitivity" for case in proposal["portfolio"])
    seed_cases = [
        case for case in proposal["portfolio"] if case["primary_role"] == "seed_sensitivity"
    ]
    assert {case["case_id"] for case in seed_cases} == {"doorway--113", "doorway--114"}
    assert all(
        case["comparison_pair_ids"] == ["doorway--113", "doorway--114"] for case in seed_cases
    )
    relation_roles = {
        "safety_boundary",
        "metric_disagreement",
        "cross_cell_inversion",
        "representative_control",
    }
    assert not any(case["primary_role"] in relation_roles for case in proposal["portfolio"])
    unavailable = {item["role"]: item["reason"] for item in proposal["unavailable_roles"]}
    assert relation_roles <= unavailable.keys()
    assert all(
        unavailable[role] == "required_relation_metric_unavailable" for role in relation_roles
    )
    assert any(item["case_id"] == "doorway--115" for item in proposal["excluded"])
    assert (output / "admission_overlay.json").is_file()
    assert (output / "viewer_blueprint.json").is_file()
    assert (output / "audit_dossier.json").is_file()
    assert (output / "audit_dossier.md").is_file()
    assert not (output / "publication" / "figure.preview.pdf").exists()
    unavailable = json.loads((output / "publication" / "UNAVAILABLE.json").read_text())
    assert unavailable["reason"] == "source_gate_receipt_missing"
    assert (output / "campaign-result-store.v2").is_dir()


def test_campaign_result_store_v2_emits_all_tables_and_unavailable_adapter(tmp_path: Path) -> None:
    """The normalized store keeps complete traces and typed v1 gaps."""

    source = tmp_path / "episodes.jsonl"
    complete = _record(113)
    legacy = _record(114)
    legacy["algorithm_metadata"] = {"simulation_step_trace": {"steps": []}}
    legacy["provenance"] = {"artifact_uri": "legacy.jsonl"}
    source.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (complete, legacy)) + "\n",
        encoding="utf-8",
    )
    result = export_campaign_result_store_v2(source, tmp_path / "store", overwrite=True)
    assert result.record_count == 2
    assert set(result.table_paths) == {
        "episodes",
        "steps",
        "actors",
        "events",
        "features",
        "cells",
        "comparisons",
    }
    assert (result.output_dir / "SHA256SUMS").is_file()
    import pyarrow.parquet as pq

    cells = pq.read_table(result.table_paths["cells"]).to_pylist()
    assert cells[0]["representative_status"] == "medoid"
    assert cells[0]["boundary_status"] in {"not_observed", "mixed_outcomes"}
    legacy_features = pq.read_table(result.table_paths["features"]).to_pylist()
    legacy_names = [
        row["feature_name"] for row in legacy_features if row["episode_id"] == "doorway--114"
    ]
    assert len(legacy_names) == len(set(legacy_names))
    proposal = analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=result.output_dir,
        output=tmp_path / "package-from-v2",
        check_determinism=True,
    )
    assert proposal["candidate_count"] == 2
    assert proposal["eligible_count"] == 1
    assert (
        tmp_path / "package-from-v2" / "campaign-result-store.v2" / "manifest.json"
    ).read_bytes() == (result.output_dir / "manifest.json").read_bytes()


def test_v2_read_adapter_rechecks_coverage_when_state_table_is_missing(tmp_path: Path) -> None:
    """A stale complete receipt cannot make a store with missing states eligible."""

    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(_record(113)) + "\n", encoding="utf-8")
    result = export_campaign_result_store_v2(source, tmp_path / "store", overwrite=True)
    (result.output_dir / "steps.parquet").unlink()

    proposal = analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=result.output_dir,
        output=tmp_path / "package-missing-steps",
        check_determinism=True,
    )
    assert proposal["eligible_count"] == 0
    assert proposal["excluded"][0]["reasons"] == ["trace_coverage:analysis_trace_fields_incomplete"]


def test_admission_overlay_is_digest_bound_and_preserves_machine_portfolio() -> None:
    """Author decisions alter admission status without erasing machine rationale."""

    proposal = {
        "schema_version": "case-workbench.v1",
        "portfolio": [
            {
                "case_id": "machine-1",
                "primary_role": "seed_sensitivity",
                "author_status": "proposed",
            }
        ],
    }
    overlay = {
        "schema_version": "case-admission-overlay.v1",
        "proposal_sha256": hashlib.sha256(canonical_json(proposal).encode("utf-8")).hexdigest(),
        "status": "admitted",
        "decisions": [{"case_id": "machine-1", "decision": "approve", "rationale": "reviewed"}],
    }
    admitted = apply_admission_overlay(proposal, overlay)
    assert admitted["portfolio"][0]["author_status"] == "approved"
    assert admitted["machine_portfolio"][0]["author_status"] == "proposed"
    assert admitted["author_admission"]["status"] == "admitted"

    replacement_trace = _trace(115)
    replacement_overlay = {
        "schema_version": "case-admission-overlay.v1",
        "proposal_sha256": overlay["proposal_sha256"],
        "status": "overridden",
        "decisions": [
            {
                "case_id": "machine-1",
                "decision": "replace",
                "rationale": "manual review selected a better trace",
                "replacement": {
                    "case_id": "author-1",
                    "scenario_id": "classic_doorway_medium",
                    "planner": "ppo",
                    "seed": 115,
                    "provenance": {"artifact_sha256": replacement_trace["artifact_sha256"]},
                    "trace": replacement_trace,
                },
            }
        ],
    }
    replaced = apply_admission_overlay(proposal, replacement_overlay)
    assert [case["case_id"] for case in replaced["portfolio"]] == ["author-1"]
    assert replaced["portfolio"][0]["machine_recommendation"] == "machine-1"

    stale = dict(overlay, proposal_sha256="0" * 64)
    try:
        apply_admission_overlay(proposal, stale)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("stale admission overlay was accepted")


def test_admitted_overlay_must_decide_every_machine_case() -> None:
    """An admitted package cannot silently leave a proposed case unresolved."""

    proposal = {
        "schema_version": "case-workbench.v1",
        "portfolio": [{"case_id": "machine-1"}, {"case_id": "machine-2"}],
    }
    overlay = {
        "schema_version": "case-admission-overlay.v1",
        "proposal_sha256": hashlib.sha256(canonical_json(proposal).encode("utf-8")).hexdigest(),
        "status": "admitted",
        "decisions": [{"case_id": "machine-1", "decision": "approve", "rationale": "reviewed"}],
    }
    with pytest.raises(ValueError, match="decide every proposed case"):
        apply_admission_overlay(proposal, overlay)


def test_campaign_result_store_v2_pair_receipt_derives_only_compatible_deltas(
    tmp_path: Path,
) -> None:
    """Matched planner traces produce absolute-time deltas without DTW alignment."""

    source = tmp_path / "paired.jsonl"
    left = _record(113)
    right = _record(113, collision=True)
    right["episode_id"] = "doorway--113--goal"
    right["algo"] = "goal"
    source.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (left, right)) + "\n",
        encoding="utf-8",
    )
    result = export_campaign_result_store_v2(source, tmp_path / "store", overwrite=True)
    import pyarrow.parquet as pq

    comparisons = pq.read_table(result.table_paths["comparisons"]).to_pylist()
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["compatibility_status"] == "compatible"
    assert comparison["shared_prefix"] is False
    assert comparison["trajectory_separation_m"] == 0.0
    assert comparison["control_sequence_difference"] == 0.0


def test_package_viewer_and_svg_preview_contracts(tmp_path: Path) -> None:
    """The package adapter satisfies the viewer schema and SVG uses valid metadata."""

    source = tmp_path / "episodes.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in (_record(113), _record(114))) + "\n",
        encoding="utf-8",
    )
    package = tmp_path / "package"
    proposal = analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=source,
        output=package,
    )
    case_id = proposal["portfolio"][0]["case_id"]
    with prepared_package_dirs(package, case_id=case_id) as bundle_dirs:
        assert len(load_episode_bundles(bundle_dirs)) == 2
    svg = tmp_path / "figure.svg"
    with pytest.raises(ValueError, match="source-integrity"):
        render_publication_figure(package, case_id=case_id, output=svg, output_format="svg")
    render_publication_figure(
        package,
        case_id=case_id,
        output=svg,
        output_format="svg",
        _allow_unverified_preview=True,
    )
    assert svg.is_file()


def test_source_gate_receipt_controls_preview_and_binds_digest(tmp_path: Path) -> None:
    """A passed source receipt enables only a diagnostic preview with a bound digest."""

    source = tmp_path / "episodes.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in (_record(113), _record(114))) + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "source-gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "case-source-integrity-gate.v1",
                "status": "passed",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package = tmp_path / "package"
    analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=source,
        output=package,
        source_gate_receipt=receipt,
    )
    gate = json.loads((package / "manifest.json").read_text(encoding="utf-8"))[
        "source_integrity_gate"
    ]
    assert gate["status"] == "passed"
    sidecar = json.loads(
        (package / "publication" / "figure.preview.pdf.json").read_text(encoding="utf-8")
    )
    assert sidecar["proposal_sha256"]
    assert sidecar["config_sha256"]
    assert sidecar["store_sha256"]


def test_candidate_rejects_tampered_trace_and_nested_fallback() -> None:
    """Copied top-level hashes and degraded nested execution never become eligible."""

    tampered = _record(113)
    tampered["algorithm_metadata"]["analysis_trace"]["artifact_sha256"] = "0" * 64
    assert not _candidate(tampered)["eligible"]
    fallback = _record(113)
    fallback["algorithm_metadata"]["execution_mode"] = "fallback"
    assert any(
        "execution_metadata:execution_mode=fallback" == reason
        for reason in _candidate(fallback)["exclusion_reasons"]
    )


def test_pair_receipt_rejects_map_digest_and_missing_timestep() -> None:
    """Physical deltas require exact map identity and a declared non-default dt."""

    left = _record(113)
    right = _record(114)
    right_trace = right["algorithm_metadata"]["analysis_trace"]
    right_trace["map_digest"] = "f" * 64
    right_trace["artifact_sha256"] = trace_artifact_sha256(right_trace)
    right["provenance"]["artifact_sha256"] = right_trace["artifact_sha256"]
    assert not is_comparison_compatible(left, right)

    missing_dt = _record(113)
    missing_trace = missing_dt["algorithm_metadata"]["analysis_trace"]
    missing_trace.pop("dt")
    missing_trace["artifact_sha256"] = trace_artifact_sha256(missing_trace)
    missing_dt["provenance"]["artifact_sha256"] = missing_trace["artifact_sha256"]
    assert "stall_duration" not in derive_episode_metrics(missing_dt)


def test_package_viewer_rejects_tampered_checksum(tmp_path: Path) -> None:
    """The interactive adapter refuses a modified package before loading traces."""

    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(_record(113)) + "\n", encoding="utf-8")
    package = tmp_path / "package"
    analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=source,
        output=package,
    )
    proposal_path = package / "proposal.json"
    proposal_path.write_text(proposal_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(TraceViewerError, match="checksum"):
        with prepared_package_dirs(package, case_id="doorway--113"):
            pass


def test_admit_package_refreshes_receipts_and_unlocks_publication(tmp_path: Path) -> None:
    """Only a passed source gate plus author decision unlocks final rendering."""

    source = tmp_path / "episodes.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in (_record(113), _record(114))) + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "source-gate.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "case-source-integrity-gate.v1",
                "status": "passed",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package = tmp_path / "package"
    proposal = analyze_cases(
        config_path="configs/analysis/case_workbench.v1.yaml",
        result_store=source,
        output=package,
        source_gate_receipt=receipt,
    )
    decisions = [
        {"case_id": case["case_id"], "decision": "approve", "rationale": "Reviewed."}
        for case in proposal["portfolio"]
    ]
    overlay = {
        "schema_version": "case-admission-overlay.v1",
        "proposal_sha256": hashlib.sha256(canonical_json(proposal).encode("utf-8")).hexdigest(),
        "status": "admitted",
        "decisions": decisions,
    }
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    admitted = admit_package(package, overlay_path)
    assert admitted["author_admission"]["status"] == "admitted"
    assert not (package / "publication" / "figure.preview.pdf").exists()
    output = tmp_path / "admitted.pdf"
    render_publication_figure(package, output=output)
    assert output.is_file()
    svg_output = tmp_path / "admitted.svg"
    render_publication_figure(package, output=svg_output, output_format="svg")
    assert svg_output.is_file()


def test_trace_metric_formulas_keep_units_and_timing_explicit() -> None:
    """Core case features derive from the trace without duration normalization."""

    trace = build_analysis_trace(
        steps=[
            {
                "step": 1,
                "time_s": 0.1,
                "robot": {
                    "position": [0.5, 0.0],
                    "heading": 0.0,
                    "velocity": [1.0, 0.0],
                },
                "pedestrians": [{"id": 0, "position": [1.5, 0.0], "velocity": [-1.0, 0.0]}],
                "planner": {
                    "amv": {
                        "applied_linear_m_s": 1.0,
                        "applied_angular_rad_s": 0.1,
                    }
                },
            },
            {
                "step": 2,
                "time_s": 0.2,
                "robot": {
                    "position": [0.7, 0.0],
                    "heading": 0.0,
                    "velocity": [0.5, 0.0],
                },
                "pedestrians": [{"id": 0, "position": [1.4, 0.0], "velocity": [-0.5, 0.0]}],
                "planner": {
                    "amv": {
                        "applied_linear_m_s": 0.5,
                        "applied_angular_rad_s": 0.2,
                    }
                },
            },
        ],
        initial_robot_position=[0.0, 0.0],
        initial_robot_heading=0.0,
        initial_pedestrians=[[2.0, 0.0]],
        initial_robot_velocity=[0.0, 0.0],
        initial_pedestrian_velocities=[[0.0, 0.0]],
        initial_pedestrian_ids=[0],
        dt=0.1,
        horizon=2,
        robot_radius_m=0.25,
        pedestrian_radius_m=0.25,
        scenario={"id": "classic_doorway_medium", "seed": 113},
        planner="ppo",
        planner_commit="commit",
        config_hash="config",
        git_hash="commit",
        termination_reason="near_miss",
        safety_events=[{"event_type": "near_miss", "time_s": 0.2}],
    )
    record = {
        "episode_id": "formula",
        "status": "near_miss",
        "outcome": {},
        "algorithm_metadata": {"analysis_trace": trace},
    }
    metrics = derive_episode_metrics(record)
    assert math.isclose(metrics["surface_clearance_min"], 0.2)
    assert math.isclose(metrics["ttc_min"], 0.2)
    assert math.isclose(metrics["cpa_min"], 0.0)
    assert math.isclose(metrics["braking_response_time"], 0.2)
    assert math.isclose(metrics["turning_response_time"], 0.1)
    assert math.isclose(metrics["progress"], 0.7)
    assert math.isclose(metrics["event_time"], 0.2)
    assert metrics["reversal_count"] == 0.0
