"""Contract tests for provenance-first case discovery."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.analysis_trace import build_analysis_trace, canonical_json, trace_coverage
from robot_sf.benchmark.case_publication_figure import render_publication_figure
from robot_sf.benchmark.case_workbench import analyze_cases, apply_admission_overlay
from robot_sf.benchmark.parquet_export import (
    derive_episode_metrics,
    export_campaign_result_store_v2,
)


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
        config_hash=f"config-{seed}",
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
        "provenance": {"artifact_uri": f"trace-{seed}.json", "artifact_sha256": f"sha-{seed}"},
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
    assert any(item["case_id"] == "doorway--115" for item in proposal["excluded"])
    assert (output / "admission_overlay.json").is_file()
    assert (output / "viewer_blueprint.json").is_file()
    assert (output / "audit_dossier.json").is_file()
    assert (output / "audit_dossier.md").is_file()
    assert (output / "publication" / "figure.pdf").is_file()
    sidecar = json.loads((output / "publication" / "figure.pdf.json").read_text(encoding="utf-8"))
    assert sidecar["panels"]["world"]["map_geometry"] == "available"
    assert sidecar["shared_prefix"] is False
    repeat_figure = tmp_path / "repeat-figure.pdf"
    render_publication_figure(output, output=repeat_figure)
    assert (output / "publication" / "figure.pdf").read_bytes() == repeat_figure.read_bytes()
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
                    "provenance": {"artifact_sha256": "author-sha"},
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
