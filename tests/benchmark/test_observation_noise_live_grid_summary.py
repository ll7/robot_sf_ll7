"""Tests for the issue #3335 observation-noise live-grid summarizer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/benchmark/summarize_observation_noise_live_grid.py")
_LOADED_MOD = None


def _load_script():
    """Load the summarizer script as a module."""

    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "summarize_observation_noise_live_grid", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["summarize_observation_noise_live_grid"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return _LOADED_MOD


def _summary(
    *,
    scenario: str,
    label: str,
    near_field: bool,
    command_changed: bool,
    closest_delta: float = 0.0,
) -> dict:
    """Build a compact native live summary fixture."""

    return {
        "schema_version": "issue_3201_observation_noise_live_smoke.v1",
        "issue": 3201,
        "inputs": {
            "clean_trace_json": "worktree-local ignored artifact summarized in this report/a.json",
            "perturbed_trace_json": "worktree-local ignored artifact summarized in this report/b.json",
        },
        "scenario": {"clean": scenario, "perturbed": scenario, "same_scenario": True},
        "seed": {"clean": 111, "perturbed": 111, "same_seed": True},
        "near_field_target": {
            "threshold_m": 2.0,
            "clean_closest_robot_ped_distance_m": 1.0 if near_field else 7.0,
            "satisfied": near_field,
        },
        "command_summary": {"sequence_changed": command_changed},
        "observation_summary": {"changed": True},
        "progress_delta": {
            "closest_robot_ped_distance": {
                "clean": 1.0,
                "perturbed": 1.0 + closest_delta,
                "delta": closest_delta,
            },
            "collision_flag_counts": {
                "clean": {"pedestrian": 0, "obstacle": 0, "robot": 0},
                "perturbed": {"pedestrian": 1 if command_changed else 0, "obstacle": 0, "robot": 0},
            },
        },
        "classification": {"label": label, "rationale": "fixture"},
    }


def _write(path: Path, payload: dict) -> Path:
    """Write one JSON payload and return its path."""

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _issue_2777_grid_summary(*, status: str, grid_label: str, closest: float) -> dict:
    """Build a compact issue #2777 grid summary fixture."""

    behavior_sensitive = grid_label == "medium_amplitude_sensitive"
    return {
        "schema_version": "issue_2777_observation_noise_live_replay.v1",
        "artifact_shape": "compact_summary_without_raw_traces",
        "issue": 2777,
        "status": status,
        "classification": {
            "label": "behavior_sensitive_diagnostic_only" if behavior_sensitive else "blocked",
            "rationale": "grid fixture",
        },
        "fixture_contract": {
            "satisfied": True,
            "matched_scenario": {
                "name": "issue_2756_occluded_emergence",
                "seeds": [111],
            },
            "scenario_matrix": "configs/scenarios/sets/example.yaml",
        },
        "run_config": {
            "scenario_matrix": "configs/scenarios/sets/example.yaml",
            "condition_set": "issue_3330_seed_amplitude_grid",
        },
        "grid_interpretation": {
            "label": grid_label,
            "summary": "grid summary",
        },
        "conditions": [
            {
                "name": "noop",
                "progress_summary": {
                    "closest_robot_ped_distance": closest,
                    "collision_flag_counts": {"pedestrian": 0},
                },
            },
            {
                "name": "medium_noise_3328",
                "behavior_change_summary": {
                    "command_sequence_changed": behavior_sensitive,
                    "progress_or_risk_changed": behavior_sensitive,
                },
                "progress_delta": {
                    "closest_robot_ped_distance": {
                        "noop": closest,
                        "condition": closest - 0.2,
                    },
                    "collision_flag_counts": {
                        "noop": {"pedestrian": 0},
                        "condition": {"pedestrian": 0},
                    },
                },
            },
        ],
    }


def test_issue_2777_grid_sources_classify_failed_second_candidate(tmp_path: Path) -> None:
    """#3335 should preserve a failed-closed second matrix as negative evidence."""

    sensitive = _write(
        tmp_path / "sensitive.json",
        _issue_2777_grid_summary(
            status="live_replay",
            grid_label="medium_amplitude_sensitive",
            closest=1.6,
        ),
    )
    failed = _write(
        tmp_path / "failed.json",
        _issue_2777_grid_summary(
            status="fail_closed",
            grid_label="unavailable_fail_closed",
            closest=21.8,
        ),
    )

    report = _load_script().build_report([sensitive, failed], command="test")

    assert report["usable_native_live_source_count"] == 1
    assert report["failed_closed_source_count"] == 1
    assert (
        report["classification"]["label"] == "fixture_candidate_failed_closed_after_sensitive_grid"
    )
    assert report["sources"][0]["status"] == "behavior_sensitive_grid"
    assert report["sources"][1]["status"] == "failed_closed"


def test_grid_classifies_near_field_sensitivity_as_fixture_specific(tmp_path: Path) -> None:
    """A sensitive near-field row plus a too-weak row should stay conservative."""

    weak = _write(
        tmp_path / "weak.json",
        _summary(
            scenario="dense_pedestrian_stress",
            label="observation_only_scenario_too_weak",
            near_field=False,
            command_changed=False,
        ),
    )
    sensitive = _write(
        tmp_path / "near_field.json",
        _summary(
            scenario="issue_3233_near_field_observation_noise",
            label="non_null_behavior_delta",
            near_field=True,
            command_changed=True,
            closest_delta=-0.2,
        ),
    )

    report = _load_script().build_report([weak, sensitive], command="test")

    assert report["usable_native_live_source_count"] == 2
    assert report["failed_closed_source_count"] == 0
    assert report["classification"]["label"] == "fixture_specific_near_field_sensitivity"
    assert report["sources"][1]["status"] == "behavior_sensitive"


def test_grid_fails_closed_for_trace_derived_summary(tmp_path: Path) -> None:
    """Trace-derived envelopes should not be accepted as native live replay evidence."""

    trace_derived = _write(
        tmp_path / "trace_derived.json",
        {
            "schema_version": "observation_noise_envelope.v1",
            "classification": {"label": "diagnostic_only"},
            "conditions": [],
        },
    )

    report = _load_script().build_report([trace_derived], command="test")

    assert report["usable_native_live_source_count"] == 0
    assert report["failed_closed_source_count"] == 1
    assert report["classification"]["label"] == "failed_closed_no_native_live_evidence"
    assert report["sources"][0]["status"] == "failed_closed"


def test_cli_writes_summary_and_markdown(tmp_path: Path) -> None:
    """The CLI writes compact durable artifacts."""

    source = _write(
        tmp_path / "source.json",
        _summary(
            scenario="issue_3233_near_field_observation_noise",
            label="non_null_behavior_delta",
            near_field=True,
            command_changed=True,
        ),
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "README.md"

    assert (
        _load_script().main(
            [
                "--source-summary",
                str(source),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_version"].endswith(".v1")
    assert "Issue #3335" in output_md.read_text(encoding="utf-8")
