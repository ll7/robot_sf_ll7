"""Tests for the issue #3201 observation-noise live smoke comparator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    REPO_ROOT / "configs/scenarios/sets/issue_3201_pedestrian_dominated_observation_noise.yaml"
)
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/compare_observation_noise_live_smoke_issue_3201.py"
TRACE_FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "dense_pedestrian_stress_episode_0000.json"
)
_LOADED_MOD = None


def _load_script():
    """Load the comparator script as a module."""
    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "compare_observation_noise_live_smoke_issue_3201", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_observation_noise_live_smoke_issue_3201"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return _LOADED_MOD


def test_issue_3201_matrix_selects_near_field_fixture() -> None:
    """The #3201 scenario set should point at the deterministic near-field fixture."""
    payload = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "robot_sf.scenario_matrix.v1"
    assert "../single/issue_3233_near_field_observation_noise.yaml" in payload["includes"]
    assert payload["select_scenarios"] == ["issue_3233_near_field_observation_noise"]
    override = payload["scenario_overrides_by_name"]["issue_3233_near_field_observation_noise"]
    assert override["seeds"] == [3233]
    assert override["metadata"]["diagnostic_role"] == (
        "pedestrian_dominated_observation_noise_retest"
    )


def test_report_input_refs_do_not_depend_on_ignored_artifacts() -> None:
    """Cataloged summaries should not point readers at worktree-local raw artifacts."""
    ref = _load_script()._durable_input_ref(
        Path("output/benchmarks/issue_3201_observation_noise_live_smoke/clean_step/trace.json")
    )

    assert "output/" not in ref
    assert "worktree-local ignored artifact" in ref


def test_dense_trace_fixture_has_near_field_observed_pedestrians() -> None:
    """The reused dense trace has observed near-field pedestrians and active action changes."""
    trace = json.loads(TRACE_FIXTURE_PATH.read_text(encoding="utf-8"))
    min_distance = min(
        (
            (ped["position"][0] - frame["robot"]["position"][0]) ** 2
            + (ped["position"][1] - frame["robot"]["position"][1]) ** 2
        )
        ** 0.5
        for frame in trace["frames"]
        for ped in frame["pedestrians"]
    )
    events = [frame["planner"]["event"] for frame in trace["frames"]]
    velocities = [
        frame["planner"]["selected_action"]["linear_velocity"] for frame in trace["frames"]
    ]

    assert trace["source"]["scenario_id"] == "dense_pedestrian_stress"
    assert all(len(frame["observed_pedestrians"]) == 3 for frame in trace["frames"])
    assert min_distance < 0.5
    assert "dense_critical" in events
    assert min(velocities) < max(velocities)


def _write_jsonl(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_comparator_classifies_non_null_behavior_delta(tmp_path: Path) -> None:
    """Metric or status differences should classify as non-null behavior deltas."""
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    base = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "status": "success",
        "termination_reason": "success",
        "steps": 10,
        "horizon": 20,
        "outcome": {"route_complete": True},
        "metrics": {"success": 1.0, "min_distance": 0.4, "path_length": 2.0},
        "observation_noise": {"enabled": False, "profile": "none"},
        "observation_noise_stats": {"steps_with_noise": 0, "pedestrians_removed": 0},
    }
    noisy = {
        **base,
        "metrics": {"success": 0.0, "min_distance": 0.2, "path_length": 1.5},
        "observation_noise": {"enabled": True, "profile": "robustness_smoke_v1"},
        "observation_noise_stats": {"steps_with_noise": 5, "pedestrians_removed": 1},
    }
    _write_jsonl(clean, base)
    _write_jsonl(perturbed, noisy)

    report = _load_script().build_report(clean, perturbed)

    assert report["classification"]["label"] == "non_null_behavior_delta"
    assert report["metric_delta"]["success"]["delta"] == -1.0
    assert report["observation_noise"]["stats_delta"]["steps_with_noise"]["delta"] == 5


def test_comparator_classifies_observation_only_delta(tmp_path: Path) -> None:
    """Noise-counter deltas without behavior deltas should be explicit."""
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    base = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "status": "success",
        "termination_reason": "success",
        "steps": 10,
        "horizon": 20,
        "outcome": {"route_complete": True},
        "metrics": {"success": 1.0, "min_distance": 0.4, "path_length": 2.0},
        "observation_noise": {"enabled": False, "profile": "none"},
        "observation_noise_stats": {"steps_with_noise": 0, "pedestrians_removed": 0},
    }
    noisy = {
        **base,
        "observation_noise": {"enabled": True, "profile": "robustness_smoke_v1"},
        "observation_noise_stats": {"steps_with_noise": 5, "pedestrians_removed": 1},
    }
    _write_jsonl(clean, base)
    _write_jsonl(perturbed, noisy)

    report = _load_script().build_report(clean, perturbed)

    assert report["classification"]["label"] == "observation_only_delta"


def test_comparator_treats_null_metric_maps_as_empty(tmp_path: Path) -> None:
    """Explicit JSON null maps should not crash comparison."""
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    row = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "status": "success",
        "termination_reason": "success",
        "steps": 10,
        "horizon": 20,
        "outcome": {"route_complete": True},
        "metrics": None,
        "observation_noise_stats": None,
    }
    _write_jsonl(clean, row)
    _write_jsonl(perturbed, row)

    report = _load_script().build_report(clean, perturbed)

    assert report["metric_delta"]["success"]["delta"] is None
    assert report["classification"]["label"] == "null_policy_insensitive"


def test_comparator_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI writes compact durable artifacts."""
    clean = tmp_path / "clean.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    row = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "status": "success",
        "termination_reason": "success",
        "steps": 10,
        "horizon": 20,
        "outcome": {"route_complete": True},
        "metrics": {"success": 1.0, "min_distance": 0.4, "path_length": 2.0},
        "observation_noise": {"enabled": False, "profile": "none"},
        "observation_noise_stats": {"steps_with_noise": 0},
    }
    _write_jsonl(clean, row)
    _write_jsonl(perturbed, row)
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "README.md"

    assert (
        _load_script().main(
            [
                "--clean-jsonl",
                str(clean),
                "--perturbed-jsonl",
                str(perturbed),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))["schema_version"].endswith(".v1")
    assert "Issue #3201" in output_md.read_text(encoding="utf-8")


def test_trace_comparator_classifies_observation_only_scenario_too_weak(tmp_path: Path) -> None:
    """Trace comparison should preserve the live-scenario-too-weak distinction."""
    clean_trace = tmp_path / "clean_trace.json"
    perturbed_trace = tmp_path / "perturbed_trace.json"
    base = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "observation_perturbation_config": {"position_noise_std_m": 0.0},
        "progress_summary": {
            "steps_observed": 2,
            "net_goal_progress": 1.0,
            "closest_robot_ped_distance": 7.0,
            "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
        },
        "steps": [
            {
                "policy_command": [1.0, 0.0],
                "observation_perturbation": {
                    "noise_profile": "none",
                    "evidence_class": "ideal_state",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "observed_actor_count": 3,
                },
            }
        ],
    }
    perturbed = {
        **base,
        "observation_perturbation_config": {
            "position_noise_std_m": 0.3,
            "missed_detection_probability": 0.5,
        },
        "steps": [
            {
                "policy_command": [1.0, 0.0],
                "observation_perturbation": {
                    "noise_profile": "bounded_gaussian",
                    "evidence_class": "perception_limited",
                    "missed_actor_count": 2,
                    "occluded_actor_count": 0,
                    "observed_actor_count": 1,
                },
            }
        ],
    }
    clean_trace.write_text(json.dumps(base), encoding="utf-8")
    perturbed_trace.write_text(json.dumps(perturbed), encoding="utf-8")

    report = _load_script().build_trace_report(clean_trace, perturbed_trace)

    assert report["classification"]["label"] == "observation_only_scenario_too_weak"
    assert report["near_field_target"] == {
        "threshold_m": 2.0,
        "clean_closest_robot_ped_distance_m": 7.0,
        "satisfied": False,
    }
    assert report["observation_summary"]["changed"] is True
    assert report["command_summary"]["sequence_changed"] is False


def test_trace_comparator_reports_false_positive_actor_observations(tmp_path: Path) -> None:
    """Trace reports should count false-positive observations separately from misses."""
    clean_trace = tmp_path / "clean_trace.json"
    perturbed_trace = tmp_path / "perturbed_trace.json"
    base = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "observation_perturbation_config": {"position_noise_std_m": 0.0},
        "progress_summary": {
            "steps_observed": 1,
            "closest_robot_ped_distance": 1.0,
            "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
        },
        "steps": [
            {
                "policy_command": [1.0, 0.0],
                "observation_perturbation": {
                    "noise_profile": "none",
                    "evidence_class": "ideal_state",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "false_positive_actor_count": 0,
                    "observed_actor_count": 2,
                },
            }
        ],
    }
    perturbed = {
        **base,
        "observation_perturbation_config": {
            "false_positive_actor_count": 1,
            "false_positive_offset_x_m": 1.0,
            "false_positive_offset_y_m": 0.0,
        },
        "steps": [
            {
                "policy_command": [1.0, 0.0],
                "observation_perturbation": {
                    "noise_profile": "false_positive_actor_injection",
                    "evidence_class": "perception_limited",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "false_positive_actor_count": 1,
                    "observed_actor_count": 3,
                },
            }
        ],
    }
    clean_trace.write_text(json.dumps(base), encoding="utf-8")
    perturbed_trace.write_text(json.dumps(perturbed), encoding="utf-8")

    report = _load_script().build_trace_report(clean_trace, perturbed_trace)

    assert report["classification"]["label"] == "observation_only_delta"
    assert report["observation_summary"]["clean"]["false_positive_actor_observations_total"] == 0
    assert (
        report["observation_summary"]["perturbed"]["false_positive_actor_observations_total"] == 1
    )
    assert report["observation_summary"]["perturbed"]["missed_actor_observations_total"] == 0


def test_trace_comparator_treats_null_trace_maps_as_empty(tmp_path: Path) -> None:
    """Explicit null progress or perturbation maps should be treated as absent metadata."""
    clean_trace = tmp_path / "clean_trace.json"
    perturbed_trace = tmp_path / "perturbed_trace.json"
    payload = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "progress_summary": None,
        "steps": [{"policy_command": [1.0, 0.0], "observation_perturbation": None}],
    }
    clean_trace.write_text(json.dumps(payload), encoding="utf-8")
    perturbed_trace.write_text(json.dumps(payload), encoding="utf-8")

    report = _load_script().build_trace_report(clean_trace, perturbed_trace)

    assert report["progress_delta"]["steps_observed"]["changed"] is False
    assert report["classification"]["label"] == "null_policy_insensitive"


def test_trace_markdown_treats_missing_near_field_distance_as_unavailable() -> None:
    """Trace Markdown should not render ``None`` as a measured distance."""
    report = {
        "claim_boundary": "diagnostic",
        "inputs": {"clean_trace_json": "clean.json", "perturbed_trace_json": "perturbed.json"},
        "scenario": {"same_scenario": True},
        "seed": {"same_seed": True},
        "classification": {"label": "null_policy_insensitive", "rationale": "no change"},
        "near_field_target": {
            "threshold_m": 2.0,
            "clean_closest_robot_ped_distance_m": None,
            "satisfied": False,
        },
        "command_summary": {
            "sequence_changed": False,
            "clean_first": [0.0, 0.0],
            "clean_last": [0.0, 0.0],
            "perturbed_first": [0.0, 0.0],
            "perturbed_last": [0.0, 0.0],
        },
        "observation_summary": {"changed": False, "clean": {}, "perturbed": {}},
        "progress_delta": {},
    }

    markdown = _load_script()._trace_markdown(report)

    assert "near-field target metadata was unavailable" in markdown
    assert "`None` m" not in markdown


def test_trace_comparator_cli_writes_trace_markdown(tmp_path: Path) -> None:
    """Trace CLI mode writes the compact report and Markdown."""
    clean_trace = tmp_path / "clean_trace.json"
    perturbed_trace = tmp_path / "perturbed_trace.json"
    payload = {
        "scenario_id": "dense_pedestrian_stress",
        "seed": 2765,
        "observation_perturbation_config": {"position_noise_std_m": 0.0},
        "progress_summary": {"steps_observed": 1, "closest_robot_ped_distance": 1.5},
        "steps": [
            {
                "policy_command": [1.0, 0.0],
                "observation_perturbation": {
                    "noise_profile": "none",
                    "evidence_class": "ideal_state",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "observed_actor_count": 1,
                },
            }
        ],
    }
    clean_trace.write_text(json.dumps(payload), encoding="utf-8")
    perturbed_trace.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "README.md"

    assert (
        _load_script().main(
            [
                "--clean-trace-json",
                str(clean_trace),
                "--perturbed-trace-json",
                str(perturbed_trace),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))["inputs"]["clean_trace_json"]
    markdown = output_md.read_text(encoding="utf-8")
    assert "Near-Field Target" in markdown
    assert "satisfied: `True`" in markdown
    assert "Progress Deltas" in markdown
