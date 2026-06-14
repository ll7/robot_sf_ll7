"""Tests for the dense-pedestrian-stress observation-noise envelope."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

from robot_sf.benchmark.pedestrian_forecast import (
    PedestrianState,
    chi_square_2d_threshold,
    constant_velocity_gaussian_baseline,
    ellipse_overlaps_point,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "dense_pedestrian_stress_episode_0000.json"
)
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_dense_stress_observation_envelope.py"
_LOADED_MOD = None


def _load_script():
    """Load the dense-stress envelope script as a module."""
    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "run_dense_stress_observation_envelope_2765", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_dense_stress_observation_envelope_2765"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return _LOADED_MOD


VALID_CLASSIFICATIONS = {
    "forecast_ambiguity_detected",
    "robustness_evidence",
    "scenario_too_weak",
    "policy_insensitive",
    "diagnostic_only",
    "inconclusive",
    "blocked",
}

EXPECTED_CONDITIONS = [
    "noop",
    "low_noise",
    "medium_noise",
    "high_noise",
    "partial_missed_detection",
    "full_missed_detection",
    "single_actor_occlusion",
    "two_actor_occlusion",
    "delay_2_steps",
    "medium_noise_with_occlusion",
]


def _load_fixture() -> dict:
    """Load the dense-pedestrian-stress fixture for test use."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_fixture_exists() -> None:
    """The dense-stress trace fixture file exists."""
    assert FIXTURE_PATH.exists(), f"Fixture not found: {FIXTURE_PATH}"


def test_fixture_has_three_pedestrians_per_frame() -> None:
    """Every frame in the fixture has exactly 3 pedestrians."""
    trace = _load_fixture()
    for frame in trace["frames"]:
        assert len(frame["pedestrians"]) == 3, (
            f"Frame {frame['step']} has {len(frame['pedestrians'])} pedestrians"
        )


def test_fixture_pedestrians_are_near_field() -> None:
    """At least one frame has a pedestrian within 2 m of the robot."""
    trace = _load_fixture()
    found_close = False
    for frame in trace["frames"]:
        robot_pos = np.array(frame["robot"]["position"])
        for ped in frame["pedestrians"]:
            dist = np.linalg.norm(robot_pos - np.array(ped["position"]))
            if dist < 2.0:
                found_close = True
                break
        if found_close:
            break
    assert found_close, "No pedestrian found within 2 m of robot"


def _check_forecast_overlap_in_trace(trace: dict) -> bool:
    """Check if any two pedestrian forecast ellipses overlap in the trace."""
    threshold = chi_square_2d_threshold(0.95)
    for frame in trace["frames"]:
        pedestrians = frame["pedestrians"]
        if len(pedestrians) < 2:
            continue
        forecasts = _build_forecasts(pedestrians)
        if _has_overlap_pair(forecasts, threshold):
            return True
    return False


def _build_forecasts(pedestrians: list) -> list:
    """Build constant-velocity forecasts for a list of pedestrians."""
    forecasts = []
    for ped in pedestrians:
        state = PedestrianState(
            id=int(ped["id"]),
            position=np.array(ped["position"], dtype=float),
            velocity=np.array(ped["velocity"], dtype=float),
        )
        forecasts.append(constant_velocity_gaussian_baseline(state, horizons_s=(0.5, 1.0)))
    return forecasts


def _has_overlap_pair(forecasts: list, threshold: float) -> bool:
    """Check if any two forecasts have overlapping ellipses."""
    for i in range(len(forecasts)):
        for j in range(i + 1, len(forecasts)):
            for fi in forecasts[i].predictions:
                for fj in forecasts[j].predictions:
                    if fi.horizon_s != fj.horizon_s:
                        continue
                    if ellipse_overlaps_point(
                        mean=fi.mean,
                        covariance=fi.covariance,
                        point=fj.mean,
                        confidence_threshold=threshold,
                        radius_m=0.3,
                    ):
                        return True
    return False


def test_fixture_forecast_ambiguity() -> None:
    """At least two pedestrian forecast ellipses overlap at some frame."""
    trace = _load_fixture()
    assert _check_forecast_overlap_in_trace(trace)


def test_all_conditions_are_evaluated() -> None:
    """Script evaluates exactly the ten required conditions."""
    mod = _load_script()
    assert list(mod.CONDITIONS.keys()) == EXPECTED_CONDITIONS


def test_report_has_required_top_level_keys() -> None:
    """JSON report contains schema_version, issue, claim_boundary, and conditions."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    results = []
    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, meta)
        results.append(result)

    fixture_meta = {"first_visible_step": 0, "frame_count": len(frames)}
    repro = {"issue": 2765}
    report = mod._build_report(results, fixture_meta, repro)

    assert report["schema_version"] == "observation_noise_envelope.v1"
    assert report["issue"] == 2765
    assert "claim_boundary" in report
    assert "conditions" in report
    assert len(report["conditions"]) == 10
    assert "summary" in report
    assert report["summary"]["pedestrian_count"] == 3


def test_condition_report_shape() -> None:
    """Each condition result has all required fields."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, meta)
        assert result["condition"] == name
        assert "description" in result
        assert "spec" in result
        assert "first_observed_step" in result
        assert "pedestrian_count" in result
        assert result["pedestrian_count"] == 3
        assert "fixture_observed_steps" in result
        assert "perturbed_observed_steps" in result
        assert "missed_actor_observations_total" in result
        assert "occluded_actor_observations_total" in result
        assert "delay_steps_configured" in result
        assert "closest_distance_m" in result
        assert "stop_yield_feasibility" in result
        assert "action_proxy_changes" in result
        assert "forecast_ambiguity" in result
        assert "classification" in result


def test_classifications_are_valid_labels() -> None:
    """All classifications use allowed labels."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, meta)
        label = result["classification"]["label"]
        assert label in VALID_CLASSIFICATIONS, f"Invalid classification '{label}' for {name}"


def test_noop_is_diagnostic_only() -> None:
    """Noop baseline is classified as diagnostic_only."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, meta)
    assert result["classification"]["label"] == "diagnostic_only"


def test_noop_observations_match_fixture() -> None:
    """Noop first-observed step matches fixture visibility (step 0)."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, meta)
    assert result["first_observed_step"] == 0
    assert result["fixture_observed_steps"][0] == 0


def test_full_missed_detection_never_observed() -> None:
    """Full missed detection never observes any pedestrian."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition(
        "full_missed_detection",
        mod.CONDITIONS["full_missed_detection"],
        frames,
        meta,
    )
    assert result["first_observed_step"] is None
    assert result["classification"]["label"] == "scenario_too_weak"


def test_forecast_ambiguity_detected_for_noise_conditions() -> None:
    """Noise conditions detect forecast ambiguity in the dense scene."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    for condition_name in [
        "low_noise",
        "medium_noise",
        "high_noise",
        "delay_2_steps",
    ]:
        result = mod.evaluate_condition(
            condition_name,
            mod.CONDITIONS[condition_name],
            frames,
            meta,
        )
        assert result["classification"]["label"] == "forecast_ambiguity_detected", (
            f"Expected forecast_ambiguity_detected for {condition_name}, "
            f"got {result['classification']['label']}"
        )


def test_forecast_ambiguity_has_overlap_events() -> None:
    """Conditions classified as forecast_ambiguity_detected have overlap events > 0."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition("medium_noise", mod.CONDITIONS["medium_noise"], frames, meta)
    assert result["forecast_ambiguity"]["total_overlap_events"] > 0


def test_closest_distance_is_positive() -> None:
    """Closest robot-pedestrian distance is always positive."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, meta)
        if result["closest_distance_m"] is not None:
            assert result["closest_distance_m"] > 0.0, f"Distance not positive for {name}"


def test_occlusion_conditions_have_correct_actor_count() -> None:
    """Partial actor occlusion conditions maintain the 3-actor frame structure."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition(
        "single_actor_occlusion",
        mod.CONDITIONS["single_actor_occlusion"],
        frames,
        meta,
    )
    assert result["spec"]["has_occlusion_mask"] is True
    assert result["spec"]["occlusion_mask_size"] == 3

    result = mod.evaluate_condition(
        "two_actor_occlusion",
        mod.CONDITIONS["two_actor_occlusion"],
        frames,
        meta,
    )
    assert result["spec"]["has_occlusion_mask"] is True
    assert result["spec"]["occlusion_mask_size"] == 3


def test_action_proxy_has_required_fields() -> None:
    """Action proxy summary has the required compact fields."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, meta)
    ap = result["action_proxy_changes"]
    assert "linear_velocity_changed" in ap
    assert "events" in ap
    assert "velocity_range" in ap


def test_report_is_compact_condition_summary() -> None:
    """Condition report omits full frame dumps from durable evidence."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, meta)
    assert "frames" not in result
    assert result["total_frames"] == len(frames)


def test_markdown_report_contains_caveats() -> None:
    """Markdown report contains diagnostic-only caveats."""
    mod = _load_script()
    trace = _load_fixture()
    frames = trace["frames"]
    meta = trace.get("dense_stress_metadata", {})

    results = []
    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, meta)
        results.append(result)

    fixture_meta = {
        "first_visible_step": 0,
        "frame_count": len(frames),
        "trace_path": "x",
        "scenario_id": "y",
    }
    repro = {
        "issue": 2765,
        "generated_at_utc": "2026-01-01",
        "command": "test",
        "repo_head": "abc",
    }
    md = mod._generate_markdown(results, fixture_meta, repro)

    assert "Not paper-facing benchmark proof" in md
    assert "Diagnostic" in md
    assert "forecast_ambiguity_detected" in md


def test_spec_summary_fields() -> None:
    """Spec summary contains the expected keys including occlusion_mask_size."""
    mod = _load_script()
    from robot_sf.benchmark.observation_perturbation import ObservationPerturbationSpec

    spec = ObservationPerturbationSpec(position_noise_std_m=0.5, seed=42)
    summary = mod._spec_summary(spec)
    assert "position_noise_std_m" in summary
    assert "noise_profile" in summary
    assert "is_noop" in summary
    assert summary["position_noise_std_m"] == 0.5
    assert summary["occlusion_mask_size"] == 0


def test_script_is_importable() -> None:
    """The script can be loaded as a module without errors."""
    mod = _load_script()
    assert hasattr(mod, "main")
    assert hasattr(mod, "CONDITIONS")
    assert hasattr(mod, "evaluate_condition")


def test_dense_stress_metadata_in_fixture() -> None:
    """Fixture contains dense_stress_metadata with expected fields."""
    trace = _load_fixture()
    meta = trace.get("dense_stress_metadata", {})
    assert meta.get("issue") == 2765
    assert meta.get("pedestrian_count") == 3
    assert meta.get("dt_s") == 0.1
    assert meta.get("total_steps") == 20
    assert len(meta.get("pedestrians", [])) == 3
