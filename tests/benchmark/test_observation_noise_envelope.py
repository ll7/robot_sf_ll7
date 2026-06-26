"""Tests for the observation-noise robustness envelope script."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "occluded_emergence_episode_0000.json"
)
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_observation_noise_envelope.py"
_LOADED_MOD = None


def _load_script():
    """Load the observation noise envelope script as a module."""
    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "run_observation_noise_envelope_issue_2755", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_observation_noise_envelope_issue_2755"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return _LOADED_MOD


VALID_CLASSIFICATIONS = {
    "robustness_evidence",
    "scenario_too_weak",
    "policy_insensitive",
    "diagnostic_only",
    "blocked",
}

EXPECTED_CONDITIONS = [
    "noop",
    "low_noise",
    "medium_noise",
    "missed_detection_only",
    "false_positive_only",
    "occlusion_only",
    "delay_only",
    "combined",
]


def test_all_conditions_are_evaluated() -> None:
    """Script evaluates exactly the seven required conditions."""
    mod = _load_script()
    assert list(mod.CONDITIONS.keys()) == EXPECTED_CONDITIONS


def test_report_has_required_top_level_keys() -> None:
    """JSON report contains schema_version, issue, claim_boundary, and conditions."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    results = []
    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, first_visible)
        results.append(result)

    fixture_meta = {"first_visible_step": first_visible, "frame_count": len(frames)}
    repro = {"issue": 2755}
    report = mod._build_report(results, fixture_meta, repro)

    assert report["schema_version"] == "observation_noise_envelope.v1"
    assert report["issue"] == 2755
    assert "claim_boundary" in report
    assert "conditions" in report
    assert len(report["conditions"]) == len(EXPECTED_CONDITIONS)
    assert "summary" in report
    assert "safety_effects" in report["summary"]
    assert (
        report["summary"]["safety_effects"]["missed_detection_only"]["false_positive"]["effect"]
        == "none_observed"
    )


def test_report_issue_follows_repro_issue() -> None:
    """Generated reports should not hard-code the legacy issue number."""
    mod = _load_script()

    report = mod._build_report([], {}, {"issue": 2927})

    assert report["issue"] == 2927


def test_condition_report_shape() -> None:
    """Each condition result has all required fields."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, first_visible)
        assert result["condition"] == name
        assert "description" in result
        assert "spec" in result
        assert "first_observed_step" in result
        assert "response_delay_steps" in result
        assert "fixture_observed_steps" in result
        assert "perturbed_observed_steps" in result
        assert "missed_actor_observations_total" in result
        assert "occluded_actor_observations_total" in result
        assert "delay_steps_configured" in result
        assert "closest_distance_m" in result
        assert "stop_yield_feasibility" in result
        assert "action_proxy_changes" in result
        assert "safety_effects" in result
        assert "false_negative" in result["safety_effects"]
        assert "false_positive" in result["safety_effects"]
        assert "classification" in result


def test_classifications_are_valid_labels() -> None:
    """All classifications use allowed labels."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, first_visible)
        label = result["classification"]["label"]
        assert label in VALID_CLASSIFICATIONS, f"Invalid classification '{label}' for {name}"


def test_noop_is_diagnostic_only() -> None:
    """Noop baseline is classified as diagnostic_only."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, first_visible)
    assert result["classification"]["label"] == "diagnostic_only"


def test_noop_first_observed_matches_fixture_visibility() -> None:
    """Noop respects the fixture visibility boundary."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, first_visible)
    assert result["first_observed_step"] == first_visible
    assert result["response_delay_steps"] == 0
    assert result["fixture_observed_steps"][0] == first_visible
    assert result["perturbed_observed_steps"][0] == first_visible


def test_missed_detection_never_observed() -> None:
    """Full missed detection never observes the pedestrian."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition(
        "missed_detection_only", mod.CONDITIONS["missed_detection_only"], frames, first_visible
    )
    assert result["first_observed_step"] is None
    assert result["classification"]["label"] == "scenario_too_weak"
    assert result["safety_effects"]["false_negative"]["effect"] == "full_miss_or_occlusion"
    assert result["safety_effects"]["false_positive"]["effect"] == "none_observed"


def test_false_positive_condition_injects_observed_only_actor() -> None:
    """False-positive condition injects a ghost actor into replay observations."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition(
        "false_positive_only", mod.CONDITIONS["false_positive_only"], frames, first_visible
    )

    assert result["false_positive_actor_observations_total"] > 0
    assert result["spec"]["false_positive_actor_count"] == 1
    assert result["safety_effects"]["false_positive"]["effect"] == "actor_injected"
    assert (
        result["safety_effects"]["false_positive"]["false_positive_actor_observations_total"]
        == result["false_positive_actor_observations_total"]
    )


def test_occlusion_only_never_observed() -> None:
    """Full occlusion never observes the pedestrian."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition(
        "occlusion_only", mod.CONDITIONS["occlusion_only"], frames, first_visible
    )
    assert result["first_observed_step"] is None
    assert result["classification"]["label"] == "scenario_too_weak"


def test_delay_only_observes_after_configured_lag() -> None:
    """Delay-only condition observes after fixture visibility plus delay."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition(
        "delay_only", mod.CONDITIONS["delay_only"], frames, first_visible
    )
    assert result["first_observed_step"] == first_visible + 2
    assert result["response_delay_steps"] == 2
    assert result["spec"]["delay_steps"] == 2
    assert result["delay_steps_configured"] == 2
    assert result["stop_yield_feasibility"]["stop_feasible_first_observed"] is False
    assert result["stop_yield_feasibility"]["yield_feasible_first_observed"] is True


def test_combined_never_observed() -> None:
    """Combined noise + occlusion still never observes (occlusion zeros position)."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition("combined", mod.CONDITIONS["combined"], frames, first_visible)
    assert result["first_observed_step"] is None
    assert result["classification"]["label"] == "scenario_too_weak"


def test_closest_distance_is_positive() -> None:
    """Closest robot-pedestrian distance is always positive for all conditions."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, first_visible)
        if result["closest_distance_m"] is not None:
            assert result["closest_distance_m"] > 0.0, f"Distance not positive for {name}"


def test_action_proxy_has_required_fields() -> None:
    """Action proxy summary stays compact but includes event and velocity signal."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, first_visible)
    ap = result["action_proxy_changes"]
    assert "linear_velocity_changed" in ap
    assert "events" in ap
    assert "velocity_range" in ap
    assert "first_action" not in ap
    assert "last_action" not in ap


def test_action_proxy_summary_ignores_missing_velocity() -> None:
    """Action proxy summary handles frames without linear velocity."""
    mod = _load_script()
    summary = mod._action_proxy_summary(
        [
            {"linear_velocity": None, "event": "missing"},
            {"linear_velocity": 0.0, "event": "stopped"},
            {"linear_velocity": 1.0, "event": "moving"},
        ]
    )

    assert summary["linear_velocity_changed"] is True
    assert summary["velocity_range"] == [0.0, 1.0]


def test_report_is_compact_condition_summary() -> None:
    """Condition report omits full frame dumps from durable evidence."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    result = mod.evaluate_condition("noop", mod.CONDITIONS["noop"], frames, first_visible)
    assert "frames" not in result
    assert result["total_frames"] == len(frames)
    assert result["fixture_observed_steps"] == list(range(first_visible, len(frames)))


def test_markdown_report_contains_caveats() -> None:
    """Markdown report contains diagnostic-only caveats."""
    mod = _load_script()
    trace = _load_fixture_for_test()
    frames = trace["frames"]
    first_visible = trace["occlusion"]["first_visible_step"]

    results = []
    for name, cfg in mod.CONDITIONS.items():
        result = mod.evaluate_condition(name, cfg, frames, first_visible)
        results.append(result)

    fixture_meta = {
        "first_visible_step": first_visible,
        "frame_count": len(frames),
        "trace_path": "x",
        "scenario_id": "y",
    }
    repro = {"issue": 2755, "generated_at_utc": "2026-01-01", "command": "test", "repo_head": "abc"}
    md = mod._generate_markdown(results, fixture_meta, repro)

    assert "Not paper-facing benchmark proof" in md
    assert "Diagnostic" in md
    assert "False-negative safety effect" in md
    assert "False-positive safety effect" in md
    assert "False-positive actors are not injected" not in md


def test_spec_summary_fields() -> None:
    """Spec summary contains the expected keys."""
    mod = _load_script()
    from robot_sf.benchmark.observation_perturbation import ObservationPerturbationSpec

    spec = ObservationPerturbationSpec(position_noise_std_m=0.5, seed=42)
    summary = mod._spec_summary(spec)
    assert "position_noise_std_m" in summary
    assert "noise_profile" in summary
    assert "is_noop" in summary
    assert summary["position_noise_std_m"] == 0.5


def test_spec_summary_includes_observation_quality_group() -> None:
    """Condition specs should carry the bounded observation-quality.v1 metadata group."""
    mod = _load_script()
    from robot_sf.benchmark.observation_perturbation import ObservationPerturbationSpec
    from robot_sf.benchmark.observation_quality import ObservationQuality

    spec = ObservationPerturbationSpec(
        position_noise_std_m=0.3,
        position_noise_bound_m=0.6,
        missed_detection_probability=0.5,
        delay_steps=2,
        seed=2927,
    )

    group = mod._spec_summary(spec)["observation_quality"]
    quality = ObservationQuality.from_dict(group["fields"])

    assert group["schema_version"] == "observation_quality.v1"
    assert quality.false_negative_rate == 0.5
    assert quality.dropout_probability == 0.5
    assert quality.false_positive_rate == 0.0
    assert quality.latency_s == 0.2
    assert "hardware-calibrated" in quality.notes


def test_script_is_importable() -> None:
    """The script can be loaded as a module without errors."""
    mod = _load_script()
    assert hasattr(mod, "main")
    assert hasattr(mod, "CONDITIONS")
    assert hasattr(mod, "evaluate_condition")


def _load_fixture_for_test() -> dict:
    """Load the occluded-emergence fixture for test use."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)
