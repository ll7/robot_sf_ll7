"""Tests for the issue #2780 simulated occluded-emergence variant fixtures."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

from robot_sf.benchmark.pedestrian_forecast import compute_batch_forecast_metrics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1"
SOURCES_DIR = FIXTURE_DIR / "sources"
GENERATOR_PATH = REPO_ROOT / "scripts/tools/generate_occluded_emergence_variants.py"
SUMMARY_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2780_occluded_emergence_variants/summary.json"
)

VARIANT_NAMES = [
    "occluded_emergence_left_close",
    "occluded_emergence_right_close",
    "occluded_emergence_late_visibility",
    "occluded_emergence_slow_pedestrian",
    "occluded_emergence_fast_pedestrian",
]

EXPECTED_FAILURE_MODES = {
    "occluded_emergence_left_close": "late_detection",
    "occluded_emergence_right_close": "wrong_source_selection",
    "occluded_emergence_late_visibility": "insufficient_braking_distance",
    "occluded_emergence_slow_pedestrian": "unnecessary_stop",
    "occluded_emergence_fast_pedestrian": "forecast_miss",
}


def _trace_path(name: str) -> pathlib.Path:
    short = name.replace("occluded_emergence_", "")
    return FIXTURE_DIR / f"occluded_emergence_{short}_episode_0000.json"


def _meta_path(name: str) -> pathlib.Path:
    return SOURCES_DIR / f"issue_2780_{name}_fixture_111_ep0000.meta.json"


def _load_trace(name: str) -> dict:
    with open(_trace_path(name)) as fh:
        return json.load(fh)


def _load_meta(name: str) -> dict:
    with open(_meta_path(name)) as fh:
        return json.load(fh)


def _load_summary() -> dict:
    with open(SUMMARY_PATH) as fh:
        return json.load(fh)


def _load_generator_module():
    module_name = "generate_occluded_emergence_variants_issue_2780"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_all_variant_trace_files_exist() -> None:
    """Every named variant has a trace JSON file."""
    for name in VARIANT_NAMES:
        path = _trace_path(name)
        assert path.exists(), f"Missing trace fixture: {path}"


def test_all_variant_metadata_files_exist() -> None:
    """Every named variant has a metadata JSON file."""
    for name in VARIANT_NAMES:
        path = _meta_path(name)
        assert path.exists(), f"Missing metadata: {path}"


def test_variant_trace_schema_version() -> None:
    """All variants use the correct schema version."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert trace["schema_version"] == "simulation_trace_export.v1"


def test_variant_evidence_boundary() -> None:
    """All variants carry the conservative evidence boundary."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert trace["evidence_boundary"] == "smoke_diagnostic_only_not_benchmark_evidence"


def test_variant_metadata_boundaries() -> None:
    """Metadata files record the same evidence boundary and variant name."""
    for name in VARIANT_NAMES:
        meta = _load_meta(name)
        assert meta["boundary"] == "smoke_diagnostic_only_not_benchmark_evidence"
        assert meta["variant_name"] == name
        assert meta["issue"] == 2780
        assert meta["parent_issue"] == 2756


def test_variant_expected_failure_modes() -> None:
    """Each variant records its expected failure mode."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert "variant" in trace
        assert trace["variant"]["expected_failure_mode"] == EXPECTED_FAILURE_MODES[name]
        assert trace["variant"]["issue"] == 2780


def test_variant_metadata_failure_modes_match() -> None:
    """Metadata failure modes match the trace variant field."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        meta = _load_meta(name)
        assert meta["expected_failure_mode"] == trace["variant"]["expected_failure_mode"]


def test_variant_variation_dimensions_nonempty() -> None:
    """Each variant specifies at least two variation dimensions."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        dims = trace["variant"]["variation_dimensions"]
        assert isinstance(dims, list)
        assert len(dims) >= 2, f"{name} has too few variation dimensions: {dims}"


def test_variant_evidence_boundary_note_present() -> None:
    """Each variant records an evidence boundary note."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert "evidence_boundary_note" in trace["variant"]
        assert trace["variant"]["evidence_boundary_note"].startswith("diagnostic")


def test_variant_safety_relevant_field_present() -> None:
    """Each variant explicitly declares safety relevance."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert "safety_relevant_under_live_replay" in trace["variant"]


def test_variant_safety_relevant_set_matches_reported_boundary() -> None:
    """Safety-relevant variants match the issue #2780 diagnostic boundary."""
    expected_safety_relevant = {
        "occluded_emergence_left_close",
        "occluded_emergence_right_close",
        "occluded_emergence_late_visibility",
        "occluded_emergence_fast_pedestrian",
    }
    actual_safety_relevant = {
        name
        for name in VARIANT_NAMES
        if _load_trace(name)["variant"]["safety_relevant_under_live_replay"]
    }
    assert actual_safety_relevant == expected_safety_relevant


def test_variant_safety_relevant_slow_pedestrian_false() -> None:
    """Slow-pedestrian variant is not safety-relevant."""
    trace = _load_trace("occluded_emergence_slow_pedestrian")
    assert trace["variant"]["safety_relevant_under_live_replay"] is False


def test_variant_files_match_generator_output() -> None:
    """Checked-in traces are exactly the deterministic generator output."""
    generator = _load_generator_module()
    generated = generator.generate_all()
    assert set(generated) == set(VARIANT_NAMES)
    for name in VARIANT_NAMES:
        assert generated[name] == _load_trace(name)


def test_variant_summary_matches_trace_and_metadata_files() -> None:
    """Summary artifact paths and fields stay synchronized with fixture files."""
    summary = _load_summary()
    summary_variants = {row["name"]: row for row in summary["variants"]}
    assert summary["variant_count"] == len(VARIANT_NAMES)
    assert set(summary_variants) == set(VARIANT_NAMES)

    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        meta = _load_meta(name)
        row = summary_variants[name]

        assert row["trace_path"] == str(_trace_path(name).relative_to(REPO_ROOT))
        assert row["metadata_path"] == str(_meta_path(name).relative_to(REPO_ROOT))
        assert row["scenario_id"] == trace["source"]["scenario_id"] == meta["scenario"]
        assert row["seed"] == trace["source"]["seed"] == meta["seed"]
        assert row["episode_id"] == trace["source"]["episode_id"] == meta["episode_id"]
        assert (
            row["expected_failure_mode"]
            == trace["variant"]["expected_failure_mode"]
            == meta["expected_failure_mode"]
        )
        assert (
            row["variation_dimensions"]
            == trace["variant"]["variation_dimensions"]
            == meta["variation_dimensions"]
        )
        assert (
            row["safety_relevant_under_live_replay"]
            == trace["variant"]["safety_relevant_under_live_replay"]
            == meta["safety_relevant_under_live_replay"]
        )


def test_variant_first_visible_step_present() -> None:
    """All variants have an occlusion metadata block with first_visible_step."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert "first_visible_step" in trace["occlusion"]
        assert isinstance(trace["occlusion"]["first_visible_step"], int)


def test_variant_first_visible_frame_matches_occlusion_transition() -> None:
    """Frame visibility flags transition at the variant first-visible step."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        first_visible_step = trace["occlusion"]["first_visible_step"]
        first_visible_frames = [frame for frame in trace["frames"] if frame["first_visible"]]
        assert [frame["step"] for frame in first_visible_frames] == [first_visible_step]

        for frame in trace["frames"]:
            expected_status = "occluded" if frame["step"] < first_visible_step else "visible"
            assert frame["occlusion_status"] == {"emerging_ped": expected_status}


def test_variant_conflict_time_matches_geometry() -> None:
    """Conflict timing metadata is derived from the initial pedestrian geometry."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        occlusion = trace["occlusion"]
        first_frame = trace["frames"][0]
        pedestrian = first_frame["pedestrians"][0]
        initial_y = pedestrian["position"][1]
        speed_y = abs(pedestrian["velocity"][1])
        conflict_y = occlusion["conflict_zone_center"][1]
        expected_conflict_time = abs(initial_y - conflict_y) / speed_y
        assert occlusion["conflict_time_s"] == round(expected_conflict_time, 4)


def test_variant_frame_conflict_time_never_rebounds_after_crossing() -> None:
    """Frame time-to-conflict decreases to zero instead of rebounding after crossing."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        times = [frame["conflict_timing"]["time_to_conflict_s"] for frame in trace["frames"]]
        assert times == sorted(times, reverse=True)
        assert times[-1] == 0.0


def test_variant_feasibility_step_metadata_matches_frame_flags() -> None:
    """Occlusion feasibility step metadata matches frame-level feasibility flags."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        frames = trace["frames"]
        occlusion = trace["occlusion"]
        stop_last_true = max(
            (frame["step"] for frame in frames if frame["conflict_timing"]["stop_feasible"]),
            default=-1,
        )
        yield_last_true = max(
            (frame["step"] for frame in frames if frame["conflict_timing"]["yield_feasible"]),
            default=-1,
        )
        assert occlusion["stop_feasible_before_step"] == stop_last_true + 1
        assert occlusion["yield_feasible_before_step"] == yield_last_true + 1


def test_variant_planner_event_does_not_restart_yield_after_conflict() -> None:
    """Planner events do not return to yield_start after the conflict-time frame."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        events = [frame["planner"]["event"] for frame in trace["frames"]]
        assert events.count("conflict_time") == 1
        conflict_index = events.index("conflict_time")
        assert "yield_start" not in events[conflict_index + 1 :]


def test_variant_emergence_sides_vary() -> None:
    """Left and right variants have different occluder x-bounds."""
    left = _load_trace("occluded_emergence_left_close")
    right = _load_trace("occluded_emergence_right_close")
    assert (
        left["occlusion"]["occluder_bounds"]["x_max"]
        < right["occlusion"]["occluder_bounds"]["x_min"]
    )


def test_variant_pedestrian_speeds_vary() -> None:
    """Slow and fast variants have different pedestrian speeds."""
    slow = _load_trace("occluded_emergence_slow_pedestrian")
    fast = _load_trace("occluded_emergence_fast_pedestrian")
    slow_speed = slow["frames"][0]["pedestrians"][0]["velocity"][1]
    fast_speed = fast["frames"][0]["pedestrians"][0]["velocity"][1]
    assert slow_speed < fast_speed


def test_variant_frame_count() -> None:
    """All variants have exactly 16 frames."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        assert len(trace["frames"]) == 16


def test_variant_frames_have_required_fields() -> None:
    """Every frame in every variant has the standard trace fields."""
    required = [
        "step",
        "time_s",
        "robot",
        "pedestrians",
        "observed_pedestrians",
        "occlusion_status",
        "first_visible",
        "conflict_timing",
        "planner",
    ]
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        for frame in trace["frames"]:
            for field in required:
                assert field in frame, f"{name} step {frame['step']} missing {field}"


def test_variant_forecast_metrics_computable() -> None:
    """All variants produce valid forecast metrics."""
    for name in VARIANT_NAMES:
        trace = _load_trace(name)
        steps = [
            {
                "step": frame["step"],
                "time_s": frame["time_s"],
                "robot": frame["robot"],
                "pedestrians": frame["pedestrians"],
            }
            for frame in trace["frames"]
        ]
        metrics = compute_batch_forecast_metrics(steps, horizons_s=[0.5, 1.0], dt_s=0.1)
        assert metrics["forecast_evaluable_samples"] > 0, f"{name} has no evaluable samples"


def test_distinct_failure_modes_across_variants() -> None:
    """Each variant has a distinct expected failure mode."""
    modes = [EXPECTED_FAILURE_MODES[name] for name in VARIANT_NAMES]
    assert len(set(modes)) == len(VARIANT_NAMES), f"Duplicate failure modes: {modes}"


def test_at_least_one_safety_relevant_variant() -> None:
    """At least one variant declares itself safety-relevant under live replay."""
    has_safety = any(
        _load_trace(name)["variant"]["safety_relevant_under_live_replay"] for name in VARIANT_NAMES
    )
    assert has_safety, "No variant is safety-relevant under live replay"
