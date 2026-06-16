"""Tests for the live same-seed forecast replay gate (issue #2902)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import robot_sf.benchmark.live_forecast_replay_gate as live_forecast_replay_gate_module
from robot_sf.analysis_workbench.simulation_trace_export import SimulationTraceExport
from robot_sf.benchmark.live_forecast_replay_gate import (
    FORECAST_VARIANTS,
    LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION,
    REQUIRED_METRICS,
    RUN_CLASSIFICATION_BLOCKED,
    RUN_CLASSIFICATION_DEGRADED,
    RUN_CLASSIFICATION_DIAGNOSTIC_ONLY,
    RUN_CLASSIFICATION_NATIVE,
    SMOKE_FORECAST_VARIANTS,
    VALID_RUN_CLASSIFICATIONS,
    LiveForecastReplayGateConfig,
    LiveForecastReplayGateError,
    _forecast_brake_needed,
    _summarize_forecast_metrics,
    build_variant_forecast_batch,
    check_native_live_path_eligibility,
    classify_live_forecast_replay_run,
    compute_baseline_closed_loop_metrics,
    format_live_forecast_replay_gate_markdown,
    load_trace_tolerant,
    run_live_forecast_replay_gate,
    run_variant_closed_loop_replay,
)


@pytest.fixture
def dense_trace() -> SimulationTraceExport:
    """Load the motion-rich dense pedestrian stress fixture."""

    path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "analysis_workbench"
        / "simulation_trace_export_v1"
        / "dense_pedestrian_stress_episode_0000.json"
    )
    return load_trace_tolerant(path)


class TestConstants:
    """Schema and registry constants should match the issue contract."""

    def test_full_matrix_variants(self) -> None:
        """The five full-matrix forecast variants must be present and ordered."""

        assert FORECAST_VARIANTS == (
            "none",
            "cv",
            "semantic",
            "interaction_aware",
            "risk_filtered",
        )

    def test_smoke_variants(self) -> None:
        """The issue #2941 default smoke must include exactly none and cv."""

        assert SMOKE_FORECAST_VARIANTS == ("none", "cv")

    def test_required_metrics(self) -> None:
        """All issue-required metrics must be declared."""

        required_by_issue_2941 = {
            "collision",
            "near_miss",
            "min_distance",
            "stop_yield_timing",
            "progress",
            "false_positive_stops",
            "runtime",
        }
        assert required_by_issue_2941.issubset(set(REQUIRED_METRICS))

    def test_schema_version_constant(self) -> None:
        """Schema version should be stable."""

        assert LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION == "LiveForecastReplayGate.v1"

    def test_valid_run_classifications(self) -> None:
        """Run classification contract must contain the four required labels."""

        assert set(VALID_RUN_CLASSIFICATIONS) == {
            RUN_CLASSIFICATION_NATIVE,
            RUN_CLASSIFICATION_BLOCKED,
            RUN_CLASSIFICATION_DEGRADED,
            RUN_CLASSIFICATION_DIAGNOSTIC_ONLY,
        }


class TestTolerantTraceLoader:
    """Tolerant trace loading should preserve diagnostic metadata."""

    def test_preserves_semantic_metadata_and_falsy_values(self, tmp_path: Path) -> None:
        """Falsy ids/labels should not be replaced with fallback values."""

        trace_path = tmp_path / "extended_trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": None,
                    "trace_id": "",
                    "source": {
                        "scenario_id": "",
                        "seed": 0,
                        "planner_id": "",
                        "episode_id": "",
                        "generated_by": None,
                    },
                    "evidence_boundary": None,
                    "coordinate_frame": "",
                    "units": None,
                    "frames": [
                        {
                            "step": 0,
                            "time_s": 0.0,
                            "robot": {
                                "position": [0.0, 0.0],
                                "heading": 0,
                                "velocity": [0.0, 0.0],
                                "goal": [2.0, 0.0],
                            },
                            "pedestrians": [
                                {
                                    "id": 0,
                                    "position": [1.0, 0.0],
                                    "velocity": [0.0, 0.0],
                                    "intent_label": "",
                                    "signal_state": "",
                                    "signal_label": "",
                                    "actor_type": "",
                                },
                                {
                                    "id": None,
                                    "position": None,
                                    "velocity": None,
                                },
                            ],
                            "planner": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        trace = load_trace_tolerant(trace_path)

        assert trace.schema_version == "simulation_trace_export.v1"
        assert trace.trace_id == ""
        assert trace.source.scenario_id == ""
        assert trace.source.seed == 0
        assert trace.source.planner_id == ""
        assert trace.source.episode_id == ""
        assert trace.source.generated_by == "tolerant_loader"
        assert trace.evidence_boundary == "analysis_workbench_only"
        assert trace.coordinate_frame == ""
        assert trace.frames[0].robot["goal"] == [2.0, 0.0]
        assert trace.frames[0].planner["selected_action"]["linear_velocity"] == 0.0
        assert trace.frames[0].pedestrians[0]["id"] == "0"
        assert trace.frames[0].pedestrians[0]["intent_label"] == ""
        assert trace.frames[0].pedestrians[0]["signal_state"] == ""
        assert trace.frames[0].pedestrians[0]["signal_label"] == ""
        assert trace.frames[0].pedestrians[0]["actor_type"] == ""
        assert trace.frames[0].pedestrians[1]["id"] == "1"
        assert trace.frames[0].pedestrians[1]["position"] == [0.0, 0.0]
        assert trace.frames[0].pedestrians[1]["velocity"] == [0.0, 0.0]

    def test_rejects_unsupported_schema_version(self, tmp_path: Path) -> None:
        """Unsupported trace schemas should fail closed before normalization."""

        trace_path = tmp_path / "unsupported_trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": "simulation_trace_export.v999",
                    "frames": [{"step": 0, "time_s": 0.0}],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(LiveForecastReplayGateError, match="unsupported trace schema"):
            load_trace_tolerant(trace_path)


class TestForecastMetricSummary:
    """Forecast metric summaries should tolerate sparse aggregate rows."""

    def test_skips_missing_metric_or_horizon(self) -> None:
        """Rows missing metric or horizon should not raise during formatting."""

        summary = _summarize_forecast_metrics(
            {
                "denominator_health": {"active_actor_count": 2},
                "aggregate_rows": [
                    {"metric": "mean_ade", "horizon_s": None, "value": 1.0},
                    {"metric": None, "horizon_s": 1.0, "value": 2.0},
                    {"metric": "mean_fde", "horizon_s": 0.0, "value": 0.0},
                ],
            }
        )

        assert summary is not None
        assert set(summary["metrics"]) == {"mean_fde_0s"}
        assert summary["metrics"]["mean_fde_0s"]["value"] == 0.0


class TestNativePathEligibility:
    """Native live path detection should recognize the new baseline components."""

    def test_live_path_is_available(self) -> None:
        """The repository now exposes a selectable-variant baseline live path."""

        eligibility = check_native_live_path_eligibility()
        assert eligibility["live_path_available"] is True
        assert not eligibility["missing_components"]
        assert eligibility["probabilistic_predictor_implementations"]
        assert eligibility["forecast_variant_config_key_present"] is True

    def test_predictor_implementation_is_baseline(self) -> None:
        """The registered predictor should be the baseline implementation."""

        eligibility = check_native_live_path_eligibility()
        assert (
            "robot_sf.nav.baseline_probabilistic_predictor.BaselineProbabilisticPredictor"
            in (eligibility["probabilistic_predictor_implementations"])
        )


class TestRunClassification:
    """Fail-closed classification of gate runs."""

    def test_current_repo_state_is_native(self, dense_trace: SimulationTraceExport) -> None:
        """The cv forecast now flows into a replay policy and differs from none."""

        report = run_live_forecast_replay_gate(dense_trace)
        classification = classify_live_forecast_replay_run(report)
        assert classification == RUN_CLASSIFICATION_NATIVE

    def test_blocked_when_missing_components(self) -> None:
        """Missing native components classify the run as blocked."""

        report = {
            "native_path_eligibility": {
                "live_path_available": False,
                "missing_components": ["missing predictor"],
            },
            "variant_results": {},
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_BLOCKED

    def test_diagnostic_only_when_no_closed_loop_metrics(self) -> None:
        """No closed-loop metrics classify the run as diagnostic_only."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": {"collision": False}},
                "cv": {"closed_loop_metrics": None},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_DIAGNOSTIC_ONLY

    def test_diagnostic_only_when_metrics_have_wrong_shape(self) -> None:
        """Malformed metric maps should fail closed instead of raising."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": ["not", "a", "dict"]},
                "cv": {"closed_loop_metrics": {"collision": False}},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_DIAGNOSTIC_ONLY

    def test_degraded_when_cv_matches_none(self) -> None:
        """Identical none and cv closed-loop metrics classify the run as degraded."""

        metrics = {"collision": False, "progress_m": 1.0}
        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": metrics},
                "cv": {"closed_loop_metrics": metrics},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_DEGRADED

    def test_replay_policy_only_difference_is_degraded(self) -> None:
        """Provenance-only replay policy differences should not promote native."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": {"collision": False, "progress_m": 1.0}},
                "cv": {
                    "closed_loop_metrics": {
                        "collision": False,
                        "progress_m": 1.0,
                        "replay_policy": "forecast_brake_replay",
                    }
                },
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_DEGRADED

    def test_bool_and_numeric_metric_values_are_not_equivalent(self) -> None:
        """A boolean metric should not compare equal to an integer metric."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": {"collision": True}},
                "cv": {"closed_loop_metrics": {"collision": 1}},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_NATIVE

    def test_degraded_tolerates_tiny_float_jitter(self) -> None:
        """Microscopic metric jitter should not promote an unchanged run to native."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": {"collision": False, "progress_m": 1.0}},
                "cv": {"closed_loop_metrics": {"collision": False, "progress_m": 1.0 + 1e-12}},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_DEGRADED

    def test_native_when_cv_differs_from_none(self) -> None:
        """Different cv closed-loop metrics classify the run as native."""

        report = {
            "native_path_eligibility": {
                "live_path_available": True,
                "missing_components": [],
            },
            "variant_results": {
                "none": {"closed_loop_metrics": {"collision": False, "progress_m": 1.0}},
                "cv": {"closed_loop_metrics": {"collision": True, "progress_m": 0.5}},
            },
        }
        assert classify_live_forecast_replay_run(report) == RUN_CLASSIFICATION_NATIVE


class TestVariantForecastBatch:
    """Forecast batch construction for each variant."""

    @pytest.mark.parametrize(
        "variant", [variant for variant in FORECAST_VARIANTS if variant != "none"]
    )
    def test_build_batch_produces_valid_artifact(
        self, dense_trace: SimulationTraceExport, variant: str
    ) -> None:
        """Each variant should produce a valid ForecastBatch.v1 artifact."""

        batch = build_variant_forecast_batch(dense_trace, variant)
        assert batch.schema_version == "ForecastBatch.v1"
        assert batch.provenance.scenario_id == dense_trace.source.scenario_id
        assert batch.provenance.seed == dense_trace.source.seed
        assert batch.provenance.predictor_family == ("none" if variant == "none" else variant)
        expected_actor_ids = {
            str(pedestrian["id"])
            for frame in dense_trace.frames
            for pedestrian in frame.pedestrians
        }
        assert len(batch.forecasts) == len(expected_actor_ids)
        assert {forecast.actor_id for forecast in batch.forecasts} == expected_actor_ids

    def test_none_variant_has_no_forecast_batch(self, dense_trace: SimulationTraceExport) -> None:
        """The none variant is represented by integrated no-forecast replay metrics."""

        with pytest.raises(LiveForecastReplayGateError, match="integrated no-forecast baseline"):
            build_variant_forecast_batch(dense_trace, "none")

    def test_unsupported_variant_raises(self, dense_trace: SimulationTraceExport) -> None:
        """Unsupported variants should raise a gate-specific error."""

        with pytest.raises(LiveForecastReplayGateError, match="unsupported forecast variant"):
            build_variant_forecast_batch(dense_trace, "unknown_variant")

    def test_risk_filtered_includes_relevance_metadata(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """Risk-filtered batch should carry relevance metadata."""

        batch = build_variant_forecast_batch(dense_trace, "risk_filtered")
        for forecast in batch.forecasts:
            summary = forecast.occupancy_summary or {}
            assert summary.get("model") == "risk_filtered_cv"
            assert summary.get("relevance_status") in {
                "robot_unavailable",
                "collision_relevant",
                "filtered_low_relevance",
            }
            assert set(summary.get("relevance_status_by_horizon_s", {})) == {
                "0.5",
                "1",
                "2",
            }


class TestBaselineClosedLoopMetrics:
    """Baseline closed-loop metric extraction from a trace."""

    def test_required_keys_present(self, dense_trace: SimulationTraceExport) -> None:
        """Baseline metrics must include every required metric key."""

        metrics = compute_baseline_closed_loop_metrics(dense_trace)
        for key in REQUIRED_METRICS:
            assert (
                key in metrics
                or f"{key}_m" in metrics
                or f"{key}_s" in metrics
                or f"{key}_steps" in metrics
            )

    def test_min_distance_is_finite(self, dense_trace: SimulationTraceExport) -> None:
        """Min distance should be a finite number for a populated trace."""

        metrics = compute_baseline_closed_loop_metrics(dense_trace)
        assert metrics["min_distance_m"] is not None
        assert np.isfinite(metrics["min_distance_m"])

    def test_runtime_is_non_negative(self, dense_trace: SimulationTraceExport) -> None:
        """Runtime should be non-negative."""

        metrics = compute_baseline_closed_loop_metrics(dense_trace)
        assert metrics["runtime_s"] >= 0.0

    def test_empty_trace_raises(self) -> None:
        """An empty trace should raise a gate error."""

        empty = SimulationTraceExport(
            schema_version="simulation_trace_export.v1",
            trace_id="empty",
            source=type(
                "Source",
                (),
                {
                    "scenario_id": "s",
                    "seed": 0,
                    "planner_id": "p",
                    "episode_id": "e",
                    "generated_by": "test",
                },
            )(),
            evidence_boundary="analysis_workbench_only",
            coordinate_frame="world",
            units={"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
            frames=[],
        )
        with pytest.raises(LiveForecastReplayGateError, match="no frames"):
            compute_baseline_closed_loop_metrics(empty)


class TestGateReport:
    """End-to-end gate report classification and formatting."""

    def test_report_classification_is_native(self, dense_trace: SimulationTraceExport) -> None:
        """With the baseline replay path the dense smoke is native."""

        report = run_live_forecast_replay_gate(dense_trace)
        assert report["status"] == RUN_CLASSIFICATION_NATIVE
        assert report["classification"] == report["status"]
        assert report["classification_reason"]
        assert report["full_matrix_expansion_recommended"] is True

    def test_report_includes_smoke_variants(self, dense_trace: SimulationTraceExport) -> None:
        """The default report should include only the smoke variants."""

        report = run_live_forecast_replay_gate(dense_trace)
        assert set(report["variant_results"]) == set(SMOKE_FORECAST_VARIANTS)
        assert report["provenance"]["variants"] == list(SMOKE_FORECAST_VARIANTS)
        assert report["variant_results"]["none"]["forecast_metrics_status"] == "not_applicable"
        assert report["variant_results"]["none"]["closed_loop_metric_source"] == (
            "no_forecast_replay"
        )
        assert report["variant_results"]["cv"]["closed_loop_metric_source"] == (
            "forecast_brake_replay"
        )

    def test_report_can_run_full_matrix(self, dense_trace: SimulationTraceExport) -> None:
        """Explicit full-matrix variants should include all five variants."""

        report = run_live_forecast_replay_gate(dense_trace, variants=FORECAST_VARIANTS)
        assert set(report["variant_results"]) == set(FORECAST_VARIANTS)
        assert report["provenance"]["variants"] == list(FORECAST_VARIANTS)

    def test_report_markdown_contains_classification(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """Markdown output should include classification and gating signal."""

        report = run_live_forecast_replay_gate(dense_trace)
        markdown = format_live_forecast_replay_gate_markdown(report)
        assert "Claim Boundary" in markdown
        assert "Classification" in markdown
        assert "Native Path Eligibility" in markdown
        assert "Recorded Trace Closed-Loop Metrics" in markdown
        assert "Variant Results" in markdown
        assert "Limitations" in markdown
        assert str(report["classification"]) in markdown

    def test_report_claim_boundary_is_fail_closed(self, dense_trace: SimulationTraceExport) -> None:
        """The claim boundary must not overclaim benchmark evidence."""

        report = run_live_forecast_replay_gate(dense_trace)
        boundary = report["claim_boundary"].lower()
        assert "native" in boundary or "blocked" in boundary or "does not" in boundary
        assert "does not claim" in boundary

    def test_config_overrides_horizons(self, dense_trace: SimulationTraceExport) -> None:
        """Custom horizons should propagate into the report provenance."""

        config = LiveForecastReplayGateConfig(horizons_s=(0.5, 1.0))
        report = run_live_forecast_replay_gate(dense_trace, config=config)
        assert report["provenance"]["horizons_s"] == [0.5, 1.0]

    def test_no_feasible_horizons_fails_closed(self, dense_trace: SimulationTraceExport) -> None:
        """Gate should not emit forecast metrics without future labels."""

        config = LiveForecastReplayGateConfig(horizons_s=(99.0,))
        with pytest.raises(LiveForecastReplayGateError, match="no requested forecast horizons"):
            run_live_forecast_replay_gate(dense_trace, config=config)

    def test_empty_variant_set_fails_closed(self, dense_trace: SimulationTraceExport) -> None:
        """An empty variant set should fail closed."""

        with pytest.raises(LiveForecastReplayGateError, match="at least one forecast variant"):
            run_live_forecast_replay_gate(dense_trace, variants=())

    def test_unsupported_variant_fails_closed(self, dense_trace: SimulationTraceExport) -> None:
        """Unsupported variants should fail closed at the gate entry."""

        with pytest.raises(LiveForecastReplayGateError, match="unsupported forecast variant"):
            run_live_forecast_replay_gate(dense_trace, variants=("none", "unknown_variant"))

    def test_required_closed_loop_metrics_present_for_smoke_variants(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """Both none and cv must carry the required closed-loop metrics."""

        report = run_live_forecast_replay_gate(dense_trace)
        for variant in SMOKE_FORECAST_VARIANTS:
            metrics = report["variant_results"][variant]["closed_loop_metrics"]
            for key in REQUIRED_METRICS:
                assert (
                    key in metrics
                    or f"{key}_m" in metrics
                    or f"{key}_s" in metrics
                    or f"{key}_steps" in metrics
                )


class TestForecastBrakeReplay:
    """Forecast-aware brake replay produces per-variant closed-loop metrics."""

    def test_none_replay_uses_integrated_no_forecast_policy(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """The none variant should use the same replay surface without forecast braking."""

        config = LiveForecastReplayGateConfig()
        replay_metrics = run_variant_closed_loop_replay(dense_trace, "none", config)
        assert replay_metrics["replay_policy"] == "no_forecast_replay"
        assert replay_metrics["runtime_s"] == pytest.approx(
            dense_trace.frames[-1].time_s - dense_trace.frames[0].time_s
        )

    def test_cv_replay_differs_from_none(self, dense_trace: SimulationTraceExport) -> None:
        """The cv replay should produce closed-loop metrics that differ from none."""

        config = LiveForecastReplayGateConfig()
        none_metrics = run_variant_closed_loop_replay(dense_trace, "none", config)
        cv_metrics = run_variant_closed_loop_replay(dense_trace, "cv", config)
        assert cv_metrics["replay_policy"] == "forecast_brake_replay"
        assert cv_metrics != none_metrics

    def test_no_brake_cv_replay_matches_none_for_classification(
        self, dense_trace: SimulationTraceExport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Classification should not become native from integration mismatch alone."""

        monkeypatch.setattr(
            live_forecast_replay_gate_module,
            "_forecast_brake_needed",
            lambda prediction, robot_position, collision_distance_m: False,
        )
        config = LiveForecastReplayGateConfig()
        none_metrics = run_variant_closed_loop_replay(dense_trace, "none", config)
        cv_metrics = run_variant_closed_loop_replay(dense_trace, "cv", config)
        cv_metrics["replay_policy"] = none_metrics["replay_policy"]

        assert cv_metrics == none_metrics

    def test_replay_uses_frame_time_deltas(self, dense_trace: SimulationTraceExport) -> None:
        """Nonuniform frame times should still produce a finite replay."""

        frames = list(dense_trace.frames)
        frames[1] = replace(frames[1], time_s=frames[0].time_s + 0.2)
        trace = replace(dense_trace, frames=frames)

        metrics = run_variant_closed_loop_replay(trace, "cv", LiveForecastReplayGateConfig())

        assert metrics["replay_policy"] == "forecast_brake_replay"
        assert metrics["runtime_s"] == pytest.approx(
            trace.frames[-1].time_s - trace.frames[0].time_s
        )

    def test_forecast_brake_triggers_when_pedestrian_predicted_close(self) -> None:
        """Brake should trigger when a predicted mean is within the threshold."""

        from robot_sf.nav.predictive_types import ProbabilisticPrediction, TrajectoryDistribution

        mean = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float32)
        prediction = ProbabilisticPrediction(
            predictions=[TrajectoryDistribution(mean=mean, pedestrian_id=0)],
            prediction_horizon=0.3,
            prediction_dt=0.1,
        )
        assert _forecast_brake_needed(prediction, np.array([0.0, 0.0]), 0.15) is True
        assert _forecast_brake_needed(prediction, np.array([1.0, 1.0]), 0.15) is False
