"""Tests for the live same-seed forecast replay gate (issue #2902)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.analysis_workbench.simulation_trace_export import SimulationTraceExport
from robot_sf.benchmark.live_forecast_replay_gate import (
    FORECAST_VARIANTS,
    LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION,
    REQUIRED_METRICS,
    LiveForecastReplayGateConfig,
    LiveForecastReplayGateError,
    _summarize_forecast_metrics,
    build_variant_forecast_batch,
    check_native_live_path_eligibility,
    compute_baseline_closed_loop_metrics,
    format_live_forecast_replay_gate_markdown,
    load_trace_tolerant,
    run_live_forecast_replay_gate,
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

    def test_required_variants(self) -> None:
        """The five required forecast variants must be present and ordered."""

        assert FORECAST_VARIANTS == (
            "none",
            "cv",
            "semantic",
            "interaction_aware",
            "risk_filtered",
        )

    def test_required_metrics(self) -> None:
        """All issue-required metrics must be declared."""

        assert set(REQUIRED_METRICS) == {
            "collision",
            "near_miss",
            "min_distance",
            "stop_yield_timing",
            "progress",
            "false_positive_stops",
            "runtime",
        }

    def test_schema_version_constant(self) -> None:
        """Schema version should be stable."""

        assert LIVE_FORECAST_REPLAY_GATE_SCHEMA_VERSION == "LiveForecastReplayGate.v1"


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
    """Native live path detection should fail closed when components are missing."""

    def test_live_path_not_available(self) -> None:
        """The repository currently lacks a native selectable-variant live path."""

        eligibility = check_native_live_path_eligibility()
        assert eligibility["live_path_available"] is False
        assert eligibility["missing_components"]
        assert not eligibility["probabilistic_predictor_implementations"]
        assert eligibility["forecast_variant_config_key_present"] is False

    def test_missing_components_named(self) -> None:
        """Missing components should be explicitly listed."""

        eligibility = check_native_live_path_eligibility()
        missing = " ".join(eligibility["missing_components"])
        assert "ProbabilisticPredictor" in missing
        assert "forecast_variant" in missing


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
        """The none variant is represented by baseline closed-loop metrics."""

        with pytest.raises(LiveForecastReplayGateError, match="recorded closed-loop baseline"):
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

    def test_report_status_is_diagnostic_only(self, dense_trace: SimulationTraceExport) -> None:
        """Without a native live path the gate must be diagnostic-only."""

        report = run_live_forecast_replay_gate(dense_trace)
        assert report["status"] == "diagnostic_only"
        assert report["classification"] == "diagnostic_only"

    def test_report_includes_all_variants(self, dense_trace: SimulationTraceExport) -> None:
        """The report should include one result per forecast variant."""

        report = run_live_forecast_replay_gate(dense_trace)
        assert set(report["variant_results"]) == set(FORECAST_VARIANTS)
        assert report["variant_results"]["none"]["forecast_metrics_status"] == "not_applicable"
        assert report["variant_results"]["none"]["closed_loop_metric_source"] == (
            "baseline_recorded_trace"
        )

    def test_report_markdown_contains_key_sections(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """Markdown output should include claim boundary, eligibility, and metrics."""

        report = run_live_forecast_replay_gate(dense_trace)
        markdown = format_live_forecast_replay_gate_markdown(report)
        assert "Claim Boundary" in markdown
        assert "Native Path Eligibility" in markdown
        assert "Baseline Closed-Loop Metrics" in markdown
        assert "Variant Results" in markdown
        assert "Limitations" in markdown

    def test_report_claim_boundary_is_diagnostic_only(
        self, dense_trace: SimulationTraceExport
    ) -> None:
        """The claim boundary must not overclaim benchmark evidence."""

        report = run_live_forecast_replay_gate(dense_trace)
        boundary = report["claim_boundary"].lower()
        assert "diagnostic-only" in boundary
        assert "no native planner" in boundary or "does not" in boundary

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
