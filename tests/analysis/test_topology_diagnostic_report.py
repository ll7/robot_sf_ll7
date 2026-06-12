"""Tests for topology diagnostic report snapshot."""

from __future__ import annotations

from pathlib import Path

from robot_sf.analysis.topology_diagnostic_report import (
    aggregate_traces,
    build_report_payload,
    render_markdown,
)
from scripts.tools.build_topology_diagnostic_report import main as report_main

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

SUMMARY_KEYS = [
    "claim_boundary",
    "trace_count",
    "total_steps",
    "diagnostic_status_counts",
    "selected_hypothesis_counts",
    "near_parity_gate_reason_counts",
    "reuse_penalty",
    "route_progress_deltas",
    "top_regressions",
    "top_unchanged",
    "terminal_outcome_counts",
]

REUSE_PENALTY_KEYS = [
    "applied_steps",
    "eligible_near_parity_alternative_steps",
    "reason_counts",
]

PAYLOAD_KEYS = [
    "report_kind",
    "label",
    "claim_boundary",
    "source_trace_count",
    "source_trace_paths",
]

MARKDOWN_SECTIONS = [
    "## Selected Hypotheses",
    "## Near-Parity Gate Reasons",
    "## Reuse-Penalty Activations",
    "## Route-Progress Deltas",
    "## Top Regressions",
    "## Top Unchanged Cases",
]


def _minimal_trace(
    *,
    selected_hypothesis: str | None = "primary_route",
    near_parity_gate_reason: str | None = "eligible_near_parity_alternative",
    progress_delta: float | None = 1.5,
    reuse_penalty_applied: bool = False,
    terminal_outcome: str = "success",
    diagnostic_status: str = "diagnostic_complete",
) -> dict:
    """Build a minimal topology diagnostic trace payload for testing."""
    step = {
        "step": 0,
        "topology_status": "ok",
        "hypothesis_count": 2,
        "topology_instrumentation": {
            "per_frame_hypothesis_count": 2,
            "alternative_hypothesis_count": 1,
            "selected_hypothesis": selected_hypothesis,
            "near_parity_gate_reason": near_parity_gate_reason,
        },
        "planner_route_corridor": {
            "topology_reuse_penalty": {
                "reuse_penalty_applied": reuse_penalty_applied,
                "eligible_near_parity_alternative_exists": True,
                "reuse_penalty_reason": ("cooldown_eligible" if reuse_penalty_applied else None),
            },
        },
    }
    summary = {
        "route_selector_selected_hypothesis_counts": (
            {selected_hypothesis: 1} if selected_hypothesis else {}
        ),
        "selected_row_near_parity_gate_reasons": (
            {near_parity_gate_reason: 1} if near_parity_gate_reason else {}
        ),
        "hypothesis_progress_by_rank": {
            "0": {
                "samples": 1,
                "first_corridor_name": "alpha",
                "last_corridor_name": "alpha",
                "first_remaining_distance_m": 5.0,
                "last_remaining_distance_m": 3.5,
                "progress_delta_m": progress_delta,
                "min_static_clearance_m": 0.4,
                "min_dynamic_clearance_m": 0.3,
            },
        },
        "corrective_behavior": {
            "terminal_outcome": {"outcome": terminal_outcome},
        },
    }
    return {
        "diagnostic_kind": "topology_hypothesis_trace",
        "diagnostic_status": diagnostic_status,
        "claim_boundary": "diagnostic_only_not_benchmark_success",
        "scenario_id": "test_scenario",
        "seed": 42,
        "summary": summary,
        "steps": [step],
    }


def test_aggregate_summary_keys_present():
    """All expected summary keys must be present and stable."""
    trace = _minimal_trace()
    result = aggregate_traces([trace])
    for key in SUMMARY_KEYS:
        assert key in result, f"Missing summary key: {key}"
    assert result["claim_boundary"] == "diagnostic_only_not_benchmark_success"
    assert result["trace_count"] == 1
    assert result["total_steps"] == 1


def test_aggregate_reuse_penalty_keys_present():
    """Reuse-penalty sub-dict must contain expected keys."""
    trace = _minimal_trace(reuse_penalty_applied=True)
    result = aggregate_traces([trace])
    rp = result["reuse_penalty"]
    for key in REUSE_PENALTY_KEYS:
        assert key in rp, f"Missing reuse_penalty key: {key}"
    assert rp["applied_steps"] == 1
    assert rp["eligible_near_parity_alternative_steps"] >= 1
    assert "cooldown_eligible" in rp["reason_counts"]


def test_aggregate_selected_hypothesis_counts():
    """Selected hypotheses should be counted across traces."""
    t1 = _minimal_trace(selected_hypothesis="primary_route")
    t2 = _minimal_trace(selected_hypothesis="divert_left")
    result = aggregate_traces([t1, t2])
    counts = result["selected_hypothesis_counts"]
    assert counts.get("primary_route") == 1
    assert counts.get("divert_left") == 1


def test_aggregate_near_parity_gate_reasons():
    """Near-parity gate reasons should be counted."""
    t1 = _minimal_trace(near_parity_gate_reason="eligible_near_parity_alternative")
    t2 = _minimal_trace(near_parity_gate_reason="route_distance_exceeds_slack")
    result = aggregate_traces([t1, t2])
    reasons = result["near_parity_gate_reason_counts"]
    assert reasons.get("eligible_near_parity_alternative") == 1
    assert reasons.get("route_distance_exceeds_slack") == 1


def test_aggregate_falls_back_to_step_counts_when_summary_omits_counts():
    """Step-level selected and gate fields are used when summary counters are absent."""
    trace = _minimal_trace(
        selected_hypothesis="divert_left",
        near_parity_gate_reason="static_clearance_below_floor",
    )
    trace["summary"].pop("route_selector_selected_hypothesis_counts")
    trace["summary"].pop("selected_row_near_parity_gate_reasons")

    result = aggregate_traces([trace])

    assert result["selected_hypothesis_counts"] == {"divert_left": 1}
    assert result["near_parity_gate_reason_counts"] == {"static_clearance_below_floor": 1}


def test_aggregate_route_progress_deltas():
    """Route-progress deltas should be extracted from hypothesis progress."""
    t = _minimal_trace(progress_delta=2.0)
    result = aggregate_traces([t])
    assert len(result["route_progress_deltas"]) == 1
    assert result["route_progress_deltas"][0]["progress_delta_m"] == 2.0


def test_aggregate_top_regressions_sorted():
    """Negative progress deltas should appear in top_regressions."""
    t = _minimal_trace(progress_delta=-3.0)
    result = aggregate_traces([t])
    assert len(result["top_regressions"]) == 1
    assert result["top_regressions"][0]["progress_delta_m"] == -3.0


def test_aggregate_top_unchanged_cases():
    """Zero progress deltas should appear in top_unchanged."""
    t = _minimal_trace(progress_delta=0.0)
    result = aggregate_traces([t])
    assert len(result["top_unchanged"]) == 1
    assert result["top_unchanged"][0]["progress_delta_m"] == 0.0


def test_build_report_payload_keys():
    """Report payload must contain top-level report keys."""
    trace_path = FIXTURES / "topology_sample.jsonl"
    payload = build_report_payload([trace_path], label="test-snapshot")
    for key in PAYLOAD_KEYS:
        assert key in payload, f"Missing payload key: {key}"
    assert payload["report_kind"] == "topology_diagnostic_report_snapshot"
    assert payload["label"] == "test-snapshot"
    assert payload["source_trace_count"] == 1


def test_render_markdown_contains_all_sections():
    """Markdown output must contain all expected section headers."""
    trace = _minimal_trace()
    agg = aggregate_traces([trace])
    payload = {"label": "test", **agg}
    md = render_markdown(payload)
    for section in MARKDOWN_SECTIONS:
        assert section in md, f"Missing Markdown section: {section}"
    assert "diagnostic_only_not_benchmark_success" in md


def test_render_markdown_handles_empty_data():
    """Markdown rendering should not crash with empty aggregation."""
    agg = aggregate_traces([])
    payload = {"label": "empty", **agg}
    md = render_markdown(payload)
    assert "(none)" in md
    assert "## Selected Hypotheses" in md


def test_build_report_payload_missing_file():
    """Missing trace files should be handled gracefully."""
    payload = build_report_payload([Path("/nonexistent/trace.jsonl")], label="missing")
    assert payload["trace_count"] == 0
    assert payload["total_steps"] == 0


def test_build_report_payload_skips_directory_inputs(tmp_path: Path):
    """Directory paths should not crash report aggregation."""
    payload = build_report_payload([tmp_path], label="directory")
    assert payload["trace_count"] == 0
    assert payload["total_steps"] == 0


def test_build_report_payload_accepts_uppercase_suffix(tmp_path: Path):
    """Uppercase JSON suffixes should load like lowercase suffixes."""
    trace_path = tmp_path / "trace.JSON"
    trace_path.write_text(json_dumps(_minimal_trace()), encoding="utf-8")

    payload = build_report_payload([trace_path], label="uppercase")

    assert payload["trace_count"] == 1
    assert payload["selected_hypothesis_counts"] == {"primary_route": 1}


def test_jsonl_scalar_rows_are_ignored(tmp_path: Path):
    """JSONL scalar rows should not be passed to trace aggregation."""
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("42\n" + json_dumps(_minimal_trace()) + "\n", encoding="utf-8")

    payload = build_report_payload([trace_path], label="jsonl")

    assert payload["trace_count"] == 1
    assert payload["total_steps"] == 1


def test_null_summary_counters_are_treated_as_zero():
    """Explicit null summary counters should not crash aggregation."""
    trace = _minimal_trace()
    trace["steps"] = []
    trace["summary"]["topology_reuse_penalty"] = {
        "applied_steps": None,
        "eligible_near_parity_alternative_steps": None,
        "reason_counts": {"missing": None},
    }

    result = aggregate_traces([trace])

    assert result["reuse_penalty"]["applied_steps"] == 0
    assert result["reuse_penalty"]["eligible_near_parity_alternative_steps"] == 0
    assert result["reuse_penalty"]["reason_counts"] == {"missing": 0}


def test_cli_rejects_directory_inputs(tmp_path: Path, capsys):
    """CLI validation should reject directories before reading them."""
    rc = report_main([str(tmp_path)])

    assert rc == 2
    assert "File not found" in capsys.readouterr().err


def json_dumps(payload: dict) -> str:
    """Serialize test payloads without importing json at module top level."""
    import json

    return json.dumps(payload)
