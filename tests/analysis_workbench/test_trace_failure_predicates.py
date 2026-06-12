"""Tests for trace failure predicate extraction and aggregate table rendering."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    aggregate_trace_failure_predicate_tables,
    extract_trace_failure_predicates,
    render_trace_failure_predicate_markdown,
)
from scripts.tools.build_trace_failure_predicate_tables import write_trace_failure_predicate_tables

TRACE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)


def _frame(
    step: int,
    time_s: float,
    *,
    robot: dict[str, object] | None = None,
    pedestrians: list[dict[str, object]] | None = None,
    action: tuple[float, float] = (0.2, 0.0),
    planner_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one synthetic frame payload."""
    robot_payload = robot or {
        "position": [0.0, 0.0],
        "heading": 0.0,
        "velocity": [0.0, 0.0],
    }
    planner_payload = {
        "selected_action": {
            "linear_velocity": action[0],
            "angular_velocity": action[1],
        },
        "event": "step",
    }
    if planner_extra:
        planner_payload.update(planner_extra)
    return {
        "step": step,
        "time_s": time_s,
        "robot": robot_payload,
        "pedestrians": pedestrians or [],
        "planner": planner_payload,
    }


def _build_trace(
    *,
    trace_id: str,
    scenario_id: str,
    seed: int,
    planner_id: str,
    frames: list[dict[str, object]],
):
    payload = load_simulation_trace_export(TRACE_FIXTURE_PATH).to_dict()
    payload["trace_id"] = trace_id
    payload["source"]["scenario_id"] = scenario_id
    payload["source"]["seed"] = seed
    payload["source"]["planner_id"] = planner_id
    payload["source"]["episode_id"] = f"{trace_id}-episode"
    payload["frames"] = frames
    return simulation_trace_export_from_dict(payload, source=TRACE_FIXTURE_PATH)


def _aggregate_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    return list(payload.get("rows", []))


def _find_row(
    rows: list[dict[str, object]],
    *,
    predicate_id: str,
) -> dict[str, object] | None:
    return next((row for row in rows if row.get("predicate_id") == predicate_id), None)


def test_aggregate_preserves_occlusion_not_available_row() -> None:
    """Missing occlusion metadata should emit and preserve `not_available` rows."""
    trace = _build_trace(
        trace_id="trace-missing-occlusion",
        scenario_id="scenario-occlusion",
        seed=11,
        planner_id="planner-a",
        frames=[
            _frame(
                0,
                0.0,
                robot={"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.1, 0.0),
                pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    payload = aggregate_trace_failure_predicate_tables([trace])
    row = _find_row(
        _aggregate_rows(payload),
        predicate_id="occlusion_triggered_near_miss",
    )

    assert row is not None
    assert row["validity_status"] == "not_available"
    assert row["severity"] == "not_available"
    assert row["predicate_count"] == 1
    assert row["trace_denominator"] == 1
    assert row["predicate_rate_per_trace"] == 1.0
    assert payload["summary"]["input_trace_count"] == 1


def test_aggregate_includes_zero_motion_timeout_predicate() -> None:
    """A timeout frame plus stationary motion should emit zero-motion predicate."""
    trace = _build_trace(
        trace_id="trace-zero-motion",
        scenario_id="scenario-zero-motion",
        seed=12,
        planner_id="planner-b",
        frames=[
            _frame(
                0,
                0.0,
                robot={"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.0, 0.0),
            ),
            _frame(
                1,
                0.1,
                robot={"position": [0.01, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                action=(0.0, 0.0),
                planner_extra={"event": "timeout"},
            ),
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="zero_motion_timeout_behavior",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["predicate_count"] == 1


def test_aggregate_includes_oscillatory_control_predicate() -> None:
    """Alternating angular signs should emit oscillatory-local-control."""
    trace = _build_trace(
        trace_id="trace-oscillation",
        scenario_id="scenario-oscillation",
        seed=13,
        planner_id="planner-c",
        frames=[
            _frame(0, 0.0, action=(0.2, 0.2)),
            _frame(1, 0.1, action=(0.2, -0.2)),
            _frame(2, 0.2, action=(0.2, 0.2)),
            _frame(3, 0.3, action=(0.2, -0.2)),
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="oscillatory_local_control",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["severity"] == "medium"


def test_aggregate_includes_bottleneck_deadlock_predicate() -> None:
    """Planner deadlock events should be preserved as aggregate predicate rows."""
    trace = _build_trace(
        trace_id="trace-bottleneck",
        scenario_id="scenario-bottleneck",
        seed=14,
        planner_id="planner-d",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="bottleneck_deadlock",
    )

    assert row is not None
    assert row["predicate_count"] == 1
    assert row["validity_status"] == "valid"


def test_aggregate_includes_clearance_critical_interaction_predicate() -> None:
    """Near-clearance observations should emit clearance-critical interaction predicates."""
    trace = _build_trace(
        trace_id="trace-clearance",
        scenario_id="scenario-clearance",
        seed=15,
        planner_id="planner-e",
        frames=[
            _frame(
                0,
                0.0,
                pedestrians=[{"id": "ped_1", "position": [0.1, 0.1], "velocity": [0.0, 0.0]}],
            )
        ],
    )
    row = _find_row(
        _aggregate_rows(aggregate_trace_failure_predicate_tables([trace])),
        predicate_id="clearance_critical_interaction",
    )

    assert row is not None
    assert row["validity_status"] == "valid"
    assert row["severity"] == "high"


def test_denominator_and_markdown_render_include_rate_fields() -> None:
    """Aggregate payload and markdown output should expose denominator and rate columns."""
    traces = [
        _build_trace(
            trace_id=f"trace-denom-{index}",
            scenario_id="scenario-denom",
            seed=16,
            planner_id="planner-f",
            frames=[
                _frame(
                    0,
                    0.0,
                    action=(0.1, 0.0),
                    pedestrians=[{"id": "ped_1", "position": [0.45, 0.0], "velocity": [0.0, 0.0]}],
                )
            ],
        )
        for index in range(2)
    ]
    payload = aggregate_trace_failure_predicate_tables(traces)
    row = _find_row(
        _aggregate_rows(payload),
        predicate_id="occlusion_triggered_near_miss",
    )

    assert row is not None
    assert row["trace_denominator"] == 2
    assert row["predicate_count"] == 2
    assert row["predicate_rate_per_trace"] == 1.0

    markdown = render_trace_failure_predicate_markdown(payload)
    assert "trace_denominator" in markdown
    assert "predicate_rate_per_trace" in markdown
    assert "scenario-denom" in markdown


def test_scenario_family_override_normalizes_group_without_rewriting_source() -> None:
    """Scenario-family overrides should label groups without hiding the source scenario."""
    trace = _build_trace(
        trace_id="trace-family-override",
        scenario_id="classic_bottleneck_medium",
        seed=17,
        planner_id="planner-g",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )

    single_trace_payload = extract_trace_failure_predicates(
        trace,
        scenario_family="bottleneck",
    )
    aggregate_payload = aggregate_trace_failure_predicate_tables(
        [trace],
        scenario_family="bottleneck",
    )

    assert single_trace_payload["source"]["scenario_id"] == "classic_bottleneck_medium"
    assert single_trace_payload["predicates"][0]["scenario_family"] == "bottleneck"
    row = _find_row(
        _aggregate_rows(aggregate_payload),
        predicate_id="bottleneck_deadlock",
    )
    assert row is not None
    assert row["scenario_family"] == "bottleneck"
    assert aggregate_payload["summary"]["input_trace_count"] == 1


def test_writer_persists_json_and_markdown_outputs(tmp_path: Path) -> None:
    """The CLI writer should persist denominator-aware JSON and diagnostic Markdown."""
    trace = _build_trace(
        trace_id="trace-writer",
        scenario_id="scenario-writer",
        seed=18,
        planner_id="planner-h",
        frames=[
            _frame(
                0,
                0.0,
                planner_extra={"event": "bottleneck_deadlock"},
            )
        ],
    )
    trace_path = tmp_path / "trace.json"
    json_output = tmp_path / "tables.json"
    markdown_output = tmp_path / "tables.md"
    trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    write_trace_failure_predicate_tables(
        traces=[trace_path],
        output_json=json_output,
        output_markdown=markdown_output,
    )

    assert '"trace_denominator"' in json_output.read_text(encoding="utf-8")
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "diagnostic-only" in markdown
    assert "trace_denominator" in markdown


def test_markdown_renderer_skips_invalid_rows_and_preserves_falsy_values() -> None:
    """Renderer should ignore malformed rows without replacing valid falsy identifiers."""
    markdown = render_trace_failure_predicate_markdown(
        {
            "schema_version": "trace_failure_predicates.v1",
            "table_kind": "aggregate_by_predicate_group",
            "summary": {"input_trace_count": 1},
            "rows": [
                None,
                "bad-row",
                {
                    "scenario_family": 0,
                    "planner_id": "",
                    "seed": 0,
                    "predicate_id": "zero_like",
                    "validity_status": "valid",
                    "severity": "low",
                    "predicate_count": 0,
                    "trace_denominator": 1,
                    "predicate_rate_per_trace": 0.0,
                },
            ],
        }
    )

    assert "| 0 |  | 0 | zero_like | valid | low | 0 | 1 | 0.0000 |" in markdown
    assert "bad-row" not in markdown
