"""Tests for paired safety and comfort release-gate reporting."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.release_gates import (
    GateSpec,
    ReleaseGateSpecError,
    build_release_gate_report,
    evaluate_release_gates,
    format_release_gate_csv,
    format_release_gate_markdown,
    load_metric_rows,
    load_release_gate_spec,
    write_release_gate_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_gate_spec(path: Path, *, collision_threshold: float = 0.0) -> Path:
    payload = {
        "schema_version": "benchmark_release_gate_spec.v1",
        "gates": [
            {
                "id": "collision_rate_zero",
                "metric": "collision_rate",
                "threshold": collision_threshold,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture_provisional_not_certification",
                "required": True,
            },
            {
                "id": "proxemic_intrusion_limit",
                "metric": "proxemic_intrusion_rate",
                "threshold": 0.1,
                "direction": "max",
                "category": "comfort",
                "provenance": "fixture_provisional_not_certification",
                "required": True,
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_release_gate_matrix_reports_pass_fail_and_not_evaluable(tmp_path: Path) -> None:
    """Evaluator emits planner x family matrix with paired category statuses."""

    gates = load_release_gate_spec(_write_gate_spec(tmp_path / "gates.yaml"))
    rows = [
        {
            "planner_key": "safe_planner",
            "scenario_family": "classic_crossing",
            "collision_rate": 0.0,
            "proxemic_intrusion_rate": 0.05,
        },
        {
            "planner_key": "unsafe_planner",
            "scenario_family": "classic_crossing",
            "collision_rate": 0.2,
            "proxemic_intrusion_rate": 0.02,
        },
        {
            "planner_key": "partial_planner",
            "scenario_family": "classic_crossing",
            "collision_rate": 0.0,
        },
        {
            "planner_key": "bool_metric_planner",
            "scenario_family": "classic_crossing",
            "collision_rate": False,
            "proxemic_intrusion_rate": 0.05,
        },
    ]

    report = evaluate_release_gates(rows, gates)

    matrix = {(row["planner_key"], row["scenario_family"]): row for row in report["matrix_rows"]}
    assert matrix[("safe_planner", "classic_crossing")]["overall_status"] == "pass"
    assert matrix[("unsafe_planner", "classic_crossing")]["overall_status"] == "fail"
    assert (
        matrix[("unsafe_planner", "classic_crossing")]["failed_gate_ids"] == "collision_rate_zero"
    )
    assert matrix[("partial_planner", "classic_crossing")]["comfort_status"] == "not_evaluable"
    assert matrix[("partial_planner", "classic_crossing")]["overall_status"] == "not_evaluable"
    assert (
        matrix[("partial_planner", "classic_crossing")]["not_evaluable_gate_ids"]
        == "proxemic_intrusion_limit"
    )
    assert matrix[("bool_metric_planner", "classic_crossing")]["safety_status"] == "not_evaluable"
    assert (
        matrix[("bool_metric_planner", "classic_crossing")]["not_evaluable_gate_ids"]
        == "collision_rate_zero"
    )
    assert "not certification" in report["claim_boundary"]


def test_threshold_change_in_yaml_changes_result_without_code_change(tmp_path: Path) -> None:
    """Thresholds are config-owned: changing YAML alone changes report outcomes."""

    rows = [
        {
            "planner_key": "candidate",
            "scenario_family": "classic_crossing",
            "collision_rate": 0.05,
            "proxemic_intrusion_rate": 0.05,
        }
    ]
    strict_gates = load_release_gate_spec(_write_gate_spec(tmp_path / "strict.yaml"))
    relaxed_gates = load_release_gate_spec(
        _write_gate_spec(tmp_path / "relaxed.yaml", collision_threshold=0.1)
    )

    strict = evaluate_release_gates(rows, strict_gates)["matrix_rows"][0]
    relaxed = evaluate_release_gates(rows, relaxed_gates)["matrix_rows"][0]

    assert strict["overall_status"] == "fail"
    assert relaxed["overall_status"] == "pass"


def test_duplicate_gate_ids_fail_closed(tmp_path: Path) -> None:
    """Duplicate gate IDs make the spec invalid instead of overwriting a gate."""

    path = _write_gate_spec(tmp_path / "gates.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["gates"].append(dict(payload["gates"][0]))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ReleaseGateSpecError, match="duplicate release gate id"):
        load_release_gate_spec(path)


def test_report_build_write_and_format_helpers_cover_success_paths(tmp_path: Path) -> None:
    """In-process report helpers write provenance-bearing JSON, CSV, and Markdown."""

    gate_spec = _write_gate_spec(tmp_path / "gates.yaml")
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            [
                {
                    "planner_key": "safe_planner",
                    "scenario_family": "classic_crossing",
                    "collision_rate": 0.0,
                    "proxemic_intrusion_rate": 0.05,
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = load_metric_rows(rows_path)
    gates = load_release_gate_spec(gate_spec)

    report = build_release_gate_report(
        rows,
        gates,
        input_path=rows_path,
        gate_spec_path=gate_spec,
        command="fixture command",
        generated_at_utc="2026-07-02T00:00:00+00:00",
    )
    paths = write_release_gate_report(
        report,
        json_path=tmp_path / "summary.json",
        csv_path=tmp_path / "matrix.csv",
        markdown_path=tmp_path / "README.md",
    )

    assert paths["json"].is_file()
    assert paths["csv"].read_text(encoding="utf-8") == format_release_gate_csv(report)
    markdown = format_release_gate_markdown(report)
    assert "No failed or not-evaluable gates." in markdown
    assert paths["markdown"].read_text(encoding="utf-8") == markdown
    assert report["provenance"]["input"]["sha256"]
    assert report["provenance"]["gate_spec"]["sha256"]


def test_load_metric_rows_accepts_known_containers_and_rejects_bad_inputs(tmp_path: Path) -> None:
    """Input loader accepts existing row containers and rejects ambiguous JSON."""

    for key in ("rows", "matrix_rows", "summary_rows", "aggregates", "summaries", "results"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({key: [{"planner": "goal"}]}), encoding="utf-8")
        assert load_metric_rows(path) == [{"planner": "goal"}]

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("42", encoding="utf-8")
    with pytest.raises(ReleaseGateSpecError, match="input JSON must be a list or mapping"):
        load_metric_rows(scalar_path)

    missing_rows_path = tmp_path / "missing_rows.json"
    missing_rows_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(ReleaseGateSpecError, match="must contain one of"):
        load_metric_rows(missing_rows_path)

    bad_row_path = tmp_path / "bad_row.json"
    bad_row_path.write_text(json.dumps({"rows": ["not a mapping"]}), encoding="utf-8")
    with pytest.raises(ReleaseGateSpecError, match="input row 0 must be a mapping"):
        load_metric_rows(bad_row_path)


def test_spec_validation_rejects_malformed_gate_entries(tmp_path: Path) -> None:
    """Gate spec validation fails closed for malformed gate definitions."""

    not_mapping = tmp_path / "not_mapping.yaml"
    not_mapping.write_text("- nope\n", encoding="utf-8")
    with pytest.raises(ReleaseGateSpecError, match="must be a YAML mapping"):
        load_release_gate_spec(not_mapping)

    empty_gates = tmp_path / "empty.yaml"
    empty_gates.write_text("schema_version: benchmark_release_gate_spec.v1\ngates: []\n")
    with pytest.raises(ReleaseGateSpecError, match="non-empty gates list"):
        load_release_gate_spec(empty_gates)

    bad_item = tmp_path / "bad_item.yaml"
    bad_item.write_text(
        "schema_version: benchmark_release_gate_spec.v1\ngates:\n  - not-a-mapping\n",
        encoding="utf-8",
    )
    with pytest.raises(ReleaseGateSpecError, match=r"gates\[0\] must be a mapping"):
        load_release_gate_spec(bad_item)

    cases = [
        ("missing", {"id": "g"}, "missing required fields"),
        (
            "empty_id",
            {
                "id": " ",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
            },
            "must be a non-empty string",
        ),
        (
            "direction",
            {
                "id": "g",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "sideways",
                "category": "safety",
                "provenance": "fixture",
            },
            "direction must be one of",
        ),
        (
            "category",
            {
                "id": "g",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "max",
                "category": "quality",
                "provenance": "fixture",
            },
            "category must be safety or comfort",
        ),
        (
            "threshold",
            {
                "id": "g",
                "metric": "collision_rate",
                "threshold": ".nan",
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
            },
            "threshold must",
        ),
        (
            "boolean_threshold",
            {
                "id": "g",
                "metric": "collision_rate",
                "threshold": True,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
            },
            "threshold must",
        ),
        (
            "scope",
            {
                "id": "g",
                "metric": "collision_rate",
                "threshold": 0.0,
                "direction": "max",
                "category": "safety",
                "provenance": "fixture",
                "scope": "classic_crossing",
            },
            "scope must be a mapping",
        ),
    ]
    for name, gate, message in cases:
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                {"schema_version": "benchmark_release_gate_spec.v1", "gates": [gate]},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with pytest.raises(ReleaseGateSpecError, match=message):
            load_release_gate_spec(path)


def test_scope_nested_rows_and_empty_gate_set_behaviors() -> None:
    """Evaluator supports scoped gates, nested metric keys, and empty-gate rejection."""

    gate = {
        "id": "nested_collision_rate_zero",
        "metric": "metrics.collision_rate",
        "threshold": 0.0,
        "direction": "max",
        "category": "safety",
        "provenance": "fixture",
        "scope": {"scenario_family": ["classic_crossing"]},
    }
    gates = [
        load_release_gate_spec_from_payload(
            {"schema_version": "benchmark_release_gate_spec.v1", "gates": [gate]}
        )[0]
    ]
    report = evaluate_release_gates(
        [
            {
                "algo": "scoped",
                "scenario_params": {"scenario_family": "classic_crossing"},
                "metrics": {"collision_rate": 0.0},
            },
            {
                "algo": "out_of_scope",
                "scenario_params": {"scenario_family": "hallway"},
                "metrics": {"collision_rate": 1.0},
            },
        ],
        gates,
    )

    matrix = {(row["planner_key"], row["scenario_family"]): row for row in report["matrix_rows"]}
    assert matrix[("scoped", "classic_crossing")]["safety_status"] == "pass"
    assert matrix[("scoped", "classic_crossing")]["comfort_status"] == "not_evaluable"
    assert matrix[("out_of_scope", "hallway")]["overall_status"] == "not_evaluable"

    with pytest.raises(ReleaseGateSpecError, match="at least one gate required"):
        evaluate_release_gates([], [])


def load_release_gate_spec_from_payload(payload: dict[str, object]) -> list:
    """Write a temporary-like payload through the real YAML loader."""

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        return load_release_gate_spec(handle.name)


def test_cli_writes_json_csv_and_markdown_outputs(tmp_path: Path) -> None:
    """CLI writes the JSON artifact, pass/fail matrix CSV, and Markdown detail report."""

    gate_spec = _write_gate_spec(tmp_path / "gates.yaml")
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "planner_key": "safe_planner",
                        "scenario_family": "classic_crossing",
                        "collision_rate": 0.0,
                        "proxemic_intrusion_rate": 0.05,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "gate_matrix.csv"
    md_path = tmp_path / "README.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_release_gate_report.py",
            "--input-json",
            str(rows_path),
            "--gate-spec",
            str(gate_spec),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--output-md",
            str(md_path),
            "--generated-at-utc",
            "2026-07-02T00:00:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "benchmark_release_gate_report.v1"
    assert report["provenance"]["gate_spec"]["sha256"]
    assert "safe_planner,classic_crossing,pass,pass,pass" in csv_path.read_text(encoding="utf-8")
    assert "Paired Safety And Comfort Release-Gate Matrix" in md_path.read_text(encoding="utf-8")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CURRENT_ROSTER_SPEC = (
    _REPO_ROOT / "configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml"
)
_CURRENT_ROSTER_INPUT = (
    _REPO_ROOT
    / "docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json"
)
_CURRENT_ROSTER_SUMMARY = (
    _REPO_ROOT / "docs/context/evidence/issue_4166_release_gates/current_roster/summary.json"
)


def test_load_metric_rows_accepts_planner_rows_container(tmp_path: Path) -> None:
    """Loader accepts the canonical camera-ready ``planner_rows`` container."""

    path = tmp_path / "campaign_summary.json"
    path.write_text(
        json.dumps({"campaign": {}, "planner_rows": [{"planner_key": "goal"}]}),
        encoding="utf-8",
    )
    assert load_metric_rows(path) == [{"planner_key": "goal"}]


def test_not_recorded_metric_and_degraded_row_fail_closed() -> None:
    """String ``nan`` degraded rows and required=false coverage gaps stay not_evaluable.

    A ``required: false`` gate on an unrecorded metric must report ``not_evaluable`` in
    detail without forcing the category to ``not_evaluable`` when the recorded required
    gate passes; a degraded all-``nan`` row must render fully ``not_evaluable``.
    """

    rows = [
        {"planner_key": "ok_planner", "collisions_mean": "0.05"},
        {"planner_key": "degraded_planner", "collisions_mean": "nan"},
    ]
    gates = [
        GateSpec(
            gate_id="mean_collisions_budget",
            metric="collisions_mean",
            threshold=0.10,
            direction="max",
            category="safety",
            provenance="provisional",
            required=True,
        ),
        GateSpec(
            gate_id="min_clearance_coverage_gap",
            metric="min_clearance_m",
            threshold=0.25,
            direction="min",
            category="safety",
            provenance="not_recorded",
            required=False,
        ),
    ]
    report = evaluate_release_gates(rows, gates)
    by_planner = {row["planner_key"]: row for row in report["results"]}

    ok_row = by_planner["ok_planner"]
    assert ok_row["safety_status"] == "pass"  # required gate passes; false gate does not gate it
    assert "min_clearance_coverage_gap" in ok_row["not_evaluable_gate_ids"]

    degraded_row = by_planner["degraded_planner"]
    assert degraded_row["safety_status"] == "not_evaluable"
    assert degraded_row["overall_status"] == "not_evaluable"


def test_shipped_current_roster_spec_loads_with_coverage_gates() -> None:
    """The shipped current-roster gate spec parses and marks coverage gaps optional."""

    gates = load_release_gate_spec(_CURRENT_ROSTER_SPEC)
    by_id = {gate.gate_id: gate for gate in gates}
    assert by_id["mean_collisions_budget"].required is True
    assert by_id["min_clearance_floor_coverage_gap"].required is False
    assert by_id["proxemic_intrusion_rate_coverage_gap"].required is False
    assert {gate.category for gate in gates} == {"safety", "comfort"}


def test_instrumented_gate_metrics_make_coverage_gaps_evaluable() -> None:
    """Recording min_clearance_m/proxemic_intrusion_rate flips coverage gaps evaluable (#4166).

    Once a campaign summary row carries the release-gate-aligned metric fields
    (`min_clearance_m`, `proxemic_intrusion_rate`) — which `_planner_report_row`
    now emits on `origin/main` after #4326/#4334 — the shipped current-roster
    coverage-gap gates must evaluate to pass/fail instead of ``not_evaluable``. A
    row that omits those fields must still fail closed as ``not_evaluable``.

    This is an end-to-end evaluability guard for issue #4166: it exercises
    ``evaluate_release_gates`` with explicit row values, so it is independent of
    how the emitter computes ``min_clearance_m`` (mean vs. worst-case) and stays
    valid against the worst-case implementation retained on main.
    """

    gates = load_release_gate_spec(_CURRENT_ROSTER_SPEC)
    rows = [
        # Instrumented row: both gate-aligned metrics recorded -> gates evaluable.
        {
            "planner_key": "instrumented",
            "scenario_family": "all",
            "collisions_mean": "0.02",
            "near_misses_mean": "1.0",
            "jerk_mean": "0.5",
            "comfort_exposure_mean": "0.02",
            "min_clearance_m": "0.40",  # >= 0.25 floor -> pass
            "proxemic_intrusion_rate": "0.20",  # > 0.10 limit -> fail
        },
        # Uninstrumented row: gate metrics absent -> coverage gaps stay not_evaluable.
        {
            "planner_key": "legacy",
            "scenario_family": "all",
            "collisions_mean": "0.02",
            "near_misses_mean": "1.0",
            "jerk_mean": "0.5",
            "comfort_exposure_mean": "0.02",
        },
    ]
    report = evaluate_release_gates(rows, gates)
    detail = {row["planner_key"]: row for row in report["results"]}

    instrumented_gates = {gate["gate_id"]: gate for gate in detail["instrumented"]["gate_results"]}
    assert instrumented_gates["min_clearance_floor_coverage_gap"]["status"] == "pass"
    assert instrumented_gates["proxemic_intrusion_rate_coverage_gap"]["status"] == "fail"

    legacy_gates = {gate["gate_id"]: gate for gate in detail["legacy"]["gate_results"]}
    assert legacy_gates["min_clearance_floor_coverage_gap"]["status"] == "not_evaluable"
    assert legacy_gates["proxemic_intrusion_rate_coverage_gap"]["status"] == "not_evaluable"


def test_current_roster_evidence_matches_retained_campaign() -> None:
    """Rebuilding from the retained campaign reproduces the committed evidence packet."""

    rows = load_metric_rows(_CURRENT_ROSTER_INPUT)
    gates = load_release_gate_spec(_CURRENT_ROSTER_SPEC)
    report = evaluate_release_gates(rows, gates)

    assert report["status_counts"] == {"pass": 2, "fail": 5, "not_evaluable": 1}
    by_planner = {row["planner_key"]: row for row in report["matrix_rows"]}
    # Degraded socnav_bench (all-nan metrics) must fail closed to not_evaluable.
    assert by_planner["socnav_bench"]["overall_status"] == "not_evaluable"
    assert by_planner["orca"]["overall_status"] == "pass"
    assert by_planner["goal"]["overall_status"] == "fail"

    committed = json.loads(_CURRENT_ROSTER_SUMMARY.read_text(encoding="utf-8"))
    assert committed["status_counts"] == report["status_counts"]
    assert committed["matrix_rows"] == report["matrix_rows"]
