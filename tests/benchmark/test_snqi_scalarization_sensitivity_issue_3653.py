"""Tests for the issue #3653 SNQI scalarization-sensitivity export wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    SENSITIVITY_PREFLIGHT_BLOCKED,
    SENSITIVITY_PREFLIGHT_MALFORMED,
    SENSITIVITY_PREFLIGHT_READY,
    build_scalarization_sensitivity_report,
    classify_scalarization_sensitivity_inputs,
)

WEIGHTS = {
    "w_success": 0.19,
    "w_time": 0.09,
    "w_collisions": 0.10,
    "w_near": 0.31,
    "w_comfort": 0.18,
    "w_force_exceed": 0.07,
    "w_jerk": 0.05,
}

BASELINE = {
    "collisions": {"med": 1.0, "p95": 3.0},
    "near_misses": {"med": 2.0, "p95": 5.0},
    "comfort_exposure": {"med": 0.2, "p95": 0.8},
    "force_exceed_events": {"med": 1.0, "p95": 4.0},
    "jerk_mean": {"med": 0.2, "p95": 0.7},
}


def _record(planner: str, scenario: str, values: dict[str, float]) -> dict[str, object]:
    return {
        "planner_key": planner,
        "planner": planner,
        "scenario_id": scenario,
        "horizon": "h500",
        "metrics": values,
    }


def _metrics(
    *,
    success: float,
    time_to_goal_norm: float,
    collisions: float,
    near_misses: float,
    comfort_exposure: float,
    force_exceed_events: float = 0.0,
    jerk_mean: float = 0.1,
) -> dict[str, float]:
    return {
        "success": success,
        "time_to_goal_norm": time_to_goal_norm,
        "collisions": collisions,
        "near_misses": near_misses,
        "comfort_exposure": comfort_exposure,
        "force_exceed_events": force_exceed_events,
        "jerk_mean": jerk_mean,
    }


def _valid_records() -> list[dict[str, object]]:
    return [
        _record(
            "orca",
            "crossing",
            _metrics(
                success=1.0,
                time_to_goal_norm=0.9,
                collisions=0.0,
                near_misses=1.0,
                comfort_exposure=0.15,
                jerk_mean=0.10,
            ),
        ),
        _record(
            "ppo",
            "crossing",
            _metrics(
                success=1.0,
                time_to_goal_norm=0.7,
                collisions=0.0,
                near_misses=3.0,
                comfort_exposure=0.55,
                force_exceed_events=1.0,
                jerk_mean=0.35,
            ),
        ),
        _record(
            "orca",
            "overtake",
            _metrics(
                success=1.0,
                time_to_goal_norm=1.0,
                collisions=0.0,
                near_misses=0.0,
                comfort_exposure=0.20,
                jerk_mean=0.12,
            ),
        ),
        _record(
            "ppo",
            "overtake",
            _metrics(
                success=1.0,
                time_to_goal_norm=0.8,
                collisions=1.0,
                near_misses=2.0,
                comfort_exposure=0.45,
                force_exceed_events=2.0,
                jerk_mean=0.30,
            ),
        ),
    ]


def _write_fixture_inputs(
    tmp_path: Path, records: list[dict[str, object]]
) -> tuple[Path, Path, Path]:
    episodes = tmp_path / "episodes.jsonl"
    baseline = tmp_path / "baseline.json"
    weights = tmp_path / "weights.json"
    episodes.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    baseline.write_text(json.dumps(BASELINE), encoding="utf-8")
    weights.write_text(json.dumps(WEIGHTS), encoding="utf-8")
    return episodes, baseline, weights


def test_ready_preflight_and_report_exports_pareto_disagreement() -> None:
    """Valid fixtures produce diagnostic disagreement and Pareto report fields."""

    records = _valid_records()

    preflight = classify_scalarization_sensitivity_inputs(
        records, weights=WEIGHTS, baseline=BASELINE
    )
    assert preflight["status"] == SENSITIVITY_PREFLIGHT_READY
    assert preflight["ready"] is True

    report = build_scalarization_sensitivity_report(
        records,
        weights=WEIGHTS,
        baseline=BASELINE,
        input_provenance={"fixture": "tests/benchmark/test_snqi_scalarization_sensitivity.py"},
    )

    assert report["evidence_kind"] == "analysis_artifact_only"
    assert report["decision_disagreement"]["pairwise_reversal_count"] >= 0
    assert report["pareto_front"]["points"]
    assert {row["planner"] for row in report["planner_rows"]} == {"orca", "ppo"}
    assert set(report["weight_zero_ablation"]) == set(WEIGHTS)
    assert set(report["weight_sweep"]) == set(WEIGHTS)


def test_preflight_blocks_missing_required_normalized_input() -> None:
    """Missing normalized time input blocks preflight and direct export."""

    records = _valid_records()
    metrics = records[0]["metrics"]
    assert isinstance(metrics, dict)
    del metrics["time_to_goal_norm"]

    preflight = classify_scalarization_sensitivity_inputs(
        records, weights=WEIGHTS, baseline=BASELINE
    )

    assert preflight["status"] == SENSITIVITY_PREFLIGHT_BLOCKED
    assert any(issue["code"] == "missing_required_term" for issue in preflight["issues"])
    with pytest.raises(ValueError, match="time_to_goal_norm"):
        build_scalarization_sensitivity_report(records, weights=WEIGHTS, baseline=BASELINE)


def test_preflight_malformed_non_finite_required_input() -> None:
    """Non-finite required metric values are malformed, not exportable evidence."""

    records = _valid_records()
    metrics = records[1]["metrics"]
    assert isinstance(metrics, dict)
    metrics["near_misses"] = float("nan")

    preflight = classify_scalarization_sensitivity_inputs(
        records, weights=WEIGHTS, baseline=BASELINE
    )

    assert preflight["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(issue["code"] == "non_finite_required_term" for issue in preflight["issues"])


def test_preflight_malformed_out_of_range_normalized_input() -> None:
    """Finite normalized terms outside [0, 1] are malformed, not exportable."""

    records = _valid_records()
    metrics = records[0]["metrics"]
    assert isinstance(metrics, dict)
    metrics["time_to_goal_norm"] = 1.2

    preflight = classify_scalarization_sensitivity_inputs(
        records, weights=WEIGHTS, baseline=BASELINE
    )

    assert preflight["status"] == SENSITIVITY_PREFLIGHT_MALFORMED
    assert any(
        issue["code"] == "out_of_range_normalized_term" for issue in preflight["issues"]
    )
    with pytest.raises(ValueError, match=r"time_to_goal_norm.*outside \[0, 1\]"):
        build_scalarization_sensitivity_report(records, weights=WEIGHTS, baseline=BASELINE)


def test_cli_refuses_blocked_inputs_and_writes_only_preflight(tmp_path: Path) -> None:
    """Blocked inputs return exit code 2 and do not create export artifacts."""

    records = _valid_records()
    metrics = records[0]["metrics"]
    assert isinstance(metrics, dict)
    del metrics["time_to_goal_norm"]
    episodes, baseline, weights = _write_fixture_inputs(tmp_path, records)
    out_dir = tmp_path / "artifacts"
    preflight_out = tmp_path / "preflight.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/snqi_scalarization_sensitivity_export.py",
            "--episodes",
            str(episodes),
            "--baseline",
            str(baseline),
            "--weights",
            str(weights),
            "--output-dir",
            str(out_dir),
            "--preflight-out",
            str(preflight_out),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert preflight_out.exists()
    assert json.loads(preflight_out.read_text(encoding="utf-8"))["status"] == (
        SENSITIVITY_PREFLIGHT_BLOCKED
    )
    assert not out_dir.exists()


def test_cli_exports_report_ready_artifacts(tmp_path: Path) -> None:
    """Ready inputs write JSON, CSV, Markdown, and SVG diagnostic artifacts."""

    episodes, baseline, weights = _write_fixture_inputs(tmp_path, _valid_records())
    out_dir = tmp_path / "artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/snqi_scalarization_sensitivity_export.py",
            "--episodes",
            str(episodes),
            "--baseline",
            str(baseline),
            "--weights",
            str(weights),
            "--output-dir",
            str(out_dir),
            "--stem",
            "issue_3653",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "issue_3653.json").exists()
    assert (out_dir / "issue_3653_planner_rows.csv").exists()
    assert (out_dir / "issue_3653_decision_disagreement.csv").exists()
    assert (out_dir / "issue_3653.md").exists()
    assert (out_dir / "issue_3653_pareto.svg").exists()

    payload = json.loads((out_dir / "issue_3653.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "snqi_scalarization_sensitivity.v1"
    assert payload["pareto_front"]["points"]
    disagreement_csv = (out_dir / "issue_3653_decision_disagreement.csv").read_text(
        encoding="utf-8"
    )
    assert "base_snqi_vs_constraints_first" in disagreement_csv
    assert "not benchmark evidence" in disagreement_csv
