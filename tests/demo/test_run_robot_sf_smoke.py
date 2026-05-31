"""Tests for the local Robot SF smoke benchmark demo."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scripts.demo import run_robot_sf_smoke as smoke

if TYPE_CHECKING:
    from pathlib import Path


def _episode(algorithm: str, *, success: float = 1.0, collisions: float = 0.0) -> dict[str, Any]:
    """Build a minimal aggregate-compatible episode record."""
    return {
        "episode_id": f"demo::{algorithm}::1",
        "scenario_id": "planner_sanity_simple",
        "seed": 0,
        "algo": algorithm,
        "scenario_params": {"algo": algorithm},
        "metrics": {
            "success": success,
            "collisions": collisions,
            "near_misses": 0.0,
            "time_to_goal_norm": 0.5,
            "path_efficiency": 1.0,
        },
    }


def test_run_demo_writes_summary_and_report(monkeypatch, tmp_path: Path) -> None:
    """The demo writes machine and Markdown summaries for two local planners."""
    calls: list[dict[str, Any]] = []

    def fake_run_batch(
        scenarios_or_path: Path,
        out_path: Path,
        schema_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del scenarios_or_path, schema_path
        algorithm = str(kwargs["algo"])
        calls.append(
            {"algorithm": algorithm, "horizon": kwargs["horizon"], "workers": kwargs["workers"]}
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_episode(algorithm)) + "\n", encoding="utf-8")
        return {"wrote": 1, "failures": [], "total_jobs": 1}

    monkeypatch.setattr(smoke, "run_batch", fake_run_batch)

    output_root = tmp_path / "smoke_benchmark"
    result = smoke.run_demo(output_root=output_root, planners=("simple_policy", "social_force"))

    assert result["passed"] is True
    assert result["output_root"] == str(output_root)
    assert result["claim_boundary"] == smoke.CLAIM_BOUNDARY
    assert [row["planner"] for row in result["planners"]] == ["simple_policy", "social_force"]
    assert calls == [
        {"algorithm": "simple_policy", "horizon": smoke.DEFAULT_HORIZON, "workers": 1},
        {"algorithm": "social_force", "horizon": smoke.DEFAULT_HORIZON, "workers": 1},
    ]

    payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot_sf_smoke_demo.v1"
    assert payload["passed"] is True
    assert payload["aggregate_summary"]["simple_policy"]["success"]["mean"] == 1.0

    report = (output_root / "report.md").read_text(encoding="utf-8")
    assert "Robot SF Smoke Benchmark Demo" in report
    assert "not durable benchmark evidence" in report
    assert "simple_policy" in report
    assert "social_force" in report


def test_run_demo_records_failed_planner_without_crashing(monkeypatch, tmp_path: Path) -> None:
    """A failed planner row is preserved in the summary and report."""

    def fake_run_batch(
        scenarios_or_path: Path,
        out_path: Path,
        schema_path: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del scenarios_or_path, out_path, schema_path
        if kwargs["algo"] == "social_force":
            raise RuntimeError("planner unavailable")
        return {"wrote": 0, "failures": [], "total_jobs": 1}

    monkeypatch.setattr(smoke, "run_batch", fake_run_batch)

    output_root = tmp_path / "smoke_benchmark"
    result = smoke.run_demo(output_root=output_root, planners=("simple_policy", "social_force"))

    assert result["passed"] is False
    assert result["planners"][0]["status"] == "failed"
    assert result["planners"][1]["status"] == "failed"
    assert "planner unavailable" in result["planners"][1]["error"]

    report = (output_root / "report.md").read_text(encoding="utf-8")
    assert "failed" in report
    assert "planner unavailable" in report
