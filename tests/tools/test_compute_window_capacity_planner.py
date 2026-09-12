"""Focused contract tests for the compute-window capacity planner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.tools.compute_window_capacity_planner import build_plan, main

FIXTURE = Path(__file__).parents[1] / "fixtures" / "compute_window_capacity_inventory.json"


def _packet() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _row(workload_id: str, *, lane: str = "cpu", transfer: int = 50) -> dict:
    return {
        "id": workload_id,
        "lane": lane,
        "status": "ready",
        "source_status": "fresh",
        "input_status": "complete",
        "authorization": "authorized",
        "execution_state": "not_running",
        "resource_demand": {
            name: int(name == lane) for name in ("cpu", "gpu", "carla", "host_pinned")
        },
        "estimates": {
            "wall_seconds": 10,
            "queue_seconds": {"expected": 10, "upper": 40},
            "transfer_bytes": transfer,
            "completion_probability": 0.9,
            "unlock_value": 50,
            "time_to_loss_seconds": 100,
            "later_executable": False,
        },
    }


def test_excludes_ineligible_rows_and_keeps_ranges_visible() -> None:
    result = build_plan(_packet(), explain=True)
    reports = {report["workload_id"]: report for report in result["workloads"]}

    assert result["status"] == "ready"
    assert {"base", "dependent"}.issubset(result["ordered_plan"])
    assert result["ordered_plan"].index("base") < result["ordered_plan"].index("dependent")
    assert "blocked" not in result["ordered_plan"]
    assert "running" not in result["ordered_plan"]
    assert "deadline_infeasible" in reports["late"]["feasibility"]["blocking_reasons"]
    assert "storage_infeasible" in reports["overflow"]["feasibility"]["blocking_reasons"]
    assert reports["dependent"]["uncertainty"]["queue"]["upper"] == 120.0
    assert reports["dependent"]["uncertainty"]["level"] == "high"
    assert reports["blocked"]["feasibility"]["eligible"] is False
    assert any(item["workload_id"] == "late" for item in result["explain"]["excluded"])


def test_resource_conflict_duplicate_and_missing_dependency_fail_closed() -> None:
    packet = _packet()
    packet["workloads"].extend(
        [
            _row("too_gpu", lane="gpu"),
            _row("missing_parent"),
            _row("duplicate"),
            _row("duplicate"),
        ]
    )
    packet["workloads"][6]["resource_demand"]["gpu"] = 2
    packet["dependencies"]["edges"].append(
        {"child": "missing_parent", "parent": "does_not_exist", "status": "required"}
    )
    reports = {report["workload_id"]: report for report in build_plan(packet)["workloads"]}

    assert "resource_infeasible" in reports["too_gpu"]["feasibility"]["blocking_reasons"]
    assert "missing_dependency" in reports["missing_parent"]["feasibility"]["blocking_reasons"]
    assert "duplicate" in reports["duplicate"]["feasibility"]["blocking_reasons"]
    assert "too_gpu" not in build_plan(packet)["ordered_plan"]


def test_plans_are_deterministic_and_ties_use_workload_id() -> None:
    packet = _packet()
    packet["workloads"] = [_row("tie_b"), _row("tie_a")]
    first = build_plan(packet)
    second = build_plan(deepcopy(packet))

    assert first == second
    assert first["ordered_plan"] == ["tie_a", "tie_b"]
    assert {plan["strategy"] for plan in first["packing_plans"]} == {
        "score_order",
        "deadline_first",
        "unlock_first",
    }
    assert set(first["lanes"]) == {"cpu", "gpu", "carla", "host_pinned"}


def test_missing_global_gate_returns_incomplete_without_selection() -> None:
    packet = _packet()
    packet["admission"]["freshness"] = "stale"
    result = build_plan(packet)

    assert result["status"] == "incomplete"
    assert result["ok"] is False
    assert result["ordered_plan"] == []
    assert result["feasibility_gates"]["global_input"] == "fail"
    assert any(item["code"] == "stale_admission" for item in result["issues"])


def test_missing_estimate_is_ineligible_and_score_shape_stays_visible() -> None:
    packet = _packet()
    packet["workloads"][0]["estimates"].pop("queue_seconds")
    packet["as_of"] = None

    result = build_plan(packet)
    report = next(item for item in result["workloads"] if item["workload_id"] == "base")

    assert result["status"] == "incomplete"
    assert report["feasibility"]["eligible"] is False
    assert "missing_estimate" in report["feasibility"]["blocking_reasons"]
    assert len(report["score"]["components"]) == 8


def test_check_mode_prints_without_writing(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plan.json"
    exit_code = main(
        ["--inventory", str(FIXTURE), "--check", "--format", "text", "--output", str(output)]
    )

    assert exit_code == 0
    assert "status=ready" in capsys.readouterr().out
    assert not output.exists()
