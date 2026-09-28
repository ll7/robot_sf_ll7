"""Acceptance checks for the diagnostic VV-5 fault-injection packet."""

from __future__ import annotations

import json
import sys

import pytest

from scripts.validation import run_issue_9735_fault_injection as vv5


def test_fault_matrix_requires_clean_controls_and_records_metric_miss() -> None:
    """Production checks detect three spawn faults but miss a doubled row total."""
    report = vv5.build_report()

    assert report["evidence_tier"] == "diagnostic_fixture_only"
    assert report["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert report["sensitivity"] == {"detected": 3, "injected": 4, "fraction": 0.75}
    assert report["missed"] == ["collision_total_doubled"]
    assert len(report["not_injected"]) == 6
    assert report["review_residuals"][0]["issue"] == 9861
    assert report["review_residuals"][0]["status"] == "not_injected"
    assert {item["case"] for item in report["review_residuals"]} == {
        "reset_pedestrian_overlap_with_completed_route",
        "reset_clearance_unavailable",
    }
    assert all(not result["control"].get("invalid_run", False) for result in report["results"])
    assert report["matrix"]["robot_start_inside_wall_radius"]["spawn_validity"] == "detected"
    assert "no simulator respawn placement" in report["results"][1]["fixture_mechanism"]
    metric = report["results"][-1]
    assert metric["control"]["gate_blocked"] is False
    assert metric["mutant"]["gate_blocked"] is False
    assert metric["mutant"]["component_sum"] != metric["mutant"]["total_collision_count"]
    assert len(report["source"]["sha256"]) == 5


def test_metric_flag_does_not_load_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected fault executes only its relevant checker path."""

    def forbidden_map_load(_path: str) -> None:
        pytest.fail("metric-only injection must not load the SVG fixture")

    monkeypatch.setattr(vv5, "convert_map", forbidden_map_load)
    report = vv5.build_report(("collision_total_doubled",))
    assert report["sensitivity"] == {"detected": 0, "injected": 1, "fraction": 0.0}
    assert report["selected_faults"] == ["collision_total_doubled"]
    assert report["matrix"]["collision_total_doubled"]["spawn_validity"] == "not_applicable"


def test_check_mode_rejects_stale_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed report cannot be silently accepted as a current result."""
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    args = [
        "vv5",
        "--fault",
        "collision_total_doubled",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert vv5.main() == 0
    assert json.loads(out_json.read_text())["selected_faults"] == ["collision_total_doubled"]
    monkeypatch.setattr(sys, "argv", [*args, "--check"])
    assert vv5.main() == 0
    out_md.write_text("stale", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        vv5.main()
