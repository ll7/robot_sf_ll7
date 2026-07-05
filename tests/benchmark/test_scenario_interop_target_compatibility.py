"""Tests target-readiness reports for scenario interop dry-run conversion."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_interop import (
    TARGET_COMPATIBILITY_SCHEMA_VERSION,
    build_target_compatibility_report,
    convert_scenario_to_ir,
)


def _axis_scenario() -> dict:
    return {
        "id": "demo-uni-low-open",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
        "groups": 0.0,
        "speed_var": "low",
        "goal_topology": "point",
        "robot_context": "embedded",
        "seeds": [1, 2, 3],
    }


def _explicit_map_scenario() -> dict:
    return {
        "name": "corridor-handoff",
        "map_file": "maps/svg_maps/corridor.svg",
        "flow": "bi",
        "simulation_config": {"step_time_s": 0.1},
        "single_pedestrians": [
            {
                "id": "ped-1",
                "start_poi": "north",
                "goal_poi": "south",
                "speed_m_s": 1.2,
            },
        ],
    }


def test_target_compatibility_report_fails_closed_for_missing_external_assets() -> None:
    """Target readiness reports name blockers instead of implying export readiness."""
    result = convert_scenario_to_ir(_explicit_map_scenario())

    reports = build_target_compatibility_report(result.ir)

    assert [report["target"] for report in reports] == ["socnavbench", "hunavsim"]
    assert all(
        report["schema_version"] == TARGET_COMPATIBILITY_SCHEMA_VERSION for report in reports
    )
    assert all(report["source_scenario_id"] == "corridor-handoff" for report in reports)
    assert all(report["ready"] is False for report in reports)
    blockers_by_target = {
        report["target"]: {blocker["code"] for blocker in report["blockers"]} for report in reports
    }
    assert "socnavbench_assets_not_staged" in blockers_by_target["socnavbench"]
    assert "hunavsim_adapter_not_staged" in blockers_by_target["hunavsim"]
    assert "unsupported_fields_present" in blockers_by_target["socnavbench"]


def test_target_compatibility_report_names_axis_scenario_export_gaps() -> None:
    """Axis-only scenarios report missing map and explicit-agent blockers."""
    result = convert_scenario_to_ir(_axis_scenario())

    [report] = build_target_compatibility_report(result.ir, targets=("socnavbench",))

    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert "map_file_missing" in blocker_codes
    assert "agents_missing" in blocker_codes
    assert report["warnings"] == []


def test_target_compatibility_report_rejects_unknown_target() -> None:
    """Unknown target names fail fast."""
    result = convert_scenario_to_ir(_axis_scenario())

    with pytest.raises(ValueError, match="unsupported target"):
        build_target_compatibility_report(result.ir, targets=("unknownsim",))
