"""Tests target-readiness reports for scenario interop dry-run conversion."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.scenario_interop import (
    TARGET_COMPATIBILITY_SCHEMA_VERSION,
    TARGET_EXPORT_MANIFEST_SCHEMA_VERSION,
    TARGET_EXPORT_PREVIEW_SCHEMA_VERSION,
    build_target_compatibility_report,
    build_target_export_manifest,
    build_target_export_preview,
    convert_scenario_to_ir,
    dump_ir,
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


def test_target_export_manifest_records_blocked_artifact_contract() -> None:
    """Export manifest preserves blockers without claiming target artifact readiness."""
    result = convert_scenario_to_ir(_explicit_map_scenario())

    manifest = build_target_export_manifest(result.ir, target="socnavbench")

    assert manifest["schema_version"] == TARGET_EXPORT_MANIFEST_SCHEMA_VERSION
    assert manifest["artifact_kind"] == "socnavbench_scenario_export_manifest"
    assert manifest["target"] == "socnavbench"
    assert manifest["source_scenario_id"] == "corridor-handoff"
    assert manifest["source_ir_schema_version"] == result.ir["schema_version"]
    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["compatibility_report_schema_version"] == TARGET_COMPATIBILITY_SCHEMA_VERSION
    assert {blocker["code"] for blocker in manifest["blockers"]} >= {
        "socnavbench_assets_not_staged",
        "unsupported_fields_present",
    }


def test_target_export_manifest_is_deterministic() -> None:
    """Identical target manifest inputs serialize byte-identically."""
    first = build_target_export_manifest(
        convert_scenario_to_ir(_explicit_map_scenario()).ir,
        target="hunavsim",
    )
    second = build_target_export_manifest(
        convert_scenario_to_ir(_explicit_map_scenario()).ir,
        target="hunavsim",
    )

    assert dump_ir(first) == dump_ir(second)


def test_socnavbench_export_preview_contains_target_shaped_payload() -> None:
    """SocNavBench preview preserves mapped sections while still blocked."""

    result = convert_scenario_to_ir(_explicit_map_scenario())
    preview = build_target_export_preview(result.ir, target="socnavbench")

    assert preview["schema_version"] == TARGET_EXPORT_PREVIEW_SCHEMA_VERSION
    assert preview["artifact_kind"] == "socnavbench_scenario_export_preview"
    assert preview["status"] == "blocked"
    assert preview["payload"]["scenario"] == {
        "name": "corridor-handoff",
        "map": "maps/svg_maps/corridor.svg",
        "environment_type": None,
        "flow": "bi",
        "density": None,
        "seed_set": None,
    }
    assert preview["payload"]["pedestrians"] == [
        {
            "id": "ped-1",
            "start": "north",
            "goal": "south",
            "preferred_speed_mps": 1.2,
        }
    ]
    assert {blocker["code"] for blocker in preview["blockers"]} >= {
        "socnavbench_assets_not_staged",
        "unsupported_fields_present",
    }


def test_hunavsim_export_preview_contains_target_shaped_payload() -> None:
    """HuNavSim preview records ROS-adapter-shaped inputs without claiming readiness."""

    result = convert_scenario_to_ir(_explicit_map_scenario())
    preview = build_target_export_preview(result.ir, target="hunavsim")

    assert preview["schema_version"] == TARGET_EXPORT_PREVIEW_SCHEMA_VERSION
    assert preview["artifact_kind"] == "hunavsim_scenario_export_preview"
    assert preview["status"] == "blocked"
    assert preview["payload"]["world"] == {
        "map_file": "maps/svg_maps/corridor.svg",
        "obstacle_topology": None,
        "flow": "bi",
    }
    assert preview["payload"]["agents"] == [
        {
            "name": "ped-1",
            "start_poi": "north",
            "goal_poi": "south",
            "behavior": {
                "preferred_speed_mps": 1.2,
                "role": None,
                "role_target_id": None,
            },
        }
    ]
    assert any(warning["code"] == "ros_semantics_unmapped" for warning in preview["warnings"])


def test_target_export_preview_is_deterministic() -> None:
    """Identical target preview inputs serialize byte-identically."""

    first = build_target_export_preview(
        convert_scenario_to_ir(_explicit_map_scenario()).ir,
        target="socnavbench",
    )
    second = build_target_export_preview(
        convert_scenario_to_ir(_explicit_map_scenario()).ir,
        target="socnavbench",
    )

    assert dump_ir(first) == dump_ir(second)


def test_target_compatibility_report_rejects_unknown_target() -> None:
    """Unknown target names fail fast."""
    result = convert_scenario_to_ir(_axis_scenario())

    with pytest.raises(ValueError, match="unsupported target"):
        build_target_compatibility_report(result.ir, targets=("unknownsim",))
