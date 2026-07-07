"""Tests target-readiness reports for scenario interop dry-run conversion."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.scenario_interop import (
    TARGET_COMPATIBILITY_SCHEMA_VERSION,
    TARGET_EXPORT_MANIFEST_SCHEMA_VERSION,
    TARGET_EXPORT_PREVIEW_SCHEMA_VERSION,
    TARGET_PREREQUISITE_REPORT_SCHEMA_VERSION,
    build_target_compatibility_report,
    build_target_export_manifest,
    build_target_export_preview,
    build_target_prerequisite_report,
    convert_scenario_to_ir,
    dump_ir,
    load_target_compatibility_schema,
    load_target_export_manifest_schema,
    load_target_export_preview_schema,
    load_target_prerequisite_report_schema,
    validate_target_compatibility_report,
    validate_target_export_manifest,
    validate_target_export_preview,
    validate_target_prerequisite_report,
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
    assert all(validate_target_compatibility_report(report) == [] for report in reports)


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
    assert validate_target_export_manifest(manifest) == []


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
    assert validate_target_export_preview(preview) == []


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
    assert validate_target_export_preview(preview) == []


def test_target_artifact_schema_loaders_and_tamper_checks() -> None:
    """Target artifact schema helpers load schemas and reject malformed payloads."""

    assert load_target_compatibility_schema()["title"]
    assert load_target_export_manifest_schema()["title"]
    assert load_target_export_preview_schema()["title"]

    result = convert_scenario_to_ir(_explicit_map_scenario())

    compatibility = build_target_compatibility_report(result.ir, targets=("socnavbench",))[0]
    broken_compatibility = dict(compatibility)
    broken_compatibility["target"] = "unknownsim"
    assert validate_target_compatibility_report(broken_compatibility) != []

    manifest = build_target_export_manifest(result.ir, target="socnavbench")
    broken_manifest = dict(manifest)
    broken_manifest["status"] = "exported"
    assert validate_target_export_manifest(broken_manifest) != []

    preview = build_target_export_preview(result.ir, target="socnavbench")
    broken_preview = dict(preview)
    broken_preview["payload"] = {"unexpected": True}
    assert validate_target_export_preview(broken_preview) != []


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


def test_target_prerequisite_report_fails_closed_for_missing_paths(tmp_path: Path) -> None:
    """Runnable-export prerequisites are reported explicitly when absent."""
    result = convert_scenario_to_ir(_explicit_map_scenario())

    report = build_target_prerequisite_report(result.ir, target="socnavbench", root=tmp_path)

    assert report["schema_version"] == TARGET_PREREQUISITE_REPORT_SCHEMA_VERSION
    assert report["artifact_kind"] == "socnavbench_scenario_export_prerequisite_report"
    assert report["target"] == "socnavbench"
    assert report["source_scenario_id"] == "corridor-handoff"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["no_download_performed"] is True
    assert {item["code"] for item in report["remaining_blockers"]} == {
        "socnavbench_control_assets",
        "socnavbench_eth_mesh",
        "socnavbench_eth_traversible_pickle",
    }
    assert {item["issue"] for item in report["remaining_blockers"]} == {1456, 1134}
    assert "benchmark evidence" in report["claim_boundary"]
    assert validate_target_prerequisite_report(report) == []


def test_target_prerequisite_report_passes_when_expected_paths_exist(tmp_path: Path) -> None:
    """Synthetic local paths exercise the ready prerequisite-report path."""
    (tmp_path / "third_party/socnavbench/wayptnav_data").mkdir(parents=True)
    (tmp_path / "third_party/socnavbench/sd3dis/stanford_building_parser_dataset/mesh/ETH").mkdir(
        parents=True
    )
    traversible = (
        tmp_path
        / "third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl"
    )
    traversible.parent.mkdir(parents=True)
    traversible.write_bytes(b"synthetic-placeholder")
    result = convert_scenario_to_ir(_explicit_map_scenario())

    report = build_target_prerequisite_report(result.ir, target="socnavbench", root=tmp_path)

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["remaining_blockers"] == []
    assert {item["status"] for item in report["prerequisites"]} == {"present"}
    assert "target exporter smoke path" in report["next_empirical_action"]
    assert validate_target_prerequisite_report(report) == []


def test_target_prerequisite_schema_loader_and_tamper_check(tmp_path: Path) -> None:
    """Prerequisite schema helper loads and rejects malformed reports."""
    assert load_target_prerequisite_report_schema()["title"]
    result = convert_scenario_to_ir(_explicit_map_scenario())
    report = build_target_prerequisite_report(result.ir, target="hunavsim", root=tmp_path)
    broken = dict(report)
    broken["no_download_performed"] = False

    assert validate_target_prerequisite_report(report) == []
    assert validate_target_prerequisite_report(broken) != []


def test_cli_writes_target_prerequisite_reports(tmp_path: Path) -> None:
    """CLI emits local-only prerequisite reports for requested targets."""
    out_dir = tmp_path / "prerequisites"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/tools/convert_scenario_interop.py",
            "--matrix",
            "configs/baselines/example_matrix.yaml",
            "--target",
            "socnavbench",
            "--target-prerequisite-out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    report_paths = sorted(out_dir.glob("*.socnavbench.prerequisite_report.json"))
    assert report_paths
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    assert report["schema_version"] == TARGET_PREREQUISITE_REPORT_SCHEMA_VERSION
    assert report["target"] == "socnavbench"
    assert report["no_download_performed"] is True
    assert validate_target_prerequisite_report(report) == []
