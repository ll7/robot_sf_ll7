"""CARLA-free tests for the T0 export CLI wrapper."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_export_t0_scenarios_main_writes_records_and_prints_manifest(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """CLI main should call library helpers and report the manifest path."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import export_t0_scenarios_main

    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "exports"
    calls = {}

    def fake_build_records(path, *, provenance):
        calls["build"] = {"path": Path(path), "provenance": dict(provenance)}
        return [{"scenario_id": "unit", "payload": {"schema_version": "carla-replay-export.v1"}}]

    def fake_write_records(records, target_dir):
        calls["write"] = {"records": list(records), "target_dir": Path(target_dir)}
        return {"schema_version": "carla-replay-export-manifest.v1", "exports": []}

    monkeypatch.setattr(cli_module, "build_export_payloads_from_scenario_file", fake_build_records)
    monkeypatch.setattr(cli_module, "write_export_records", fake_write_records)

    exit_code = export_t0_scenarios_main(
        [
            "--scenario-file",
            str(scenario_path),
            "--output-dir",
            str(output_dir),
            "--robot-sf-commit",
            "abc123",
            "--created-by",
            "unit-test",
        ]
    )

    assert exit_code == 0
    assert calls["build"]["path"] == scenario_path
    assert calls["build"]["provenance"] == {
        "robot_sf_commit": "abc123",
        "created_by": "unit-test",
        "certificate_generator": "scenario_cert.v1",
    }
    assert calls["write"]["target_dir"] == output_dir
    assert "manifest.json" in capsys.readouterr().out


def test_validate_t0_manifest_main_reads_manifest_and_prints_count(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Manifest validator CLI should report the number of exported payloads."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import validate_t0_manifest_main

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    calls = {}

    def fake_read_manifest(path):
        calls["path"] = Path(path)
        return {
            "schema_version": "carla-replay-export-manifest.v1",
            "exports": [{"scenario_id": "unit", "path": "unit.json"}],
        }

    monkeypatch.setattr(cli_module, "read_export_manifest", fake_read_manifest)

    exit_code = validate_t0_manifest_main(["--manifest", str(manifest_path)])

    assert exit_code == 0
    assert calls["path"] == manifest_path
    assert "1 export" in capsys.readouterr().out


def test_validate_t0_export_batch_main_loads_payloads_and_prints_count(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Batch validator CLI should load every manifest payload and report the count."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import validate_t0_export_batch_main

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    calls = {}

    def fake_load_payloads(path):
        calls["path"] = Path(path)
        return [
            {"scenario_id": "first", "path": tmp_path / "first.json", "payload": {}},
            {"scenario_id": "second", "path": tmp_path / "second.json", "payload": {}},
        ]

    monkeypatch.setattr(cli_module, "load_export_manifest_payloads", fake_load_payloads)

    exit_code = validate_t0_export_batch_main(["--manifest", str(manifest_path)])

    assert exit_code == 0
    assert calls["path"] == manifest_path
    assert "2 payloads" in capsys.readouterr().out


def test_validate_t0_export_batch_main_prints_json_summary(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """Batch validator CLI should support deterministic machine-readable output."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import validate_t0_export_batch_main

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    def fake_load_payloads(path):
        assert Path(path) == manifest_path
        return [
            {"scenario_id": "first", "path": tmp_path / "first.json", "payload": {}},
            {"scenario_id": "second", "path": tmp_path / "second.json", "payload": {}},
        ]

    monkeypatch.setattr(cli_module, "load_export_manifest_payloads", fake_load_payloads)

    exit_code = validate_t0_export_batch_main(["--manifest", str(manifest_path), "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "manifest": manifest_path.as_posix(),
        "payload_count": 2,
        "scenario_ids": ["first", "second"],
    }


def test_check_carla_availability_main_prints_json_status(monkeypatch, capsys) -> None:
    """CARLA availability CLI should expose deterministic machine-readable status."""
    from robot_sf_carla_bridge.cli import check_carla_availability_main

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "carla" else object(),
    )

    exit_code = check_carla_availability_main(["--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "available": False,
        "dependency": "carla",
        "reason": "CARLA Python API package 'carla' is not importable",
        "status": "not-available",
    }


def test_check_carla_availability_main_prints_text_status(monkeypatch, capsys) -> None:
    """CARLA availability CLI should keep a concise human-readable output."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import check_carla_availability_main

    monkeypatch.setattr(
        cli_module,
        "check_carla_availability",
        lambda: {
            "status": "available",
            "reason": "CARLA Python API package is importable",
            "dependency": "carla",
        },
    )

    exit_code = check_carla_availability_main([])

    assert exit_code == 0
    assert capsys.readouterr().out == "carla: available - CARLA Python API package is importable\n"


def test_check_carla_availability_main_require_fails_when_unavailable(
    monkeypatch,
    capsys,
) -> None:
    """Require mode should fail closed when CARLA is unavailable."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import check_carla_availability_main

    monkeypatch.setattr(
        cli_module,
        "check_carla_availability",
        lambda: {
            "status": "not-available",
            "reason": "CARLA Python API package 'carla' is not importable",
            "dependency": "carla",
        },
    )

    exit_code = check_carla_availability_main(["--require", "--json"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "dependency": "carla",
        "reason": "CARLA Python API package 'carla' is not importable",
        "status": "not-available",
    }


def test_check_carla_availability_main_require_passes_when_available(
    monkeypatch,
    capsys,
) -> None:
    """Require mode should still succeed when CARLA is available."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import check_carla_availability_main

    monkeypatch.setattr(
        cli_module,
        "check_carla_availability",
        lambda: {
            "status": "available",
            "reason": "CARLA Python API package is importable",
            "dependency": "carla",
        },
    )

    exit_code = check_carla_availability_main(["--require"])

    assert exit_code == 0
    assert capsys.readouterr().out == "carla: available - CARLA Python API package is importable\n"


def test_export_t0_cli_is_packaged_as_project_script() -> None:
    """Project metadata should expose the CLI and include the bridge package."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["robot-sf-export-carla-t0"] == (
        "robot_sf_carla_bridge.cli:export_t0_scenarios_main"
    )
    assert pyproject["project"]["scripts"]["robot-sf-validate-carla-t0-manifest"] == (
        "robot_sf_carla_bridge.cli:validate_t0_manifest_main"
    )
    assert pyproject["project"]["scripts"]["robot-sf-validate-carla-t0-batch"] == (
        "robot_sf_carla_bridge.cli:validate_t0_export_batch_main"
    )
    assert pyproject["project"]["scripts"]["robot-sf-check-carla"] == (
        "robot_sf_carla_bridge.cli:check_carla_availability_main"
    )
    hatchling_packages = pyproject["tool"]["hatchling"]["build"]["targets"]["wheel"]["packages"]
    assert {"include": "robot_sf_carla_bridge"} in hatchling_packages
    assert (
        "/robot_sf_carla_bridge"
        in pyproject["tool"]["hatchling"]["build"]["targets"]["sdist"]["include"]
    )
