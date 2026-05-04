"""CARLA-free tests for the T0 export CLI wrapper."""

from __future__ import annotations

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


def test_export_t0_cli_is_packaged_as_project_script() -> None:
    """Project metadata should expose the CLI and include the bridge package."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["robot-sf-export-carla-t0"] == (
        "robot_sf_carla_bridge.cli:export_t0_scenarios_main"
    )
    hatchling_packages = pyproject["tool"]["hatchling"]["build"]["targets"]["wheel"]["packages"]
    assert {"include": "robot_sf_carla_bridge"} in hatchling_packages
    assert (
        "/robot_sf_carla_bridge"
        in pyproject["tool"]["hatchling"]["build"]["targets"]["sdist"]["include"]
    )


def test_export_t0_scenarios_main_rejects_parent_relative_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """CLI main should fail fast on parent-relative file arguments."""
    import robot_sf_carla_bridge.cli as cli_module
    from robot_sf_carla_bridge.cli import export_t0_scenarios_main

    build_called = False
    write_called = False

    def fake_build_records(path, *, provenance):
        nonlocal build_called
        build_called = True
        return []

    def fake_write_records(records, target_dir):
        nonlocal write_called
        write_called = True
        return {"schema_version": "carla-replay-export-manifest.v1", "exports": []}

    monkeypatch.setattr(cli_module, "build_export_payloads_from_scenario_file", fake_build_records)
    monkeypatch.setattr(cli_module, "write_export_records", fake_write_records)

    exit_code = export_t0_scenarios_main(
        [
            "--scenario-file",
            "../unsafe.yaml",
            "--output-dir",
            str(tmp_path / "exports"),
            "--robot-sf-commit",
            "abc123",
        ]
    )

    assert exit_code == 1
    assert not build_called
    assert not write_called
    assert "Invalid scenario file path" in capsys.readouterr().err
