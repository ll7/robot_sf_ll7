"""Tests for config-derived scenario denominator manifests."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.scenario_denominator_manifest import (
    DenominatorManifestError,
    build_scenario_denominator_manifest,
    check_denominator_table,
    check_manifest,
    denominator_table_rows,
    write_denominator_table,
    write_manifest,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "build_scenario_denominator_manifest.py"
)
_SPEC = importlib.util.spec_from_file_location("build_scenario_denominator_manifest", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def _write_yaml(path: Path, payload: object) -> None:
    """Write YAML fixtures with deterministic key order."""

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_fixture_campaign(tmp_path: Path, *, seed_policy: dict[str, object]) -> Path:
    """Create a compact campaign config and scenario matrix fixture."""

    scenarios_path = tmp_path / "scenarios.yaml"
    _write_yaml(
        scenarios_path,
        {
            "scenarios": [
                {
                    "name": "crossing_low",
                    "map_file": "map.svg",
                    "metadata": {"archetype": "crossing", "density": "low"},
                    "seeds": [1, 2],
                },
                {
                    "name": "crossing_high",
                    "map_file": "map.svg",
                    "metadata": {"archetype": "crossing", "density": "high"},
                    "seeds": [3],
                },
                {
                    "name": "bottleneck_low",
                    "map_file": "map.svg",
                    "metadata": {"archetype": "bottleneck", "density": "low"},
                    "seeds": [4, 5],
                },
            ]
        },
    )
    config_path = tmp_path / "campaign.yaml"
    _write_yaml(
        config_path,
        {
            "name": "fixture_campaign",
            "scenario_matrix": "scenarios.yaml",
            "seed_policy": seed_policy,
            "kinematics_matrix": ["differential_drive", "holonomic"],
            "planners": [
                {"key": "goal", "algo": "goal"},
                {"key": "social_force", "algo": "social_force"},
                {"key": "disabled_ppo", "algo": "ppo", "enabled": False},
            ],
        },
    )
    return config_path


def test_manifest_resolves_seed_policy_and_family_planner_denominators(tmp_path: Path) -> None:
    """Manifest denominators come from scenario cells, resolved seeds, and enabled planners."""

    config_path = _write_fixture_campaign(
        tmp_path,
        seed_policy={"mode": "fixed-list", "seeds": [11, 12, 13]},
    )

    manifest = build_scenario_denominator_manifest([config_path], repo_root=tmp_path)

    config = manifest["configs"][0]
    assert config["summary"]["scenario_count"] == 3
    assert config["summary"]["family_count"] == 2
    assert config["summary"]["seed_count"] == 3
    assert config["summary"]["planner_count"] == 2
    assert config["summary"]["kinematics_count"] == 2
    assert config["summary"]["episode_denominator_without_planner"] == 9
    assert config["summary"]["planner_episode_denominator"] == 18
    assert config["summary"]["planner_kinematics_episode_denominator"] == 36
    assert [planner["key"] for planner in config["disabled_planners"]] == ["disabled_ppo"]

    rows = denominator_table_rows(manifest)
    crossing_goal = next(
        row for row in rows if row["family"] == "crossing" and row["planner"] == "goal"
    )
    assert crossing_goal["scenario_count"] == "2"
    assert crossing_goal["seed_count"] == "3"
    assert crossing_goal["denominator_episodes"] == "6"
    assert crossing_goal["denominator_episodes_with_kinematics"] == "12"
    assert crossing_goal["densities"] == "high;low"


def test_manifest_uses_scenario_default_seeds_and_candidate_filter(tmp_path: Path) -> None:
    """Scenario-default seed policy preserves per-cell seeds after candidate filtering."""

    config_path = _write_fixture_campaign(tmp_path, seed_policy={"mode": "scenario-default"})
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["scenario_candidates"] = ["crossing_low", "bottleneck_low"]
    _write_yaml(config_path, payload)

    manifest = build_scenario_denominator_manifest([config_path], repo_root=tmp_path)

    config = manifest["configs"][0]
    assert config["summary"]["scenario_count"] == 2
    assert config["summary"]["seeds"] == [1, 2, 4, 5]
    assert config["summary"]["episode_denominator_without_planner"] == 4
    assert {cell["scenario_id"] for cell in config["cells"]} == {
        "crossing_low",
        "bottleneck_low",
    }


def test_manifest_check_modes_fail_closed_on_mismatch(tmp_path: Path) -> None:
    """Manifest and table checks reject drift from config-derived denominators."""

    config_path = _write_fixture_campaign(
        tmp_path,
        seed_policy={"mode": "fixed-list", "seeds": [11, 12, 13]},
    )
    manifest = build_scenario_denominator_manifest([config_path], repo_root=tmp_path)
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    table_path = write_denominator_table(manifest, tmp_path / "denominators.csv")

    check_manifest(manifest, manifest_path)
    check_denominator_table(manifest, table_path)

    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    headers = list(rows[0])
    markdown_path = tmp_path / "denominators.md"
    markdown_path.write_text(
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(row[header] for header in headers) + " |" for row in rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    check_denominator_table(manifest, markdown_path)

    rows[0]["denominator_episodes"] = "999"
    bad_table_path = tmp_path / "bad_denominators.csv"
    with bad_table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(DenominatorManifestError, match="Denominator table mismatch"):
        check_denominator_table(manifest, bad_table_path)


def test_manifest_fails_closed_on_invalid_config_inputs(tmp_path: Path) -> None:
    """Malformed canonical config inputs fail before producing denominator counts."""

    bad_mapping = tmp_path / "bad_mapping.yaml"
    _write_yaml(bad_mapping, ["not", "a", "mapping"])
    with pytest.raises(DenominatorManifestError, match="YAML mapping"):
        build_scenario_denominator_manifest([bad_mapping], repo_root=tmp_path)

    config_path = _write_fixture_campaign(tmp_path, seed_policy={"mode": "scenario-default"})
    scenarios_path = tmp_path / "scenarios.yaml"
    _write_yaml(
        scenarios_path,
        {
            "scenarios": [
                {
                    "name": "seedless",
                    "map_file": "map.svg",
                    "metadata": {"archetype": "crossing", "density": "low"},
                }
            ]
        },
    )
    with pytest.raises(DenominatorManifestError, match="no seeds"):
        build_scenario_denominator_manifest([config_path], repo_root=tmp_path)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["seed_policy"] = {"mode": "fixed-list", "seeds": [11]}
    payload["scenario_candidates"] = ["missing_scenario"]
    _write_yaml(config_path, payload)
    with pytest.raises(DenominatorManifestError, match="scenario_candidates"):
        build_scenario_denominator_manifest([config_path], repo_root=tmp_path)

    payload["scenario_candidates"] = []
    payload["seed_policy"] = {"mode": "seed-set", "seed_set": "missing"}
    seed_sets = tmp_path / "seed_sets.yaml"
    _write_yaml(seed_sets, {"eval": [1, 2]})
    payload["seed_policy"]["seed_sets_path"] = "seed_sets.yaml"
    _write_yaml(config_path, payload)
    with pytest.raises(DenominatorManifestError, match="Unknown seed set"):
        build_scenario_denominator_manifest([config_path], repo_root=tmp_path)


def test_table_check_fails_closed_on_malformed_consumers(tmp_path: Path) -> None:
    """Consumer tables must expose complete integer denominator rows."""

    config_path = _write_fixture_campaign(
        tmp_path,
        seed_policy={"mode": "fixed-list", "seeds": [11, 12, 13]},
    )
    manifest = build_scenario_denominator_manifest([config_path], repo_root=tmp_path)
    rows = denominator_table_rows(manifest)

    missing_column_path = tmp_path / "missing_column.json"
    missing_column_path.write_text(
        json.dumps({"rows": [{"config_name": rows[0]["config_name"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(DenominatorManifestError, match="missing columns"):
        check_denominator_table(manifest, missing_column_path)

    bad_int_path = tmp_path / "bad_int.csv"
    bad_rows = [dict(rows[0])]
    bad_rows[0]["denominator_episodes"] = "not-an-int"
    with bad_int_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bad_rows[0].keys())
        writer.writeheader()
        writer.writerows(bad_rows)
    with pytest.raises(DenominatorManifestError, match="must be integer"):
        check_denominator_table(manifest, bad_int_path)

    no_table_path = tmp_path / "no_table.md"
    no_table_path.write_text("No denominator table here.\n", encoding="utf-8")
    with pytest.raises(DenominatorManifestError, match="No denominator table"):
        check_denominator_table(manifest, no_table_path)

    unsupported_path = tmp_path / "table.txt"
    unsupported_path.write_text("plain text\n", encoding="utf-8")
    with pytest.raises(DenominatorManifestError, match="Unsupported denominator table format"):
        check_denominator_table(manifest, unsupported_path)


def test_cli_generates_outputs_and_checks_table(tmp_path: Path) -> None:
    """Script entry point supports generate and fail-closed check mode."""

    config_path = _write_fixture_campaign(
        tmp_path,
        seed_policy={"mode": "fixed-list", "seeds": [11, 12, 13]},
    )
    manifest_path = tmp_path / "manifest.json"
    table_path = tmp_path / "denominators.csv"

    assert (
        _SCRIPT.main(
            [
                "--config",
                str(config_path),
                "--output",
                str(manifest_path),
                "--table",
                str(table_path),
            ]
        )
        == 0
    )
    assert manifest_path.exists()
    assert table_path.exists()
    assert _SCRIPT.main(["--config", str(config_path), "--check-table", str(table_path)]) == 0
