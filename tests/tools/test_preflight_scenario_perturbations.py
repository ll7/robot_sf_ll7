"""CLI tests for the scenario perturbation preflight (issue #4907 bad-input UX)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.scenario_certification import PERTURBATION_MANIFEST_SCHEMA_VERSION
from scripts.tools.preflight_scenario_perturbations import main

if TYPE_CHECKING:
    from pathlib import Path


def _write_valid_manifest(path: Path) -> Path:
    """Write a tiny perturbation manifest targeting an existing simple scenario."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_perturbation_manifest",
        "scenario_config": "configs/scenarios/single/planner_sanity_simple.yaml",
        "seed_controls": {
            "baseline_seeds": [101, 102],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": 0.5,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": "planner_sanity_simple_noop",
                "scenario_id": "planner_sanity_simple",
                "family": "noop",
                "seeds": [101, 102],
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_main_nonexistent_manifest_prints_actionable_error(monkeypatch, capsys) -> None:
    """A nonexistent manifest path prints one actionable line and exits non-zero."""
    missing = "/nonexistent/perturbation_manifest.json"
    monkeypatch.setattr("sys.argv", ["preflight_scenario_perturbations.py", missing])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "FAILED:" in captured.err
    assert "No such file or directory" in captured.err
    assert missing in captured.err
    assert "Traceback" not in captured.err


def test_main_non_mapping_manifest_prints_actionable_error(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A non-mapping manifest prints one actionable line and exits non-zero."""
    bad_manifest = tmp_path / "bad.yaml"
    bad_manifest.write_text("- a\n- b\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["preflight_scenario_perturbations.py", str(bad_manifest)])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert f"FAILED: Perturbation manifest must contain a mapping: {bad_manifest}" in (captured.err)
    assert "Traceback" not in captured.err


def test_main_valid_manifest_emits_report_and_exits_zero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """A valid manifest still emits the JSON report and exits 0 (happy path unchanged)."""
    manifest = _write_valid_manifest(tmp_path / "perturbations.yaml")
    monkeypatch.setattr("sys.argv", ["preflight_scenario_perturbations.py", str(manifest)])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FAILED:" not in captured.err
    payload = json.loads(captured.out)
    assert payload["manifest_id"] == "test_perturbation_manifest"
