"""Regression coverage for the issue #6070 S30 population-contract canary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "benchmark" / "s30_campaign_canary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("s30_campaign_canary", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(scenario_id: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "population_arm": "heterogeneous",
        "planner": "goal",
        "seed": 101,
        "density": 0.02,
        "map_file": "maps/svg_maps/classic_crossing.svg",
        "arm_population": {"counts": {"cautious": 1, "standard": 2}},
    }


def test_canary_cli_reports_pass_and_every_named_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A two-cell manifest keeps a pass and the broken cell's full count tuple."""

    module = _load_module()
    manifest_path = tmp_path / "micro-manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_rows": [_row("passing_cell"), _row("broken_cell")]}),
        encoding="utf-8",
    )

    def fake_run_manifest_cell(row: dict[str, Any], **_kwargs: Any) -> dict[str, int]:
        if row["scenario_id"] == "broken_cell":
            return {
                "declared_population": 3,
                "instantiated_pedestrians": 3,
                "emitted_labels": 3,
                "trace_rows": 2,
            }
        return {
            "declared_population": 3,
            "instantiated_pedestrians": 3,
            "emitted_labels": 3,
            "trace_rows": 3,
        }

    monkeypatch.setattr(module, "run_manifest_cell", fake_run_manifest_cell)
    monkeypatch.setattr(
        sys,
        "argv",
        [MODULE_PATH.name, "--manifest", str(manifest_path), "--max-steps", "2"],
    )

    assert module.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["failed_cell_count"] == 1
    assert report["cells"][0]["status"] == "passed"
    assert report["cells"][1] == {
        "declared_population": 3,
        "emitted_labels": 3,
        "instantiated_pedestrians": 3,
        "planner": "goal",
        "population_arm": "heterogeneous",
        "reason": "declared!=instantiated!=labels!=traces",
        "row_index": 1,
        "scenario": "broken_cell",
        "seed": 101,
        "status": "failed",
        "trace_rows": 2,
    }


def test_load_manifest_rows_rejects_ambiguous_or_empty_manifest(tmp_path: Path) -> None:
    """The canary fails closed instead of manufacturing cells from a config."""

    module = _load_module()
    manifest_path = tmp_path / "ambiguous.json"
    manifest_path.write_text(json.dumps({"scenarios": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_rows"):
        module.load_manifest_rows(manifest_path)


def test_canary_reports_missing_spawn_synthesis_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unmerged spawn-synthesis dependency stays a named failed cell."""

    module = _load_module()

    def missing_spawn_synthesis(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        raise ValueError(
            "force_population_size requires a pedestrian route or crowded zone for the "
            "remaining 11 background pedestrians"
        )

    monkeypatch.setattr(module, "run_manifest_cell", missing_spawn_synthesis)

    report = module.run_canary([_row("geometryless_cell")], manifest_path=tmp_path, max_steps=2)

    assert report["passed"] is False
    assert report["cells"][0]["scenario"] == "geometryless_cell"
    assert report["cells"][0]["declared_population"] == 3
    assert report["cells"][0]["status"] == "unrealizable_without_spawn_synthesis"
