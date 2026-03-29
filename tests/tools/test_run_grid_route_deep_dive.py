"""Regression tests for the grid-route deep-dive validation runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import scripts.validation.run_grid_route_deep_dive as deep_dive

if TYPE_CHECKING:
    from pathlib import Path


def test_manifest_path_label_falls_back_for_external_path(tmp_path: Path) -> None:
    """Use absolute-path fallback labels for manifests outside the repository root."""
    external_manifest = tmp_path / "external_set.yaml"
    external_manifest.write_text("scenarios: []\n", encoding="utf-8")

    assert deep_dive._manifest_path_label(external_manifest) == str(external_manifest)


def test_main_returns_nonzero_when_any_set_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return a failing exit code while still writing summary artifacts on set errors."""
    scenario_set = tmp_path / "scenario_set.yaml"
    scenario_set.write_text("scenarios: []\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    def _args():
        return type(
            "Args",
            (),
            {
                "scenario_dir": tmp_path,
                "scenario_sets": [str(scenario_set)],
                "schema": tmp_path / "episode.schema.v1.json",
                "algo_config": tmp_path / "grid_route.yaml",
                "output_dir": out_dir,
                "algo": "grid_route",
                "dt": 0.1,
                "horizon": 32,
                "workers": 1,
            },
        )()

    monkeypatch.setattr(
        deep_dive,
        "parse_args",
        _args,
    )
    monkeypatch.setattr(deep_dive, "load_scenarios", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        deep_dive,
        "run_map_batch",
        lambda *_args, **_kwargs: {
            "benchmark_availability": {
                "availability_status": "not_available",
                "availability_reason": "episodes output not written",
            }
        },
    )

    result = deep_dive.main()

    assert result == 1
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["overall"]["sets_failed"] == 1
    assert payload["sets"][0]["manifest"] == str(scenario_set)
    assert payload["sets"][0]["status"] == "error"
    assert payload["sets"][0]["error"] == "not_available: episodes output not written"
