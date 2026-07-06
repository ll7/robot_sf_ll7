"""Tests for the SocNavBench ETH shape-contract audit command."""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING

import numpy as np

from robot_sf.data.external import socnavbench_eth
from scripts.validation import check_socnavbench_eth_shape_contract as checker

if TYPE_CHECKING:
    from pathlib import Path


def _stage_minimal_eth_fixture(root: Path) -> None:
    """Create a tiny layout-compatible SocNavBench ETH fixture."""
    layout = socnavbench_eth.expected_layout(root)
    layout.mesh_dir.mkdir(parents=True)
    (layout.mesh_dir / "mesh.obj").write_text("# synthetic mesh marker\n", encoding="utf-8")
    layout.traversible_pickle.parent.mkdir(parents=True)
    with layout.traversible_pickle.open("wb") as handle:
        pickle.dump(
            {
                "resolution": 5.0,
                "traversible": np.array([[True, False], [True, True]], dtype=bool),
            },
            handle,
            protocol=2,
        )


def test_build_report_missing_data_fails_closed(tmp_path: Path) -> None:
    """Missing staged data returns a fail-closed report."""
    report, exit_code = checker.build_report(tmp_path / "socnavbench")

    assert exit_code == 2
    assert report["schema"] == checker.REPORT_SCHEMA
    assert report["issue"] == 4279
    assert report["asset_id"] == "socnavbench-s3dis-eth"
    assert report["ok"] is False
    assert report["status"] == "missing"
    assert report["no_download_performed"] is True
    assert report["required_paths"]["mesh_dir"]["exists"] is False
    assert report["required_paths"]["traversible_pickle"]["exists"] is False


def test_build_report_staged_data_passes_shape_contract(tmp_path: Path) -> None:
    """Synthetic staged data exercises the passing shape-contract path."""
    root = tmp_path / "socnavbench"
    _stage_minimal_eth_fixture(root)

    report, exit_code = checker.build_report(root)

    assert exit_code == 0
    assert report["ok"] is True
    assert report["status"] == "passed"
    assert report["shape_contract"] == {
        "resolution": 5.0,
        "traversible_dtype": "bool",
        "traversible_shape": [2, 2],
    }


def test_main_writes_json_out_and_returns_missing(tmp_path: Path, capsys) -> None:
    """CLI writes the same missing-data report to stdout and JSON output."""
    report_path = tmp_path / "reports" / "audit.json"

    exit_code = checker.main(
        ["--root", str(tmp_path / "socnavbench"), "--json-out", str(report_path)]
    )

    assert exit_code == 2
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert file_report["status"] == "missing"
