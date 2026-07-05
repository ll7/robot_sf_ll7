"""Tests for the SocNavBench ETH stage/generate/convert/smoke wrapper."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np

from scripts.tools import stage_socnavbench_eth_traversible_svg as stage_svg

if TYPE_CHECKING:
    from pathlib import Path


def _write_socnav_eth_fixture(root: Path, traversible: np.ndarray) -> None:
    """Write the minimal staged SocNavBench ETH layout used by the smoke wrapper."""
    mesh_dir = root / "sd3dis" / "stanford_building_parser_dataset" / "mesh" / "ETH"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh.obj").write_text("# fixture mesh marker\n", encoding="utf-8")
    pkl_path = (
        root / "sd3dis" / "stanford_building_parser_dataset" / "traversibles" / "ETH" / "data.pkl"
    )
    pkl_path.parent.mkdir(parents=True)
    with pkl_path.open("wb") as handle:
        pickle.dump({"resolution": 2.0, "traversible": traversible}, handle, protocol=2)


def test_staged_eth_traversible_generates_parser_smoke_svg(tmp_path: Path) -> None:
    """A locally staged ETH traversible is converted and smoke-validated."""
    root = tmp_path / "socnavbench"
    _write_socnav_eth_fixture(
        root,
        np.array(
            [
                [False, False, False, False, False, False],
                [True, True, True, False, True, True],
                [False, False, True, True, True, False],
                [False, False, False, False, False, False],
            ],
            dtype=bool,
        ),
    )
    output_svg = tmp_path / "socnavbench_eth.svg"
    report_json = tmp_path / "report.json"

    exit_code, report = stage_svg.stage_generate_convert_smoke(
        socnav_root=root,
        output_svg=output_svg,
        report_json=report_json,
        dry_run=False,
        force_generate=False,
    )

    assert exit_code == 0
    assert report["status"] == "ready"
    assert report["generation"]["status"] == "already_present"
    assert report["conversion_ready"] is True
    assert report["smoke_ready"] is True
    assert output_svg.is_file()
    assert report_json.is_file()


def test_missing_eth_mesh_fails_closed_without_svg(tmp_path: Path) -> None:
    """Missing licensed SocNavBench inputs report a blocked state instead of a placeholder SVG."""
    output_svg = tmp_path / "socnavbench_eth.svg"
    report_json = tmp_path / "blocked.json"

    exit_code, report = stage_svg.stage_generate_convert_smoke(
        socnav_root=tmp_path / "missing_socnavbench",
        output_svg=output_svg,
        report_json=report_json,
        dry_run=False,
        force_generate=False,
    )

    assert exit_code == stage_svg.EXIT_BLOCKED
    assert report["status"] == "blocked_missing_mesh"
    assert report["conversion_ready"] is False
    assert report["smoke_ready"] is False
    assert not output_svg.exists()
    assert report["generation_preflight"]["status"] == "blocked_missing_mesh"
