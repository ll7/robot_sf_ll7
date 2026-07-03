"""Skip-if-absent shape-contract tests for SocNavBench ETH external data."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.data.external import socnavbench_eth

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


def test_socnavbench_eth_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without SocNavBench ETH bytes should skip, not fail."""

    monkeypatch.setenv(socnavbench_eth.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not socnavbench_eth.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied ETH contract")


def test_socnavbench_eth_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """Available data path checks layout and traversible pickle structure."""

    root = tmp_path / "socnavbench"
    _stage_minimal_eth_fixture(root)

    assert socnavbench_eth.is_available(root)
    layout = socnavbench_eth.require_available(root)
    assert layout.mesh_dir.is_dir()
    assert layout.traversible_pickle.is_file()

    shape = socnavbench_eth.load_shape_contract(root)
    assert shape.resolution == 5.0
    assert shape.traversible_shape == (2, 2)
    assert shape.traversible_dtype == "bool"


def test_socnavbench_eth_shape_contract_rejects_malformed_pickle(tmp_path: Path) -> None:
    """Malformed staged traversible files fail closed with an actionable error."""

    root = tmp_path / "socnavbench"
    _stage_minimal_eth_fixture(root)
    layout = socnavbench_eth.expected_layout(root)
    with layout.traversible_pickle.open("wb") as handle:
        pickle.dump({"resolution": 5.0}, handle, protocol=2)

    with pytest.raises(socnavbench_eth.SocNavBenchEthDataError, match="missing required keys"):
        socnavbench_eth.load_shape_contract(root)
