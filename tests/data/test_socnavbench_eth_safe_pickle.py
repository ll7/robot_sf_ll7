"""Tests for safe pickle integration in socnavbench_eth module."""

from __future__ import annotations

import os
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


class TestSocNavBenchEthSafePickle:
    """Tests for safe pickle in socnavbench_eth.load_shape_contract."""

    def test_legitimate_traversible_loads(self, tmp_path: Path) -> None:
        """Verify that a legitimate traversible pickle loads successfully."""
        root = tmp_path / "socnavbench"
        _stage_minimal_eth_fixture(root)

        shape = socnavbench_eth.load_shape_contract(root)
        assert shape.resolution == 5.0
        assert shape.traversible_shape == (2, 2)
        assert shape.traversible_dtype == "bool"

    def test_malicious_pickle_rejected(self, tmp_path: Path) -> None:
        """Verify that malicious pickle is rejected before execution."""

        class Evil:
            def __reduce__(self):
                return (os.system, ("echo should-not-run",))

        root = tmp_path / "socnavbench"
        _stage_minimal_eth_fixture(root)
        layout = socnavbench_eth.expected_layout(root)

        with layout.traversible_pickle.open("wb") as handle:
            pickle.dump(Evil(), handle, protocol=2)

        with pytest.raises(socnavbench_eth.SocNavBenchEthDataError, match="Unsafe pickle rejected"):
            socnavbench_eth.load_shape_contract(root)

    def test_malicious_subprocess_rejected(self, tmp_path: Path) -> None:
        """Verify that malicious subprocess pickle is rejected."""
        import subprocess

        class Evil:
            def __reduce__(self):
                return (subprocess.call, (["echo", "should-not-run"],))

        root = tmp_path / "socnavbench"
        _stage_minimal_eth_fixture(root)
        layout = socnavbench_eth.expected_layout(root)

        with layout.traversible_pickle.open("wb") as handle:
            pickle.dump(Evil(), handle, protocol=2)

        with pytest.raises(socnavbench_eth.SocNavBenchEthDataError):
            socnavbench_eth.load_shape_contract(root)

    def test_numpy_array_traversible_loads(self, tmp_path: Path) -> None:
        """Verify that a traversible with float array loads successfully."""
        root = tmp_path / "socnavbench"
        layout = socnavbench_eth.expected_layout(root)
        layout.mesh_dir.mkdir(parents=True)
        (layout.mesh_dir / "mesh.obj").write_text("# synthetic mesh marker\n", encoding="utf-8")
        layout.traversible_pickle.parent.mkdir(parents=True)
        with layout.traversible_pickle.open("wb") as handle:
            pickle.dump(
                {
                    "resolution": 0.05,
                    "traversible": np.zeros((10, 10), dtype=np.float32),
                },
                handle,
                protocol=2,
            )

        shape = socnavbench_eth.load_shape_contract(root)
        assert shape.resolution == 0.05
        assert shape.traversible_shape == (10, 10)
        assert shape.traversible_dtype == "float32"
