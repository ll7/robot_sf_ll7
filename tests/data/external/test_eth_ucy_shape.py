"""Skip-if-absent shape-contract tests for ETH/UCY external data.

These tests never require the license-gated ETH/UCY bytes. Exactly one test path
depends on locally staged real data and skips when it is absent; every other test
builds a synthetic layout under ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import eth_ucy

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    """Create parent directories and write a small text fixture file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# A minimal well-formed obsmat block: eight whitespace columns (frame, id,
# x, z, y, vx, vz, vy) matching the canonical BIWI layout, two data rows.
_OBSMAT_ROWS = "1.0 1.0 2.5 0.0 4.5 0.1 0.0 0.2\n11.0 1.0 2.6 0.0 4.6 0.1 0.0 0.2\n"

# A minimal normalized UCY trajectory .txt block: four columns (frame, id, x, y).
_TXT_ROWS = "1 1 2.0 3.0\n10 1 2.1 3.1\n1 2 5.0 6.0\n"

# A minimal UCY .vsp block: integer agent-count header plus numeric control rows.
_VSP_ROWS = "2\n0 1 2.0 3.0\n10 1 2.1 3.1\n"


def _stage_minimal_dataset(root: Path) -> None:
    """Create a tiny documented ETH/UCY layout with all five splits.

    ETH splits use ``obsmat.txt``; UCY splits exercise both accepted formats
    (``.txt`` for univ/zara01 and ``.vsp`` for zara02).
    """

    _write(root / "eth" / "obsmat.txt", _OBSMAT_ROWS)
    _write(root / "hotel" / "obsmat.txt", _OBSMAT_ROWS)
    _write(root / "univ" / "univ.txt", _TXT_ROWS)
    _write(root / "zara01" / "zara01.txt", _TXT_ROWS)
    _write(root / "zara02" / "crowds_zara02.vsp", _VSP_ROWS)


def test_eth_ucy_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without staged ETH/UCY bytes skip instead of failing."""

    monkeypatch.setenv(eth_ucy.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not eth_ucy.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied ETH/UCY contract")


def test_eth_ucy_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """A complete synthetic layout resolves and produces per-split shape metadata."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)

    assert eth_ucy.is_available(root)
    dataset = eth_ucy.require_available(root)
    assert {split.split for split in dataset.splits} == {
        "eth",
        "hotel",
        "univ",
        "zara01",
        "zara02",
    }

    contract = eth_ucy.load_shape_contract(root)
    assert contract["asset_id"] == "eth-ucy"
    assert contract["docs_path"] == "docs/datasets/eth-ucy.md"
    splits = contract["splits"]

    assert splits["eth"]["group"] == "eth"
    assert splits["eth"]["format"] == "obsmat"
    assert splits["eth"]["row_count"] == 2
    assert splits["eth"]["column_count"] == 8
    assert splits["eth"]["delimiter"] == "whitespace"

    assert splits["univ"]["group"] == "ucy"
    assert splits["univ"]["format"] == "txt"
    assert splits["univ"]["row_count"] == 3
    assert splits["univ"]["column_count"] == 4

    assert splits["zara02"]["format"] == "vsp"
    assert splits["zara02"]["row_count"] == 2
    assert splits["zara02"]["agent_count"] == 2


def test_eth_ucy_accepts_comma_delimited_rows(tmp_path: Path) -> None:
    """Comma-delimited normalized trajectory files parse and report the delimiter."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "univ" / "univ.txt", "1,1,2.0,3.0\n10,1,2.1,3.1\n")

    contract = eth_ucy.load_shape_contract(root)
    assert contract["splits"]["univ"]["delimiter"] == "comma"
    assert contract["splits"]["univ"]["column_count"] == 4


def test_eth_ucy_missing_split_is_unavailable(tmp_path: Path) -> None:
    """A missing split file makes the dataset unavailable and raises with docs pointer."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    (root / "zara02" / "crowds_zara02.vsp").unlink()

    assert not eth_ucy.is_available(root)
    with pytest.raises(eth_ucy.EthUcyDataError, match="zara02"):
        eth_ucy.require_available(root)


def test_eth_ucy_wrong_layout_is_unavailable(tmp_path: Path) -> None:
    """A wrong directory layout is unavailable and the error names the docs page."""

    root = tmp_path / "eth-ucy"
    # Stage trajectory files at the root instead of the documented split dirs.
    _write(root / "obsmat.txt", _OBSMAT_ROWS)

    assert not eth_ucy.is_available(root)
    with pytest.raises(eth_ucy.EthUcyDataError, match="docs/datasets/eth-ucy.md"):
        eth_ucy.load_shape_contract(root)


def test_eth_ucy_empty_file_fails_closed(tmp_path: Path) -> None:
    """An empty staged split file fails closed with an actionable error."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "eth" / "obsmat.txt", "")

    with pytest.raises(eth_ucy.EthUcyDataError, match="no numeric data rows"):
        eth_ucy.load_shape_contract(root)


def test_eth_ucy_non_numeric_value_fails_closed(tmp_path: Path) -> None:
    """A non-numeric trajectory value fails closed with an actionable error."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "hotel" / "obsmat.txt", "1.0 1.0 x 0.0 4.5 0.1 0.0 0.2\n")

    with pytest.raises(eth_ucy.EthUcyDataError, match="non-numeric"):
        eth_ucy.load_shape_contract(root)


def test_eth_ucy_non_finite_value_fails_closed(tmp_path: Path) -> None:
    """A non-finite trajectory value fails closed with an actionable error."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "hotel" / "obsmat.txt", "1.0 1.0 2.5 0.0 inf 0.1 0.0 0.2\n")

    with pytest.raises(eth_ucy.EthUcyDataError, match="non-finite"):
        eth_ucy.load_shape_contract(root)


def test_eth_ucy_too_few_columns_fails_closed(tmp_path: Path) -> None:
    """A trajectory file narrower than the structural floor fails closed."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "zara01" / "zara01.txt", "1 2.0 3.0\n10 2.1 3.1\n")

    with pytest.raises(eth_ucy.EthUcyDataError, match="columns"):
        eth_ucy.load_shape_contract(root)


def test_eth_ucy_vsp_bad_header_fails_closed(tmp_path: Path) -> None:
    """A .vsp file without an integer agent-count header fails closed."""

    root = tmp_path / "eth-ucy"
    _stage_minimal_dataset(root)
    _write(root / "zara02" / "crowds_zara02.vsp", "header\n0 1 2.0 3.0\n")

    with pytest.raises(eth_ucy.EthUcyDataError, match="integer agent/spline count"):
        eth_ucy.load_shape_contract(root)
