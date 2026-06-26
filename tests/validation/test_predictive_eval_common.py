"""Tests for shared predictive validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.predictive_eval_common import load_seed_manifest, make_subset_scenarios


def _write_matrix(tmp_path: Path) -> Path:
    """Write a small scenario matrix with relative assets."""
    matrix = tmp_path / "matrix.yaml"
    map_path = tmp_path / "maps" / "open.svg"
    map_path.parent.mkdir()
    map_path.write_text("<svg />", encoding="utf-8")
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "planner_sanity_simple",
                        "map_file": "maps/open.svg",
                        "seeds": [101],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return matrix


def test_make_subset_scenarios_applies_manifest_and_resolves_paths(tmp_path: Path) -> None:
    """Valid manifests should select scenarios and preserve existing path resolution."""
    matrix = _write_matrix(tmp_path)

    subset = make_subset_scenarios(matrix, {"planner_sanity_simple": [111, 112]})

    assert len(subset) == 1
    assert subset[0]["name"] == "planner_sanity_simple"
    assert subset[0]["seeds"] == [111, 112]
    assert Path(subset[0]["map_file"]) == (tmp_path / "maps" / "open.svg").resolve()


def test_make_subset_scenarios_rejects_unknown_manifest_keys(tmp_path: Path) -> None:
    """Seed manifests should fail fast when they do not match the matrix."""
    matrix = _write_matrix(tmp_path)

    with pytest.raises(ValueError, match="bogus_scenario"):
        make_subset_scenarios(
            matrix,
            {
                "planner_sanity_simple": [111],
                "bogus_scenario": [111],
            },
        )


def test_make_subset_scenarios_rejects_empty_selection(tmp_path: Path) -> None:
    """An empty manifest should not produce a silently empty validation slice."""
    matrix = _write_matrix(tmp_path)

    with pytest.raises(ValueError, match="selected no scenarios"):
        make_subset_scenarios(matrix, {})


def test_load_seed_manifest_rejects_invalid_entries(tmp_path: Path) -> None:
    """Hard-case seed manifests fail closed before campaign execution."""
    manifest = tmp_path / "hard.yaml"
    manifest.write_text(
        yaml.safe_dump({"planner_sanity_simple": [101, 101]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate seeds"):
        load_seed_manifest(manifest)

    manifest.write_text(yaml.safe_dump({"planner_sanity_simple": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no seeds"):
        load_seed_manifest(manifest)
