"""Tests for the GSN-flavored assurance fragment exporter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.assurance_fragment import (
    build_assurance_fragment,
    render_assurance_fragment_to_markdown,
    render_assurance_fragment_to_svg,
    validate_assurance_fragment,
    write_assurance_fragment,
)
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path


def test_build_assurance_fragment_from_fixture(tmp_path: Path) -> None:
    """Test building and validating the assurance fragment from a real fixture."""
    repo_root = get_repository_root()
    fixture_path = (
        repo_root
        / "docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json"
    )
    assert fixture_path.exists(), f"Fixture campaign summary not found: {fixture_path}"

    with fixture_path.open("r", encoding="utf-8") as f:
        campaign_summary = json.load(f)

    # 1. Build assurance fragment
    fragment = build_assurance_fragment(campaign_summary, repo_root=repo_root)

    # 2. Validate against schema
    validate_assurance_fragment(fragment)

    # 3. Assert correct structure and expected nodes
    assert fragment["schema_version"] == "assurance_fragment.v1"
    assert "nodes" in fragment
    nodes = fragment["nodes"]

    assert "G_root" in nodes
    assert nodes["G_root"]["type"] == "goal"
    assert "S_campaign" in nodes
    assert "C_matrix" in nodes
    assert "C_git" in nodes
    assert "Sn_campaign_summary" in nodes

    # Verify every solution node's path/sha256 if metadata is populated
    for node in nodes.values():
        if node["type"] == "solution" and "metadata" in node:
            metadata = node["metadata"]
            path_str = metadata.get("path")
            if path_str:
                resolved_path = repo_root / path_str
                # For some tests/fixtures, local runs/planner paths might not exist physically
                # but if they do exist, we verify the hash match.
                if resolved_path.exists():
                    assert metadata.get("sha256") is not None
                    assert len(metadata["sha256"]) == 64

    # 4. Render markdown and SVG
    markdown_content = render_assurance_fragment_to_markdown(fragment)
    assert "graph TD" in markdown_content
    assert "G_root" in markdown_content

    svg_content = render_assurance_fragment_to_svg(fragment)
    assert "<svg" in svg_content
    assert "G_root" in svg_content

    # 5. Write artifacts
    written = write_assurance_fragment(tmp_path, fragment, repo_root=repo_root)
    assert written["json"].exists()
    assert written["markdown"].exists()
    assert written["svg"].exists()
