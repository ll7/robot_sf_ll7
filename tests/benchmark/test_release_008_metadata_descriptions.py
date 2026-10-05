"""Catch publication wording drift using the real F2 calibration-bound identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.release_protocol import _replace_identity_tokens

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs/context/evidence/2026-10-05_freeze008_box4_seed_admission"
CALIBRATION = ROOT / "docs/context/evidence/2026-10-04_freeze008_f2_calibration"
TEMPLATES = {
    "main": "benchmark_data_release_s30_h600_zenodo_metadata.template.json",
    "doorway": "three_width_doorway_release_0_0_8_v1_zenodo_metadata.template.json",
}


def _render(track: str) -> tuple[str, dict]:
    """Render current template bytes with authentic, already resolved F2 coordinates."""
    identity = json.loads((PACKET / track / "release_identity.resolved.json").read_bytes())
    template = json.loads((ROOT / "configs/benchmarks/releases" / TEMPLATES[track]).read_bytes())
    rendered = _replace_identity_tokens(
        template,
        {
            "source_sha": identity["source_commit"],
            "latest_main_base_commit": identity["resolved_manifest"]["latest_main_base_commit"],
            "release_tag": identity["release_tag"],
            "concept_doi": identity["publication"]["concept_doi"],
            "version_doi": identity["publication"]["version_doi"],
        },
    )
    return rendered["metadata"]["description"], identity


@pytest.mark.parametrize("track", TEMPLATES)
def test_v2_calibration_context_description_does_not_claim_calibration_failed(track):
    """Acquired dev-anchor context cannot be described as failed calibration."""
    description, identity = _render(track)
    anchors = json.loads((CALIBRATION / "anchors.v2.0.acquired.json").read_bytes())
    receipt = json.loads((CALIBRATION / "determinism-receipt.json").read_bytes())
    assert anchors["calibration"]["source_commit"] == identity["source_commit"]
    assert anchors["calibration"]["seeds"] == [1001, 1002]
    assert receipt["source_commit"] == identity["source_commit"]
    assert identity["determinism_receipt"]["sha256"] == (
        "cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464"
    )
    assert "snqi_v2_binding" in identity["resolved_manifest"]["metrics"]
    assert identity["resolved_manifest"]["release_contract"]["snqi_claim_policy"] == (
        "advisory_no_ranking"
    )
    assert "calibration failed" not in description.casefold(), description
    assert all(term in description.casefold() for term in ("snqi", "advisory", "ranking"))


@pytest.mark.parametrize("track", TEMPLATES)
def test_non_main_freeze_first_parent_is_not_described_as_mainline_base(track):
    """F2's independently checked non-main parent must retain its actual role."""
    description, identity = _render(track)
    # This exact first parent is a freeze-branch commit, not a main ancestor
    # (author's F2 ruling / source audit, 2026-10-05). No Git fetch is needed in CI.
    parent = "1261295887901e566a93da87a4559ea1040808a7"
    assert identity["resolved_manifest"]["latest_main_base_commit"] == parent
    assert "mainline base" not in description.casefold(), description
    assert f"freeze-branch base (first parent of the source commit) {parent}" in description
