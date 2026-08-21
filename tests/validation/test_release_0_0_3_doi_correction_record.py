"""Regression tests for issue #7445 release 0.0.3 DOI correction record."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

CORRECTION = Path(
    "docs/context/evidence/issue_4364_release_0_0_3_post1/release_0_0_3_doi_correction.v1.json"
)
RELEASE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml"
)
SOURCE_DIGEST = "3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7"
PLACEHOLDER = "10.5281/zenodo.<record-id>"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_correction_record_binds_tag_and_source_asset_digest() -> None:
    """The correction record must bind tag 0.0.3 and the exact source asset digest."""
    payload = _load_json(CORRECTION)

    assert payload["schema_version"] == "release_doi_correction.v1"
    assert payload["release_id"] == "paper_experiment_matrix_v2_h600_s30_v0_0_3"
    assert payload["release_tag"] == "0.0.3"
    assert payload["release_url"] == "https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.3"
    assert payload["publication_date"] == "2026-07-13"
    assert payload["asset"]["sha256"] == SOURCE_DIGEST
    assert payload["asset"]["name"] == (
        "paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final_publication_bundle.tar.gz"
    )


def test_correction_record_declares_doi_not_assigned_with_reason() -> None:
    """The record must declare doi_status not_assigned and explain the immutable placeholder."""
    payload = _load_json(CORRECTION)

    assert payload["doi_status"] == "not_assigned"
    assert "not rewritten" in payload["doi_status_reason"].lower()
    assert "no retroactive zenodo deposit is authorized" in payload["doi_status_reason"].lower()
    placeholder = payload["placeholder"]
    assert placeholder["literal"] == PLACEHOLDER
    assert placeholder["disposition"] == "retained_as_immutable_history_not_rewritten"


def test_correction_record_keeps_reproduction_not_run() -> None:
    """reproduction_status must stay not_run and carry a hash-bound promotion condition."""
    payload = _load_json(CORRECTION)

    assert payload["reproduction_status"] == "not_run"
    assert "clean environment" in payload["reproduction_status_promotion_condition"]
    assert "hash-bound" in payload["reproduction_status_promotion_condition"]


def test_release_manifest_keeps_placeholder_and_references_correction() -> None:
    """The 0.0.3 manifest must keep the placeholder and point at the correction record."""
    manifest = _load_manifest(RELEASE_MANIFEST)

    assert manifest["release_tag"] == "0.0.3"
    assert manifest["provenance"]["doi"] == PLACEHOLDER
    assert manifest["provenance"]["doi_correction_record"] == str(CORRECTION)
    assert manifest["provenance"]["reproduction_status"] == "not_run"
