"""Focused tests for the SDD bring-your-own (BYO) staging preflight (Issue #1497).

Issue #1497 owns the provenance-safe staging gate for licensed Stanford Drone Dataset (SDD)
annotations. Under the BYO-dataset reframe (#3065) the contributor supplies a copy they already
have rights to; the repository never downloads, redistributes, or commits the annotations. These
tests cover the two new manifest fields (an ordered ``retrieval_recipe`` and a
``license_acknowledgment`` opt-in) and the ``sdd-preflight`` readiness gate.

The preflight must fail closed: it is ``ready`` only when the license acknowledgment is affirmed
AND the annotation files are present locally. None of these tests download or ingest any data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.tools import manage_external_data as canonical

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(
    tmp_path: Path,
    *,
    staging_dir: Path,
    retrieval_recipe: list[str] | None = ("step one", "step two"),
    license_acknowledgment: dict | None = None,
) -> Path:
    """Write a minimal SDD staging manifest with the BYO opt-in fields under test."""
    manifest: dict = {
        "schema": "robot_sf_sdd_staging_manifest.v1",
        "asset_id": "sdd",
        "title": "Stanford Drone Dataset annotations",
        "version_tag": "sdd-test",
        "source_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "license": "CC BY-NC-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "readme_pointer": "docs/context/issue_2657_sdd_staging.md",
        "staging_dir": str(staging_dir),
        "download_url": None,
        "access_note": "License-gated; manual acquisition only.",
        "expected_files": [
            {
                "pattern": "**/annotations.txt",
                "kind": "file",
                "description": "Original SDD annotation text file.",
                "min_count": 1,
            }
        ],
        "expected_total_size_bytes": 1024,
        "checksums": {"algorithm": "SHA-256", "tree_sha256": None, "expected_tree_sha256": None},
        "local_availability": "missing",
    }
    if retrieval_recipe is not None:
        manifest["retrieval_recipe"] = list(retrieval_recipe)
    if license_acknowledgment is not None:
        manifest["license_acknowledgment"] = license_acknowledgment
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _stage_annotation(staging_dir: Path) -> None:
    """Create a valid SDD annotation file under the staging dir (no real data)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "annotations.txt").write_text(
        "1 0 0 10 10 0 0 0 0 Pedestrian\n", encoding="utf-8"
    )


# --- manifest field parsing ------------------------------------------------------------------


def test_repo_manifest_ships_unacknowledged_byo_contract() -> None:
    """The committed manifest must never imply redistribution rights (acknowledged stays false)."""
    spec = canonical.load_sdd_staging_spec()
    ack = spec.license_acknowledgment
    assert ack.required is True
    assert ack.acknowledged is False
    assert ack.satisfied is False
    assert spec.retrieval_recipe  # concrete ordered acquisition steps are present
    assert spec.license  # license/provenance fields are carried for the preflight


def test_missing_byo_fields_default_to_blocked(tmp_path: Path) -> None:
    """A manifest without the BYO fields defaults to required-but-unacknowledged (fail closed)."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        retrieval_recipe=None,
        license_acknowledgment=None,
    )
    spec = canonical.load_sdd_staging_spec(manifest)
    assert spec.retrieval_recipe == ()
    assert spec.license_acknowledgment.required is True
    assert spec.license_acknowledgment.satisfied is False


def test_non_boolean_acknowledgment_fails_closed(tmp_path: Path) -> None:
    """A YAML string like 'true' must be rejected so the gate cannot be bypassed by a typo."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        license_acknowledgment={"required": True, "acknowledged": "true"},
    )
    with pytest.raises(canonical.ExternalDataError, match="acknowledged.*boolean"):
        canonical.load_sdd_staging_spec(manifest)


def test_required_false_is_rejected_so_gate_cannot_be_disabled(tmp_path: Path) -> None:
    """`required: false` must fail closed: the SDD acknowledgment is mandatory and non-disableable."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        license_acknowledgment={"required": False, "acknowledged": False},
    )
    with pytest.raises(canonical.ExternalDataError, match="required.*cannot be disabled"):
        canonical.load_sdd_staging_spec(manifest)


def test_non_string_statement_fails_closed(tmp_path: Path) -> None:
    """A non-string statement (e.g. a list) must be rejected, not silently coerced."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        license_acknowledgment={"required": True, "acknowledged": False, "statement": ["a", "b"]},
    )
    with pytest.raises(canonical.ExternalDataError, match="statement.*string"):
        canonical.load_sdd_staging_spec(manifest)


def test_malformed_retrieval_recipe_fails_closed(tmp_path: Path) -> None:
    """A retrieval recipe with an empty step must fail closed."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        retrieval_recipe=["valid step", "  "],
    )
    with pytest.raises(canonical.ExternalDataError, match="retrieval_recipe"):
        canonical.load_sdd_staging_spec(manifest)


# --- preflight readiness gate ----------------------------------------------------------------


def test_preflight_blocked_when_unacknowledged_and_unstaged(tmp_path: Path) -> None:
    """Both prerequisites unmet -> blocked external input, not ready."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        license_acknowledgment={"required": True, "acknowledged": False},
    )
    spec = canonical.load_sdd_staging_spec(manifest)
    report = canonical.build_sdd_preflight(spec)
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert set(report["unmet_prerequisites"]) == {"license_acknowledgment", "annotations_staged"}
    assert report["scenario_prior_mode"] == canonical.SDD_MODE_PROXY


def test_preflight_blocked_when_acknowledged_but_unstaged(tmp_path: Path) -> None:
    """Acknowledged license alone is insufficient; missing annotations keep it blocked."""
    manifest = _write_manifest(
        tmp_path,
        staging_dir=tmp_path / "sdd",
        license_acknowledgment={"required": True, "acknowledged": True},
    )
    spec = canonical.load_sdd_staging_spec(manifest)
    report = canonical.build_sdd_preflight(spec)
    assert report["ready"] is False
    assert report["unmet_prerequisites"] == ["annotations_staged"]
    assert report["license_acknowledgment"]["satisfied"] is True


def test_preflight_blocked_when_staged_but_unacknowledged(tmp_path: Path) -> None:
    """Staged annotations without an affirmed license stay blocked (license gate is real)."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    manifest = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        license_acknowledgment={"required": True, "acknowledged": False},
    )
    spec = canonical.load_sdd_staging_spec(manifest)
    report = canonical.build_sdd_preflight(spec)
    assert report["ready"] is False
    assert report["unmet_prerequisites"] == ["license_acknowledgment"]


def test_preflight_ready_when_acknowledged_and_staged(tmp_path: Path) -> None:
    """Both prerequisites satisfied -> ready, unblocked, recipe surfaced."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    manifest = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        retrieval_recipe=["read license", "download", "extract"],
        license_acknowledgment={"required": True, "acknowledged": True},
    )
    spec = canonical.load_sdd_staging_spec(manifest)
    report = canonical.build_sdd_preflight(spec)
    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["unmet_prerequisites"] == []
    assert report["retrieval_recipe"] == ["read license", "download", "extract"]


def test_preflight_cli_exit_codes(tmp_path: Path) -> None:
    """The sdd-preflight CLI returns 2 when blocked and 0 when ready."""
    staging_dir = tmp_path / "sdd"
    blocked_manifest = _write_manifest(tmp_path, staging_dir=staging_dir)
    assert canonical._handle_sdd_preflight(_ns(manifest=blocked_manifest)) == 2

    _stage_annotation(staging_dir)
    ready_manifest = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        license_acknowledgment={"required": True, "acknowledged": True},
    )
    assert canonical._handle_sdd_preflight(_ns(manifest=ready_manifest)) == 0


class _ns:
    """Tiny argparse.Namespace stand-in for the manifest argument."""

    def __init__(self, *, manifest: Path) -> None:
        self.manifest = manifest
        self.json = True
