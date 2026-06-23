"""Migration tests for the SDD staging consolidation (Issue #3473).

PR #3457 (Issue #2657) added an issue-specific SDD staging script and manifest. Issue #3473
consolidates that logic into the canonical external-data subsystem
(``scripts/tools/manage_external_data.py``). These tests prove that:

* the proxy-vs-dataset-backed gate decision is byte-for-byte identical whether resolved through the
  canonical functions or through the preserved ``stage_sdd_dataset_issue_2657`` wrapper;
* the no-auto-download safety contract from #2657 is preserved by the canonical subsystem;
* the canonical ``sdd`` asset specification carries the checksum policy and the manifest pointer so
  there is a single source of truth.

They never touch the network and never download anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.data import stage_sdd_dataset_issue_2657 as wrapper
from scripts.tools import manage_external_data as canonical

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(
    tmp_path: Path,
    *,
    staging_dir: Path,
    expected_tree_sha256: str | None = None,
    expected_size_bytes: int = 1024,
    download_url: str | None = None,
) -> Path:
    """Write a minimal SDD staging manifest pointing at a temp staging dir."""
    manifest = {
        "schema": "robot_sf_sdd_staging_manifest.v1",
        "asset_id": "sdd",
        "title": "Stanford Drone Dataset annotations",
        "version_tag": "sdd-test",
        "source_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "license": "CC BY-NC-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "readme_pointer": "docs/context/issue_2657_sdd_staging.md",
        "staging_dir": str(staging_dir),
        "download_url": download_url,
        "access_note": "License-gated; manual acquisition only.",
        "expected_files": [
            {
                "pattern": "**/annotations.txt",
                "kind": "file",
                "description": "Original SDD annotation text file.",
                "min_count": 1,
            }
        ],
        "expected_total_size_bytes": expected_size_bytes,
        "checksums": {
            "algorithm": "SHA-256",
            "tree_sha256": None,
            "expected_tree_sha256": expected_tree_sha256,
        },
        "local_availability": "missing",
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _stage_annotation(staging_dir: Path) -> None:
    """Create a valid SDD annotation file under the staging dir."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "annotations.txt").write_text(
        "1 0 0 10 10 0 0 0 0 Pedestrian\n", encoding="utf-8"
    )


# --- single source of truth ------------------------------------------------------------------


def test_sdd_asset_carries_checksum_policy_and_manifest_pointer() -> None:
    """The canonical ``sdd`` asset declares the staging manifest + availability gate states."""
    asset = canonical._get_asset("sdd")
    assert asset.staging_manifest_path == canonical.DEFAULT_SDD_STAGING_MANIFEST
    assert asset.proxy_mode == canonical.SDD_MODE_PROXY
    assert asset.dataset_backed_mode == canonical.SDD_MODE_DATASET_BACKED


def test_wrapper_delegates_to_canonical_symbols() -> None:
    """The #2657 wrapper re-exports the canonical implementations (no second source of truth)."""
    assert wrapper.load_manifest is canonical.load_sdd_staging_spec
    assert wrapper.validate_staging is canonical.validate_sdd_staging
    assert wrapper.resolve_scenario_prior_mode is canonical.resolve_sdd_scenario_prior_mode
    assert wrapper.MODE_PROXY == canonical.SDD_MODE_PROXY
    assert wrapper.MODE_DATASET_BACKED == canonical.SDD_MODE_DATASET_BACKED
    assert wrapper.SddStagingError is canonical.ExternalDataError


# --- proxy decision parity -------------------------------------------------------------------


def test_proxy_decision_identical_canonical_and_wrapper(tmp_path: Path) -> None:
    """A missing SDD copy yields the same proxy gate via canonical and wrapper paths."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")

    canonical_gate = canonical.resolve_sdd_scenario_prior_mode(manifest_path=manifest_path)
    wrapper_gate = wrapper.resolve_scenario_prior_mode(manifest_path=manifest_path)

    assert canonical_gate == wrapper_gate
    assert canonical_gate["mode"] == canonical.SDD_MODE_PROXY
    assert canonical_gate["dataset_backed"] is False


# --- dataset-backed decision parity ----------------------------------------------------------


def test_dataset_backed_decision_identical_canonical_and_wrapper(tmp_path: Path) -> None:
    """A staged+validated copy yields the same dataset-backed gate via both paths."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    unpinned = canonical.load_sdd_staging_spec(_write_manifest(tmp_path, staging_dir=staging_dir))
    unpinned_report = canonical.validate_sdd_staging(unpinned)
    manifest_path = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        expected_tree_sha256=unpinned_report["tree_sha256"],
    )

    canonical_gate = canonical.resolve_sdd_scenario_prior_mode(manifest_path=manifest_path)
    wrapper_gate = wrapper.resolve_scenario_prior_mode(manifest_path=manifest_path)

    assert canonical_gate == wrapper_gate
    assert canonical_gate["mode"] == canonical.SDD_MODE_DATASET_BACKED
    assert canonical_gate["dataset_backed"] is True
    assert canonical_gate["tree_sha256"]


def test_pinned_checksum_mismatch_fails_closed_in_canonical(tmp_path: Path) -> None:
    """A pinned-but-wrong checksum refuses dataset-backed mode through the canonical path."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    spec = canonical.load_sdd_staging_spec(
        _write_manifest(tmp_path, staging_dir=staging_dir, expected_tree_sha256="0" * 64)
    )

    report = canonical.validate_sdd_staging(spec)

    assert report["ok"] is False
    assert report["checksum_match"] is False
    assert report["mode"] == canonical.SDD_MODE_PROXY


# --- no-auto-download safety preserved -------------------------------------------------------


def test_canonical_download_refuses_without_confirmation(tmp_path: Path) -> None:
    """The canonical guarded download fails closed without explicit confirmation."""
    spec = canonical.load_sdd_staging_spec(_write_manifest(tmp_path, staging_dir=tmp_path / "sdd"))

    with pytest.raises(canonical.ExternalDataError, match="never auto-download"):
        canonical.run_sdd_download(spec, confirm_download=False, yes=False)


def test_canonical_download_fails_closed_on_insufficient_disk(tmp_path: Path) -> None:
    """Confirmed download still fails closed when free disk space is insufficient."""
    spec = canonical.load_sdd_staging_spec(
        _write_manifest(
            tmp_path,
            staging_dir=tmp_path / "sdd",
            expected_size_bytes=10**18,
            download_url="https://example.invalid/sdd.zip",
        )
    )

    with pytest.raises(canonical.ExternalDataError, match="Insufficient disk space"):
        canonical.run_sdd_download(spec, confirm_download=True, yes=True)


def test_canonical_download_fails_closed_without_url(tmp_path: Path) -> None:
    """Even confirmed + disk-OK, a missing download URL fails closed (license-gated)."""
    spec = canonical.load_sdd_staging_spec(
        _write_manifest(tmp_path, staging_dir=tmp_path / "sdd", download_url=None)
    )

    with pytest.raises(canonical.ExternalDataError, match="No download URL is configured"):
        canonical.run_sdd_download(spec, confirm_download=True, yes=True)


def test_canonical_plan_never_downloads_and_reports_disk_check(tmp_path: Path) -> None:
    """A canonical plan reports the disk check and declares no download."""
    spec = canonical.load_sdd_staging_spec(_write_manifest(tmp_path, staging_dir=tmp_path / "sdd"))

    plan = canonical.build_sdd_plan(spec)

    assert plan["would_download"] is False
    assert plan["auto_download"] is False
    assert plan["scenario_prior_mode"] == canonical.SDD_MODE_PROXY
    assert plan["disk_check"]["required_bytes"] == spec.expected_total_size_bytes


def test_repo_sdd_manifest_loads_through_canonical_asset() -> None:
    """The shipped manifest referenced by the asset parses and declares no download URL."""
    spec = canonical.load_sdd_staging_spec()

    assert spec.asset_id == "sdd"
    assert spec.download_url is None
    assert spec.local_availability_declared == "missing"
    assert spec.expected_total_size_bytes > 0


def test_manifest_null_staging_dir_fails_closed(tmp_path: Path) -> None:
    """Explicit YAML null for the required staging dir must not become the string 'None'."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["staging_dir"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(canonical.ExternalDataError, match="missing required `staging_dir`"):
        canonical.load_sdd_staging_spec(manifest_path)


def test_manifest_null_optional_strings_remain_empty_or_unpinned(tmp_path: Path) -> None:
    """Explicit YAML nulls for optional manifest fields should stay missing, not 'None'."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["source_url"] = None
    manifest["download_url"] = None
    manifest["local_availability"] = None
    manifest["checksums"]["algorithm"] = None
    manifest["checksums"]["expected_tree_sha256"] = None
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    spec = canonical.load_sdd_staging_spec(manifest_path)

    assert spec.source_url == ""
    assert spec.download_url is None
    assert spec.local_availability_declared == "missing"
    assert spec.checksum_algorithm == "SHA-256"
    assert spec.expected_tree_sha256 is None


@pytest.mark.parametrize("pattern", ["../secret.txt", "/tmp/secret.txt"])
def test_manifest_rejects_expected_file_patterns_outside_staging_dir(
    tmp_path: Path, pattern: str
) -> None:
    """Expected-file globs must not escape the staging directory."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_files"][0]["pattern"] = pattern
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(canonical.ExternalDataError, match="must stay within staging_dir"):
        canonical.load_sdd_staging_spec(manifest_path)


def test_manifest_rejects_invalid_expected_file_kind(tmp_path: Path) -> None:
    """Expected-file kind should fail during manifest parsing, before glob matching."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_files"][0]["kind"] = "symlink"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(canonical.ExternalDataError, match="must be one of"):
        canonical.load_sdd_staging_spec(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_total_size_bytes", "not-an-int", "must be an integer"),
        ("checksums", [], "must be a mapping"),
    ],
)
def test_manifest_type_errors_raise_external_data_error(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Manifest type errors should use the CLI's fail-closed ExternalDataError path."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(canonical.ExternalDataError, match=message):
        canonical.load_sdd_staging_spec(manifest_path)


def test_plan_guidance_uses_registered_sdd_cli_command_names(tmp_path: Path) -> None:
    """User guidance should name the actual sdd-* subcommands."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    spec = canonical.load_sdd_staging_spec(_write_manifest(tmp_path, staging_dir=staging_dir))

    plan = canonical.build_sdd_plan(spec)

    assert "sdd-validate" in plan["validation"]["checksum_skipped"]
    assert "sdd-validate" in plan["validation"]["action"]
    assert "sdd-download --confirm-download" in plan["next_step"]


def test_cached_sdd_gate_revalidates_current_tree_checksum(tmp_path: Path) -> None:
    """A stale status file must not unlock dataset-backed mode after files change."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    unpinned = canonical.load_sdd_staging_spec(_write_manifest(tmp_path, staging_dir=staging_dir))
    unpinned_report = canonical.validate_sdd_staging(unpinned)
    manifest_path = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        expected_tree_sha256=unpinned_report["tree_sha256"],
    )
    spec = canonical.load_sdd_staging_spec(manifest_path)
    valid_report = canonical.validate_sdd_staging(spec)
    canonical.write_sdd_staging_status(spec, valid_report)
    (staging_dir / "annotations.txt").write_text(
        "1 0 0 99 99 0 0 0 0 Pedestrian\n", encoding="utf-8"
    )

    gate = canonical.resolve_sdd_scenario_prior_mode(manifest_path=manifest_path)

    assert gate["dataset_backed"] is False
    assert gate["mode"] == canonical.SDD_MODE_PROXY
    assert gate["report"]["checksum_match"] is False
