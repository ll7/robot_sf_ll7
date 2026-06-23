"""Tests for the reproducible SDD staging tool (Issue #2657).

These tests use temp dirs and explicit manifest overrides. They never touch the network and never
download anything. They verify the safety contract:

* default invocation plans/reports only (no download, no network);
* download refuses without explicit confirmation;
* the disk-space check fails closed when free space is insufficient;
* availability status reports ``missing`` when SDD is absent and ``staged`` once validated;
* the scenario-prior mode gate forces ``proxy_schema_smoke`` when SDD is absent/unvalidated and
  unlocks ``dataset_backed_prior`` only when a validated copy is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from scripts.data import stage_sdd_dataset_issue_2657 as stage


def _write_manifest(
    tmp_path: Path,
    *,
    staging_dir: Path,
    expected_size_bytes: int = 1024,
    download_url: str | None = None,
    expected_tree_sha256: str | None = None,
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


def _download_args(*, confirm: bool = False, yes: bool = False) -> argparse.Namespace:
    """Build a download-subcommand args namespace."""
    return argparse.Namespace(confirm_download=confirm, yes=yes)


# --- plan / default behavior -----------------------------------------------------------------


def test_default_run_plans_only_and_never_downloads(tmp_path: Path, capsys) -> None:
    """A default invocation must only plan/report and never download."""
    staging_dir = tmp_path / "sdd"
    manifest_path = _write_manifest(tmp_path, staging_dir=staging_dir)

    rc = stage.main(["--manifest", str(manifest_path), "--json"])

    assert rc == 0
    out = capsys.readouterr().out
    assert '"would_download": false' in out
    assert '"auto_download": false' in out
    assert '"scenario_prior_mode": "proxy_schema_smoke"' in out
    # Plan-only must not create the staging dir or any data.
    assert not staging_dir.exists()


def test_plan_reports_disk_check(tmp_path: Path) -> None:
    """The plan payload reports the pre-download disk-space check."""
    manifest = stage.load_manifest(_write_manifest(tmp_path, staging_dir=tmp_path / "sdd"))
    plan = stage.build_plan(manifest)

    assert plan["would_download"] is False
    assert "disk_check" in plan
    assert plan["disk_check"]["required_bytes"] == manifest.expected_total_size_bytes


def test_plan_skips_tree_checksum_for_present_files(tmp_path: Path) -> None:
    """Plan-only output must not hash a potentially large staged dataset."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    manifest = stage.load_manifest(_write_manifest(tmp_path, staging_dir=staging_dir))

    plan = stage.build_plan(manifest)

    assert plan["validation"]["checksum_skipped"]
    assert plan["local_availability"] == "present_unvalidated"
    assert plan["scenario_prior_mode"] == stage.MODE_PROXY


def test_manifest_rejects_staging_dir_escape(tmp_path: Path) -> None:
    """A manifest cannot point staging outside the repo output or manifest-local roots."""
    outside = tmp_path.parent / "outside-sdd"
    manifest_path = _write_manifest(tmp_path, staging_dir=outside)

    with pytest.raises(stage.SddStagingError, match="outside allowed roots"):
        stage.load_manifest(manifest_path)


def test_manifest_rejects_path_traversal_segments(tmp_path: Path) -> None:
    """Relative traversal segments in manifest paths fail closed before resolution."""
    manifest_path = _write_manifest(tmp_path, staging_dir=Path("../outside-sdd"))

    with pytest.raises(stage.SddStagingError, match="path traversal"):
        stage.load_manifest(manifest_path)


def test_manifest_rejects_staging_dir_symlink(tmp_path: Path) -> None:
    """The unresolved staging dir cannot be a symlink."""
    target = tmp_path / "real-sdd"
    target.mkdir()
    link = tmp_path / "linked-sdd"
    link.symlink_to(target, target_is_directory=True)
    manifest_path = _write_manifest(tmp_path, staging_dir=link)

    with pytest.raises(stage.SddStagingError, match="must not be a symlink"):
        stage.load_manifest(manifest_path)


def test_manifest_parse_errors_are_sdd_staging_errors(tmp_path: Path) -> None:
    """Malformed expected-file entries should not leak KeyError/TypeError tracebacks."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["expected_files"] = [{"kind": "file"}]
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(stage.SddStagingError, match=r"expected_files\[0\]\.pattern"):
        stage.load_manifest(manifest_path)


# --- download confirmation gate --------------------------------------------------------------


def test_download_refuses_without_confirmation(tmp_path: Path) -> None:
    """Download must fail closed when not explicitly confirmed (default = no download)."""
    manifest = stage.load_manifest(_write_manifest(tmp_path, staging_dir=tmp_path / "sdd"))

    with pytest.raises(stage.SddStagingError, match="never auto-download"):
        stage.run_download(manifest, _download_args(confirm=False))


def test_confirm_flag_alone_still_requires_prompt(monkeypatch, tmp_path: Path) -> None:
    """`--confirm-download` without `--yes` requires an interactive y/N that defaults to no."""
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert stage._confirm_download(_download_args(confirm=True, yes=False)) is False
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert stage._confirm_download(_download_args(confirm=True, yes=False)) is True


def test_yes_flag_confirms_without_prompt() -> None:
    """`--yes` provides non-interactive confirmation only alongside `--confirm-download`."""
    assert stage._confirm_download(_download_args(confirm=True, yes=True)) is True
    assert stage._confirm_download(_download_args(confirm=False, yes=True)) is False


# --- disk-space fail-closed ------------------------------------------------------------------


def test_disk_check_fails_closed_when_insufficient(tmp_path: Path) -> None:
    """An impossibly large expected size makes the disk check refuse the download."""
    manifest = stage.load_manifest(
        _write_manifest(
            tmp_path,
            staging_dir=tmp_path / "sdd",
            expected_size_bytes=10**18,  # ~1 EiB, larger than any real free space
            download_url="https://example.invalid/sdd.zip",
        )
    )

    disk = stage.check_disk_space(manifest)
    assert disk["sufficient"] is False

    with pytest.raises(stage.SddStagingError, match="Insufficient disk space"):
        stage.run_download(manifest, _download_args(confirm=True, yes=True))


def test_download_with_no_url_fails_closed_after_confirmation(tmp_path: Path) -> None:
    """Even confirmed + disk-OK, a missing download URL fails closed (license-gated)."""
    manifest = stage.load_manifest(
        _write_manifest(tmp_path, staging_dir=tmp_path / "sdd", download_url=None)
    )

    with pytest.raises(stage.SddStagingError, match="No download URL is configured"):
        stage.run_download(manifest, _download_args(confirm=True, yes=True))


# --- availability status ---------------------------------------------------------------------


def test_status_reports_missing_when_absent(tmp_path: Path) -> None:
    """Availability is `missing` and mode is proxy when SDD is not staged."""
    manifest = stage.load_manifest(_write_manifest(tmp_path, staging_dir=tmp_path / "sdd"))
    report = stage.validate_staging(manifest)

    assert report["ok"] is False
    assert report["local_availability"] == "missing"
    assert report["mode"] == stage.MODE_PROXY
    assert report["availability"]["state"] == "missing"
    assert report["availability"]["mode"] == stage.MODE_PROXY
    assert report["availability"]["proxy_only"] is True
    assert report["missing_expected"]


def test_status_reports_staged_when_validated(tmp_path: Path) -> None:
    """A staged + valid annotation file flips availability to `staged` with a checksum."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    unpinned = stage.load_manifest(_write_manifest(tmp_path, staging_dir=staging_dir))
    unpinned_report = stage.validate_staging(unpinned)
    manifest = stage.load_manifest(
        _write_manifest(
            tmp_path,
            staging_dir=staging_dir,
            expected_tree_sha256=unpinned_report["tree_sha256"],
        )
    )

    report = stage.validate_staging(manifest)

    assert report["ok"] is True
    assert report["local_availability"] == "staged"
    assert report["mode"] == stage.MODE_DATASET_BACKED
    assert report["availability"]["state"] == "dataset_backed"
    assert report["availability"]["mode"] == stage.MODE_DATASET_BACKED
    assert report["availability"]["dataset_backed"] is True
    assert report["availability"]["validated"] is True
    assert report["tree_sha256"]
    assert report["matched_files"] == ["annotations.txt"]


def test_unpinned_checksum_fails_closed_for_dataset_backed_mode(tmp_path: Path) -> None:
    """Present files without a pinned trusted checksum do not become dataset-backed evidence."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    manifest = stage.load_manifest(_write_manifest(tmp_path, staging_dir=staging_dir))

    report = stage.validate_staging(manifest)

    assert report["ok"] is False
    assert report["local_availability"] == "present_unpinned_checksum"
    assert report["mode"] == stage.MODE_PROXY
    assert report["availability"]["state"] == "proxy_only"
    assert report["availability"]["mode"] == stage.MODE_PROXY
    assert report["tree_sha256"]


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    """A pinned-but-wrong expected checksum refuses to mark SDD as staged."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    manifest = stage.load_manifest(
        _write_manifest(
            tmp_path,
            staging_dir=staging_dir,
            expected_tree_sha256="0" * 64,
        )
    )

    report = stage.validate_staging(manifest)

    assert report["ok"] is False
    assert report["local_availability"] == "missing"
    assert report["checksum_match"] is False


# --- scenario-prior mode gate ----------------------------------------------------------------


def test_mode_gate_forces_proxy_when_sdd_absent(tmp_path: Path) -> None:
    """The mode gate forces proxy_schema_smoke and is not dataset-backed when SDD is absent."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")

    gate = stage.resolve_scenario_prior_mode(manifest_path=manifest_path)

    assert gate["mode"] == stage.MODE_PROXY
    assert gate["dataset_backed"] is False


def test_mode_gate_unlocks_dataset_backed_when_staged(tmp_path: Path) -> None:
    """The mode gate unlocks dataset_backed_prior only with a validated staged copy."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    unpinned = stage.load_manifest(_write_manifest(tmp_path, staging_dir=staging_dir))
    unpinned_report = stage.validate_staging(unpinned)
    manifest_path = _write_manifest(
        tmp_path,
        staging_dir=staging_dir,
        expected_tree_sha256=unpinned_report["tree_sha256"],
    )

    gate = stage.resolve_scenario_prior_mode(manifest_path=manifest_path)

    assert gate["mode"] == stage.MODE_DATASET_BACKED
    assert gate["dataset_backed"] is True
    assert gate["tree_sha256"]


def test_mode_gate_uses_cached_status_without_rehashing(monkeypatch, tmp_path: Path) -> None:
    """A valid staging-status cache avoids re-hashing the staged tree on import paths."""
    staging_dir = tmp_path / "sdd"
    _stage_annotation(staging_dir)
    unpinned = stage.load_manifest(_write_manifest(tmp_path, staging_dir=staging_dir))
    unpinned_report = stage.validate_staging(unpinned)
    manifest = stage.load_manifest(
        _write_manifest(
            tmp_path,
            staging_dir=staging_dir,
            expected_tree_sha256=unpinned_report["tree_sha256"],
        )
    )
    report = stage.validate_staging(manifest)
    stage._write_staging_status(manifest, report)

    def fail_checksum(*_args, **_kwargs):
        raise AssertionError("tree checksum should not run when cached status is valid")

    monkeypatch.setattr(stage, "_tree_checksum", fail_checksum)

    gate = stage.resolve_scenario_prior_mode(manifest)

    assert gate["mode"] == stage.MODE_DATASET_BACKED
    assert gate["dataset_backed"] is True
    assert gate["availability"]["state"] == "dataset_backed"
    assert gate["tree_sha256"] == report["tree_sha256"]


# --- shipped manifest sanity -----------------------------------------------------------------


def test_repo_manifest_loads_and_defaults_to_missing() -> None:
    """The committed repo manifest parses and declares no auto-download URL."""
    manifest = stage.load_manifest()

    assert manifest.asset_id == "sdd"
    assert manifest.download_url is None
    assert manifest.local_availability_declared == "missing"
    assert manifest.expected_total_size_bytes > 0


def test_default_manifest_shape_reports_proxy_mode(tmp_path: Path) -> None:
    """A missing staged copy reports proxy/not-dataset-backed without depending on repo output."""
    manifest_path = _write_manifest(tmp_path, staging_dir=tmp_path / "sdd")

    gate = stage.resolve_scenario_prior_mode(manifest_path=manifest_path)

    assert gate["mode"] == stage.MODE_PROXY
    assert gate["dataset_backed"] is False
