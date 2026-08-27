"""Tests for the published-release audit (issue #7936)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.published_release_audit import (
    SCHEMA,
    _extract_members,
    _verify_internal_checksums,
    audit_published,
)

_CLI_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "published_release_audit.py"
)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_bundle(
    path: Path, *, member: str = "manifest.json", data: bytes = b"bundle-bytes"
) -> None:
    """Write a real zip bundle with one member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, data)


def test_cross_channel_byte_identity_passes(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
        _write_bytes(channel / "checksums.sha256", b"checksum-bytes")
    receipt = audit_published(
        tag="paper-matrix-v2-h600-s30",
        doi="10.5281/zenodo.1234567",
        github_dir=github,
        zenodo_dir=zenodo,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["ok"] is True
    assert receipt["status"] == "pass"
    assert receipt["observations"]["common_asset_names"] == ["bundle.zip", "checksums.sha256"]
    assert receipt["problems"] == []


def test_cross_channel_mismatch_fails(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_bundle(github / "bundle.zip", data=b"same")
    _make_bundle(zenodo / "bundle.zip", data=b"different")
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is False
    assert any("cross-channel byte mismatch" in problem for problem in receipt["problems"])


def test_missing_channel_reports_unavailable(tmp_path: Path) -> None:
    github = tmp_path / "github"
    github.mkdir()
    receipt = audit_published(
        tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=tmp_path / "empty"
    )
    assert receipt["ok"] is False
    assert any("Zenodo channel has no assets" in problem for problem in receipt["problems"])


def test_doi_validation(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    receipt = audit_published(tag="t", doi="", github_dir=github, zenodo_dir=zenodo)
    assert any("version DOI is missing" in problem for problem in receipt["problems"])
    receipt2 = audit_published(tag="t", doi="not-a-doi", github_dir=github, zenodo_dir=zenodo)
    assert any("version DOI is malformed" in problem for problem in receipt2["problems"])


def test_bundle_extraction_and_internal_checksums(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    github.mkdir(parents=True)
    zenodo.mkdir(parents=True)
    bundle_path = github / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr("manifest.json", b"member-data")
    checksum_line = f"{sha256_file(bundle_path)} bundle.zip\n"
    # A sidecar checksum file inside the bundle:
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("checksums.sha256", checksum_line)
    _write_bytes(zenodo / "bundle.zip", bundle_path.read_bytes())
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is True
    assert receipt["observations"]["bundle"] == "bundle.zip"
    assert receipt["observations"]["bundle_member_count"] == 2


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(ValueError, match="path escape"):
        _extract_members(evil, tmp_path / "dest")


def test_unsupported_archive_fails_closed(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"not-a-real-archive")
    with pytest.raises(ValueError, match="unsupported archive|extraction failed"):
        _extract_members(bogus, tmp_path / "dest")


def test_internal_checksum_mismatch_detected(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "file.txt").write_text("content")
    (extracted / "checksums.sha256").write_text("0" * 64 + "  file.txt\n")
    problems = _verify_internal_checksums(extracted, ["file.txt", "checksums.sha256"])
    assert any("internal checksum mismatch" in problem for problem in problems)


def test_source_sha_tag_binding_enforced(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    receipt = audit_published(
        tag="paper-matrix-abcdef1234567890abcdef1234567890abcdef12",
        doi="10.5281/zenodo.1",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="0" * 40,
    )
    assert receipt["ok"] is False
    assert any("disagrees with" in problem for problem in receipt["problems"])


def test_receipt_is_deterministic(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
        _write_bytes(channel / "checksums.sha256", b"y")
    first = json.dumps(
        audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo),
        sort_keys=True,
    )
    second = json.dumps(
        audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo),
        sort_keys=True,
    )
    assert first == second


def test_tar_bundle_extraction(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    github.mkdir(parents=True)
    zenodo.mkdir(parents=True)
    bundle_path = github / "bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        data = b"member-data"
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    _write_bytes(zenodo / "bundle.tar.gz", bundle_path.read_bytes())
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is True
    assert receipt["observations"]["bundle"] == "bundle.tar.gz"
    assert receipt["observations"]["bundle_member_count"] == 1


def test_checksums_json_sidecar(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "file.txt").write_text("content")
    (extracted / "checksums.json").write_text(
        json.dumps({"file.txt": sha256_file(extracted / "file.txt")})
    )
    problems = _verify_internal_checksums(extracted, ["file.txt", "checksums.json"])
    assert problems == []


def test_checksums_json_malformed_reports(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "checksums.json").write_text("{not-json")
    problems = _verify_internal_checksums(extracted, ["checksums.json"])
    assert any("not valid JSON" in problem for problem in problems)


def test_cli_main_passes(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI_SCRIPT),
            "--tag",
            "paper-matrix-v2-h600-s30",
            "--doi",
            "10.5281/zenodo.1234567",
            "--github-dir",
            str(github),
            "--zenodo-dir",
            str(zenodo),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is True


def test_cli_main_missing_channel_returns_one(tmp_path: Path) -> None:
    github = tmp_path / "github"
    github.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI_SCRIPT),
            "--tag",
            "t",
            "--doi",
            "10.5281/zenodo.1",
            "--github-dir",
            str(github),
            "--zenodo-dir",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is False
