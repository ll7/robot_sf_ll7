"""Focused operational tests for the deterministic artifact archive tool."""

# fmt: off
from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

import pytest

from scripts.tools import build_artifact_archive as mod

if TYPE_CHECKING:
    from pathlib import Path


def _fixture(tmp_path: Path, data: dict[str, bytes] | None = None) -> tuple[Path, Path]:
    data = data or {"rows/a.jsonl": b'{"row": 1}\n', "trace.bin": b"trace\0bytes"}
    source = tmp_path / "source"
    inventory = []
    for relative, content in data.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        inventory.append({"relative_path": relative, "sha256": hashlib.sha256(content).hexdigest(), "byte_size": len(content), "retention_class": "durable_required", "status": "present"})
    source.mkdir(exist_ok=True)
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps({"schema_version": "terminal_job_harvest.v1", "receipt_id": "a" * 64, "status": "ready", "artifact_status": "complete", "scheduler": {"terminal": True}, "problems": [], "inventory": inventory}, sort_keys=True), encoding="utf-8")
    return source, manifest


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "bundle.tar.gz", tmp_path / "bundle.members.json"


def _build(tmp_path: Path, **kwargs):
    source, manifest = _fixture(tmp_path)
    archive, member_manifest = _paths(tmp_path)
    return mod.build_archive(manifest, source, archive, member_manifest=member_manifest, **kwargs), source, archive, member_manifest


def _code(call) -> str:
    with pytest.raises(mod.ArchiveError) as error:
        call()
    return error.value.code


def test_repeat_is_byte_deterministic_and_extracts_checksums(tmp_path: Path) -> None:
    first, source, archive, member_manifest = _build(tmp_path / "one")
    second, _source2, archive2, member_manifest2 = _build(tmp_path / "two")
    assert first["status"] == second["status"] == "verified"
    assert archive.read_bytes() == archive2.read_bytes()
    assert member_manifest.read_bytes() == member_manifest2.read_bytes()
    extracted = tmp_path / "extracted"
    report = mod.verify_archive(archive, member_manifest, extraction_root=extracted)
    assert report == {"status": "verified", "schema_version": mod.ARCHIVE_SCHEMA, "archive_sha256": first["archive_sha256"], "member_count": 2, "extracted": True}
    assert (extracted / "rows/a.jsonl").read_bytes() == (source / "rows/a.jsonl").read_bytes()
    assert (extracted / "trace.bin").read_bytes() == (source / "trace.bin").read_bytes()


def test_cli_check_is_the_canonical_verifier_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _result, _source, archive, member_manifest = _build(tmp_path)
    assert mod.main(["--check", "--manifest", str(member_manifest), "--archive", str(archive), "--format", "json"]) == mod.EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


@pytest.mark.parametrize("raw", ["../escape", "/absolute", "a/../b", "dir/bad:name", "\ud800"])
def test_traversal_is_rejected(tmp_path: Path, raw: str) -> None:
    source, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"][0]["relative_path"] = raw
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive, member_manifest = _paths(tmp_path)
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) in {"path_escape", "path_invalid", "private_locator"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "blocked"), ("status", "running"), ("artifact_status", "partial"), ("scheduler", {"terminal": False}), ("problems", [{"code": "blocked"}])],
)
def test_incomplete_harvest_receipts_are_rejected(tmp_path: Path, field: str, value) -> None:
    source, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive, member_manifest = _paths(tmp_path)
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "source_not_ready"


@pytest.mark.parametrize("alias", ["same", "symlink"])
def test_output_aliases_are_rejected_before_temp_files(tmp_path: Path, alias: str) -> None:
    source, manifest = _fixture(tmp_path)
    archive, member_manifest = _paths(tmp_path)
    if alias == "same":
        member_manifest = archive
    else:
        member_manifest.symlink_to(archive)
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "output_alias"
    assert not any(path.name.endswith(".partial") for path in tmp_path.iterdir())


def test_duplicate_manifest_path_is_rejected(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"].append(dict(payload["inventory"][0]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive, member_manifest = _paths(tmp_path)
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "ambiguous_manifest"


def test_extraction_restores_executable_mode(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path)
    os.chmod(source / "trace.bin", 0o755)  # noqa: S103 - executable-mode fixture
    archive, member_manifest = _paths(tmp_path)
    mod.build_archive(manifest, source, archive, member_manifest=member_manifest)
    extracted = tmp_path / "extracted"
    mod.verify_archive(archive, member_manifest, extraction_root=extracted)
    assert (extracted / "trace.bin").stat().st_mode & 0o777 == 0o755


def test_source_drift_during_copy_is_rejected(tmp_path: Path, monkeypatch) -> None:
    source, manifest = _fixture(tmp_path)
    archive, member_manifest = _paths(tmp_path)
    original = mod._write_tar

    def mutate(path, sources):
        (source / "trace.bin").write_bytes(b"changed!!!!")
        original(path, sources)

    monkeypatch.setattr(mod, "_write_tar", mutate)
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "source_drift"
    assert not archive.exists() and not member_manifest.exists()


def test_symlink_device_active_writer_source_drift_and_capacity_fail_closed(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path / "symlink")
    (source / "link").symlink_to(source / "trace.bin")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"][0]["relative_path"] = "link"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive, member_manifest = _paths(tmp_path / "symlink")
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "symlink_rejected"

    source, manifest = _fixture(tmp_path / "special")
    os.mkfifo(source / "pipe")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"][0].update(relative_path="pipe", sha256="0" * 64, byte_size=0)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive, member_manifest = _paths(tmp_path / "special")
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "special_file_rejected"

    source, manifest = _fixture(tmp_path / "active")
    (source / ".active-writer").touch()
    archive, member_manifest = _paths(tmp_path / "active")
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "active_writer"

    source, manifest = _fixture(tmp_path / "drift")
    (source / "trace.bin").write_bytes(b"changed!!!!")
    archive, member_manifest = _paths(tmp_path / "drift")
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest)) == "source_checksum_mismatch"

    source, manifest = _fixture(tmp_path / "capacity")
    archive, member_manifest = _paths(tmp_path / "capacity")
    assert _code(lambda: mod.build_archive(manifest, source, archive, member_manifest=member_manifest, capacity_bytes=0)) == "capacity_insufficient"


def test_partial_archive_and_checksum_tamper_are_rejected(tmp_path: Path) -> None:
    _result, _source, archive, member_manifest = _build(tmp_path)
    archive.with_name(f"{archive.name}.partial").write_bytes(b"partial")
    assert _code(lambda: mod.verify_archive(archive, member_manifest)) == "partial_archive"
    archive.with_name(f"{archive.name}.partial").unlink()
    broken = tmp_path / "broken.tar.gz"
    broken.write_bytes(archive.read_bytes()[:-5])
    assert _code(lambda: mod.verify_archive(broken, member_manifest)) == "archive_checksum_mismatch"
