"""Focused tests for the fixture-based artifact transfer verifier (issue #8825)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_verify_artifact_transfer", REPO / "scripts/validation/verify_artifact_transfer.py"
)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixture(tmp_path: Path, layout: dict[str, bytes]) -> tuple[Path, Path]:
    source = tmp_path / "source"
    inventory = []
    for relative, data in layout.items():
        path = _write(source, relative, data)
        inventory.append(
            {
                "relative_path": relative,
                "retention_class": "durable_required",
                "sha256": MOD.sha256_file(path),
                "byte_size": len(data),
                "status": "present",
            }
        )
    payload = {
        "schema_version": MOD.HARVEST_SCHEMA,
        "receipt_id": "a" * 64,
        "ownership": {"owner": "ll7", "campaign_id": "fixture"},
        "inventory": inventory,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, source


def _staging_manifest(tmp_path: Path, source: Path, layout: dict[str, bytes]) -> Path:
    members = [
        {"path": r, "sha256": MOD.sha256_file(_write(source, r, d)), "byte_size": len(d)}
        for r, d in layout.items()
    ]
    body = {"schema_version": MOD.BUNDLE_SCHEMA, "owner": "ll7", "members": members}
    body["manifest_sha256"] = MOD.stable_hash(body)
    manifest = tmp_path / "staging.json"
    manifest.write_text(json.dumps(body), encoding="utf-8")
    return manifest


def _reasons(manifest: Path, source: Path, destination: Path, **kwargs) -> list[str]:
    return MOD.build_transfer_report(manifest, source, destination, **kwargs)["reason_codes"]


def test_apply_copies_verifies_and_emits_deterministic_receipt(tmp_path: Path, monkeypatch) -> None:
    layout = {"rows/row-1.json": b"one", "report.md": b"two", "empty.txt": b""}
    manifest, source = _fixture(tmp_path, layout)
    destination = tmp_path / "destination"
    report = MOD.build_transfer_report(manifest, source, destination, apply=True)
    assert report["status"] == "verified"
    assert report["counts"] == {
        "members": 3,
        "copied": 3,
        "already_verified": 0,
        "pending": 0,
        "conflicts": 0,
    }
    assert report["manifest"]["schema_version"] == "terminal_job_harvest"
    assert (destination / "report.md").read_bytes() == b"two"
    assert (destination / "empty.txt").read_bytes() == b""
    written = json.loads((destination / MOD.DEFAULT_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert written["receipt_id"] == report["receipt_id"]
    checked = MOD.build_transfer_report(manifest, source, destination)
    again = MOD.build_transfer_report(manifest, source, destination)
    assert checked["status"] == "verified" and checked["counts"]["already_verified"] == 3
    assert json.dumps(checked, sort_keys=True) == json.dumps(again, sort_keys=True)
    calls: list[str] = []
    monkeypatch.setattr(MOD, "_copy_member", lambda *args: calls.append(args[2].relative_path))
    repeat = MOD.build_transfer_report(manifest, source, destination, apply=True)
    assert repeat["status"] == "verified"
    assert repeat["counts"]["already_verified"] == 3 and repeat["counts"]["copied"] == 0
    assert calls == []


def test_interrupted_copy_resumes_without_recopying_verified_files(tmp_path, monkeypatch) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha", "b.txt": b"beta"})
    destination = tmp_path / "destination"
    original = MOD._copy_member

    def fail_on_b(source_path, target, member):
        if member.relative_path == "b.txt":
            raise OSError("interrupted")
        original(source_path, target, member)

    monkeypatch.setattr(MOD, "_copy_member", fail_on_b)
    blocked = MOD.build_transfer_report(manifest, source, destination, apply=True)
    assert blocked["status"] == "blocked" and "copy_failed" in blocked["reason_codes"]
    assert (destination / "a.txt").read_bytes() == b"alpha"
    assert not (destination / "b.txt").exists()
    calls: list[str] = []

    def tracked(source_path, target, member):
        calls.append(member.relative_path)
        original(source_path, target, member)

    monkeypatch.setattr(MOD, "_copy_member", tracked)
    resumed = MOD.build_transfer_report(manifest, source, destination, apply=True)
    assert resumed["status"] == "verified"
    assert calls == ["b.txt"]
    assert resumed["counts"]["already_verified"] == 1 and resumed["counts"]["copied"] == 1


def test_capacity_partial_and_unexpected_members_fail_closed(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    limited = MOD.build_transfer_report(manifest, source, destination, apply=True, capacity_bytes=1)
    assert "destination_capacity_exceeded" in limited["reason_codes"]
    assert not (destination / "a.txt").exists()
    partial = _write(destination, f".a.txt.{'0' * 12}{MOD.PARTIAL_SUFFIX}", b"part")
    checked = MOD.build_transfer_report(manifest, source, destination)
    assert "partial_transfer" in checked["reason_codes"] and partial.exists()
    applied = MOD.build_transfer_report(
        manifest, source, destination, apply=True, capacity_bytes=1 << 20
    )
    assert applied["status"] == "verified" and not partial.exists()
    assert applied["destination"]["partial_members_cleaned"] == 1
    _write(destination, "extra.txt", b"x")
    assert "unexpected_destination_member" in _reasons(manifest, source, destination, apply=True)


def test_blocked_apply_never_writes_or_overwrites_receipt(tmp_path: Path, monkeypatch) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    receipt = MOD.DEFAULT_RECEIPT_NAME

    def blocked(destination: Path, **kwargs) -> None:
        report = MOD.build_transfer_report(manifest, source, destination, apply=True, **kwargs)
        assert report["status"] == "blocked"
        assert not (destination / receipt).exists()

    blocked(tmp_path / "limited", capacity_bytes=1)
    unexpected = tmp_path / "unexpected"
    _write(unexpected, "extra.txt", b"extra")
    blocked(unexpected)
    mismatch = tmp_path / "source-mismatch"
    (source / "a.txt").write_bytes(b"changed")
    blocked(mismatch)
    (source / "a.txt").write_bytes(b"alpha")

    def fail_copy(*_args) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(MOD, "_copy_member", fail_copy)
    blocked(tmp_path / "copy-failed")
    monkeypatch.undo()

    valid = tmp_path / "valid"
    assert MOD.build_transfer_report(manifest, source, valid, apply=True)["status"] == "verified"
    before = (valid / receipt).read_bytes()
    _write(valid, "extra.txt", b"extra")
    retry = MOD.build_transfer_report(manifest, source, valid, apply=True)
    assert retry["status"] == "blocked"
    assert (valid / receipt).read_bytes() == before


def test_source_integrity_failures_block_before_any_copy(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    destination.mkdir()
    (source / "a.txt").write_bytes(b"ALPHA")
    assert "source_checksum_mismatch" in _reasons(manifest, source, destination, apply=True)
    assert not (destination / "a.txt").exists()
    (source / "a.txt").unlink()
    assert "source_missing" in _reasons(manifest, source, destination, apply=True)


def test_destination_conflicts_never_overwrite_existing_bytes(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    _write(destination, "a.txt", b"other")
    assert "destination_conflict" in _reasons(manifest, source, destination, apply=True)
    assert (destination / "a.txt").read_bytes() == b"other"


def test_manifest_ambiguity_unavailable_members_and_schema_fail_closed(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"].append(dict(payload["inventory"][0]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert "ambiguous_manifest" in _reasons(manifest, source, destination, apply=True)
    manifest.write_text(json.dumps({"schema_version": "other.v1", "inventory": []}))
    assert "unsupported_manifest_schema" in _reasons(manifest, source, destination, apply=True)
    nested, nested_source = _fixture(tmp_path / "nested", {"a.txt": b"alpha"})
    payload = json.loads(nested.read_text(encoding="utf-8"))
    payload["inventory"][0]["status"] = "missing"
    nested.write_text(json.dumps(payload), encoding="utf-8")
    blocked = MOD.build_transfer_report(nested, nested_source, tmp_path / "dst-b", apply=True)
    assert "manifest_member_unavailable" in blocked["reason_codes"]


def test_staging_manifest_round_trip_and_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _staging_manifest(tmp_path, source, {"configs/a.yaml": b"a: 1\n"})
    report = MOD.build_transfer_report(manifest, source, tmp_path / "destination", apply=True)
    assert report["status"] == "verified"
    assert report["manifest"]["schema_version"] == "compute_staging_bundle"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["members"][0]["byte_size"] = 999
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    broken = MOD.build_transfer_report(manifest, source, tmp_path / "dst2", apply=True)
    assert "manifest_identity_mismatch" in broken["reason_codes"]


def test_path_escape_private_locator_and_symlinks_fail_closed(tmp_path: Path) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    for raw, code in (("../escape.txt", "path_escape"), ("~/.ssh/key", "private_locator")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["inventory"][0]["relative_path"] = raw
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        assert code in _reasons(manifest, source, destination, apply=True)
    (source / "link.txt").symlink_to(source / "a.txt")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inventory"][0]["relative_path"] = "link.txt"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert "symlink_rejected" in _reasons(manifest, source, destination, apply=True)


def test_cli_modes_and_exit_codes(tmp_path: Path, capsys) -> None:
    manifest, source = _fixture(tmp_path, {"a.txt": b"alpha"})
    destination = tmp_path / "destination"
    base = ["--manifest", str(manifest), "--source-root", str(source), "--destination"]
    assert MOD.main(["--apply", *base, str(destination)]) == MOD.EXIT_VERIFIED
    capsys.readouterr()
    assert MOD.main(["--check", *base, str(destination), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    (destination / "a.txt").write_bytes(b"conflict!")
    assert MOD.main(["--check", *base, str(destination)]) == MOD.EXIT_BLOCKED
    assert MOD.main(["--check", "--apply", *base, str(destination)]) == MOD.EXIT_MALFORMED
    missing = ["--check", "--manifest", str(tmp_path / "nope.json"), *base[2:], str(destination)]
    assert MOD.main(missing) == MOD.EXIT_MALFORMED
