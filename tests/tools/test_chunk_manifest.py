"""Focused tests for the deterministic chunk-manifest tool (issue #8915)."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from typing import TYPE_CHECKING

import pytest

from scripts.tools import chunk_manifest as cm

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _manifest(tmp_path: Path, files, *, chunk_size: int = 1024, extra: tuple = (), capsys=None):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    _write(root, files)
    output = tmp_path / "manifest.json"
    common = ["manifest", "--root", str(root), "--output", str(output)]
    assert cm.main([*common, "--chunk-size", str(chunk_size), *extra]) == cm.EXIT_OK
    if capsys is not None:
        capsys.readouterr()
    return root, output, json.loads(output.read_text(encoding="utf-8"))


def _build(root: Path, *, chunk_size: int = 64):
    policy = cm.ChunkingPolicy(chunk_size_bytes=chunk_size)
    return cm.build_manifest(
        root, artifact=cm.ArtifactIdentity("fixture", "v1", "sha256:fixture-root"), policy=policy
    )


def _verify(root: Path, output: Path) -> int:
    return cm.main(["verify", "--root", str(root), "--manifest", str(output), "--json"])


def test_schema_semantics_and_relative_path_only_output(tmp_path: Path) -> None:
    data = b"hello"
    _root, output, manifest = _manifest(tmp_path, {"small.bin": data})
    record = manifest["files"][0]
    assert manifest["schema_version"] == cm.SCHEMA_VERSION
    assert record["path"] == "small.bin" and record["mode"] == "full"
    assert record["content_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["tree_sha256"] == cm.compute_tree_digest(manifest["files"])
    assert manifest["manifest_id"] == cm.compute_manifest_id(manifest)
    assert manifest["artifact"]["retention_role"] == "unspecified"
    assert len(manifest["files"]) == 1
    text = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and '"path": "small.bin"' in text


def test_fixed_chunk_boundaries_and_order_worker_determinism(tmp_path: Path) -> None:
    data = bytes(index % 251 for index in range(2500))
    root, output, first = _manifest(tmp_path, {"data.bin": data}, chunk_size=1024)
    record = first["files"][0]
    assert record["mode"] == "chunked"
    sizes = [(c["offset"], c["length"]) for c in record["chunks"]]
    assert sizes == [(0, 1024), (1024, 1024), (2048, 452)]
    for chunk in record["chunks"]:
        start, length = chunk["offset"], chunk["length"]
        assert chunk["sha256"] == hashlib.sha256(data[start : start + length]).hexdigest()
    second = tmp_path / "workers.json"
    args = ["manifest", "--root", str(root), "--output", str(second), "--chunk-size", "1024"]
    assert cm.main([*args, "--workers", "4"]) == cm.EXIT_OK
    other = json.loads(second.read_text(encoding="utf-8"))
    assert other["manifest_id"] == first["manifest_id"]
    assert other["tree_sha256"] == first["tree_sha256"]
    assert cm.compute_tree_digest(list(reversed(first["files"]))) == first["tree_sha256"]
    assert _verify(root, output) == cm.EXIT_OK


def test_changed_middle_chunk_reports_exact_location(tmp_path: Path, capsys) -> None:
    data = bytes(index % 251 for index in range(2500))
    root, output, _first = _manifest(tmp_path, {"data.bin": data}, chunk_size=1024, capsys=capsys)
    changed = bytearray(data)
    changed[1100] ^= 0xFF
    (root / "data.bin").write_bytes(bytes(changed))
    assert _verify(root, output) == cm.EXIT_FAILED
    failure = json.loads(capsys.readouterr().out)["failures"][0]
    assert failure["code"] == "chunk_digest_mismatch"
    assert (failure["file"], failure["chunk_index"], failure["offset"]) == ("data.bin", 1, 1024)


def test_truncation_fails_closed(tmp_path: Path, capsys) -> None:
    root, output, _first = _manifest(tmp_path, {"data.bin": b"x" * 500}, capsys=capsys)
    path = root / "data.bin"
    path.write_bytes(path.read_bytes()[:-50])
    assert _verify(root, output) == cm.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["failures"][0]["code"] == "size_mismatch"


def test_path_rules_duplicates_and_case_collision(tmp_path: Path) -> None:
    with pytest.raises(cm.ChunkManifestError) as escape:
        cm.normalize_relative_path("../escape")
    assert escape.value.code == "path_escape"
    _root, output, _manifest_data = _manifest(tmp_path, {"a.bin": b"abc"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "/abs/path.bin"
    assert "path_not_relative" in {issue["code"] for issue in cm.validate_manifest(payload)}
    duplicated = json.loads(output.read_text(encoding="utf-8"))
    duplicated["files"] = [duplicated["files"][0], dict(duplicated["files"][0])]
    assert "duplicate_path" in {issue["code"] for issue in cm.validate_manifest(duplicated)}
    case_root = tmp_path / "case"
    case_root.mkdir()
    (case_root / "A.txt").write_bytes(b"1")
    (case_root / "a.txt").write_bytes(b"2")
    with pytest.raises(cm.ChunkManifestError) as collision:
        _build(case_root)
    assert collision.value.code == "case_collision"


def test_partial_and_tampered_manifest_fail_closed(tmp_path: Path) -> None:
    _root, output, _manifest_data = _manifest(tmp_path, {"a.bin": b"abc"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["files"][0]["content_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(cm.ChunkManifestError):
        cm.load_manifest_file(tampered)
    broken = tmp_path / "broken.json"
    broken.write_text(output.read_text(encoding="utf-8")[:20], encoding="utf-8")
    with pytest.raises(cm.ChunkManifestError) as partial:
        cm.load_manifest_file(broken)
    assert partial.value.code == "partial_manifest"


def _make_symlink(root: Path) -> None:
    (root / "real.bin").write_bytes(b"x" * 100)
    os.symlink(root / "real.bin", root / "link.bin")


def _make_hardlink(root: Path) -> None:
    (root / "real.bin").write_bytes(b"x" * 100)
    os.link(root / "real.bin", root / "hard.bin")


def _make_special(root: Path) -> None:
    os.mkfifo(root / "pipe")


def _make_sparse(root: Path) -> None:
    path = root / "sparse.bin"
    with path.open("wb") as handle:
        handle.seek(1024 * 1024)
        handle.write(b"x")
    stat_result = path.stat()
    if getattr(stat_result, "st_blocks", 0) * 512 >= stat_result.st_size:
        pytest.skip("filesystem does not report sparse allocation")


def _preallocate_beyond_eof(path: Path) -> None:
    """Hide a real in-file hole from the st_blocks hint with kept-size allocation."""
    fallocate = getattr(ctypes.CDLL(None, use_errno=True), "fallocate", None)
    if fallocate is None:
        pytest.skip("fallocate is unavailable")
    fallocate.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong)
    fallocate.restype = ctypes.c_int
    size = path.stat().st_size
    fd = os.open(path, os.O_RDWR)
    try:
        if fallocate(fd, 1, size, size) != 0:  # FALLOC_FL_KEEP_SIZE
            pytest.skip(f"kept-size preallocation unavailable: {ctypes.get_errno()}")
    finally:
        os.close(fd)
    if path.stat().st_blocks * 512 < size:
        pytest.skip("preallocation did not mask the low-block hint")


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (_make_symlink, "symlink_rejected"),
        (_make_hardlink, "hardlink_rejected"),
        (_make_special, "unsupported_special_file"),
        (_make_sparse, "sparse_file"),
    ],
)
def test_unsafe_members_fail_closed(tmp_path: Path, setup, code: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    setup(root)
    with pytest.raises(cm.ChunkManifestError) as error:
        _build(root)
    assert error.value.code == code


def test_dense_file_is_never_checked_for_sparse_holes(tmp_path: Path) -> None:
    path = tmp_path / "dense.bin"
    path.write_bytes(hashlib.shake_256(b"dense fixture").digest(128 * 1024))
    if cm._looks_sparse_by_blocks(path.stat()):
        pytest.skip("filesystem reports low allocation for the dense fixture")
    assert cm._stat_checked(path, path.name)[0] == path.stat().st_size


def test_fiemap_full_encoded_mapping_is_accepted(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "compressed.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(0, size, flags)]
    )
    assert cm._has_sparse_hole(path, size) is False


def test_fiemap_gap_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "gap.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(100, size - 100, flags)]
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_unavailable_mapping_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "unavailable.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    monkeypatch.setattr(cm, "_fiemap_extents", lambda _fd, _size: None)
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_non_linux_platform_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(cm.sys, "platform", "darwin")
    assert cm._fiemap_extents(-1, 4096) is None


def test_fiemap_ambiguous_extent_flags_are_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ambiguous.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_UNWRITTEN | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(0, size, flags)]
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_incomplete_coverage_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "incomplete.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(0, size - 100, flags)]
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_without_encoded_extent_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "no_encoded.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    monkeypatch.setattr(
        cm,
        "_fiemap_extents",
        lambda _fd, _size: [cm._FiemapExtent(0, size, cm._FIEMAP_EXTENT_LAST)],
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_zero_length_trailing_extent_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "trailing_zero.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED
    last_flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm,
        "_fiemap_extents",
        lambda _fd, _size: [
            cm._FiemapExtent(0, size, flags),
            cm._FiemapExtent(size, 0, last_flags),
        ],
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_missing_terminal_last_flag_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "no_last.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(0, size, flags)]
    )
    assert cm._has_sparse_hole(path, size) is True


def test_fiemap_last_flag_on_earlier_extent_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "early_last.bin"
    path.write_bytes(b"x" * 8192)
    size = path.stat().st_size
    half = size // 2
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(
        cm,
        "_fiemap_extents",
        lambda _fd, _size: [
            cm._FiemapExtent(0, half, flags),
            cm._FiemapExtent(half, size - half, flags),
        ],
    )
    assert cm._has_sparse_hole(path, size) is True


def test_sparse_file_rejected_even_if_seek_data_hole_report_dense(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _make_sparse(root)
    size = (root / "sparse.bin").stat().st_size

    def fake_lseek(_fd, offset, whence):
        if whence == os.SEEK_DATA:
            return 0
        if whence == os.SEEK_HOLE:
            return size
        raise AssertionError(f"unexpected whence: {whence}")

    monkeypatch.setattr(cm.os, "lseek", fake_lseek)
    with pytest.raises(cm.ChunkManifestError) as error:
        _build(root)
    assert error.value.code == "sparse_file"


def test_stat_checked_accepts_low_block_hint_confirmed_by_fiemap(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "compressed.bin"
    path.write_bytes(b"x" * 4096)
    size = path.stat().st_size
    flags = cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST
    monkeypatch.setattr(cm, "_looks_sparse_by_blocks", lambda _st: True)
    monkeypatch.setattr(
        cm, "_fiemap_extents", lambda _fd, _size: [cm._FiemapExtent(0, size, flags)]
    )
    identity = cm._stat_checked(path, path.name)
    assert identity[0] == size


def _gpfs_like_source(tmp_path: Path, monkeypatch, data: bytes) -> tuple[Path, Path, dict]:
    """Make FIEMAP return EOPNOTSUPP for one low-allocation source inode."""
    if cm.fcntl is None:
        pytest.skip("FIEMAP needs fcntl")
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    source_file = source_root / "output.bin"
    source_file.write_bytes(data)
    source_inode = source_file.stat().st_ino
    real_hint = cm._looks_sparse_by_blocks
    real_ioctl = cm.fcntl.ioctl
    monkeypatch.setattr(
        cm, "_looks_sparse_by_blocks", lambda st: st.st_ino == source_inode or real_hint(st)
    )

    def unsupported_source(fd, request, buffer, mutate):
        if os.fstat(fd).st_ino == source_inode:
            raise OSError(errno.EOPNOTSUPP, "FIEMAP unsupported")
        return real_ioctl(fd, request, buffer, mutate)

    monkeypatch.setattr(cm.fcntl, "ioctl", unsupported_source)
    manifest = _build(source_root, chunk_size=128)
    return source_root, destination_root, manifest


def test_gpfs_like_source_dense_destination_passes_custody(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data = b"compressed output" * 50
    source_root, destination_root, manifest = _gpfs_like_source(tmp_path, monkeypatch, data)
    (destination_root / "output.bin").write_bytes(data)
    record = manifest["files"][0]
    expected_sha = hashlib.sha256(data).hexdigest()
    assert record["allocation_status"] == cm.ALLOCATION_UNVERIFIED
    assert record["file_sha256"] == expected_sha
    assert record["mode"] == "chunked"

    source = cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)
    destination = cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    assert source["status"] == destination["status"] == "ok"
    assert source["files"][0]["allocation_status"] == cm.ALLOCATION_UNVERIFIED
    assert destination["files"][0]["allocation_status"] == cm.ALLOCATION_VERIFIED
    assert source["files"][0]["file_sha256"] == destination["files"][0]["file_sha256"]
    receipt = cm.build_custody_receipt(
        manifest, source, destination, source_root=source_root, destination_root=destination_root
    )
    assert receipt["allocation_unverified_source"] == [
        {
            "path": "output.bin",
            "source_sha256": expected_sha,
            "destination_sha256": expected_sha,
            "destination_allocation_status": cm.ALLOCATION_VERIFIED,
        }
    ]

    manifest_path = tmp_path / "manifest.json"
    source_path = tmp_path / "source-verify.json"
    destination_path = tmp_path / "destination-verify.json"
    receipt_path = tmp_path / "custody.json"
    for path, payload in (
        (manifest_path, manifest),
        (source_path, source),
        (destination_path, destination),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    source_args = ["verify", "--root", str(source_root), "--manifest", str(manifest_path), "--json"]
    assert cm.main(source_args) == cm.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == (
        "allocation_unverifiable_destination"
    )
    assert cm.main([*source_args, "--side", "source"]) == cm.EXIT_OK
    assert json.loads(capsys.readouterr().out)["allocation_policy"] == "source"
    assert (
        cm.main(
            [
                "verify",
                "--side",
                "destination",
                "--root",
                str(destination_root),
                "--manifest",
                str(manifest_path),
                "--json",
            ]
        )
        == cm.EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["allocation_policy"] == "destination"
    assert (
        cm.main(
            [
                "custody",
                "--manifest",
                str(manifest_path),
                "--source-verification",
                str(source_path),
                "--destination-verification",
                str(destination_path),
                "--source-root",
                str(source_root),
                "--destination-root",
                str(destination_root),
                "--output",
                str(receipt_path),
            ]
        )
        == cm.EXIT_OK
    )
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = written.pop("receipt_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(written, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_gpfs_like_source_destination_hash_mismatch_fails(tmp_path: Path, monkeypatch) -> None:
    data = b"compressed output" * 50
    source_root, destination_root, manifest = _gpfs_like_source(tmp_path, monkeypatch, data)
    changed = bytearray(data)
    changed[200] ^= 1
    (destination_root / "output.bin").write_bytes(changed)
    source = cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)
    destination = cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    assert destination["status"] == "failed"
    assert "file_digest_mismatch" in {failure["code"] for failure in destination["failures"]}
    with pytest.raises(cm.ChunkManifestError) as error:
        cm.build_custody_receipt(
            manifest,
            source,
            destination,
            source_root=source_root,
            destination_root=destination_root,
        )
    assert error.value.code == "custody_verification_invalid"


def test_gpfs_like_source_sparse_destination_fails(tmp_path: Path, monkeypatch) -> None:
    data = b"\0" * (1024 * 1024) + b"x"
    source_root, destination_root, manifest = _gpfs_like_source(tmp_path, monkeypatch, data)
    destination_file = destination_root / "output.bin"
    with destination_file.open("wb") as handle:
        handle.seek(len(data) - 1)
        handle.write(b"x")
    if not cm._looks_sparse_by_blocks(destination_file.stat()):
        pytest.skip("filesystem does not report sparse allocation")
    assert (
        cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)["status"]
        == "ok"
    )
    with pytest.raises(cm.ChunkManifestError) as error:
        cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    assert error.value.code == "sparse_file"


@pytest.mark.parametrize("without_fiemap", (False, True))
def test_destination_hole_fails_even_with_blocks_beyond_eof(
    tmp_path: Path, monkeypatch, without_fiemap: bool
) -> None:
    data = b"\0" * (1024 * 1024) + b"x"
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    (source_root / "output.bin").write_bytes(data)
    manifest = _build(source_root)
    destination_file = destination_root / "output.bin"
    with destination_file.open("wb") as handle:
        handle.seek(len(data) - 1)
        handle.write(b"x")
    _preallocate_beyond_eof(destination_file)
    assert destination_file.stat().st_blocks * 512 >= len(data)
    if without_fiemap:
        monkeypatch.setattr(cm, "_fiemap_extents", lambda _fd, _size: None)
    else:
        with destination_file.open("rb") as handle:
            extents = cm._fiemap_extents(handle.fileno(), len(data))
        if extents is None:
            pytest.skip("FIEMAP unavailable on test filesystem")
        assert extents and extents[0].logical > 0
    with pytest.raises(cm.ChunkManifestError) as error:
        cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    assert error.value.code == "sparse_file"


def test_destination_without_fiemap_rejects_coarse_seek_on_sparse_file(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    data = b"\0" * (1024 * 1024) + b"x"
    (source_root / "output.bin").write_bytes(data)
    manifest = _build(source_root)
    destination_file = destination_root / "output.bin"
    with destination_file.open("wb") as handle:
        handle.seek(len(data) - 1)
        handle.write(b"x")
    if not cm._looks_sparse_by_blocks(destination_file.stat()):
        pytest.skip("filesystem does not report sparse allocation")
    monkeypatch.setattr(cm, "_fiemap_extents", lambda _fd, _size: None)

    def coarse_seek(_fd, _offset, whence):
        return 0 if whence == os.SEEK_DATA else len(data)

    monkeypatch.setattr(cm.os, "lseek", coarse_seek)
    with pytest.raises(cm.ChunkManifestError) as error:
        cm.verify_manifest(destination_root, manifest=manifest)
    assert error.value.code == "allocation_unverifiable_destination"


def test_destination_without_fiemap_rejects_dense_file(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "destination"
    root.mkdir()
    (root / "output.bin").write_bytes(b"dense data" * 100)
    manifest = _build(root)
    monkeypatch.setattr(cm, "_fiemap_extents", lambda _fd, _size: None)
    with pytest.raises(cm.ChunkManifestError) as error:
        cm.verify_manifest(root, manifest=manifest)
    assert error.value.code == "allocation_unverifiable_destination"


def test_source_mapping_with_holes_still_fails(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    path = root / "output.bin"
    path.write_bytes(b"x" * 4096)
    monkeypatch.setattr(cm, "_looks_sparse_by_blocks", lambda _st: True)
    monkeypatch.setattr(
        cm,
        "_fiemap_extents",
        lambda _fd, size: [
            cm._FiemapExtent(64, size - 64, cm._FIEMAP_EXTENT_ENCODED | cm._FIEMAP_EXTENT_LAST)
        ],
    )
    with pytest.raises(cm.ChunkManifestError) as error:
        _build(root)
    assert error.value.code == "sparse_file"


@pytest.mark.parametrize(
    "tamper", ("missing_destination", "unverified_destination", "wrong_hash", "cached_receipt")
)
def test_custody_rejects_incomplete_destination_proof(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    data = b"compressed output" * 50
    source_root, destination_root, manifest = _gpfs_like_source(tmp_path, monkeypatch, data)
    (destination_root / "output.bin").write_bytes(data)
    source = cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)
    destination = cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    if tamper == "missing_destination":
        destination["files"] = []
    elif tamper == "unverified_destination":
        destination["files"][0]["allocation_status"] = cm.ALLOCATION_UNVERIFIED
    elif tamper == "cached_receipt":
        destination["state_ref"] = {"kind": cm.STATE_SCHEMA_VERSION}
    else:
        destination["files"][0]["file_sha256"] = "0" * 64
    with pytest.raises(cm.ChunkManifestError):
        cm.build_custody_receipt(
            manifest,
            source,
            destination,
            source_root=source_root,
            destination_root=destination_root,
        )


@pytest.mark.parametrize(
    ("side", "mutation"),
    (
        ("source", "delete"),
        ("source", "modify"),
        ("destination", "delete"),
        ("destination", "modify"),
        ("destination", "sparse"),
    ),
)
def test_custody_rechecks_roots_after_verification(
    tmp_path: Path, side: str, mutation: str
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    data = b"\0" * (1024 * 1024) + b"x" if mutation == "sparse" else b"copy evidence" * 100
    (source_root / "output.bin").write_bytes(data)
    destination_file = destination_root / "output.bin"
    destination_file.write_bytes(data)
    manifest = _build(source_root)
    source = cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)
    destination = cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    assert source["status"] == destination["status"] == "ok"
    manifest_path = tmp_path / "manifest.json"
    source_path = tmp_path / "source-verify.json"
    destination_path = tmp_path / "destination-verify.json"
    receipt_path = tmp_path / "custody.json"
    for path, payload in (
        (manifest_path, manifest),
        (source_path, source),
        (destination_path, destination),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    changed_file = (source_root / "output.bin") if side == "source" else destination_file
    if mutation == "delete":
        changed_file.unlink()
    elif mutation == "sparse":
        with changed_file.open("wb") as handle:
            handle.seek(len(data) - 1)
            handle.write(b"x")
        _preallocate_beyond_eof(changed_file)
    else:
        changed_file.write_bytes(b"x" * len(data))
    result = cm.main(
        [
            "custody",
            "--manifest",
            str(manifest_path),
            "--source-verification",
            str(source_path),
            "--destination-verification",
            str(destination_path),
            "--source-root",
            str(source_root),
            "--destination-root",
            str(destination_root),
            "--output",
            str(receipt_path),
        ]
    )
    assert result == cm.EXIT_FAILED
    assert not receipt_path.exists()


@pytest.mark.parametrize("side", ("source", "destination"))
def test_custody_cli_rejects_symlinked_root(tmp_path: Path, capsys, side: str) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    data = b"copy evidence" * 100
    (source_root / "output.bin").write_bytes(data)
    (destination_root / "output.bin").write_bytes(data)
    manifest = _build(source_root)
    source = cm.verify_manifest(source_root, manifest=manifest, require_allocation=False)
    destination = cm.verify_manifest(destination_root, manifest=manifest, require_allocation=True)
    paths = {
        "manifest": tmp_path / "manifest.json",
        "source": tmp_path / "source-verify.json",
        "destination": tmp_path / "destination-verify.json",
    }
    for path, payload in (
        (paths["manifest"], manifest),
        (paths["source"], source),
        (paths["destination"], destination),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    linked_root = tmp_path / f"{side}-link"
    linked_root.symlink_to(
        source_root if side == "source" else destination_root, target_is_directory=True
    )
    roots = {"source": source_root, "destination": destination_root}
    roots[side] = linked_root
    output = tmp_path / "custody.json"
    assert (
        cm.main(
            [
                "custody",
                "--manifest",
                str(paths["manifest"]),
                "--source-verification",
                str(paths["source"]),
                "--destination-verification",
                str(paths["destination"]),
                "--source-root",
                str(roots["source"]),
                "--destination-root",
                str(roots["destination"]),
                "--output",
                str(output),
                "--json",
            ]
        )
        == cm.EXIT_FAILED
    )
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "symlink_root"
    assert not output.exists()


def test_source_mutation_guard_fails_closed(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"f.bin": b"x" * 100})
    real = cm._stat_checked
    calls = {"count": 0}

    def flaky(path: Path, relative: str):
        calls["count"] += 1
        identity = list(real(path, relative))
        if calls["count"] == 2:
            identity[0] += 1
        return tuple(identity)

    monkeypatch.setattr(cm, "_stat_checked", flaky)
    with pytest.raises(cm.ChunkManifestError) as mutated:
        _build(root)
    assert mutated.value.code == "source_mutated"


def test_missing_and_unexpected_members_fail_closed(tmp_path: Path, capsys) -> None:
    root, output, _first = _manifest(tmp_path, {"a.bin": b"a", "b.bin": b"b"}, capsys=capsys)
    (root / "b.bin").unlink()
    (root / "c.bin").write_bytes(b"c")
    assert _verify(root, output) == cm.EXIT_FAILED
    codes = {failure["code"] for failure in json.loads(capsys.readouterr().out)["failures"]}
    assert codes == {"missing_member", "unexpected_member"}


def test_excluded_member_reason_and_output_inside_root(tmp_path: Path, capsys) -> None:
    root, output, manifest = _manifest(
        tmp_path,
        {"keep.bin": b"k", "nested/drop.tmp": b"d"},
        extra=("--exclude", "*.tmp"),
        capsys=capsys,
    )
    assert manifest["excluded"] == [
        {"path": "nested/drop.tmp", "member_type": "file", "reason": "excluded_by_pattern:*.tmp"}
    ]
    assert _verify(root, output) == cm.EXIT_OK
    capsys.readouterr()
    args = ["manifest", "--root", str(root), "--output", str(root / "m.json"), "--json"]
    assert cm.main(args) == cm.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "path_inside_root"


def _manifest_for(root: Path, output: Path, *, chunk_size: int = 1024) -> dict:
    assert (
        cm.main(
            [
                "manifest",
                "--root",
                str(root),
                "--output",
                str(output),
                "--chunk-size",
                str(chunk_size),
            ]
        )
        == cm.EXIT_OK
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _resume_state(path: Path, *, root: Path, chunk_size: int = 1024) -> cm.ResumeState:
    policy = cm.ChunkingPolicy(chunk_size_bytes=chunk_size)
    return cm.ResumeState(
        root_identity=cm._default_root_identity(root),
        chunk_size_bytes=chunk_size,
        full_digest_threshold_bytes=cm._threshold(policy),
        path=path,
    )


def test_schema_extras_are_emitted_and_validated(tmp_path: Path) -> None:
    data = bytes(index % 251 for index in range(2500))
    _root, output, manifest = _manifest(tmp_path, {"data.bin": data}, chunk_size=1024)
    record = manifest["files"][0]

    assert record["digest_kind"] == cm.DIGEST_KIND_CHUNKED
    assert manifest["chunking"]["read_size_bytes"] == cm.READ_SIZE
    assert manifest["summary"] == {
        "member_count": 1,
        "total_bytes": len(data),
        "chunk_count": 3,
        "chunked_members": 1,
        "full_members": 0,
        "excluded_member_count": 0,
    }
    assert cm.validate_manifest(manifest) == []

    kind = json.loads(output.read_text(encoding="utf-8"))
    kind["files"][0]["digest_kind"] = "other"
    assert "manifest_digest_kind" in {issue["code"] for issue in cm.validate_manifest(kind)}

    summary = json.loads(output.read_text(encoding="utf-8"))
    summary["summary"]["member_count"] = 99
    assert "manifest_summary_mismatch" in {issue["code"] for issue in cm.validate_manifest(summary)}

    read_size = json.loads(output.read_text(encoding="utf-8"))
    read_size["chunking"]["read_size_bytes"] = 0
    assert "manifest_chunk_algorithm" in {
        issue["code"] for issue in cm.validate_manifest(read_size)
    }


def test_legacy_manifest_without_extras_still_validates(tmp_path: Path) -> None:
    _root, _output, manifest = _manifest(tmp_path, {"small.bin": b"hello"})
    legacy = {key: value for key, value in manifest.items() if key != "summary"}
    legacy["files"] = [
        {key: value for key, value in record.items() if key != "digest_kind"}
        for record in manifest["files"]
    ]
    legacy["chunking"] = {
        key: value for key, value in manifest["chunking"].items() if key != "read_size_bytes"
    }
    legacy["manifest_id"] = cm.compute_manifest_id(legacy)

    assert cm.validate_manifest(legacy) == []


def test_resume_reuses_cached_digests_and_rehashes_only_changed_members(
    tmp_path: Path, capsys
) -> None:
    data = bytes(index % 251 for index in range(2500))
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"data.bin": data, "small.bin": b"hello"})
    first = _manifest_for(root, tmp_path / "first.json")
    capsys.readouterr()
    state_path = tmp_path / "state.json"

    def run(output: Path) -> tuple[dict, dict]:
        assert (
            cm.main(
                [
                    "resume",
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--state",
                    str(state_path),
                    "--chunk-size",
                    "1024",
                    "--json",
                ]
            )
            == cm.EXIT_OK
        )
        receipt = json.loads(capsys.readouterr().out)
        return receipt, json.loads(output.read_text(encoding="utf-8"))

    warm_receipt, warmed = run(tmp_path / "warm.json")
    assert warm_receipt["resume"]["hashed_files"] == 2
    assert warmed["manifest_id"] == first["manifest_id"]

    reused_receipt, reused = run(tmp_path / "reused.json")
    assert reused_receipt["resume"] == {
        "cached_files": 2,
        "hashed_files": 0,
        "reused_chunks": 4,
        "reused_files": 2,
    }
    assert reused["manifest_id"] == first["manifest_id"]

    (root / "data.bin").write_bytes(data[:-10])
    changed_receipt, changed = run(tmp_path / "changed.json")
    assert changed_receipt["resume"]["reused_files"] == 1
    assert changed_receipt["resume"]["hashed_files"] == 1
    assert changed["manifest_id"] != first["manifest_id"]
    assert changed["tree_sha256"] != first["tree_sha256"]


def test_resume_state_fails_closed_when_untrusted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"a.bin": b"x" * 100})
    state = _resume_state(tmp_path / "state.json", root=root)
    state.record_file(
        "a.bin",
        identity=cm._stat_checked(root / "a.bin", "a.bin"),
        size_bytes=100,
        mode="full",
        chunks=[{"index": 0, "offset": 0, "length": 100, "sha256": "0" * 64}],
        content_sha256="0" * 64,
        complete=True,
        force_persist=True,
    )
    state_path = state.path
    assert state_path is not None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    identity = payload["root_identity"]
    geometry = {
        "root_identity": identity,
        "chunk_size_bytes": 1024,
        "full_digest_threshold_bytes": 1024,
    }

    state_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(cm.ChunkManifestError) as invalid:
        cm.ResumeState.load(state_path, **geometry)
    assert invalid.value.code == "state_invalid"

    state_path.write_text(
        json.dumps({**payload, "schema_version": "chunk_manifest.state.v2"}), "utf-8"
    )
    with pytest.raises(cm.ChunkManifestError) as schema:
        cm.ResumeState.load(state_path, **geometry)
    assert schema.value.code == "state_schema_unsupported"

    tampered = json.loads(json.dumps(payload))
    tampered["files"][0]["size_bytes"] = 999
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(cm.ChunkManifestError) as digest:
        cm.ResumeState.load(state_path, **geometry)
    assert digest.value.code == "state_invalid"

    with pytest.raises(cm.ChunkManifestError) as policy:
        cm.ResumeState.load(
            state_path,
            root_identity=identity,
            chunk_size_bytes=2048,
            full_digest_threshold_bytes=1024,
        )
    assert policy.value.code == "state_policy_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("chunk_size_bytes", "full_digest_threshold_bytes")
        for value in (True, float("nan"), float("inf"), -1, "not-an-int")
    ],
)
def test_invalid_state_geometry_returns_state_invalid_receipt(
    tmp_path: Path, capsys, field: str, value
) -> None:
    """Malformed persisted geometry returns the documented fail-closed receipt."""
    root = tmp_path / "root"
    root.mkdir()
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "manifest.json"
    chunking = {
        "algorithm": cm.ALGORITHM,
        "chunk_size_bytes": 1024,
        "full_digest_threshold_bytes": 1024,
    }
    chunking[field] = value
    state_path.write_text(
        json.dumps(
            {
                "schema_version": cm.STATE_SCHEMA_VERSION,
                "root_identity": cm._default_root_identity(root),
                "chunking": chunking,
                "files": [],
                "state_id": "not-reached",
            }
        ),
        encoding="utf-8",
    )

    result = cm.main(
        [
            "resume",
            "--root",
            str(root),
            "--output",
            str(output_path),
            "--state",
            str(state_path),
            "--chunk-size",
            "1024",
            "--json",
        ]
    )

    assert result == cm.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "state_invalid"


def test_resume_state_root_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"a.bin": b"x" * 100})
    state_path = tmp_path / "state.json"
    cm.ResumeState(
        root_identity="sha256:somewhere-else",
        chunk_size_bytes=1024,
        full_digest_threshold_bytes=1024,
        path=state_path,
    ).flush()

    with pytest.raises(cm.ChunkManifestError) as mismatch:
        cm.ResumeState.load(
            state_path,
            root_identity=cm._default_root_identity(root),
            chunk_size_bytes=1024,
            full_digest_threshold_bytes=1024,
        )
    assert mismatch.value.code == "state_root_mismatch"


def test_partial_chunk_checkpoint_resumes_mid_file(tmp_path: Path) -> None:
    data = bytes(index % 251 for index in range(2500))
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"data.bin": data})
    policy = cm.ChunkingPolicy(chunk_size_bytes=1024)
    artifact = cm.ArtifactIdentity("fixture", "v1", "sha256:fixture-root")
    full = cm.build_manifest(root, artifact=artifact, policy=policy)
    identity = cm._stat_checked(root / "data.bin", "data.bin")
    state_path = tmp_path / "state.json"
    state = _resume_state(state_path, root=root)
    state.record_file(
        "data.bin",
        identity=identity,
        size_bytes=len(data),
        mode="chunked",
        chunks=[full["files"][0]["chunks"][0]],
        content_sha256=None,
        complete=False,
        force_persist=True,
    )

    loaded = cm.ResumeState.load(
        state_path,
        root_identity=cm._default_root_identity(root),
        chunk_size_bytes=1024,
        full_digest_threshold_bytes=1024,
    )
    resumed = cm.build_manifest(root, artifact=artifact, policy=policy, state=loaded)

    assert resumed["manifest_id"] == full["manifest_id"]
    assert loaded.stats()["reused_chunks"] == 1
    assert loaded.stats()["reused_files"] == 0
    assert loaded.stats()["hashed_files"] == 1


def test_verify_with_state_reuses_digests_and_records_changes(tmp_path: Path, capsys) -> None:
    data = bytes(index % 251 for index in range(2500))
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"data.bin": data})
    manifest = _manifest_for(root, tmp_path / "manifest.json")
    capsys.readouterr()
    state_path = tmp_path / "state.json"
    args = [
        "verify",
        "--root",
        str(root),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--state",
        str(state_path),
        "--json",
    ]

    assert cm.main(args) == cm.EXIT_OK
    first = json.loads(capsys.readouterr().out)
    assert first["resume"]["hashed_files"] == 1
    assert cm.main(args) == cm.EXIT_OK
    second = json.loads(capsys.readouterr().out)
    assert second["resume"]["reused_files"] == 1
    assert second["state_ref"]["kind"] == cm.STATE_SCHEMA_VERSION

    changed = bytearray(data)
    changed[1100] ^= 0xFF
    (root / "data.bin").write_bytes(bytes(changed))
    assert cm.main(args) == cm.EXIT_FAILED
    failed = json.loads(capsys.readouterr().out)
    assert failed["failures"][0]["code"] == "chunk_digest_mismatch"
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["files"][0]["complete"] is True
    assert updated["files"][0]["content_sha256"] != manifest["files"][0]["content_sha256"]
    assert failed["manifest_id"] == manifest["manifest_id"]


def test_resume_state_and_output_must_differ(tmp_path: Path, capsys) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"a.bin": b"x"})
    same = tmp_path / "both.json"
    args = [
        "resume",
        "--root",
        str(root),
        "--output",
        str(same),
        "--state",
        str(same),
        "--json",
    ]
    assert cm.main(args) == cm.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "state_manifest_conflict"


def test_compare_reports_added_removed_and_changed_deterministically(
    tmp_path: Path, capsys
) -> None:
    left_root, right_root = tmp_path / "left", tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    _write(
        left_root, {"keep.bin": b"same", "gone.bin": b"x", "size.bin": b"one", "swap.bin": b"aaa"}
    )
    _write(
        right_root,
        {"keep.bin": b"same", "new.bin": b"y", "size.bin": b"three", "swap.bin": b"bbb"},
    )
    left = _manifest_for(left_root, tmp_path / "left.json")
    right = _manifest_for(right_root, tmp_path / "right.json")

    assert (
        cm.main(
            [
                "compare",
                "--left",
                str(tmp_path / "left.json"),
                "--right",
                str(tmp_path / "left.json"),
            ]
        )
        == cm.EXIT_OK
    )
    capsys.readouterr()
    assert (
        cm.main(
            [
                "compare",
                "--left",
                str(tmp_path / "left.json"),
                "--right",
                str(tmp_path / "right.json"),
                "--json",
            ]
        )
        == cm.EXIT_FAILED
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "different"
    assert result["added"] == ["new.bin"]
    assert result["removed"] == ["gone.bin"]
    assert [entry["path"] for entry in result["changed"]] == ["size.bin", "swap.bin"]
    assert [entry["reason"] for entry in result["changed"]] == ["size_changed", "content_changed"]
    assert result["counts"] == {"added": 1, "removed": 1, "changed": 2, "unchanged": 1}
    assert result["left_manifest_id"] == left["manifest_id"]
    assert result["right_manifest_id"] == right["manifest_id"]
    assert str(tmp_path) not in json.dumps(result)


def test_failure_list_is_truncated_with_true_count(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cm, "MAX_FAILURES", 1)
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"a.bin": b"aaa", "b.bin": b"bbb", "c.bin": b"ccc"})
    _manifest_for(root, tmp_path / "manifest.json")
    capsys.readouterr()
    _write(root, {"a.bin": b"zzzz", "b.bin": b"yyyy", "c.bin": b"xxxx"})

    assert _verify(root, tmp_path / "manifest.json") == cm.EXIT_FAILED
    result = json.loads(capsys.readouterr().out)
    assert len(result["failures"]) == 1
    assert result["failure_count"] == 3
    assert result["failures_truncated"] is True


def test_atomic_manifest_write_keeps_previous_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "out.json"
    target.write_text('{"previous": true}', encoding="utf-8")

    def crash(source, destination):
        raise OSError("simulated crash")

    monkeypatch.setattr(cm.os, "replace", crash)
    with pytest.raises(OSError):
        cm._write_json(target, {"next": True})

    assert target.read_text(encoding="utf-8") == '{"previous": true}'
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.json"]


def test_progress_output_is_bounded_and_stdout_stays_json(tmp_path: Path, capsys) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(root, {"a.bin": b"a", "b.bin": b"b", "c.bin": b"c"})
    assert (
        cm.main(
            [
                "manifest",
                "--root",
                str(root),
                "--output",
                str(tmp_path / "manifest.json"),
                "--progress-every",
                "1",
                "--json",
            ]
        )
        == cm.EXIT_OK
    )
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("progress:")]
    assert lines == [
        "progress: 1/3 members, 1 bytes",
        "progress: 2/3 members, 2 bytes",
        "progress: 3/3 members, 3 bytes",
    ]
    assert json.loads(captured.out)["status"] == "ok"
