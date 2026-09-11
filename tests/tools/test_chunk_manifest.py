"""Focused tests for the deterministic chunk-manifest tool (issue #8915)."""

from __future__ import annotations

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
