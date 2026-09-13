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
