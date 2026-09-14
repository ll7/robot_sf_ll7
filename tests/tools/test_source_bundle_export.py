"""Tests for source_bundle_export (#8851).

Fixture scenarios: clean commit, dirty tree, admitted patch replay, missing object,
shallow clone, subproject mismatch, generated-file drift, private remote, byte-stable
verify, complete checksum coverage, and fail-closed malformed inputs. Additional
adversarial fixtures cover ignored state, linked shallow worktrees, encoded URL suffix
redaction, gitlinks, and source/bundle path escapes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.tools.source_bundle_export import (
    export_bundle,
    main,
    public_status,
    verify_bundle,
)

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=tester"]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*GIT, "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _repo(tmp_path: Path, vendored: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    if vendored:
        (repo / "third_party" / "vend").mkdir(parents=True)
        (repo / "third_party" / "vend" / "lib.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_clean_commit_exports_and_verifies(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "workload-a")
    assert manifest["status"] == "pass"
    assert (out / "source.bundle").is_file()
    assert (out / "SHA256SUMS").is_file()
    assert manifest["inventory_count"] == 1
    report = verify_bundle(out, tmp_path / "restore")
    assert report["status"] == "pass"
    assert report["commit"] == manifest["commit"]


def test_dirty_and_untracked_state_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    manifest = export_bundle(repo, tmp_path / "b1", "dirty")
    assert manifest["status"] == "blocked"
    assert "dirty_tree_not_admitted" in manifest["reasons"]

    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    manifest = export_bundle(repo, tmp_path / "b2", "untracked")
    assert manifest["status"] == "blocked"
    assert "untracked_state_not_admitted" in manifest["reasons"]


def test_ignored_state_is_rejected_and_does_not_create_partial_bundle(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("*.secret\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore secret fixtures")
    (repo / "credentials.secret").write_text("do-not-export\n", encoding="utf-8")

    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "ignored")

    assert manifest["status"] == "blocked"
    assert "ignored_state_not_admitted" in manifest["reasons"]
    assert manifest["private_state_rejected"] is True
    assert not (out / "source.bundle").exists()


def test_admitted_patch_is_checksum_bound(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("patched\n", encoding="utf-8")
    patch = tmp_path / "admitted.patch"
    patch.write_text(_git(repo, "diff").stdout, encoding="utf-8")

    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "patched", patch_path=patch)
    assert manifest["status"] == "pass"
    assert manifest["patch"]["sha256"] == manifest["patch"]["captured_diff_sha256"]
    assert (out / "admitted.patch").is_file()
    report = verify_bundle(out, tmp_path / "restore")
    assert report["status"] == "pass"
    assert (tmp_path / "restore" / "a.txt").read_text(encoding="utf-8") == "patched\n"


def test_admitted_patch_must_match_current_dirty_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("first\n", encoding="utf-8")
    patch = tmp_path / "admitted.patch"
    patch.write_text(_git(repo, "diff").stdout, encoding="utf-8")
    (repo / "a.txt").write_text("different\n", encoding="utf-8")

    manifest = export_bundle(repo, tmp_path / "bundle", "patch-mismatch", patch_path=patch)

    assert manifest["status"] == "blocked"
    assert "patch_not_bound" in manifest["reasons"]
    assert not (tmp_path / "bundle").exists()


def test_admitted_staged_patch_replays_with_index_binding(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    patch = tmp_path / "staged.patch"
    patch.write_text(_git(repo, "diff", "--cached").stdout, encoding="utf-8")

    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "staged", patch_path=patch)

    assert manifest["status"] == "pass"
    assert verify_bundle(out, tmp_path / "restore")["status"] == "pass"


def test_missing_object_and_shallow_clone_are_blocked(tmp_path: Path) -> None:
    missing = export_bundle(tmp_path / "not-a-repo", tmp_path / "b0", "missing")
    assert missing["status"] == "blocked"
    assert "missing_object" in missing["reasons"]

    repo = _repo(tmp_path)
    head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".git" / "shallow").write_text(f"{head_sha}\n", encoding="utf-8")
    manifest = export_bundle(repo, tmp_path / "b1", "shallow")
    assert manifest["status"] == "blocked"
    assert "shallow_clone_rejected" in manifest["reasons"]


def test_shallow_linked_worktree_is_blocked_via_git_common_state(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", "--detach", str(linked))
    assert (linked / ".git").is_file()

    shallow = Path(_git(linked, "rev-parse", "--git-path", "shallow").stdout.strip())
    shallow.write_text(f"{_git(linked, 'rev-parse', 'HEAD').stdout.strip()}\n", encoding="utf-8")

    manifest = export_bundle(linked, tmp_path / "bundle", "linked-shallow")

    assert manifest["status"] == "blocked"
    assert "shallow_clone_rejected" in manifest["reasons"]


def test_gitlink_inventory_is_stable_and_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init", "-q", "-b", "main")
    (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
    _git(nested, "add", "nested.txt")
    _git(nested, "commit", "-qm", "nested")
    nested_head = _git(nested, "rev-parse", "HEAD").stdout.strip()
    (repo / "third_party").mkdir()
    (repo / "third_party" / "nested").mkdir()
    _git(
        repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{nested_head},third_party/nested",
    )
    _git(repo, "commit", "-qm", "add gitlink")

    manifest = export_bundle(repo, tmp_path / "bundle", "gitlink")

    assert manifest["status"] == "blocked"
    assert "gitlink_rejected" in manifest["reasons"]
    row = next(row for row in manifest["inventory"] if row["path"] == "third_party/nested")
    assert row["size_bytes"] is None


def test_declared_git_subproject_is_explicitly_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path, vendored=True)
    marker = repo / "third_party" / "vend" / ".git"
    marker.write_text("gitdir: ../.git/modules/vend\n", encoding="utf-8")

    manifest = export_bundle(
        repo, tmp_path / "bundle", "git-subproject", subprojects=["third_party/vend"]
    )

    assert manifest["status"] == "blocked"
    assert "git_subproject_rejected" in manifest["reasons"]
    assert manifest["subprojects"][0]["kind"] == "git_rejected"


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://user:secret@github.com/org/repo.git",
        "https://github.com/org/repo.git?access_token=secret#password=also-secret",
        "https://github.com/org/repo.git#token=secret",
    ],
)
def test_remote_digest_is_retained_while_url_suffixes_stay_private(
    tmp_path: Path, remote_url: str
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "remote", "add", "origin", remote_url)

    manifest = export_bundle(repo, tmp_path / "bundle", "remote")
    status = public_status(manifest)

    expected_digest = hashlib.sha256(remote_url.encode("utf-8")).hexdigest()
    assert manifest["remote_digest"] == expected_digest
    assert status["remote_public"] is False
    assert status["remote"] is None
    assert status["remote_digest"] == expected_digest
    assert "secret" not in json.dumps(status)
    assert "access_token" not in json.dumps(status)
    assert "password" not in json.dumps(status)


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://github.com/org/repo.git%3Ftoken%3Dsecret",
        "https://github.com/org/repo.git%23password%3Dsecret",
        "https://user%3Asecret%40github.com/org/repo.git",
    ],
)
def test_encoded_remote_credentials_and_suffixes_stay_private(
    tmp_path: Path, remote_url: str
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "remote", "add", "origin", remote_url)

    manifest = export_bundle(repo, tmp_path / "bundle", "encoded-remote")
    status = public_status(manifest)

    assert manifest["remote_public"] is False
    assert status["remote"] is None
    assert "secret" not in json.dumps(status)


@pytest.mark.parametrize("declaration", ["generated", "subproject"])
@pytest.mark.parametrize("path_kind", ["absolute", "parent", "symlink"])
def test_declared_paths_cannot_escape_repository(
    tmp_path: Path, declaration: str, path_kind: str
) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    if path_kind == "absolute":
        declared_path = str(outside)
    elif path_kind == "parent":
        declared_path = "../outside.txt"
    else:
        link = repo / "escape-link"
        link.symlink_to(outside)
        declared_path = "escape-link"

    kwargs = (
        {"generated": [{"path": declared_path}]}
        if declaration == "generated"
        else {"subprojects": [declared_path]}
    )
    manifest = export_bundle(repo, tmp_path / "bundle", "path-escape", **kwargs)
    status_text = json.dumps(public_status(manifest))

    assert manifest["status"] == "blocked"
    assert "path_escape" in manifest["reasons"]
    assert str(outside) not in status_text
    assert outside.read_text(encoding="utf-8") == "outside-secret\n"


def test_vendored_symlink_escape_is_not_hashed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    vendored = repo / "third_party" / "vend"
    vendored.mkdir(parents=True)
    (vendored / "escape-link").symlink_to(tmp_path / "outside.txt")
    (tmp_path / "outside.txt").write_text("outside-secret\n", encoding="utf-8")

    manifest = export_bundle(
        repo, tmp_path / "bundle", "vendored-escape", subprojects=["third_party/vend"]
    )

    assert manifest["status"] == "blocked"
    assert "path_escape" in manifest["reasons"]
    assert manifest["subprojects"][0]["identity"] is None


def test_subproject_mismatch_and_generated_drift(tmp_path: Path) -> None:
    repo = _repo(tmp_path, vendored=True)
    out = tmp_path / "bundle"
    manifest = export_bundle(
        repo, out, "v", subprojects=["third_party/vend"], generated=[{"path": "a.txt"}]
    )
    assert manifest["status"] == "pass"
    assert verify_bundle(out, tmp_path / "restore")["status"] == "pass"

    tampered = out / "manifest.json"
    data = json.loads(tampered.read_text(encoding="utf-8"))
    data["subprojects"][0]["identity"] = "0" * 64
    data["generated"][0]["sha256"] = "1" * 64
    tampered.write_text(json.dumps(data), encoding="utf-8")
    report = verify_bundle(out, tmp_path / "restore2")
    assert report["status"] == "fail"
    assert "bundle_checksum_mismatch" in report["reasons"]
    assert not (tmp_path / "restore2").exists()


def test_private_remote_is_never_published(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "remote", "add", "origin", "https://user:secret@git.internal.example/org/repo.git")
    manifest = export_bundle(repo, tmp_path / "bundle", "private")
    status = public_status(manifest)
    assert status["remote"] is None
    assert status["remote_public"] is False
    assert "secret" not in json.dumps(status)
    assert "git.internal.example" not in json.dumps(status)


def test_export_is_byte_stable_and_cli_returns_codes(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    first, second = tmp_path / "one", tmp_path / "two"
    assert export_bundle(repo, first, "stable")["status"] == "pass"
    assert export_bundle(repo, second, "stable")["status"] == "pass"
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()

    assert (
        main(
            [
                "--export",
                "--repo",
                str(repo),
                "--out",
                str(third := tmp_path / "three"),
                "--workload-id",
                "cli",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert (
        main(
            [
                "--verify",
                "--bundle",
                str(third),
                "--workdir",
                str(tmp_path / "restore"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert (
        main(
            [
                "--export",
                "--repo",
                str(repo),
                "--out",
                str(tmp_path / "four"),
                "--workload-id",
                "cli",
            ]
        )
        == 0
    )


def test_ref_context_is_stable_when_local_ref_names_change(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = export_bundle(repo, first, "same-commit")
    _git(repo, "checkout", "-qb", "alternate")
    second_manifest = export_bundle(repo, second, "same-commit")

    assert first_manifest["ref_context"] == "resolved_commit"
    assert second_manifest["ref_context"] == first_manifest["ref_context"]
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_writer_errors_return_stable_status_without_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _repo(tmp_path)

    def fail_writer(*args, **kwargs):
        raise OSError("simulated writer failure")

    monkeypatch.setattr("scripts.tools.source_bundle_export._write_bundle_files", fail_writer)
    manifest = export_bundle(repo, tmp_path / "bundle", "writer-failure")

    assert manifest["status"] == "blocked"
    assert "output_write_failed" in manifest["reasons"]
    assert not (tmp_path / "bundle").exists()


def test_cli_exposes_validated_generated_source_input(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "generated-bundle"

    assert (
        main(
            [
                "--export",
                "--repo",
                str(repo),
                "--out",
                str(out),
                "--generated",
                "a.txt",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    data = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert data["generated"][0]["path"] == "a.txt"


def test_invalid_generated_input_is_blocked_without_writing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    manifest = export_bundle(
        repo,
        tmp_path / "bundle",
        "bad-generated",
        generated=[{"path": 3}],  # type: ignore[list-item]
    )

    assert manifest["status"] == "blocked"
    assert "generated_input_invalid" in manifest["reasons"]
    assert not (tmp_path / "bundle").exists()


def test_preexisting_output_root_is_rejected_and_untouched(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    marker = out / "unrelated.txt"
    marker.write_text("keep\n", encoding="utf-8")

    manifest = export_bundle(repo, out, "contaminated")

    assert manifest["status"] == "blocked"
    assert "output_contaminated" in manifest["reasons"]
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_reserved_patch_member_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    patch = tmp_path / "manifest.json"
    patch.write_text(_git(repo, "diff").stdout, encoding="utf-8")

    manifest = export_bundle(repo, tmp_path / "bundle", "reserved", patch_path=patch)

    assert manifest["status"] == "blocked"
    assert "output_member_collision" in manifest["reasons"]


def test_public_status_redacts_private_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manifest = export_bundle(repo, tmp_path / "bundle", "redact")
    status = public_status(manifest)
    text = json.dumps(status)
    assert str(tmp_path) not in text
    assert status["status"] == "pass"


def test_public_status_redacts_unsafe_subproject_path_and_blocks(tmp_path: Path) -> None:
    private_path = tmp_path / "private-secret.txt"
    status = public_status(
        {
            "status": "pass",
            "workload_id": "safe",
            "remote_public": False,
            "subprojects": [{"path": str(private_path), "kind": "vendored", "identity": "0" * 64}],
            "reasons": [],
        }
    )

    assert status["status"] == "blocked"
    assert status["subprojects"][0]["path"] == "[REDACTED_PATH]"
    assert str(private_path) not in json.dumps(status)
    assert "sanitization_failed" in status["reasons"]


def test_public_status_rejects_unsafe_scalar_projection() -> None:
    status = public_status(
        {
            "status": "pass",
            "workload_id": "../private-secret",
            "remote_public": True,
            "remote": "https://github.com/org/repo.git?token=secret",
            "inventory_count": "/private/inventory",
            "reasons": [],
        }
    )

    assert status["status"] == "blocked"
    assert status["workload_id"] == "[REDACTED]"
    assert status["remote_public"] is False
    assert status["remote"] is None
    assert status["inventory_count"] == 0
    assert "secret" not in json.dumps(status)


def test_verify_rejects_manifest_path_escape_without_reading_outside(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "verify-path")["status"] == "pass"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["generated"] = [{"path": str(outside), "sha256": "0" * 64}]
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert "manifest_invalid" in report["reasons"]
    assert outside.read_text(encoding="utf-8") == "outside-secret\n"


def test_verify_rejects_symlinked_manifest_without_reading_target(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "verify-symlink")["status"] == "pass"
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("{}\n", encoding="utf-8")
    manifest_path = out / "manifest.json"
    manifest_path.unlink()
    manifest_path.symlink_to(outside)

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert report["reasons"] == ["path_escape"]
    assert outside.read_text(encoding="utf-8") == "{}\n"


def test_verify_requires_complete_checksum_coverage(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "checksum")["status"] == "pass"
    sums = out / "SHA256SUMS"
    rows = sums.read_text(encoding="utf-8").splitlines()
    sums.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert "checksum_incomplete" in report["reasons"]


def test_verify_rejects_checksum_rows_for_unowned_members(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "checksum-extra")["status"] == "pass"
    sums = out / "SHA256SUMS"
    sums.write_text(sums.read_text(encoding="utf-8") + f"{'0' * 64}  not-owned\n", encoding="utf-8")

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert "checksum_incomplete" in report["reasons"]
    assert not (tmp_path / "restore").exists()


def test_verify_rejects_undecodable_checksum_input_stably(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "checksum-invalid")["status"] == "pass"
    (out / "SHA256SUMS").write_bytes(b"\xff\xfe\n")

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert report["reasons"] == ["checksum_invalid"]


def test_verify_rejects_malformed_manifest_without_raising(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "malformed")["status"] == "pass"
    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("commit")
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    report = verify_bundle(out, tmp_path / "restore")

    assert report["status"] == "blocked"
    assert report["reasons"] == ["manifest_invalid"]


def test_verify_rejects_non_directory_restore_workdir_stably(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "bundle"
    assert export_bundle(repo, out, "restore-path")["status"] == "pass"
    restore = tmp_path / "restore"
    restore.write_text("do not overwrite\n", encoding="utf-8")

    report = verify_bundle(out, restore)

    assert report["status"] == "blocked"
    assert report["reasons"] == ["restore_workdir_unavailable"]
    assert restore.read_text(encoding="utf-8") == "do not overwrite\n"
