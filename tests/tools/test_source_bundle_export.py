"""Tests for source_bundle_export (#8851, repaired by #9131).

Coverage: clean commit, dirty tree, admitted patch binding, missing object, shallow
clone, vendored subproject, gitlink subproject, generated-file confinement, private
remote redaction, ignored state, byte-stable export, and CLI round trip.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from scripts.tools.source_bundle_export import (
    export_bundle,
    main,
    public_status,
    verify_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

GIT = ["git", "-c", "user.email=t@example.com", "-c", "user.name=tester"]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*GIT, "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _repo(tmp_path: Path, vendored: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
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
    assert manifest["ref"] == "HEAD"
    assert (out / "source.bundle").is_file()
    report = verify_bundle(out, tmp_path / "restore")
    assert report["status"] == "pass"


def test_dirty_untracked_and_ignored_state_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    assert export_bundle(repo, tmp_path / "b1", "dirty")["reasons"] == ["dirty_tree_not_admitted"]

    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert export_bundle(repo, tmp_path / "b2", "untracked")["reasons"] == [
        "untracked_state_not_admitted"
    ]

    (repo / "scratch.txt").unlink()
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore")
    (repo / "ignored.txt").write_text("secret-ish\n", encoding="utf-8")
    manifest = export_bundle(repo, tmp_path / "b3", "ignored")
    assert manifest["status"] == "blocked"
    assert "ignored_state_not_admitted" in manifest["reasons"]


def test_admitted_patch_must_match_captured_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("patched\n", encoding="utf-8")
    patch = tmp_path / "admitted.patch"
    patch.write_text(_git(repo, "diff").stdout, encoding="utf-8")

    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "patched", patch_path=patch)
    assert manifest["status"] == "pass"
    assert manifest["patch"]["matches_captured_diff"] is True
    assert manifest["patch"]["captured_diff_sha256"]
    report = verify_bundle(out, tmp_path / "restore")
    assert report["status"] == "pass", report["discrepancies"]
    assert "patched" in (tmp_path / "restore" / "a.txt").read_text(encoding="utf-8")

    _git(repo, "checkout", "--", "a.txt")
    mismatched = tmp_path / "foreign.patch"
    mismatched.write_text(
        "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-hello\n+other\n", encoding="utf-8"
    )
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    manifest = export_bundle(repo, tmp_path / "b2", "mismatch", patch_path=mismatched)
    assert manifest["status"] == "blocked"
    assert "patch_does_not_match_dirty_state" in manifest["reasons"]


def test_missing_object_and_shallow_clone_are_blocked(tmp_path: Path) -> None:
    missing = export_bundle(tmp_path / "not-a-repo", tmp_path / "b0", "missing")
    assert missing["status"] == "blocked"
    assert "missing_object" in missing["reasons"]

    repo = _repo(tmp_path)
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--no-local", str(repo), str(shallow)],
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = export_bundle(shallow, tmp_path / "b1", "shallow")
    assert manifest["status"] == "blocked"
    assert "shallow_clone_rejected" in manifest["reasons"]


def test_vendored_subproject_mismatch(tmp_path: Path) -> None:
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
    assert "subproject_mismatch" in report["reasons"]
    assert "generated_drift" in report["reasons"]


def test_gitlink_subproject_is_explicitly_rejected_and_never_crashes(tmp_path: Path) -> None:
    """A gitlink is not carried by the bundle: it must be rejected, not crash."""
    inner = _repo(tmp_path / "inner")
    repo = _repo(tmp_path / "outer")
    subprocess.run(
        [
            *GIT,
            "-C",
            str(repo),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(inner),
            "sub",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(repo, "commit", "-qm", "add submodule")
    manifest = export_bundle(repo, tmp_path / "bundle", "submodular", subprojects=["sub"])
    assert manifest["status"] == "blocked"
    assert "subproject_not_restorable" in manifest["reasons"]
    assert any(
        row["kind"] == "git" and row["restorable"] is False for row in manifest["subprojects"]
    )
    # inventory must tolerate the non-numeric gitlink size
    assert any(
        row["kind"] == "commit" and row["size_bytes"] is None for row in manifest["inventory"]
    )


def test_generated_source_custody_is_confined(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = repo / "link.txt"
    link.symlink_to(outside)

    escaped = export_bundle(repo, tmp_path / "b1", "escape", generated=[{"path": "../outside.txt"}])
    assert escaped["status"] == "blocked"
    assert "generated_path_escape" in escaped["reasons"]

    linked = export_bundle(repo, tmp_path / "b2", "link", generated=[{"path": "link.txt"}])
    assert linked["status"] == "blocked"
    assert "generated_path_escape" in linked["reasons"]
    link.unlink()

    confined = export_bundle(repo, tmp_path / "b3", "ok", generated=[{"path": "a.txt"}])
    assert confined["status"] == "pass"
    assert confined["generated"][0]["confined"] is True


def test_private_remote_is_redacted_to_a_digest(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://user:secret@git.internal.example/org/repo.git?access_token=abc",
    )
    manifest = export_bundle(repo, tmp_path / "bundle", "private")
    status = public_status(manifest)
    text = json.dumps(status)
    assert status["remote"] is None
    assert status["remote_public"] is False
    assert status["remote_digest"]
    for secret in ("secret", "git.internal.example", "access_token", "abc"):
        assert secret not in text
    assert str(tmp_path) not in text


def test_export_is_byte_stable_and_cli_round_trips(tmp_path: Path, capsys) -> None:
    repo = _repo(tmp_path)
    first, second = tmp_path / "one", tmp_path / "two"
    assert export_bundle(repo, first, "stable")["status"] == "pass"
    assert export_bundle(repo, second, "stable")["status"] == "pass"
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()

    third = tmp_path / "three"
    assert (
        main(
            [
                "--export",
                "--repo",
                str(repo),
                "--out",
                str(third),
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
