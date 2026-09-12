"""Tests for source_bundle_export (#8851).

Fixture scenarios: clean commit, dirty tree, admitted patch, missing object, shallow
clone, subproject mismatch, generated-file drift, private remote, byte-stable verify.
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


def test_admitted_patch_is_checksum_bound(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("patched\n", encoding="utf-8")
    patch = tmp_path / "admitted.patch"
    patch.write_text(_git(repo, "diff").stdout, encoding="utf-8")
    _git(repo, "checkout", "--", "a.txt")

    out = tmp_path / "bundle"
    manifest = export_bundle(repo, out, "patched", patch_path=patch)
    assert manifest["status"] == "pass"
    assert manifest["patch"]["sha256"]
    assert (out / "admitted.patch").is_file()
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
    assert "subproject_mismatch" in report["reasons"]
    assert "generated_drift" in report["reasons"]


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


def test_public_status_redacts_private_paths(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manifest = export_bundle(repo, tmp_path / "bundle", "redact")
    status = public_status(manifest)
    text = json.dumps(status)
    assert str(tmp_path) not in text
    assert status["status"] == "pass"
