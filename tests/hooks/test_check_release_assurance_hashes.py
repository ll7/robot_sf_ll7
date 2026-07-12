"""Contract tests for the release-assurance pre-commit hook."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).resolve().parents[2] / "hooks" / "check_release_assurance_hashes.py"
_SPEC = importlib.util.spec_from_file_location("check_release_assurance_hashes", _HOOK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HOOK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HOOK)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _write_evidence(repo: Path, source_path: str, digest: str) -> Path:
    evidence_path = repo / _HOOK.EVIDENCE_PATH
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps({"evidence": [{"path": source_path, "sha256": digest}]}) + "\n",
        encoding="utf-8",
    )
    return evidence_path


@pytest.fixture
def indexed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Create a small repository whose evidence and source are staged."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "gate@example.invalid")
    _git(tmp_path, "config", "user.name", "Gate")
    source = tmp_path / "docs" / "RELEASE.md"
    source.parent.mkdir()
    source.write_text("old release\n", encoding="utf-8")
    _write_evidence(tmp_path, "docs/RELEASE.md", hashlib.sha256(source.read_bytes()).hexdigest())
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")
    monkeypatch.chdir(tmp_path)
    return tmp_path, source


def test_updates_and_stages_hash_from_index_not_unstaged_worktree(
    indexed_repo: tuple[Path, Path],
) -> None:
    """A staged source digest wins over conflicting unstaged bytes."""
    repo, source = indexed_repo
    source.write_text("staged release\n", encoding="utf-8")
    _git(repo, "add", "docs/RELEASE.md")
    source.write_text("unstaged release\n", encoding="utf-8")

    assert _HOOK.main() == 1

    staged_evidence = json.loads(_git(repo, "show", f":{_HOOK.EVIDENCE_PATH.as_posix()}").stdout)
    assert (
        staged_evidence["evidence"][0]["sha256"] == hashlib.sha256(b"staged release\n").hexdigest()
    )


def test_rejects_non_file_evidence_path(indexed_repo: tuple[Path, Path]) -> None:
    """Directory-valued evidence paths fail closed rather than being hashed."""
    repo, _ = indexed_repo
    _write_evidence(repo, "docs", "ignored")
    _git(repo, "add", _HOOK.EVIDENCE_PATH.as_posix())

    assert _HOOK.main() == 1
