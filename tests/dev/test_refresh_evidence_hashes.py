"""Tests for the evidence-hash refresh helper (issue #8390).

The helper refreshes stale ``sha256`` declarations in evidence-registry JSON
files without regenerating the machine baseline. These tests lock the
fail-closed contract: baseline files are refused, ambiguous pins are skipped,
rewrites round-trip byte-exact, and check mode exits nonzero on stale input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "refresh_evidence_hashes.py"

_spec = importlib.util.spec_from_file_location("refresh_evidence_hashes", SCRIPT)
assert _spec is not None and _spec.loader is not None
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)

STALE_HASH = "0" * 64


def _write_evidence(tmp_path: Path, name: str, artifact: str, declared: str) -> Path:
    """Write a minimal evidence JSON pinning one artifact hash."""
    target = tmp_path / name
    target.write_text(
        json.dumps({"evidence": [{"path": artifact, "sha256": declared}]}),
        encoding="utf-8",
    )
    return target


def test_stale_declaration_detected_with_actual_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatched declaration is reported with the recomputed value."""
    target = _write_evidence(tmp_path, "evidence.json", "docs/README.md", STALE_HASH)
    monkeypatch.setattr(helper, "_actual_hash", lambda _root, _art: "ab" * 32)
    stale, skipped = helper._stale_declarations(tmp_path, target)
    assert skipped == []
    assert len(stale) == 1
    assert stale[0]["artifact"] == "docs/README.md"
    assert stale[0]["declared"] == STALE_HASH
    assert stale[0]["actual"] == "ab" * 32


def test_matching_declaration_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A declaration matching the artifact hash produces no findings."""
    target = _write_evidence(tmp_path, "evidence.json", "docs/README.md", "ab" * 32)
    monkeypatch.setattr(helper, "_actual_hash", lambda _root, _art: "ab" * 32)
    found, skipped = helper._stale_declarations(tmp_path, target)
    assert found == []
    assert skipped == []


def test_ambiguous_double_pin_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same artifact pinned twice with different values is not rewritten."""
    target = tmp_path / "evidence.json"
    target.write_text(
        json.dumps(
            {
                "evidence": [
                    {"path": "docs/README.md", "sha256": STALE_HASH},
                    {"path": "docs/README.md", "sha256": "ff" * 32},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(helper, "_actual_hash", lambda _root, _art: "ab" * 32)
    _stale, skipped = helper._stale_declarations(tmp_path, target)
    assert len(skipped) == 1
    assert "ambiguous" in skipped[0]


def test_rewrite_round_trip_is_byte_exact(tmp_path: Path) -> None:
    """Rewriting one stale value preserves every other byte."""
    target = _write_evidence(tmp_path, "evidence.json", "docs/README.md", STALE_HASH)
    before = target.read_text(encoding="utf-8")
    refreshed = helper._rewrite_file(
        target,
        [
            {
                "artifact": "docs/README.md",
                "hash_key": "sha256",
                "declared": STALE_HASH,
                "actual": "ab" * 32,
            }
        ],
    )
    assert refreshed == 1
    assert target.read_text(encoding="utf-8") == before.replace(STALE_HASH, "ab" * 32)
    json.loads(target.read_text(encoding="utf-8"))


def test_main_refuses_baseline_files() -> None:
    """Baseline paths fail closed instead of being rewritten."""
    code = helper.main(["--write", "--path", "scripts/validation/evidence_registry_baseline.json"])
    assert code == 2


def test_check_clean_pinned_file_passes() -> None:
    """Check mode exits zero for the currently pinned assurance-case example."""
    code = helper.main(
        ["--path", "docs/context/evidence/issue_4683_release_assurance_case_example.json"]
    )
    assert code == 0


def test_real_artifact_hash_matches_helper_logic() -> None:
    """The helper hash function agrees with hashlib on a tracked file."""
    repo_root = ROOT
    expected = hashlib.sha256((repo_root / "README.md").read_bytes()).hexdigest()
    assert helper._actual_hash(repo_root, "README.md") == expected
