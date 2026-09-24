"""Digest-equivalence tests for the issue #8948 SHA-256 helper consolidation."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.adversarial import feasibility_first_real, held_out_preflight, provenance
from robot_sf.benchmark import artifact_catalog
from robot_sf.evidence.writers import sha256_file
from scripts.adversarial.run_proposal_vs_random_issue_2921 import _raw_file_sha256
from scripts.coverage.check_changed_files_coverage import _sha256_file as coverage_sha256_file

_FIXTURE_BYTES = b"issue-8948 sha256 fixture payload\n"

# Delegated helpers that now share the canonical ``robot_sf.evidence`` behavior.
_DELEGATED_HELPERS = (
    artifact_catalog.sha256_file,
    provenance.sha256_of_file,
    held_out_preflight.raw_sha256,
    feasibility_first_real._sha256_file,
    _raw_file_sha256,
)


@pytest.fixture()
def fixture_file(tmp_path: Path) -> Path:
    """Write one deterministic fixture file."""
    path = tmp_path / "fixture.bin"
    path.write_bytes(_FIXTURE_BYTES)
    return path


def _helper_id(helper: object) -> str:
    return f"{helper.__module__}.{helper.__name__}"  # type: ignore[attr-defined]


def test_canonical_digest_matches_hashlib(fixture_file: Path) -> None:
    expected = hashlib.sha256(_FIXTURE_BYTES).hexdigest()
    actual = sha256_file(fixture_file)
    assert actual == expected
    assert actual == actual.lower()
    assert len(actual) == 64


@pytest.mark.parametrize("helper", _DELEGATED_HELPERS, ids=_helper_id)
def test_delegated_helpers_return_identical_digest(helper, fixture_file: Path) -> None:
    assert helper(fixture_file) == sha256_file(fixture_file)


@pytest.mark.parametrize("helper", _DELEGATED_HELPERS, ids=_helper_id)
def test_delegated_helpers_fail_like_canonical_on_missing_path(helper, tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        sha256_file(missing)
    with pytest.raises(FileNotFoundError):
        helper(missing)


def test_delegated_helpers_reject_directories_like_canonical(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        sha256_file(tmp_path)
    for helper in _DELEGATED_HELPERS:
        with pytest.raises(IsADirectoryError):
            helper(tmp_path)


def test_sha256_of_file_accepts_str_and_path(fixture_file: Path) -> None:
    assert provenance.sha256_of_file(str(fixture_file)) == sha256_file(fixture_file)
    assert provenance.sha256_of_file(fixture_file) == sha256_file(fixture_file)


def test_coverage_gate_helper_keeps_fail_soft_contract(fixture_file: Path, tmp_path: Path) -> None:
    assert coverage_sha256_file(fixture_file) == sha256_file(fixture_file)
    assert coverage_sha256_file(tmp_path / "missing.bin") is None
    assert coverage_sha256_file(tmp_path) is None
