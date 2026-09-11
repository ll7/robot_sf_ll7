"""Focused contract tests for the package SHA-256 helper consolidation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.adversarial import feasibility_first_real, held_out_preflight, provenance
from robot_sf.benchmark import artifact_catalog
from robot_sf.evidence.writers import sha256_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_PACKAGE_HELPERS: tuple[tuple[str, Callable[[Path], str]], ...] = (
    ("artifact_catalog", artifact_catalog.sha256_file),
    ("provenance", provenance.sha256_of_file),
    ("held_out_preflight", held_out_preflight.raw_sha256),
    ("feasibility_first_real", feasibility_first_real._sha256_file),
)


@pytest.mark.parametrize(("name", "helper"), _PACKAGE_HELPERS)
def test_package_helpers_match_canonical_digest(
    name: str, helper: Callable[[Path], str], tmp_path: Path
) -> None:
    """All listed package helpers return the canonical digest for fixture bytes."""
    fixture = tmp_path / f"{name}.bin"
    fixture.write_bytes(b"sha256 helper consolidation fixture\x00\xff")

    assert helper(fixture) == sha256_file(fixture)


def test_provenance_helper_retains_string_path_support(tmp_path: Path) -> None:
    """The provenance helper continues to accept its documented string input."""
    fixture = tmp_path / "string-input.bin"
    fixture.write_bytes(b"string path")

    assert provenance.sha256_of_file(str(fixture)) == sha256_file(fixture)


@pytest.mark.parametrize(("name", "helper"), _PACKAGE_HELPERS)
def test_package_helpers_match_canonical_missing_path_failure(
    name: str, helper: Callable[[Path], str], tmp_path: Path
) -> None:
    """All listed package helpers preserve the canonical missing-path failure."""
    missing = tmp_path / f"missing-{name}.bin"

    with pytest.raises(FileNotFoundError) as canonical_error:
        sha256_file(missing)
    with pytest.raises(FileNotFoundError) as helper_error:
        helper(missing)

    assert type(helper_error.value) is type(canonical_error.value)
    assert helper_error.value.args == canonical_error.value.args
