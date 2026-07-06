"""Tests for the release artifact badging validator and functional smoke checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation.check_release_artifact_badging import validate_badging_claims
from scripts.validation.run_release_functional_badge_smoke import (
    execute_functional_smoke_checks,
    verify_bundle_checksums,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_validator_claimed_level_none() -> None:
    """Validator should pass for claimed_level none."""
    manifest = {
        "artifact_badging": {
            "claimed_level": "none",
        }
    }
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "passed"
    assert not blockers


def test_validator_claimed_level_invalid() -> None:
    """Validator fails closed for invalid claimed levels."""
    manifest = {
        "artifact_badging": {
            "claimed_level": "invalid-level",
        }
    }
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "failed"
    assert any("invalid" in b.lower() for b in blockers)


def test_validator_available_badge_requirements() -> None:
    """Verify available badge checks archive, checksum and durable id."""
    # Fails if missing checksum or durable identifier or archive
    manifest = {
        "artifact_badging": {
            "claimed_level": "available",
        }
    }
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "failed"
    assert len(blockers) >= 2

    # Passes with valid channels and files
    valid_manifest = {
        "artifact_badging": {
            "claimed_level": "available",
            "checklist_path": "docs/checklist.md",
        },
        "publication_channels": {
            "doi": "10.5281/zenodo.12345",
            "release_url": "https://github.com/ll7/robot_sf_ll7/releases/tag/v1.0.0",
        },
        "files": [{"path": "summary.json", "sha256": "abc123hash"}],
    }
    status, blockers, _ = validate_badging_claims(valid_manifest, None)
    assert status == "passed"
    assert not blockers


def test_validator_fails_on_raw_output_pointers() -> None:
    """Verify available badge fails if pointers are local raw output/ folders."""
    manifest_raw_pointer = {
        "artifact_badging": {
            "claimed_level": "available",
            "checklist_path": "docs/checklist.md",
        },
        "publication_channels": {
            "doi": "output/run_1234",  # local output pointer
        },
        "files": [{"path": "summary.json", "sha256": "abc"}],
    }
    status, blockers, _ = validate_badging_claims(manifest_raw_pointer, None)
    assert status == "failed"
    assert any("durable identifier" in b.lower() for b in blockers)


def test_validator_functional_badge_requirements() -> None:
    """Verify functional badge checks smoke command and passed status."""
    manifest = {
        "artifact_badging": {
            "claimed_level": "functional",
            "checklist_path": "docs/checklist.md",
            "functional_smoke_status": "failed",  # failed smoke
        },
        "publication_channels": {
            "doi": "10.5281/zenodo.12345",
        },
        "files": [{"path": "summary.json", "sha256": "abc"}],
    }
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "failed"
    assert any("functional smoke test" in b.lower() for b in blockers)

    # Passes if smoke test passed
    manifest["artifact_badging"]["functional_smoke_status"] = "passed"
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "passed"
    assert not blockers


def test_validator_reproduced_badge_requirements() -> None:
    """Verify reproduced badge checks tolerances, command, and status."""
    manifest = {
        "artifact_badging": {
            "claimed_level": "reproduced",
            "checklist_path": "docs/checklist.md",
            "functional_smoke_status": "passed",
            "reproduction_status": "failed",  # failed reproduction
            "claim_boundary": "reproduced-headline",
        },
        "publication_channels": {
            "doi": "10.5281/zenodo.12345",
        },
        "files": [{"path": "summary.json", "sha256": "abc"}],
    }
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "failed"
    assert any("reproduction run must have passed" in b.lower() for b in blockers)

    # Passes if reproduction passed
    manifest["artifact_badging"]["reproduction_status"] = "passed"
    status, blockers, _ = validate_badging_claims(manifest, None)
    assert status == "passed"
    assert not blockers


def test_functional_smoke_checks_on_synthetic_bundle(tmp_path: Path) -> None:
    """Verify checksums and smoke checks on a synthetic bundle fixture."""
    bundle_dir = tmp_path / "synthetic_bundle"
    payload_dir = bundle_dir / "payload"
    payload_dir.mkdir(parents=True)

    # Write files
    (payload_dir / "summary.json").write_text(json.dumps({"success_rate": 0.85}), encoding="utf-8")
    (payload_dir / "episodes.jsonl").write_text('{"episode_id": "ep1"}\n', encoding="utf-8")

    # Generate hashes
    import hashlib

    def get_sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    files = [
        {"path": "summary.json", "sha256": get_sha(payload_dir / "summary.json")},
        {"path": "episodes.jsonl", "sha256": get_sha(payload_dir / "episodes.jsonl")},
    ]

    manifest = {
        "schema_version": "publication-bundle.v1",
        "files": files,
    }
    (bundle_dir / "publication_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Verify checksums (should be ok)
    errors = verify_bundle_checksums(bundle_dir, files)
    assert not errors

    # Check checksum mismatch
    corrupted_files = [
        {"path": "summary.json", "sha256": "corruptedhashvalue12345"},
    ]
    errors_corrupted = verify_bundle_checksums(bundle_dir, corrupted_files)
    assert len(errors_corrupted) == 1
    assert "checksum mismatch" in errors_corrupted[0].lower()

    # Verify execute_functional_smoke_checks
    smoke_errors = execute_functional_smoke_checks(bundle_dir)
    assert not smoke_errors
