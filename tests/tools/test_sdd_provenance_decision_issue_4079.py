"""Documentation tests for SDD provenance decision (Issue #4079).

Issue #4079 investigates whether the Kaggle-reduced SDD annotation package is
byte-equivalent to the official Stanford SDD archive. These tests validate that
the documentation correctly records the provenance decision: the official archive
is unreachable, and the Kaggle source remains local BYO staging only.

They never touch the network and never download anything.
"""

from __future__ import annotations

from pathlib import Path

# Test file is at tests/tools/test_xxx.py, so repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sdd_staging_doc_records_kaggle_provenance_decision() -> None:
    """Validate that the SDD staging documentation records the Kaggle provenance decision.

    Issue #4079 documents that the official Stanford SDD archive is unreachable and
    the Kaggle-reduced source remains local BYO staging only. This test validates
    that the documentation explicitly states this decision so future contributors
    understand the provenance boundary.

    Regression test for: https://github.com/ll7/robot_sf_ll7/issues/4079
    """
    doc_path = REPO_ROOT / "docs" / "context" / "issue_2657_sdd_staging.md"
    assert doc_path.is_file(), "SDD staging documentation must exist"

    content = doc_path.read_text(encoding="utf-8")

    # The documentation must explicitly mention issue #4079
    assert "Issue #4079" in content, (
        "Documentation must cite issue #4079 for the provenance decision"
    )

    # Must record the official archive unavailability
    assert "blocked_official_source_unavailable" in content, (
        "Documentation must classify the official source as unavailable"
    )
    assert "http://vatic2.stanford.edu/stanford_campus_dataset.zip" in content, (
        "Documentation must record the official archive URL for future verification"
    )

    # Must record the Kaggle source and its classification
    assert "aryashah2k/stanford-drone-dataset" in content, (
        "Documentation must cite the Kaggle source"
    )
    assert "local_byo_staging_only" in content or "local BYO staging only" in content, (
        "Documentation must classify Kaggle source as local BYO only"
    )

    # Must include the Kaggle checksum for reference
    assert "66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf" in content, (
        "Documentation must record the Kaggle checksum (even if not canonical)"
    )

    # Must state that the Kaggle checksum is NOT promoted to canonical
    assert "NOT" in content and "canonical" in content, (
        "Documentation must explicitly state that Kaggle is not canonical-equivalent"
    )

    # Must include the benchmark claim caveat
    assert "provenance caveat" in content or "caveat" in content, (
        "Documentation must include a benchmark claim caveat"
    )


def test_sdd_staging_doc_excludes_kaggle_checksum_from_manifest() -> None:
    """Validate that the SDD manifest does NOT promote the Kaggle checksum.

    Issue #4079 explicitly rejects promoting the Kaggle checksum to canonical status
    because byte-equivalence cannot be verified. This test validates that the
    committed manifest does not carry the Kaggle checksum as `expected_tree_sha256`.

    Regression test for: https://github.com/ll7/robot_sf_ll7/issues/4079
    """
    manifest_path = REPO_ROOT / "configs" / "data" / "sdd_staging_manifest.yaml"
    assert manifest_path.is_file(), "SDD staging manifest must exist"

    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), "Manifest must be a mapping"

    checksums = manifest.get("checksums")
    assert isinstance(checksums, dict), "Manifest must contain a 'checksums' mapping"
    assert "expected_tree_sha256" in checksums, (
        "Manifest checksums must include expected_tree_sha256"
    )
    expected_tree_sha256 = checksums["expected_tree_sha256"]

    # The Kaggle checksum must NOT be pinned as the expected checksum
    kaggle_checksum = "66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf"
    assert expected_tree_sha256 != kaggle_checksum, (
        "The Kaggle checksum must not be promoted to canonical status. "
        "The manifest should either leave expected_tree_sha256 as null/placeholder "
        "or use a checksum verified against the official Stanford archive."
    )


def test_sdd_staging_doc_has_verification_date() -> None:
    """Validate that the documentation records when the official archive was verified unreachable.

    The provenance decision depends on when the official archive was last verified as
    unavailable. This test ensures the documentation includes a verification date so
    future contributors can judge whether a re-verification attempt is warranted.

    Regression test for: https://github.com/ll7/robot_sf_ll7/issues/4079
    """
    doc_path = REPO_ROOT / "docs" / "context" / "issue_2657_sdd_staging.md"
    content = doc_path.read_text(encoding="utf-8")

    # Must include a verification date (2026-07-02 or later)
    assert "2026-07-02" in content or "2026-" in content, (
        "Documentation must record the verification date for the official source unavailability"
    )

    # Must describe the network observation
    assert "TCP" in content or "refused" in content or "filtered" in content, (
        "Documentation must describe the network failure mode (TCP refused/filtered)"
    )
