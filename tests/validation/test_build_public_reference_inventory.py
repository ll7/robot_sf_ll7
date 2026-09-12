"""Tests for the declared public reference inventory (issue #8929)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.validation.build_public_reference_inventory import (
    DEFAULT_DECLARATION,
    build_inventory,
    main,
)
from scripts.validation.check_durable_artifact_locality import audit_locality

if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_DECLARATION_TEMPLATE = """\
schema_version: public_reference_inventory_sources.v1
sources:
  - id: fixture_release
    kind: release_checksum_manifest
    path: {manifest_path}
    consumer_path: {manifest_path}
    consumer_status: active
    retention_class: release_facing
"""

_MANIFEST_TEMPLATE = """\
schema_version: release-checksum-manifest.v1
release_id: benchmark_release_fixture
release_tag: "9.9.9"
artifact_set:
  bundle:
    directory: docs/context/evidence/fixture
    files:
      - path: docs/context/evidence/fixture/README.md
        sha256: {digest_a}
        description: Fixture artifact.
embedded_artifacts:
  checksums:
    path_in_archive: bundle/checksums.sha256
    sha256: {digest_b}
    description: Fixture embedded checksum list.
"""


def _write_fixture(tmp_path: Path, *, digest_a: str, digest_b: str) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        _MANIFEST_TEMPLATE.format(digest_a=digest_a, digest_b=digest_b), encoding="utf-8"
    )
    declaration = tmp_path / "declaration.yaml"
    declaration.write_text(
        _DECLARATION_TEMPLATE.format(manifest_path="manifest.yaml"), encoding="utf-8"
    )
    return declaration


def test_real_declaration_builds_clean_inventory() -> None:
    """The checked-in declaration resolves identities with no findings."""
    payload, findings = build_inventory(DEFAULT_DECLARATION, root=REPO_ROOT)

    assert findings == ()
    assert payload["schema"] == "public_reference_inventory.v1"
    assert payload["status"] == "ok"
    assert len(payload["references"]) >= 10
    for reference in payload["references"]:
        assert reference["digest"]
        assert "://" not in reference["consumer_path"]
        assert not reference["consumer_path"].startswith("/")
        assert reference["consumer_status"] in {"active", "inactive"}


def test_round_trip_feeds_locality_audit(tmp_path: Path) -> None:
    """A generated inventory round-trips through the #8907 locality audit."""
    declaration = _write_fixture(tmp_path, digest_a="a" * 64, digest_b="b" * 64)
    payload, findings = build_inventory(declaration, root=tmp_path)
    assert findings == ()
    assert len(payload["references"]) == 2

    artifacts = [
        {
            "artifact_id": reference["artifact_id"],
            "version": reference["version"],
            "digest": reference["digest"],
            "locators": [
                {
                    "locator_class": "public_release",
                    "verification": "verified",
                    "verified_at": "2026-09-12",
                    "failure_domain_id": "release-public",
                    "mutable_alias": False,
                },
                {
                    "locator_class": "cloud_durable",
                    "verification": "verified",
                    "verified_at": "2026-09-12",
                    "failure_domain_id": "cloud-region-b",
                    "mutable_alias": False,
                },
            ],
        }
        for reference in payload["references"]
    ]
    packet = {
        "schema": "durable_artifact_locator_projection.v1",
        "generated_at": "2026-09-12",
        "verification_max_age_days": 30,
        "minimum_release_copies": 2,
        "references": payload["references"],
        "artifacts": artifacts,
    }
    packet_path = tmp_path / "projection.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    report = audit_locality(packet_path)

    assert report.ok, report.to_dict()
    assert report.status == "pass"


def test_unreadable_source_fails_closed(tmp_path: Path) -> None:
    """A declared surface that cannot be read yields source_unreadable."""
    declaration = tmp_path / "declaration.yaml"
    declaration.write_text(
        _DECLARATION_TEMPLATE.format(manifest_path="missing.yaml"), encoding="utf-8"
    )

    payload, findings = build_inventory(declaration, root=tmp_path)

    assert [finding.code for finding in findings] == ["source_unreadable", "empty_inventory"]
    assert payload["status"] == "fail"


def test_invalid_digest_fails_closed(tmp_path: Path) -> None:
    """A non-64-hex digest is rejected with invalid_digest."""
    declaration = _write_fixture(tmp_path, digest_a="abc", digest_b="b" * 64)

    payload, findings = build_inventory(declaration, root=tmp_path)

    codes = [finding.code for finding in findings]
    assert "invalid_digest" in codes
    assert len(payload["references"]) == 1


def test_private_locator_path_is_rejected(tmp_path: Path) -> None:
    """Absolute or URL-like surface paths are refused."""
    declaration = tmp_path / "declaration.yaml"
    declaration.write_text(
        _DECLARATION_TEMPLATE.format(manifest_path="/absolute/manifest.yaml"), encoding="utf-8"
    )

    _payload, findings = build_inventory(declaration, root=tmp_path)

    assert any(finding.code == "private_locator_value" for finding in findings)


def test_main_check_exits_nonzero_on_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check exits 1 and emits deterministic JSON when identities are unresolved."""
    declaration = _write_fixture(tmp_path, digest_a="short", digest_b="b" * 64)
    output = tmp_path / "inventory.json"

    exit_code = main(
        [
            "--declaration",
            str(declaration),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--check",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["schema"] == "public_reference_inventory.v1"
    captured = capsys.readouterr()
    assert "invalid_digest" in captured.err
