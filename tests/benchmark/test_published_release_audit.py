"""Tests for the published-release audit (issue #7936)."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from robot_sf.benchmark import published_release_audit as published_audit_module
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.published_release_audit import (
    NETWORK_SCHEMA,
    SCHEMA,
    _extract_members,
    _verify_internal_checksums,
    audit_published,
    audit_published_network,
)
from robot_sf.benchmark.release_erratum import (
    ErratumContract,
    PredecessorEvidence,
    build_erratum_receipt,
    snapshot_campaign,
)

_CLI_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "published_release_audit.py"
)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_bundle(
    path: Path, *, member: str = "manifest.json", data: bytes = b"bundle-bytes"
) -> None:
    """Write a real zip bundle with one member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, data)


def _make_archive(path: Path, archive_kind: str, entries: list[tuple[str, bytes]]) -> None:
    """Write a small ZIP or TAR archive fixture for bounded-extraction tests."""
    if archive_kind == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)
        return
    with tarfile.open(path, "w") as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _make_zip64_archive(path: Path) -> tuple[Path, int, int]:
    """Write a small ZIP whose end records use the ZIP64 metadata structures."""
    _make_bundle(path)
    raw = path.read_bytes()
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    (
        _signature,
        _disk_number,
        _directory_disk,
        entries_on_disk,
        entries_total,
        central_directory_size,
        central_directory_offset,
        _comment_size,
    ) = struct.unpack_from("<4s4H2LH", raw, eocd_offset)
    zip64_offset = eocd_offset
    zip64_record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entries_on_disk,
        entries_total,
        central_directory_size,
        central_directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_offset, 1)
    zip_eocd = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    path.write_bytes(raw[:eocd_offset] + zip64_record + locator + zip_eocd)
    return path, zip64_offset, zip64_offset + len(zip64_record)


def _make_erratum_assets(
    github: Path,
    zenodo: Path,
    *,
    source_sha: str,
    tag: str,
    doi: str,
    payload_files: dict[str, bytes],
    root_extra: dict[str, bytes] | None = None,
    manifest_overrides: dict[str, object] | None = None,
) -> Path:
    """Write matching archive/manifest/checksum/custody assets for an erratum fixture."""
    receipt_data = payload_files.get("provenance/benchmark_release_erratum.json", b"{}")
    parsed_receipt = json.loads(receipt_data)
    receipt = parsed_receipt if isinstance(parsed_receipt, dict) else {}
    root_name = "bundle"
    entries = []
    checksum_lines = []
    for relative, data in sorted(payload_files.items()):
        digest = hashlib.sha256(data).hexdigest()
        entries.append(
            {
                "path": relative,
                "size_bytes": len(data),
                "sha256": digest,
                "kind": "provenance",
            }
        )
        checksum_lines.append(f"{digest}  payload/{relative}\n")
    checksums = "".join(checksum_lines).encode()
    manifest_payload: dict[str, object] = {
        "schema_version": "benchmark-publication-bundle.v2",
        "bundle_name": root_name,
        "publication_channels": {
            "repository_url": "https://github.com/ll7/robot_sf_ll7",
            "release_tag": tag,
            "release_url": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}",
            "doi": doi,
        },
        "provenance": {"repository": {"commit": source_sha}},
        "totals": {
            "file_count": len(entries),
            "total_bytes": sum(len(data) for data in payload_files.values()),
        },
        "files": entries,
    }
    manifest_payload.update(manifest_overrides or {})
    manifest = json.dumps(manifest_payload, sort_keys=True).encode()
    readme = b"# Erratum fixture\n"
    bundle = github / "bundle.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w") as archive:
        for relative, data in sorted(payload_files.items()):
            archive.writestr(f"{root_name}/payload/{relative}", data)
        archive.writestr(f"{root_name}/checksums.sha256", checksums)
        archive.writestr(f"{root_name}/publication_manifest.json", manifest)
        archive.writestr(f"{root_name}/README.md", readme)
        for relative, data in sorted((root_extra or {}).items()):
            archive.writestr(f"{root_name}/{relative}", data)

    custody = {
        "schema_version": "benchmark-publication-custody.v1",
        "source_execution_commit": source_sha,
        "archive": {
            "path": bundle.name,
            "sha256": sha256_file(bundle),
            "size_bytes": bundle.stat().st_size,
        },
        "bundle": {
            "path": root_name,
            "publication_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "checksums_sha256": hashlib.sha256(checksums).hexdigest(),
        },
        "archive_self_digest_policy": "archive digest is external to the bundle; no cycle",
        "credentials": "not_recorded",
        "erratum": {
            "correction_id": receipt.get("correction_id"),
            "correction_scope": receipt.get("correction_scope"),
            "supersedes": receipt.get("supersedes"),
            "successor": receipt.get("successor"),
            "scientific_equality": receipt.get("scientific_equality"),
            "embedded_receipt_path": (
                f"{root_name}/payload/provenance/benchmark_release_erratum.json"
            ),
            "embedded_receipt_sha256": hashlib.sha256(
                payload_files.get("provenance/benchmark_release_erratum.json", b"")
            ).hexdigest(),
        },
    }
    sidecars = {
        bundle.name: bundle.read_bytes(),
        "publication_manifest.json": manifest,
        "checksums.sha256": checksums,
        "publication_custody.json": json.dumps(custody, sort_keys=True).encode(),
    }
    for channel in (github, zenodo):
        for name, data in sidecars.items():
            _write_bytes(channel / name, data)
    return bundle


def _full_erratum_payload(
    tmp_path: Path,
) -> tuple[dict[str, bytes], dict[str, object], str, str]:
    """Build a one-cell payload that passes the real preflight and cold validator.

    Returns:
        Payload bytes, correction receipt, successor tag, and successor DOI.
    """
    source_sha = "5" * 40
    builder_sha = "a" * 40
    orchestration_sha = "b" * 40
    predecessor_doi = "10.5281/zenodo.7"
    concept_doi = "10.5281/zenodo.6"
    successor_doi = "10.5281/zenodo.8"
    predecessor_tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}"
    successor_tag = f"{predecessor_tag}-erratum.1"
    campaign = tmp_path / "cold-campaign"
    episodes = campaign / "runs/orca__differential_drive/episodes.jsonl"
    row = {
        "algo": "orca",
        "scenario_id": "crossing",
        "seed": 111,
        "episode_id": "crossing--111--fixture",
        "status": "success",
        "outcome": "goal_reached",
        "git_hash": source_sha,
        "provenance": {"git_hash": source_sha},
        "result_provenance": {"repo_commit": source_sha},
        "event_ledger": {
            "software_commit": source_sha,
            "exact_events": {"goal_reached": True, "timeout": False},
        },
        "metrics": {"collisions": 0, "snqi": 0.5},
    }
    _write_bytes(episodes, (json.dumps(row, sort_keys=True) + "\n").encode())
    metadata_path = campaign / "release/zenodo_metadata.erratum.json"
    metadata = {
        "metadata": {
            "title": "Robot SF benchmark erratum",
            "upload_type": "dataset",
            "access_right": "open",
            "license": "GPL-3.0-only",
            "description": (
                "Derived metadata erratum. All scientific rows are unchanged and no simulation "
                "rerun occurred. SNQI remains advisory and supports no planner ranking claim."
            ),
            "creators": [{"name": "Luttkus, Lennart"}],
            "related_identifiers": [
                {
                    "identifier": (
                        "https://github.com/ll7/robot_sf_ll7/releases/tag/" + successor_tag
                    ),
                    "relation": "isSupplementTo",
                    "scheme": "url",
                },
                {
                    "identifier": predecessor_doi,
                    "relation": "isNewVersionOf",
                    "scheme": "doi",
                },
            ],
        }
    }
    _write_bytes(metadata_path, json.dumps(metadata, sort_keys=True).encode())
    predecessor = tmp_path / "immutable-predecessor.tar.gz"
    with tarfile.open(predecessor, "w:gz") as archive:
        archive.add(
            episodes,
            arcname="fixture_bundle/payload/runs/orca__differential_drive/episodes.jsonl",
        )
    contract = ErratumContract(
        correction_id="fixture-derived-metadata-erratum.1",
        predecessor_version_doi=predecessor_doi,
        predecessor_archive_sha256=sha256_file(predecessor),
        predecessor_archive_size_bytes=predecessor.stat().st_size,
        predecessor_github_release_tag=predecessor_tag,
        source_sha=source_sha,
        planner_arms=1,
        scenario_count=1,
        seed_count=1,
        episode_rows=1,
        builder_sha=builder_sha,
        validator_sha=builder_sha,
        orchestration_sha=orchestration_sha,
        concept_doi=concept_doi,
        successor_version_doi=successor_doi,
        successor_github_release_tag=successor_tag,
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
    )
    erratum_block = {
        "correction_id": contract.correction_id,
        "correction_scope": "derived_publication_metadata_only",
        "predecessor_version_doi": predecessor_doi,
        "predecessor_github_release_tag": predecessor_tag,
        "concept_doi": concept_doi,
        "source_sha": source_sha,
        "scientific_source_unchanged": True,
        "simulation_rerun": False,
    }
    publication = {
        "release_tag": successor_tag,
        "release_id": successor_tag,
        "doi": successor_doi,
        "concept_doi": concept_doi,
        "version_doi": successor_doi,
        "predecessor_version_doi": predecessor_doi,
        "bundle_metadata_path": "release/zenodo_metadata.erratum.json",
        "metadata_sha256": contract.metadata_sha256,
        "correction_scope": "derived_publication_metadata_only",
    }
    provenance = {
        "release_tag": successor_tag,
        "release_id": successor_tag,
        "doi": successor_doi,
        "version_doi": successor_doi,
        "concept_doi": concept_doi,
        "scientific_source_sha": source_sha,
        "erratum_builder_sha": builder_sha,
        "erratum_validator_sha": builder_sha,
        "erratum_orchestration_sha": orchestration_sha,
    }
    current = {
        "release_tag": successor_tag,
        "release_id": successor_tag,
        "doi": successor_doi,
        "version_doi": successor_doi,
        "concept_doi": concept_doi,
        "publication": publication,
        "provenance": provenance,
        "erratum": erratum_block,
    }
    execution = {
        "release_tag": predecessor_tag,
        "release_id": predecessor_tag,
        "doi": predecessor_doi,
        "version_doi": predecessor_doi,
        "concept_doi": concept_doi,
    }
    _write_bytes(
        campaign / "release/release_manifest.resolved.json",
        json.dumps(current, sort_keys=True).encode(),
    )
    release_result = {
        **current,
        "benchmark_release": current,
        "resolved_manifest": current,
        "scientific_execution_benchmark_release": execution,
        "scientific_execution_resolved_manifest": execution,
        "derivation": {
            "builder_sha": builder_sha,
            "validator_sha": builder_sha,
            "orchestration_sha": orchestration_sha,
            "scientific_source_sha": source_sha,
            "simulation_rerun": False,
            "correction_id": contract.correction_id,
            "predecessor_version_doi": predecessor_doi,
        },
        "status": "accepted",
        "evidence_status": "valid",
        "total_episodes": 1,
        "successful_runs": 1,
        "publication_preflight_status": "pass",
        "publication_preflight_violations": [],
        "release_status": "ok",
        "ranking_claims_admitted": False,
    }
    _write_bytes(
        campaign / "release/release_result.json",
        json.dumps(release_result, sort_keys=True).encode(),
    )
    summary_campaign = {
        **current,
        "status": "accepted",
        "evidence_status": "valid",
        "total_episodes": 1,
        "successful_runs": 1,
        "scientific_execution_release_identity": {
            "release_tag": predecessor_tag,
            "doi": predecessor_doi,
            "source_sha": source_sha,
        },
    }
    _write_bytes(
        campaign / "reports/campaign_summary.json",
        json.dumps(
            {
                "benchmark_release": current,
                "campaign": summary_campaign,
                "publication_erratum": erratum_block,
            },
            sort_keys=True,
        ).encode(),
    )
    _write_bytes(
        campaign / "provenance/derived_revalidation_receipt.json",
        json.dumps(
            {
                "schema_version": "benchmark-derived-revalidation.v1",
                "source": {"execution_commit": source_sha},
                "validator": {"commit": builder_sha},
            },
            sort_keys=True,
        ).encode(),
    )
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=snapshot,
        successor=snapshot,
    )
    _write_bytes(
        campaign / "provenance/benchmark_release_erratum.json",
        json.dumps(receipt, sort_keys=True).encode(),
    )
    payload_files = {
        path.relative_to(campaign).as_posix(): path.read_bytes()
        for path in campaign.rglob("*")
        if path.is_file()
    }
    return payload_files, receipt, successor_tag, successor_doi


def _predecessor_evidence(tmp_path: Path, receipt: dict[str, object]) -> PredecessorEvidence:
    """Build detached evidence for the complete erratum fixture."""
    supersedes = receipt["supersedes"]
    assert isinstance(supersedes, dict)
    archive = tmp_path / "immutable-predecessor.tar.gz"
    return PredecessorEvidence(
        archive_path=archive,
        version_doi=str(supersedes["version_doi"]),
        concept_doi=str(receipt["successor"]["concept_doi"]),
        github_release_tag=str(supersedes["github_release_tag"]),
        archive_sha256=str(supersedes["archive_sha256"]),
        archive_size_bytes=int(supersedes["archive_size_bytes"]),
    )


def test_erratum_audit_runs_real_preflight_and_cold_validation(tmp_path: Path) -> None:
    """The production validators authenticate one complete cold-start archive."""
    payload_files, correction_receipt, tag, doi = _full_erratum_payload(tmp_path)
    predecessor_evidence = _predecessor_evidence(tmp_path, correction_receipt)
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
        predecessor_evidence=predecessor_evidence,
    )

    assert result["status"] == "pass"
    assert result["problems"] == []
    assert result["observations"]["erratum_bundle_inventory"]["preflight_status"] == "pass"
    assert result["observations"]["erratum"]["episode_rows"] == 1
    assert result["observations"]["erratum_custody"]["status"] == "pass"
    assert correction_receipt["scientific_equality"]["status"] == "identical"


def test_erratum_audit_requires_exact_tag_target_before_bundle_validation(tmp_path: Path) -> None:
    """A canonical erratum cannot be audited without its immutable source target."""
    payload_files, _receipt, tag, doi = _full_erratum_payload(tmp_path)
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha=None,
    )

    assert result["status"] == "fail"
    assert any("exact GitHub tag target SHA" in problem for problem in result["problems"])


def test_erratum_audit_requires_detached_predecessor_evidence(tmp_path: Path) -> None:
    """A complete successor bundle cannot self-authenticate its predecessor."""
    payload_files, _receipt, tag, doi = _full_erratum_payload(tmp_path)
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
    )

    assert result["status"] == "fail"
    assert any("predecessor evidence is required" in problem for problem in result["problems"])


def test_erratum_audit_rejects_manifest_file_size_tampering(tmp_path: Path) -> None:
    """A manifest entry cannot authenticate bytes with a false declared size."""
    payload_files, _receipt, tag, doi = _full_erratum_payload(tmp_path)
    entries = [
        {
            "path": relative,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "kind": "provenance",
        }
        for relative, data in sorted(payload_files.items())
    ]
    entries[0]["size_bytes"] = int(entries[0]["size_bytes"]) + 1
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
        manifest_overrides={"files": entries},
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
    )

    assert result["status"] == "fail"
    assert any(
        "size_bytes disagrees with payload bytes" in problem for problem in result["problems"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("release_tag", "old-release", "release tag is not bound"),
        (
            "release_url",
            "https://github.com/ll7/robot_sf_ll7/releases/tag/old-release",
            "release_url is not bound",
        ),
    ],
)
def test_erratum_audit_rejects_tampered_publication_channel(
    tmp_path: Path, field: str, value: str, expected: str
) -> None:
    """Publication-channel coordinates must match the requested tag and DOI."""
    payload_files, _receipt, tag, doi = _full_erratum_payload(tmp_path)
    channels = {
        "repository_url": "https://github.com/ll7/robot_sf_ll7",
        "release_tag": tag,
        "release_url": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}",
        "doi": doi,
    }
    channels[field] = value
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
        manifest_overrides={"publication_channels": channels},
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
    )

    assert result["status"] == "fail"
    assert any(expected in problem for problem in result["problems"])


@pytest.mark.parametrize("tamper", ["root_tag", "nested_doi"])
def test_erratum_audit_rejects_stale_optional_publication_document(
    tmp_path: Path, tamper: str
) -> None:
    """Copied optional metadata cannot retain predecessor tag or DOI aliases."""
    payload_files, receipt, tag, doi = _full_erratum_payload(tmp_path)
    predecessor_evidence = _predecessor_evidence(tmp_path, receipt)
    predecessor = receipt["supersedes"]
    predecessor_tag = str(predecessor["github_release_tag"])
    predecessor_doi = str(predecessor["version_doi"])
    document = json.loads(payload_files["release/release_manifest.resolved.json"])
    if tamper == "root_tag":
        document["release_tag"] = predecessor_tag
    else:
        document["publication"]["version_doi"] = predecessor_doi
    payload_files["manifest.json"] = json.dumps(document, sort_keys=True).encode()

    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi=doi,
        payload_files=payload_files,
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
        predecessor_evidence=predecessor_evidence,
    )

    assert result["status"] == "fail"
    expected = "stale release-tag alias" if tamper == "root_tag" else "stale version-DOI alias"
    assert any(expected in problem for problem in result["problems"])


def test_cross_channel_byte_identity_passes(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
        _write_bytes(channel / "checksums.sha256", b"checksum-bytes")
    receipt = audit_published(
        tag="paper-matrix-v2-h600-s30",
        doi="10.5281/zenodo.1234567",
        github_dir=github,
        zenodo_dir=zenodo,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["ok"] is True
    assert receipt["status"] == "pass"
    assert receipt["observations"]["common_asset_names"] == ["bundle.zip", "checksums.sha256"]
    assert receipt["problems"] == []


def test_erratum_audit_requires_and_routes_embedded_correction_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Erratum tags trigger cold scientific-receipt validation from the bundle payload."""
    source_sha = "5" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    doi = "10.5281/zenodo.8"
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    correction_receipt = {
        "schema_version": "benchmark-release-erratum-receipt.v1",
        "correction_id": "fixture-erratum.1",
        "correction_scope": "derived_publication_metadata_only",
        "supersedes": {
            "version_doi": "10.5281/zenodo.7",
            "archive_sha256": "0" * 64,
            "archive_size_bytes": 1,
            "github_release_tag": f"paper-matrix-v2-h600-s30-2026-09-{source_sha}",
        },
        "successor": {"version_doi": doi, "github_release_tag": tag},
        "predecessor_version_doi": "10.5281/zenodo.7",
        "concept_doi": "10.5281/zenodo.6",
        "scientific_equality": {"status": "identical"},
    }
    receipt_bytes = json.dumps(correction_receipt, sort_keys=True).encode()
    metadata_bytes = b"{}"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha=source_sha,
        tag=tag,
        doi=doi,
        payload_files={
            "provenance/benchmark_release_erratum.json": receipt_bytes,
            "release/zenodo_metadata.erratum.json": metadata_bytes,
        },
    )
    predecessor_archive = tmp_path / "predecessor.tar.gz"
    predecessor_archive.write_bytes(b"fixture")
    predecessor_evidence = PredecessorEvidence(
        archive_path=predecessor_archive,
        version_doi="10.5281/zenodo.7",
        concept_doi="10.5281/zenodo.6",
        github_release_tag=f"paper-matrix-v2-h600-s30-2026-09-{source_sha}",
        archive_sha256="0" * 64,
        archive_size_bytes=1,
    )
    calls: list[tuple[Path, Path, Path, str, str, str, str | None]] = []
    evidence_seen: list[PredecessorEvidence | None] = []

    def fake_validate(
        receipt_path: Path,
        *,
        campaign_root: Path,
        metadata_path: Path,
        archive_name: str,
        expected_tag: str,
        expected_doi: str,
        expected_source_sha: str | None,
        predecessor_evidence: PredecessorEvidence | None,
    ) -> dict[str, object]:
        evidence_seen.append(predecessor_evidence)
        calls.append(
            (
                receipt_path,
                campaign_root,
                metadata_path,
                archive_name,
                expected_tag,
                expected_doi,
                expected_source_sha,
            )
        )
        return {"status": "pass", "episode_rows": 20_160}

    monkeypatch.setattr(
        published_audit_module, "validate_erratum_receipt_against_campaign", fake_validate
    )
    monkeypatch.setattr(
        published_audit_module,
        "verify_publication_bundle_preflight",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    receipt = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha=source_sha,
        predecessor_evidence=predecessor_evidence,
    )

    assert receipt["status"] == "pass"
    assert receipt["observations"]["erratum"]["episode_rows"] == 20_160
    assert calls[0][1].name == "payload"
    assert calls[0][2].name == "zenodo_metadata.erratum.json"
    assert calls[0][3:] == ("bundle.zip", tag, doi, source_sha)
    assert evidence_seen == [predecessor_evidence]


def test_erratum_audit_rejects_bundle_without_correction_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching two-channel archive is insufficient for an erratum without its proof."""
    tag = f"paper-matrix-v2-h600-s30-2026-09-{'5' * 40}-erratum.1"
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha="5" * 40,
        tag=tag,
        doi="10.5281/zenodo.8",
        payload_files={"release/zenodo_metadata.erratum.json": b"{}"},
    )
    monkeypatch.setattr(
        published_audit_module,
        "verify_publication_bundle_preflight",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    receipt = audit_published(
        tag=tag,
        doi="10.5281/zenodo.8",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="5" * 40,
    )

    assert receipt["status"] == "fail"
    assert any("canonical receipt or metadata" in problem for problem in receipt["problems"])


@pytest.mark.parametrize("suffix", ["-erratum.01", "-Erratum.1"])
def test_erratum_audit_rejects_malformed_suffix(tmp_path: Path, suffix: str) -> None:
    source_sha = "5" * 40
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")

    result = audit_published(
        tag=f"paper-matrix-v2-h600-s30-2026-09-{source_sha}{suffix}",
        doi="10.5281/zenodo.8",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha=source_sha,
    )

    assert result["status"] == "fail"
    assert any("erratum tag is malformed" in problem for problem in result["problems"])


def test_erratum_audit_rejects_decoy_correction_receipt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second lookalike receipt cannot broaden the canonical proof boundary."""
    source_sha = "5" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    correction_receipt = {
        "correction_id": "fixture-erratum.1",
        "correction_scope": "derived_publication_metadata_only",
        "supersedes": {},
        "successor": {},
        "scientific_equality": {},
    }
    receipt = json.dumps(correction_receipt).encode()
    metadata = b"{}"
    _make_erratum_assets(
        github,
        zenodo,
        source_sha=source_sha,
        tag=tag,
        doi="10.5281/zenodo.8",
        payload_files={
            "provenance/benchmark_release_erratum.json": receipt,
            "release/zenodo_metadata.erratum.json": metadata,
        },
        root_extra={"decoy/benchmark_release_erratum.json": receipt},
    )
    monkeypatch.setattr(
        published_audit_module,
        "verify_publication_bundle_preflight",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    result = audit_published(
        tag=tag,
        doi="10.5281/zenodo.8",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha=source_sha,
    )

    assert result["status"] == "fail"
    assert any("unlisted or missing bundle member" in problem for problem in result["problems"])


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("manifest_schema", "manifest schema"),
        ("unsigned_payload", "payload/checksum inventory"),
        ("external_manifest", "differs from the archived sidecar"),
        ("stale_custody", "custody archive identity is stale"),
    ],
)
def test_erratum_audit_rejects_incomplete_bundle_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    expected: str,
) -> None:
    """Manifest, payload inventory, public sidecars, and custody all fail closed."""
    source_sha = "5" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    doi = "10.5281/zenodo.8"
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    correction_receipt = {
        "correction_id": "fixture-erratum.1",
        "correction_scope": "derived_publication_metadata_only",
        "supersedes": {},
        "successor": {},
        "scientific_equality": {},
    }
    receipt_bytes = json.dumps(correction_receipt).encode()
    _make_erratum_assets(
        github,
        zenodo,
        source_sha=source_sha,
        tag=tag,
        doi=doi,
        payload_files={
            "provenance/benchmark_release_erratum.json": receipt_bytes,
            "release/zenodo_metadata.erratum.json": b"{}",
        },
        root_extra=({"payload/unlisted.json": b"{}"} if tamper == "unsigned_payload" else None),
        manifest_overrides=(
            {"schema_version": "unsupported"} if tamper == "manifest_schema" else None
        ),
    )
    if tamper == "external_manifest":
        for channel in (github, zenodo):
            (channel / "publication_manifest.json").write_bytes(b"{}")
    if tamper == "stale_custody":
        for channel in (github, zenodo):
            path = channel / "publication_custody.json"
            custody = json.loads(path.read_text(encoding="utf-8"))
            custody["archive"]["sha256"] = "0" * 64
            path.write_text(json.dumps(custody), encoding="utf-8")

    monkeypatch.setattr(
        published_audit_module,
        "verify_publication_bundle_preflight",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        published_audit_module,
        "validate_erratum_receipt_against_campaign",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    result = audit_published(
        tag=tag,
        doi=doi,
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha=source_sha,
    )

    assert result["status"] == "fail"
    assert any(expected in problem for problem in result["problems"])


def test_cross_channel_mismatch_fails(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    _make_bundle(github / "bundle.zip", data=b"same")
    _make_bundle(zenodo / "bundle.zip", data=b"different")
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is False
    assert any("cross-channel byte mismatch" in problem for problem in receipt["problems"])


def test_missing_channel_reports_unavailable(tmp_path: Path) -> None:
    github = tmp_path / "github"
    github.mkdir()
    receipt = audit_published(
        tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=tmp_path / "empty"
    )
    assert receipt["ok"] is False
    assert any("Zenodo channel has no assets" in problem for problem in receipt["problems"])


def test_channel_symlink_asset_fails_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.zip"
    _make_bundle(outside)
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    github.mkdir()
    (github / "bundle.zip").symlink_to(outside)
    _write_bytes(zenodo / "bundle.zip", outside.read_bytes())

    receipt = audit_published(
        tag="release", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo
    )

    assert receipt["ok"] is False
    assert any("must not be a symlink" in problem for problem in receipt["problems"])


def test_doi_validation(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    receipt = audit_published(tag="t", doi="", github_dir=github, zenodo_dir=zenodo)
    assert any("version DOI is missing" in problem for problem in receipt["problems"])
    receipt2 = audit_published(tag="t", doi="not-a-doi", github_dir=github, zenodo_dir=zenodo)
    assert any("version DOI is malformed" in problem for problem in receipt2["problems"])


def test_bundle_extraction_and_internal_checksums(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    github.mkdir(parents=True)
    zenodo.mkdir(parents=True)
    bundle_path = github / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr("manifest.json", b"member-data")
    checksum_line = f"{sha256_file(bundle_path)} bundle.zip\n"
    # A sidecar checksum file inside the bundle:
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("checksums.sha256", checksum_line)
    _write_bytes(zenodo / "bundle.zip", bundle_path.read_bytes())
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is True
    assert receipt["observations"]["bundle"] == "bundle.zip"
    assert receipt["observations"]["bundle_member_count"] == 2


def test_zip_central_directory_size_is_preflighted_before_zipfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A huge declared central directory is rejected before stdlib parsing."""
    archive_path = tmp_path / "oversized-central-directory.zip"
    _make_bundle(archive_path)
    raw = bytearray(archive_path.read_bytes())
    eocd_offset = raw.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    struct.pack_into(
        "<L",
        raw,
        eocd_offset + 12,
        published_audit_module.DEFAULT_MAX_ZIP_CENTRAL_DIRECTORY_BYTES + 1,
    )
    archive_path.write_bytes(raw)

    zipfile_calls: list[tuple[object, ...]] = []

    def unexpected_zipfile(*args: object, **kwargs: object) -> object:
        zipfile_calls.append(args)
        raise AssertionError("ZipFile must not parse oversized central-directory metadata")

    monkeypatch.setattr(published_audit_module.zipfile, "ZipFile", unexpected_zipfile)
    with pytest.raises(ValueError, match="central directory exceeds limit"):
        _extract_members(archive_path, tmp_path / "oversized-central-directory-dest")
    assert zipfile_calls == []


def test_zip64_entry_count_is_preflighted_before_zipfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZIP64 entry counts cannot force an unbounded ``infolist`` allocation."""
    archive_path, zip64_offset, _locator_offset = _make_zip64_archive(
        tmp_path / "oversized-entry-count.zip"
    )
    raw = bytearray(archive_path.read_bytes())
    enormous_count = 1 << 63
    struct.pack_into("<Q", raw, zip64_offset + 24, enormous_count)
    struct.pack_into("<Q", raw, zip64_offset + 32, enormous_count)
    archive_path.write_bytes(raw)

    zipfile_calls: list[tuple[object, ...]] = []

    def unexpected_zipfile(*args: object, **kwargs: object) -> object:
        zipfile_calls.append(args)
        raise AssertionError("ZipFile must not parse oversized ZIP64 entry metadata")

    monkeypatch.setattr(published_audit_module.zipfile, "ZipFile", unexpected_zipfile)
    with pytest.raises(ValueError, match="member count exceeds limit"):
        _extract_members(archive_path, tmp_path / "oversized-entry-count-dest")
    assert zipfile_calls == []


def test_zip64_central_directory_size_is_preflighted_before_zipfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ZIP64 central-directory sizes are bounded before stdlib parsing."""
    archive_path, zip64_offset, _locator_offset = _make_zip64_archive(
        tmp_path / "oversized-zip64-central-directory.zip"
    )
    raw = bytearray(archive_path.read_bytes())
    struct.pack_into(
        "<Q",
        raw,
        zip64_offset + 40,
        published_audit_module.DEFAULT_MAX_ZIP_CENTRAL_DIRECTORY_BYTES + 1,
    )
    archive_path.write_bytes(raw)

    zipfile_calls: list[tuple[object, ...]] = []

    def unexpected_zipfile(*args: object, **kwargs: object) -> object:
        zipfile_calls.append(args)
        raise AssertionError("ZipFile must not parse oversized ZIP64 central-directory metadata")

    monkeypatch.setattr(published_audit_module.zipfile, "is_zipfile", lambda _path: True)
    monkeypatch.setattr(published_audit_module.zipfile, "ZipFile", unexpected_zipfile)
    with pytest.raises(ValueError, match="central directory exceeds limit"):
        _extract_members(archive_path, tmp_path / "oversized-zip64-central-directory-dest")
    assert zipfile_calls == []


def test_valid_zip64_archive_extracts_with_existing_streaming_limits(tmp_path: Path) -> None:
    """A structurally valid ZIP64 archive still uses the streaming extractor."""
    archive_path, _zip64_offset, _locator_offset = _make_zip64_archive(tmp_path / "valid-zip64.zip")

    members = _extract_members(archive_path, tmp_path / "valid-zip64-dest")

    assert members == ["manifest.json"]
    assert (tmp_path / "valid-zip64-dest" / "manifest.json").read_bytes() == b"bundle-bytes"


def test_prefixed_zip_keeps_supported_central_directory_offsets(tmp_path: Path) -> None:
    """A valid self-extracting-style prefix remains compatible with ZIP parsing."""
    archive_path = tmp_path / "prefixed.zip"
    _make_bundle(archive_path)
    prefixed_path = tmp_path / "prefixed-with-stub.zip"
    prefixed_path.write_bytes(b"self-extracting-stub\n" + archive_path.read_bytes())

    members = _extract_members(prefixed_path, tmp_path / "prefixed-dest")

    assert members == ["manifest.json"]
    assert (tmp_path / "prefixed-dest" / "manifest.json").read_bytes() == b"bundle-bytes"


def test_prefixed_zip64_keeps_supported_central_directory_offsets(tmp_path: Path) -> None:
    """A ZIP64 archive with a self-extracting-style prefix remains readable."""
    archive_path, _zip64_offset, _locator_offset = _make_zip64_archive(
        tmp_path / "prefixed-zip64.zip"
    )
    prefixed_path = tmp_path / "prefixed-zip64-with-stub.zip"
    prefixed_path.write_bytes(b"self-extracting-stub\n" + archive_path.read_bytes())

    members = _extract_members(prefixed_path, tmp_path / "prefixed-zip64-dest")

    assert members == ["manifest.json"]
    assert (tmp_path / "prefixed-zip64-dest" / "manifest.json").read_bytes() == b"bundle-bytes"


def test_zip_preflight_rejects_truncated_eocd_comment(tmp_path: Path) -> None:
    """A full EOCD that declares bytes past EOF is not passed to ``ZipFile``."""
    archive_path = tmp_path / "truncated-eocd.zip"
    archive_path.write_bytes(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 1))

    with pytest.raises(ValueError, match="end-of-central-directory record"):
        _extract_members(archive_path, tmp_path / "truncated-eocd-dest")


@pytest.mark.parametrize("case", ["missing-locator", "gap", "bad-offset", "oversized-record"])
def test_zip64_preflight_rejects_malformed_metadata(
    tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed ZIP64 locators/records fail before central-directory parsing."""
    archive_path, zip64_offset, locator_offset = _make_zip64_archive(
        tmp_path / f"malformed-{case}.zip"
    )
    raw = bytearray(archive_path.read_bytes())
    if case == "missing-locator":
        raw[locator_offset : locator_offset + 4] = b"NOPE"
    elif case == "gap":
        raw[locator_offset:locator_offset] = b"X"
    elif case == "bad-offset":
        struct.pack_into("<Q", raw, locator_offset + 8, (1 << 64) - 1)
    else:
        struct.pack_into(
            "<Q",
            raw,
            zip64_offset + 4,
            published_audit_module._MAX_ZIP64_EOCD_RECORD_BYTES + 1,
        )
    archive_path.write_bytes(raw)

    monkeypatch.setattr(published_audit_module.zipfile, "is_zipfile", lambda _path: True)
    with pytest.raises(ValueError, match="zip64 end-of-central-directory"):
        _extract_members(archive_path, tmp_path / f"malformed-{case}-dest")


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(ValueError, match="path escape"):
        _extract_members(evil, tmp_path / "dest")


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_archive_member_count_limit_removes_partial_output(
    tmp_path: Path, archive_kind: str
) -> None:
    """Extraction rejects oversized member inventories before committing output."""
    archive_path = tmp_path / f"member-count.{archive_kind}"
    _make_archive(archive_path, archive_kind, [(f"member-{index}.txt", b"x") for index in range(3)])
    destination = tmp_path / "member-count-dest"

    with pytest.raises(ValueError, match="member count exceeds limit"):
        _extract_members(archive_path, destination, max_members=2)

    assert not destination.exists()
    assert not list(tmp_path.glob(".member-count-dest-*"))


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_archive_per_file_expanded_limit_removes_partial_output(
    tmp_path: Path, archive_kind: str
) -> None:
    """Extraction bounds each member's expanded bytes before writing a bomb."""
    archive_path = tmp_path / f"per-file.{archive_kind}"
    _make_archive(archive_path, archive_kind, [("large.txt", b"1234")])
    destination = tmp_path / "per-file-dest"

    with pytest.raises(ValueError, match="per-file expanded byte limit"):
        _extract_members(archive_path, destination, max_member_expanded_bytes=3)

    assert not destination.exists()
    assert not list(tmp_path.glob(".per-file-dest-*"))


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_archive_cumulative_expanded_limit_removes_partial_output(
    tmp_path: Path, archive_kind: str
) -> None:
    """Extraction bounds total expanded bytes across all regular members."""
    archive_path = tmp_path / f"cumulative.{archive_kind}"
    _make_archive(
        archive_path,
        archive_kind,
        [("first.txt", b"123"), ("second.txt", b"456")],
    )
    destination = tmp_path / "cumulative-dest"

    with pytest.raises(ValueError, match="cumulative expanded byte limit"):
        _extract_members(archive_path, destination, max_expanded_bytes=5)

    assert not destination.exists()
    assert not list(tmp_path.glob(".cumulative-dest-*"))


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_sibling_prefix_escape_fails_closed(tmp_path: Path, archive_kind: str) -> None:
    """A sibling whose name begins with the destination prefix is still outside it."""
    archive_path = tmp_path / f"evil.{archive_kind}"
    member_name = "../dest_evil/file.txt"
    if archive_kind == "zip":
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(member_name, b"x")
    else:
        with tarfile.open(archive_path, "w") as archive:
            info = tarfile.TarInfo(member_name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="path escape"):
        _extract_members(archive_path, tmp_path / "dest")


def test_zip_duplicate_and_symlink_members_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("same.txt", b"one")
            archive.writestr("same.txt", b"two")
    with pytest.raises(ValueError, match="duplicate"):
        _extract_members(duplicate, tmp_path / "duplicate-dest")

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="symbolic link"):
        _extract_members(symlink, tmp_path / "symlink-dest")


def test_zip_special_unix_member_types_fail_closed(tmp_path: Path) -> None:
    """ZIP device-like entries cannot become files during release-audit extraction."""
    special = tmp_path / "special.zip"
    info = zipfile.ZipInfo("device")
    info.create_system = 3
    info.external_attr = (stat.S_IFCHR | 0o600) << 16
    with zipfile.ZipFile(special, "w") as archive:
        archive.writestr(info, b"device-bytes")

    with pytest.raises(ValueError, match="non-regular member"):
        _extract_members(special, tmp_path / "special-dest")


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
@pytest.mark.parametrize("alias", ["bundle/payload//file.txt", "bundle/payload/./file.txt"])
def test_archive_noncanonical_member_aliases_fail_closed(
    tmp_path: Path, archive_kind: str, alias: str
) -> None:
    """Two raw member names cannot collapse onto one authenticated output path."""
    archive_path = tmp_path / f"collision.{archive_kind}"
    canonical = "bundle/payload/file.txt"
    if archive_kind == "zip":
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(canonical, b"trusted")
            archive.writestr(alias, b"replacement")
    else:
        with tarfile.open(archive_path, "w") as archive:
            for name, data in ((canonical, b"trusted"), (alias, b"replacement")):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    with pytest.raises(ValueError, match="path escape|colliding member"):
        _extract_members(archive_path, tmp_path / "collision-dest")


def test_extraction_rejects_preexisting_destination_symlink(tmp_path: Path) -> None:
    """Offline audit extraction cannot be redirected outside its lexical directory."""
    archive_path = tmp_path / "bundle.zip"
    _make_bundle(archive_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "_extracted"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="destination must not be a symlink"):
        _extract_members(archive_path, destination)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_tar_link_and_device_like_members_fail_closed(tmp_path: Path, member_type: bytes) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("unsafe-member")
        member.type = member_type
        member.linkname = "target"
        archive.addfile(member)

    with pytest.raises(ValueError, match="non-regular member"):
        _extract_members(archive_path, tmp_path / "unsafe-dest")


def test_unsupported_archive_fails_closed(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"not-a-real-archive")
    with pytest.raises(ValueError, match="unsupported archive|extraction failed"):
        _extract_members(bogus, tmp_path / "dest")


def test_internal_checksum_mismatch_detected(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "file.txt").write_text("content")
    (extracted / "checksums.sha256").write_text("0" * 64 + "  file.txt\n")
    problems = _verify_internal_checksums(extracted, ["file.txt", "checksums.sha256"])
    assert any("internal checksum mismatch" in problem for problem in problems)


def test_source_sha_tag_binding_enforced(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    receipt = audit_published(
        tag="paper-matrix-abcdef1234567890abcdef1234567890abcdef12",
        doi="10.5281/zenodo.1",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="0" * 40,
    )
    assert receipt["ok"] is False
    assert any("disagrees with" in problem for problem in receipt["problems"])


def test_historical_precontract_tag_retains_exact_read_only_exception(tmp_path: Path) -> None:
    """The one documented August tag remains auditable at its known immutable source."""
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    receipt = audit_published(
        tag="paper-matrix-v2-h600-s30-2026-08-cd831d7582c1",
        doi="10.5281/zenodo.1",
        github_dir=github,
        zenodo_dir=zenodo,
        source_sha="b1d5ab6de708385c0828c99501a9d1c29727ec11",
    )
    assert receipt["ok"] is True


def test_receipt_is_deterministic(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
        _write_bytes(channel / "checksums.sha256", b"y")
    first = json.dumps(
        audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo),
        sort_keys=True,
    )
    second = json.dumps(
        audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo),
        sort_keys=True,
    )
    assert first == second


def test_tar_bundle_extraction(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    github.mkdir(parents=True)
    zenodo.mkdir(parents=True)
    bundle_path = github / "bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        data = b"member-data"
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    _write_bytes(zenodo / "bundle.tar.gz", bundle_path.read_bytes())
    receipt = audit_published(tag="t", doi="10.5281/zenodo.1", github_dir=github, zenodo_dir=zenodo)
    assert receipt["ok"] is True
    assert receipt["observations"]["bundle"] == "bundle.tar.gz"
    assert receipt["observations"]["bundle_member_count"] == 1


def test_checksums_json_sidecar(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "file.txt").write_text("content")
    (extracted / "checksums.json").write_text(
        json.dumps({"file.txt": sha256_file(extracted / "file.txt")})
    )
    problems = _verify_internal_checksums(extracted, ["file.txt", "checksums.json"])
    assert problems == []


def test_checksums_json_malformed_reports(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "checksums.json").write_text("{not-json")
    problems = _verify_internal_checksums(extracted, ["checksums.json"])
    assert any("not valid JSON" in problem for problem in problems)


def test_cli_main_passes(tmp_path: Path) -> None:
    github = tmp_path / "github"
    zenodo = tmp_path / "zenodo"
    for channel in (github, zenodo):
        _make_bundle(channel / "bundle.zip")
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI_SCRIPT),
            "--tag",
            "paper-matrix-v2-h600-s30",
            "--doi",
            "10.5281/zenodo.1234567",
            "--github-dir",
            str(github),
            "--zenodo-dir",
            str(zenodo),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is True


def test_cli_main_requires_all_predecessor_arguments_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Standalone cold-audit predecessor inputs are an all-or-none contract."""
    status = published_audit_module.main(
        [
            "--tag",
            "tag",
            "--doi",
            "10.5281/zenodo.1",
            "--github-dir",
            str(tmp_path / "github"),
            "--zenodo-dir",
            str(tmp_path / "zenodo"),
            "--predecessor-archive",
            str(tmp_path / "predecessor.tar.gz"),
        ]
    )

    assert status == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "must be provided together" in payload["error"]


def test_cli_main_constructs_predecessor_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Standalone CLI forwards every supplied predecessor coordinate to the offline core."""
    predecessor = tmp_path / "predecessor.tar.gz"
    predecessor.write_bytes(b"fixture")
    seen: dict[str, object] = {}

    def fake_audit(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"schema": SCHEMA, "ok": True, "status": "pass"}

    monkeypatch.setattr(published_audit_module, "audit_published", fake_audit)
    status = published_audit_module.main(
        [
            "--tag",
            "tag",
            "--doi",
            "10.5281/zenodo.1",
            "--github-dir",
            str(tmp_path / "github"),
            "--zenodo-dir",
            str(tmp_path / "zenodo"),
            "--source-sha",
            "a" * 40,
            "--predecessor-archive",
            str(predecessor),
            "--predecessor-doi",
            "10.5281/zenodo.2",
            "--predecessor-concept-doi",
            "10.5281/zenodo.1",
            "--predecessor-tag",
            "predecessor-tag",
            "--predecessor-sha256",
            "b" * 64,
            "--predecessor-size-bytes",
            "7",
        ]
    )

    assert status == 0
    evidence = seen["predecessor_evidence"]
    assert isinstance(evidence, PredecessorEvidence)
    assert evidence.archive_path == predecessor
    assert evidence.version_doi == "10.5281/zenodo.2"
    assert evidence.concept_doi == "10.5281/zenodo.1"
    assert evidence.github_release_tag == "predecessor-tag"
    assert evidence.archive_sha256 == "b" * 64
    assert evidence.archive_size_bytes == 7
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_main_missing_channel_returns_one(tmp_path: Path) -> None:
    github = tmp_path / "github"
    github.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI_SCRIPT),
            "--tag",
            "t",
            "--doi",
            "10.5281/zenodo.1",
            "--github-dir",
            str(github),
            "--zenodo-dir",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is False


class _PublicResponse:
    """Small response double for public discovery and streamed downloads."""

    def __init__(
        self,
        *,
        payload: object = None,
        chunks: tuple[bytes, ...] = (),
        url: str,
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self._chunks = chunks
        self.url = url
        self.status_code = status_code
        self.closed = False

    def json(self) -> object:
        return self._payload

    def iter_content(self, *, chunk_size: int):
        del chunk_size
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _PublicSession:
    """Route-only session double that records every request and its options."""

    def __init__(self, routes: dict[str, _PublicResponse | Exception]) -> None:
        self.routes = routes
        self.headers = {"Authorization": "Bearer should-not-be-sent", "X-token": "secret"}
        self.auth = object()
        self.cookies = {"session": "secret"}
        self.params = {"token": "secret"}
        self.proxies = {"https": "https://user:secret@proxy.test"}
        self.trust_env = True
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def get(self, url: str, **kwargs: object) -> _PublicResponse:
        self.calls.append((url, kwargs, dict(self.headers)))
        route = self.routes[url]
        if isinstance(route, Exception):
            raise route
        return route


def _network_fixture(
    tmp_path: Path,
    *,
    zenodo_name: str = "bundle.zip",
    zenodo_doi: str = "10.5281/zenodo.1234567",
    release_tag: str = "paper-matrix-v2-h600-s30",
    predecessor_doi: str | None = None,
    corrupt_member_checksum: bool = False,
) -> tuple[_PublicSession, bytes, str, str, str]:
    """Build a complete mocked GitHub/Zenodo public response set."""
    del tmp_path
    github_base = "https://github.test"
    zenodo_base = "https://zenodo.test/api"
    tag = release_tag
    source_sha = "b" * 40
    bundle_buffer = io.BytesIO()
    with zipfile.ZipFile(bundle_buffer, "w") as archive:
        archive.writestr("manifest.json", b"network-fixture")
        if corrupt_member_checksum:
            archive.writestr("checksums.sha256", f"{'0' * 64}  manifest.json\n")
    bundle = bundle_buffer.getvalue()
    digest = hashlib.sha256(bundle).hexdigest()
    github_release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{tag}"
    github_ref_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/ref/tags/{tag}"
    github_asset_url = f"https://cdn.github.test/{tag}/bundle.zip"
    zenodo_record_url = f"{zenodo_base}/records/1234567"
    zenodo_asset_url = "https://zenodo.test/api/records/1234567/files/bundle.zip/content"
    source_tag_url = f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}"
    related_identifiers = [
        {
            "identifier": source_tag_url,
            "relation": "isSupplementTo",
            "scheme": "url",
        }
    ]
    if predecessor_doi is not None:
        related_identifiers.append(
            {
                "identifier": predecessor_doi,
                "relation": "isNewVersionOf",
                "scheme": "doi",
            }
        )
    routes: dict[str, _PublicResponse | Exception] = {
        github_release_url: _PublicResponse(
            payload={
                "id": 7944,
                "tag_name": tag,
                "draft": False,
                "prerelease": False,
                "body": f"Source SHA: {source_sha}",
                "assets": [
                    {
                        "name": "bundle.zip",
                        "size": len(bundle),
                        "digest": f"sha256:{digest}",
                        "browser_download_url": github_asset_url,
                    }
                ],
            },
            url=github_release_url,
        ),
        github_ref_url: _PublicResponse(
            payload={
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": source_sha},
            },
            url=github_ref_url,
        ),
        zenodo_record_url: _PublicResponse(
            payload={
                "id": 1234567,
                "doi": zenodo_doi,
                "conceptdoi": "10.5281/zenodo.1234566",
                "state": "done",
                "status": "published",
                "metadata": {
                    "doi": zenodo_doi,
                    "conceptdoi": "10.5281/zenodo.1234566",
                    "related_identifiers": related_identifiers,
                },
                "files": [
                    {
                        "filename": None,
                        "key": zenodo_name,
                        "size": len(bundle),
                        "links": {"self": zenodo_asset_url},
                    }
                ],
            },
            url=zenodo_record_url,
        ),
        github_asset_url: _PublicResponse(
            chunks=(bundle[:3], bundle[3:]),
            url="https://cdn.github.test/final/bundle.zip",
        ),
        zenodo_asset_url: _PublicResponse(
            chunks=(bundle[:5], bundle[5:]),
            url="https://zenodo.test/cdn/final/bundle.zip",
        ),
    }
    if predecessor_doi is not None:
        predecessor_tag = tag.removesuffix("-erratum.1")
        predecessor_bundle = bundle + b"-predecessor"
        predecessor_digest = hashlib.sha256(predecessor_bundle).hexdigest()
        predecessor_release_url = (
            f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{predecessor_tag}"
        )
        predecessor_ref_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/ref/tags/{predecessor_tag}"
        predecessor_record_id = predecessor_doi.rsplit(".", 1)[-1]
        predecessor_record_url = f"{zenodo_base}/records/{predecessor_record_id}"
        predecessor_github_asset_url = f"https://cdn.github.test/{predecessor_tag}/predecessor.zip"
        predecessor_zenodo_asset_url = (
            f"https://zenodo.test/cdn/{predecessor_record_id}/predecessor.zip"
        )
        predecessor_source_tag_url = (
            f"https://github.com/ll7/robot_sf_ll7/releases/tag/{predecessor_tag}"
        )
        predecessor_related_identifiers = [
            {
                "identifier": predecessor_source_tag_url,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ]
        routes.update(
            {
                predecessor_release_url: _PublicResponse(
                    payload={
                        "id": 7943,
                        "tag_name": predecessor_tag,
                        "draft": False,
                        "prerelease": False,
                        "body": f"Source SHA: {source_sha}",
                        "assets": [
                            {
                                "name": "predecessor.zip",
                                "size": len(predecessor_bundle),
                                "digest": f"sha256:{predecessor_digest}",
                                "browser_download_url": predecessor_github_asset_url,
                            }
                        ],
                    },
                    url=predecessor_release_url,
                ),
                predecessor_ref_url: _PublicResponse(
                    payload={
                        "ref": f"refs/tags/{predecessor_tag}",
                        "object": {"type": "commit", "sha": source_sha},
                    },
                    url=predecessor_ref_url,
                ),
                predecessor_record_url: _PublicResponse(
                    payload={
                        "id": int(predecessor_record_id),
                        "doi": predecessor_doi,
                        "conceptdoi": "10.5281/zenodo.1234566",
                        "state": "done",
                        "status": "published",
                        "metadata": {
                            "doi": predecessor_doi,
                            "conceptdoi": "10.5281/zenodo.1234566",
                            "related_identifiers": predecessor_related_identifiers,
                        },
                        "files": [
                            {
                                "filename": "predecessor.zip",
                                "key": "predecessor.zip",
                                "size": len(predecessor_bundle),
                                "links": {"self": predecessor_zenodo_asset_url},
                            }
                        ],
                    },
                    url=predecessor_record_url,
                ),
                predecessor_github_asset_url: _PublicResponse(
                    chunks=(predecessor_bundle[:3], predecessor_bundle[3:]),
                    url=predecessor_github_asset_url,
                ),
                predecessor_zenodo_asset_url: _PublicResponse(
                    chunks=(predecessor_bundle[:5], predecessor_bundle[5:]),
                    url=predecessor_zenodo_asset_url,
                ),
            }
        )
    return _PublicSession(routes), bundle, tag, github_base, zenodo_base


def test_network_audit_discovers_and_streams_public_assets(tmp_path: Path) -> None:
    session, bundle, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
        download_chunk_size=7,
    )
    assert receipt["schema"] == NETWORK_SCHEMA
    assert receipt["status"] == "pass"
    assert receipt["ok"] is True
    assert receipt["source_sha"] == "b" * 40
    assert receipt["downloads"]["bytes"] == len(bundle) * 2
    assert receipt["discovery"]["common_asset_names"] == ["bundle.zip"]
    assert all(not headers for _, _, headers in session.calls)
    assert session.cookies == {}
    assert session.params == {}
    assert session.proxies == {}
    assert session.trust_env is False
    assert all(kwargs["allow_redirects"] is True for _, kwargs, _ in session.calls)
    assert all("stream" in kwargs for url, kwargs, _ in session.calls if "bundle.zip" in url)
    assert "robot-sf-published-audit-" not in json.dumps(receipt)


def test_network_invalid_receipt_sanitizes_private_temp_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid receipts retain bundle diagnostics without private temporary roots."""
    session, _, tag, github_base, zenodo_base = _network_fixture(
        tmp_path, corrupt_member_checksum=True
    )
    original_verify = published_audit_module._verify_internal_checksums
    seen: dict[str, str] = {}

    def leaking_verify(extracted_dir: Path, members: list[str]) -> list[str]:
        problems = original_verify(extracted_dir, members)
        assert problems
        private_root = extracted_dir.parents[1]
        seen["private_root"] = str(private_root)
        seen["resolved_private_root"] = str(private_root.resolve())
        return [*problems, f"diagnostic path: {extracted_dir / 'manifest.json'}"]

    monkeypatch.setattr(published_audit_module, "_verify_internal_checksums", leaking_verify)
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["status"] == "invalid"
    assert seen["private_root"] not in serialized
    assert seen["resolved_private_root"] not in serialized
    assert "github-bundle/manifest.json" in serialized


def test_network_erratum_resolves_and_authenticates_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical successor binds a separately downloaded predecessor archive."""
    predecessor_doi = "10.5281/zenodo.1234565"
    session, bundle, tag, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=f"paper-matrix-v2-h600-s30-2026-09-{'b' * 40}-erratum.1",
        predecessor_doi=predecessor_doi,
    )
    seen: dict[str, object] = {}

    def fake_audit(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        evidence = kwargs["predecessor_evidence"]
        assert isinstance(evidence, PredecessorEvidence)
        return {
            "ok": True,
            "status": "pass",
            "problems": [],
            "observations": {
                "erratum": {
                    "predecessor_version_doi": predecessor_doi,
                    "concept_doi": "10.5281/zenodo.1234566",
                }
            },
        }

    monkeypatch.setattr(published_audit_module, "audit_published", fake_audit)
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
        download_chunk_size=7,
    )

    assert receipt["status"] == "pass"
    assert receipt["ok"] is True
    assert receipt["predecessor"] == {
        "version_doi": predecessor_doi,
        "concept_doi": "10.5281/zenodo.1234566",
        "github_release_tag": tag.removesuffix("-erratum.1"),
        "source_sha": "b" * 40,
        "archive_sha256": hashlib.sha256(bundle + b"-predecessor").hexdigest(),
        "archive_size_bytes": len(bundle) + len(b"-predecessor"),
    }
    assert seen["tag"] == tag
    assert seen["doi"] == "10.5281/zenodo.1234567"
    assert receipt["downloads"]["bytes"] == 2 * len(bundle) + 2 * (len(bundle) + 12)
    assert [record["name"] for record in receipt["downloads"]["github"]] == [
        "bundle.zip",
        "predecessor.zip",
    ]
    assert [record["name"] for record in receipt["downloads"]["zenodo"]] == [
        "bundle.zip",
        "predecessor.zip",
    ]
    assert "archive_path" not in json.dumps(receipt)


def test_network_audit_rejects_renamed_channel_asset_before_download(tmp_path: Path) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(
        tmp_path, zenodo_name="renamed.zip"
    )
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "invalid"
    assert any("named public GitHub" in problem for problem in receipt["problems"])
    assert not any("bundle.zip" in url for url, _, _ in session.calls[2:])


def test_network_audit_records_every_zenodo_download(tmp_path: Path) -> None:
    """The durable receipt must enumerate every downloaded Zenodo asset."""
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    github_release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{tag}"
    zenodo_record_url = f"{zenodo_base}/records/1234567"
    github_response = session.routes[github_release_url]
    zenodo_response = session.routes[zenodo_record_url]
    assert isinstance(github_response, _PublicResponse)
    assert isinstance(zenodo_response, _PublicResponse)
    github_payload = copy.deepcopy(github_response._payload)
    zenodo_payload = copy.deepcopy(zenodo_response._payload)
    note = b"public release note"
    digest = hashlib.sha256(note).hexdigest()
    github_note_url = "https://cdn.github.test/final/notes.txt"
    zenodo_note_url = "https://zenodo.test/cdn/final/notes.txt"
    github_payload["assets"].append(
        {
            "name": "notes.txt",
            "size": len(note),
            "digest": f"sha256:{digest}",
            "browser_download_url": github_note_url,
        }
    )
    zenodo_payload["files"].append(
        {
            "filename": "notes.txt",
            "key": "notes.txt",
            "size": len(note),
            "links": {"self": zenodo_note_url},
        }
    )
    github_response._payload = github_payload
    zenodo_response._payload = zenodo_payload
    session.routes[github_note_url] = _PublicResponse(chunks=(note,), url=github_note_url)
    session.routes[zenodo_note_url] = _PublicResponse(chunks=(note,), url=zenodo_note_url)

    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert receipt["status"] == "pass"
    assert [record["name"] for record in receipt["downloads"]["zenodo"]] == [
        "bundle.zip",
        "notes.txt",
    ]


@pytest.mark.parametrize("size", [None, 0])
def test_network_audit_rejects_missing_or_empty_published_zenodo_size(
    tmp_path: Path, size: int | None
) -> None:
    """Published Zenodo records must advertise a positive file size."""
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    record_url = f"{zenodo_base}/records/1234567"
    response = session.routes[record_url]
    assert isinstance(response, _PublicResponse)
    payload = copy.deepcopy(response._payload)
    payload["files"][0]["size"] = size
    response._payload = payload

    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert receipt["status"] == "invalid"
    assert any("positive advertised size" in problem for problem in receipt["problems"])
    assert not any("bundle.zip" in url for url, _, _ in session.calls[2:])


def test_network_audit_separates_transport_unavailability(tmp_path: Path) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    first_url = next(iter(session.routes))
    session.routes[first_url] = OSError("network down")
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "unavailable"
    assert receipt["ok"] is False
    assert receipt["audit"] is None


def test_network_audit_rejects_partial_stream(tmp_path: Path) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    asset_url = next(url for url in session.routes if "cdn.github.test" in url)
    response = session.routes[asset_url]
    assert isinstance(response, _PublicResponse)
    response._chunks = (b"partial",)
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "invalid"
    assert any("size mismatch" in problem for problem in receipt["problems"])


def test_network_audit_resolves_annotated_tag(tmp_path: Path) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    ref_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/ref/tags/{tag}"
    annotation_sha = "c" * 40
    source_sha = "b" * 40
    annotated_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/tags/{annotation_sha}"
    ref_response = session.routes[ref_url]
    assert isinstance(ref_response, _PublicResponse)
    ref_response._payload = {
        "ref": f"refs/tags/{tag}",
        "object": {"type": "tag", "sha": annotation_sha},
    }
    session.routes[annotated_url] = _PublicResponse(
        payload={"object": {"type": "commit", "sha": source_sha}},
        url=annotated_url,
    )
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "pass"
    assert receipt["source_sha"] == source_sha


def test_network_audit_rejects_doi_drift_and_secret_headers(tmp_path: Path) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(
        tmp_path, zenodo_doi="10.5281/zenodo.7654321"
    )
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "invalid"
    assert any("DOI does not match" in problem for problem in receipt["problems"])
    assert all("Authorization" not in json.dumps(headers) for _, _, headers in session.calls)
    assert "secret" not in json.dumps(receipt)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("source_scheme", "not related"),
        ("missing_predecessor", "lacks one distinct predecessor version DOI"),
        ("predecessor_scheme", "predecessor-version relation is malformed"),
        ("duplicate_predecessor", "predecessor-version relation is malformed"),
    ],
)
def test_network_erratum_requires_exact_relation_schemes_and_cardinality(
    tmp_path: Path, fault: str, expected: str
) -> None:
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, _, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    record_url = f"{zenodo_base}/records/1234567"
    response = session.routes[record_url]
    assert isinstance(response, _PublicResponse)
    payload = copy.deepcopy(response._payload)
    related = payload["metadata"]["related_identifiers"]
    if fault == "source_scheme":
        related[0]["scheme"] = "doi"
    elif fault == "missing_predecessor":
        payload["metadata"]["related_identifiers"] = related[:1]
    elif fault == "predecessor_scheme":
        related[1]["scheme"] = "url"
    else:
        related.append(copy.deepcopy(related[1]))
    response._payload = payload

    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert result["status"] == "invalid"
    assert any(expected in problem for problem in result["problems"])


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("source", "predecessor GitHub tag source SHA differs"),
        ("concept", "predecessor Zenodo concept DOI differs"),
    ],
)
def test_network_erratum_requires_predecessor_source_and_concept_identity(
    tmp_path: Path, fault: str, expected: str
) -> None:
    """The independently resolved predecessor must retain both identities."""
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, _, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    predecessor_tag = tag.removesuffix("-erratum.1")
    if fault == "source":
        ref_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/ref/tags/{predecessor_tag}"
        release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{predecessor_tag}"
        ref_response = session.routes[ref_url]
        release_response = session.routes[release_url]
        assert isinstance(ref_response, _PublicResponse)
        assert isinstance(release_response, _PublicResponse)
        ref_payload = copy.deepcopy(ref_response._payload)
        release_payload = copy.deepcopy(release_response._payload)
        ref_payload["object"]["sha"] = "c" * 40
        release_payload["body"] = f"Source SHA: {'c' * 40}"
        ref_response._payload = ref_payload
        release_response._payload = release_payload
    else:
        record_url = f"{zenodo_base}/records/1234565"
        record_response = session.routes[record_url]
        assert isinstance(record_response, _PublicResponse)
        record_payload = copy.deepcopy(record_response._payload)
        record_payload["conceptdoi"] = "10.5281/zenodo.9999999"
        record_payload["metadata"]["conceptdoi"] = "10.5281/zenodo.9999999"
        record_response._payload = record_payload

    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert result["status"] == "invalid"
    assert any(expected in problem for problem in result["problems"])


@pytest.mark.parametrize("variant", ["no_common", "ambiguous"])
def test_network_erratum_requires_one_common_predecessor_archive(
    tmp_path: Path, variant: str
) -> None:
    """Predecessor custody must identify exactly one shared archive before downloading."""
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, _, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    predecessor_tag = tag.removesuffix("-erratum.1")
    release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{predecessor_tag}"
    record_url = f"{zenodo_base}/records/1234565"
    release_response = session.routes[release_url]
    record_response = session.routes[record_url]
    assert isinstance(release_response, _PublicResponse)
    assert isinstance(record_response, _PublicResponse)
    release_payload = copy.deepcopy(release_response._payload)
    record_payload = copy.deepcopy(record_response._payload)
    if variant == "no_common":
        release_payload["assets"][0]["name"] = "predecessor.txt"
        record_payload["files"][0]["filename"] = "predecessor.txt"
        record_payload["files"][0]["key"] = "predecessor.txt"
    else:
        second_url = "https://cdn.github.test/predecessor/second.zip"
        release_payload["assets"].append(
            {
                "name": "second.zip",
                "size": release_payload["assets"][0]["size"],
                "digest": release_payload["assets"][0]["digest"],
                "browser_download_url": second_url,
            }
        )
        record_payload["files"].append(
            {
                "filename": "second.zip",
                "key": "second.zip",
                "size": record_payload["files"][0]["size"],
                "links": {"self": "https://zenodo.test/cdn/1234565/second.zip"},
            }
        )
    release_response._payload = release_payload
    record_response._payload = record_payload

    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert result["status"] == "invalid"
    expected = "no common predecessor archive" if variant == "no_common" else "exactly one"
    assert any(expected in problem for problem in result["problems"])
    assert not any("predecessor.zip" in url for url, _, _ in session.calls)


@pytest.mark.parametrize("channel", ["github", "zenodo"])
@pytest.mark.parametrize("size", [None, 0])
def test_network_erratum_requires_positive_predecessor_archive_size(
    tmp_path: Path, channel: str, size: int | None
) -> None:
    """Both public predecessor channels must advertise a positive archive size."""
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, _, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    predecessor_tag = tag.removesuffix("-erratum.1")
    if channel == "github":
        release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{predecessor_tag}"
        response = session.routes[release_url]
        assert isinstance(response, _PublicResponse)
        payload = copy.deepcopy(response._payload)
        payload["assets"][0]["size"] = size
    else:
        response = session.routes[f"{zenodo_base}/records/1234565"]
        assert isinstance(response, _PublicResponse)
        payload = copy.deepcopy(response._payload)
        payload["files"][0]["size"] = size
    response._payload = payload

    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert result["status"] == "invalid"
    assert any("positive advertised size" in problem for problem in result["problems"])


@pytest.mark.parametrize("fault", ["size", "digest"])
def test_network_erratum_reconciles_predecessor_archive_channels(
    tmp_path: Path, fault: str
) -> None:
    """The two independently downloaded predecessor archives must be byte-identical."""
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, bundle, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    predecessor_asset_url = "https://zenodo.test/cdn/1234565/predecessor.zip"
    response = session.routes[predecessor_asset_url]
    assert isinstance(response, _PublicResponse)
    predecessor_bundle = bundle + b"-predecessor"
    if fault == "size":
        altered = predecessor_bundle + b"x"
        response._chunks = (altered[:5], altered[5:])
        record_response = session.routes[f"{zenodo_base}/records/1234565"]
        assert isinstance(record_response, _PublicResponse)
        record_payload = copy.deepcopy(record_response._payload)
        record_payload["files"][0]["size"] = len(altered)
        record_response._payload = record_payload
    else:
        altered = bytes([predecessor_bundle[0] ^ 1]) + predecessor_bundle[1:]
        response._chunks = (altered[:5], altered[5:])

    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )

    assert result["status"] == "invalid"
    expected = "size mismatch" if fault == "size" else "digest mismatch"
    assert any(expected in problem for problem in result["problems"])


def test_network_erratum_applies_cumulative_predecessor_byte_limit(tmp_path: Path) -> None:
    """The predecessor pair shares the existing cumulative download cap."""
    source_sha = "b" * 40
    tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}-erratum.1"
    session, bundle, _, github_base, zenodo_base = _network_fixture(
        tmp_path,
        release_tag=tag,
        predecessor_doi="10.5281/zenodo.1234565",
    )
    total_advertised = 2 * len(bundle) + 2 * (len(bundle) + len(b"-predecessor"))
    result = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
        max_download_bytes=total_advertised - 1,
    )

    assert result["status"] == "invalid"
    assert any("advertised public assets exceed" in problem for problem in result["problems"])
    assert not any("cdn." in url for url, _, _ in session.calls)


def test_network_erratum_lineage_reconciliation_fails_closed() -> None:
    core = {
        "ok": True,
        "status": "pass",
        "problems": [],
        "observations": {
            "erratum": {
                "predecessor_version_doi": "10.5281/zenodo.7",
                "concept_doi": "10.5281/zenodo.6",
            }
        },
    }

    published_audit_module._reconcile_zenodo_erratum_lineage(
        core,
        tag=f"release-{'b' * 40}-erratum.1",
        zenodo={"predecessor_doi": "10.5281/zenodo.9", "concept_doi": "10.5281/zenodo.6"},
    )

    assert core["ok"] is False
    assert core["status"] == "fail"
    assert core["problems"] == ["Zenodo API lineage differs from the embedded erratum receipt"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tag": "bad/tag"}, "path-safe"),
        ({"repo": "bad-repository"}, "owner/name"),
        ({"doi": "not-a-doi"}, "10.5281"),
        ({"max_download_bytes": 0}, "max_download_bytes"),
        ({"download_chunk_size": 0}, "download_chunk_size"),
        ({"timeout": 0}, "timeout"),
        ({"github_api_base": "https://github.test?token=secret"}, "query"),
    ],
)
def test_network_audit_rejects_invalid_inputs(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    _, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    kwargs: dict[str, object] = {
        "tag": tag,
        "doi": "10.5281/zenodo.1234567",
        "github_api_base": github_base,
        "zenodo_api_base": zenodo_base,
    }
    kwargs.update(overrides)
    receipt = audit_published_network(**kwargs)  # type: ignore[arg-type]
    assert receipt["status"] == "invalid"
    assert message in receipt["problems"][0]


@pytest.mark.parametrize(
    "status_code, status", [(503, "unavailable"), (404, "invalid"), (302, "invalid")]
)
def test_network_audit_maps_public_http_statuses(
    tmp_path: Path, status_code: int, status: str
) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    release_url = next(url for url in session.routes if "/releases/tags/" in url)
    release = session.routes[release_url]
    assert isinstance(release, _PublicResponse)
    release.status_code = status_code
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == status
    assert receipt["audit"] is None


@pytest.mark.parametrize("variant", ["no_assets", "malformed_asset", "duplicate", "size", "digest"])
def test_network_audit_rejects_malformed_github_assets(tmp_path: Path, variant: str) -> None:
    session, _, tag, github_base, zenodo_base = _network_fixture(tmp_path)
    release_url = next(url for url in session.routes if "/releases/tags/" in url)
    release = session.routes[release_url]
    assert isinstance(release, _PublicResponse)
    payload = copy.deepcopy(release._payload)
    assert isinstance(payload, dict)
    if variant == "no_assets":
        payload["assets"] = []
    elif variant == "malformed_asset":
        payload["assets"] = ["not-an-object"]
    elif variant == "duplicate":
        payload["assets"] = [payload["assets"][0], copy.deepcopy(payload["assets"][0])]
    elif variant == "size":
        payload["assets"][0]["size"] = -1
    else:
        payload["assets"][0]["digest"] = "sha256:not-a-digest"
    release._payload = payload
    receipt = audit_published_network(
        tag=tag,
        doi="10.5281/zenodo.1234567",
        session=session,
        github_api_base=github_base,
        zenodo_api_base=zenodo_base,
    )
    assert receipt["status"] == "invalid"


def test_network_audit_public_helpers_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(published_audit_module, "try_import", lambda _: None)
    with pytest.raises(published_audit_module.PublishedAuditUnavailable, match="requests"):
        published_audit_module._prepare_public_session(None)


def test_network_failure_receipt_redacts_invalid_identifiers() -> None:
    receipt = audit_published_network(
        tag="https://user:secret@example.test/release?token=secret",
        doi="https://doi.org/10.5281/zenodo.1234567?token=secret",
    )
    assert receipt["status"] == "invalid"
    assert receipt["tag"] == "<invalid-tag>"
    assert receipt["doi"] == "<invalid-doi>"
    assert "secret" not in json.dumps(receipt)


def test_release_cli_exposes_network_audit_and_writes_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from robot_sf import cli

    receipt = {
        "schema": NETWORK_SCHEMA,
        "ok": True,
        "status": "pass",
        "tag": "tag",
        "doi": "10.5281/zenodo.1",
        "source_sha": "a" * 40,
        "problems": [],
    }
    seen: dict[str, object] = {}

    def fake_audit(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return receipt

    monkeypatch.setattr(
        "robot_sf.release_cli.published_release_audit.audit_published_network", fake_audit
    )
    output = tmp_path / "receipt.json"
    code = cli.main(
        [
            "release",
            "audit-published",
            "--tag",
            "tag",
            "--doi",
            "10.5281/zenodo.1",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert seen["tag"] == "tag"
    assert json.loads(output.read_text()) == receipt
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.parametrize("status, expected_code", [("invalid", 1), ("unavailable", 2)])
def test_release_cli_maps_network_failure_statuses(
    tmp_path: Path, monkeypatch, capsys, status: str, expected_code: int
) -> None:
    from robot_sf import cli

    receipt = {
        "schema": NETWORK_SCHEMA,
        "ok": False,
        "status": status,
        "tag": "tag",
        "doi": "10.5281/zenodo.1",
        "problems": ["public service condition"],
    }
    monkeypatch.setattr(
        "robot_sf.release_cli.published_release_audit.audit_published_network",
        lambda **kwargs: receipt,
    )
    code = cli.main(
        [
            "release",
            "audit-published",
            "--tag",
            "tag",
            "--doi",
            "10.5281/zenodo.1",
        ]
    )
    assert code == expected_code
    assert json.loads(capsys.readouterr().out) == receipt


def test_release_cli_returns_two_when_receipt_write_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from robot_sf import cli

    receipt = {
        "schema": NETWORK_SCHEMA,
        "ok": True,
        "status": "pass",
        "tag": "tag",
        "doi": "10.5281/zenodo.1",
        "problems": [],
    }
    monkeypatch.setattr(
        "robot_sf.release_cli.published_release_audit.audit_published_network",
        lambda **kwargs: receipt,
    )
    monkeypatch.setattr(
        "robot_sf.release_cli.published_release_audit.write_network_receipt",
        lambda *args: (_ for _ in ()).throw(OSError("write denied")),
    )
    code = cli.main(
        [
            "release",
            "audit-published",
            "--tag",
            "tag",
            "--doi",
            "10.5281/zenodo.1",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "error"
    assert "write denied" not in json.dumps(output)
