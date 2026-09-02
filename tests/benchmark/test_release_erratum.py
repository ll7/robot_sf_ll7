"""Scientific-equality and identity tests for benchmark release errata."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.release_erratum import (
    SCIENTIFIC_CANONICALIZATION,
    ErratumContract,
    ReleaseErratumError,
    build_erratum_receipt,
    compare_scientific_snapshots,
    load_erratum_contract,
    snapshot_campaign,
    snapshot_predecessor_archive,
    validate_erratum_receipt_against_campaign,
)

SOURCE_SHA = "59577bad289dd692ba3580e1600c4a649ae27880"
BUILDER_SHA = "a4aaf1f06860cf632d0173c5a13e11ad855b6df2"
ORCHESTRATION_SHA = "b" * 40
OLD_TAG = f"paper-matrix-v2-h600-s30-2026-09-{SOURCE_SHA}"
NEW_TAG = f"{OLD_TAG}-erratum.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row(arm: str, scenario: str, seed: int) -> dict[str, Any]:
    episode_id = f"{scenario}--{seed}--fixture"
    status = "success" if scenario == "crossing" and seed == 111 else "collision"
    if scenario == "bottleneck":
        status = "failure"
    return {
        "algo": arm,
        "scenario_id": scenario,
        "seed": seed,
        "episode_id": episode_id,
        "status": status,
        "outcome": "goal_reached",
        "git_hash": SOURCE_SHA,
        "provenance": {"git_hash": SOURCE_SHA},
        "result_provenance": {"repo_commit": SOURCE_SHA},
        "event_ledger": {"software_commit": SOURCE_SHA},
        "metrics": {
            "collisions": 0,
            "snqi": seed / 100.0,
            "unavailable_nan": float("nan"),
            "unbounded_positive": float("inf"),
            "unbounded_negative": float("-inf"),
        },
    }


def _write_campaign(root: Path) -> None:
    for arm in ("goal__differential_drive", "orca__differential_drive"):
        path = root / "runs" / arm / "episodes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            _row(arm.split("__", 1)[0], scenario, seed)
            for scenario in ("crossing", "bottleneck")
            for seed in (111, 112)
        ]
        path.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")


def _archive_campaign(campaign: Path, archive: Path) -> None:
    with tarfile.open(archive, mode="w:gz") as bundle:
        for path in sorted((campaign / "runs").glob("*/episodes.jsonl")):
            bundle.add(
                path,
                arcname=f"fixture_bundle/payload/runs/{path.parent.name}/episodes.jsonl",
            )


def _contract(archive: Path) -> ErratumContract:
    return ErratumContract(
        correction_id="september-2026-derived-metadata-erratum.1",
        predecessor_version_doi="10.5281/zenodo.22227035",
        predecessor_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        predecessor_archive_size_bytes=archive.stat().st_size,
        predecessor_github_release_tag=OLD_TAG,
        source_sha=SOURCE_SHA,
        planner_arms=2,
        scenario_count=2,
        seed_count=2,
        episode_rows=8,
        builder_sha=BUILDER_SHA,
        validator_sha=BUILDER_SHA,
        orchestration_sha=ORCHESTRATION_SHA,
        concept_doi="10.5281/zenodo.22227034",
        successor_version_doi="10.5281/zenodo.22229999",
        successor_github_release_tag=NEW_TAG,
        metadata_path=Path("metadata.json"),
        metadata_sha256="a" * 64,
    )


def _with_bundle_metadata(campaign: Path, contract: ErratumContract) -> ErratumContract:
    metadata = campaign / "release/zenodo_metadata.erratum.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    updated = replace(
        contract,
        metadata_path=metadata,
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    provenance = {
        "release_tag": updated.successor_github_release_tag,
        "release_id": updated.successor_github_release_tag,
        "doi": updated.successor_version_doi,
        "version_doi": updated.successor_version_doi,
        "concept_doi": updated.concept_doi,
        "metadata_path": "release/zenodo_metadata.erratum.json",
        "metadata_sha256": updated.metadata_sha256,
        "scientific_source_sha": updated.source_sha,
    }
    current = {
        "release_tag": updated.successor_github_release_tag,
        "release_id": updated.successor_github_release_tag,
        "doi": updated.successor_version_doi,
        "version_doi": updated.successor_version_doi,
        "concept_doi": updated.concept_doi,
        "provenance": provenance,
    }
    (campaign / "release/release_manifest.resolved.json").write_text(
        json.dumps(current), encoding="utf-8"
    )
    (campaign / "release/release_result.json").write_text(
        json.dumps(
            {
                **current,
                "benchmark_release": current,
                "resolved_manifest": current,
                "publication_preflight_status": "pass",
                "publication_preflight_violations": [],
                "release_status": "ok",
                "ranking_claims_admitted": False,
            }
        ),
        encoding="utf-8",
    )
    summary = campaign / "reports/campaign_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"benchmark_release": current, "campaign": current}), encoding="utf-8"
    )
    return updated


def test_predecessor_and_successor_scientific_leaves_match(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))

    predecessor = snapshot_predecessor_archive(archive, contract=contract)
    successor = snapshot_campaign(campaign, contract=contract)
    equality = compare_scientific_snapshots(predecessor, successor)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=predecessor,
        successor=successor,
    )

    assert equality["status"] == "identical"
    assert equality["episode_rows"] == 8
    assert receipt["scientific_identity"]["component_leaf_manifest_sha256"]
    assert receipt["derivation"] == {
        "builder_sha": BUILDER_SHA,
        "validator_sha": BUILDER_SHA,
        "orchestration_sha": ORCHESTRATION_SHA,
        "scientific_source_sha": SOURCE_SHA,
        "simulation_rerun": False,
    }
    assert receipt["corrected_verdict"]["ranking_claims_admitted"] is False
    assert receipt["scientific_canonicalization"]["schema"] == SCIENTIFIC_CANONICALIZATION


def test_scientific_equality_distinguishes_nonfinite_float_categories(tmp_path: Path) -> None:
    """NaN and signed infinities are retained as distinct scientific leaf values."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)
    predecessor = snapshot_predecessor_archive(archive, contract=contract)

    path = campaign / "runs" / "goal__differential_drive" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["metrics"]["unbounded_positive"] = float("-inf")
    path.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")
    successor = snapshot_campaign(campaign, contract=contract)

    with pytest.raises(ReleaseErratumError, match="scientific leaves differ"):
        compare_scientific_snapshots(predecessor, successor)


def test_cold_erratum_receipt_recomputes_successor_leaves(tmp_path: Path) -> None:
    """A downloaded successor must reproduce every scientific digest in its receipt."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=snapshot,
        successor=snapshot,
    )
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    observed = validate_erratum_receipt_against_campaign(
        receipt_path,
        campaign_root=campaign,
        metadata_path=contract.metadata_path,
        expected_tag=NEW_TAG,
        expected_doi=contract.successor_version_doi,
    )

    assert observed["status"] == "pass"
    assert observed["episode_rows"] == 8
    assert observed["canonical_row_manifest_sha256"] == snapshot.canonical_row_manifest_sha256


def test_cold_erratum_receipt_rejects_tampered_successor_row(tmp_path: Path) -> None:
    """A valid-looking receipt cannot mask a changed downloaded component leaf."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(
                contract=contract,
                predecessor=snapshot,
                successor=snapshot,
            )
        ),
        encoding="utf-8",
    )
    episodes = campaign / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["metrics"]["collisions"] = 99
    episodes.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="differ from its receipt"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("github_source", "GitHub tag target"),
        ("derivation_source", "derivation source"),
        ("metadata", "metadata SHA-256"),
    ],
)
def test_cold_erratum_receipt_rejects_identity_or_metadata_drift(
    tmp_path: Path, fault: str, match: str
) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=snapshot,
        successor=snapshot,
    )
    expected_source_sha = SOURCE_SHA
    if fault == "github_source":
        expected_source_sha = "0" * 40
    elif fault == "derivation_source":
        receipt["derivation"]["scientific_source_sha"] = "0" * 40
    else:
        contract.metadata_path.write_text("{}\n", encoding="utf-8")
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=match):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
            expected_source_sha=expected_source_sha,
        )


def test_cold_erratum_helper_requires_canonical_payload_paths(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=snapshot,
        successor=snapshot,
    )
    canonical_receipt = campaign / "provenance/benchmark_release_erratum.json"
    canonical_receipt.parent.mkdir(parents=True)
    canonical_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    external_receipt = tmp_path / "benchmark_release_erratum.json"
    external_receipt.write_bytes(canonical_receipt.read_bytes())
    external_metadata = tmp_path / "zenodo_metadata.erratum.json"
    external_metadata.write_bytes(contract.metadata_path.read_bytes())

    with pytest.raises(ReleaseErratumError, match="receipt is outside"):
        validate_erratum_receipt_against_campaign(
            external_receipt,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )
    with pytest.raises(ReleaseErratumError, match="metadata is outside"):
        validate_erratum_receipt_against_campaign(
            canonical_receipt,
            campaign_root=campaign,
            metadata_path=external_metadata,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_receipt_rejects_stale_release_document_alias(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(
                contract=contract,
                predecessor=snapshot,
                successor=snapshot,
            )
        ),
        encoding="utf-8",
    )
    result_path = campaign / "release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["version_doi"] = contract.predecessor_version_doi
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="stale version-DOI alias"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_scientific_equality_rejects_changed_component_metric(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)
    predecessor = snapshot_predecessor_archive(archive, contract=contract)

    path = campaign / "runs" / "goal__differential_drive" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["metrics"]["collisions"] = 1
    path.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")
    successor = snapshot_campaign(campaign, contract=contract)

    with pytest.raises(ReleaseErratumError, match="scientific leaves differ"):
        compare_scientific_snapshots(predecessor, successor)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_source", "failed"])
def test_scientific_snapshot_rejects_invalid_matrix(tmp_path: Path, fault: str) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)
    path = campaign / "runs" / "goal__differential_drive" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "wrong_source":
        rows[0]["git_hash"] = "0" * 40
    else:
        rows[0]["status"] = "degraded"
    path.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")

    with pytest.raises(ReleaseErratumError):
        snapshot_campaign(campaign, contract=contract)


def test_scientific_snapshot_rejects_nested_episode_file_and_ledger_mismatch(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)

    nested = campaign / "runs/goal__differential_drive/nested/episodes.jsonl"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReleaseErratumError, match="outside runs/<arm>"):
        snapshot_campaign(campaign, contract=contract)
    nested.unlink()

    episodes = campaign / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["event_ledger"]["software_commit"] = "0" * 40
    episodes.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")
    with pytest.raises(ReleaseErratumError, match="event_ledger.software_commit"):
        snapshot_campaign(campaign, contract=contract)


@pytest.mark.parametrize(
    ("contract_updates", "match"),
    [
        ({"source_sha": "0" * 40}, "predecessor tag"),
        (
            {
                "predecessor_github_release_tag": "semantic-release",
                "successor_github_release_tag": "semantic-release-erratum.1",
            },
            "predecessor tag",
        ),
        (
            {
                "predecessor_github_release_tag": OLD_TAG.upper(),
                "successor_github_release_tag": f"{OLD_TAG.upper()}-erratum.1",
            },
            "predecessor tag",
        ),
        ({"successor_github_release_tag": f"{OLD_TAG}-erratum.01"}, "successor tag"),
    ],
)
def test_direct_contract_rejects_noncanonical_tag_lineage(
    tmp_path: Path, contract_updates: dict[str, object], match: str
) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = replace(_contract(archive), **contract_updates)

    with pytest.raises(ReleaseErratumError, match=match):
        snapshot_campaign(campaign, contract=contract)


def test_predecessor_archive_rejects_hash_size_and_unsafe_members(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)

    with pytest.raises(ReleaseErratumError, match="byte count"):
        snapshot_predecessor_archive(
            archive,
            contract=replace(contract, predecessor_archive_size_bytes=archive.stat().st_size + 1),
        )
    with pytest.raises(ReleaseErratumError, match="SHA-256"):
        snapshot_predecessor_archive(
            archive,
            contract=replace(contract, predecessor_archive_sha256="0" * 64),
        )

    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, mode="w:gz") as bundle:
        member = tarfile.TarInfo("bundle/payload/runs/goal__differential_drive/episodes.jsonl")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        bundle.addfile(member, io.BytesIO())
    unsafe_contract = replace(
        contract,
        predecessor_archive_sha256=hashlib.sha256(unsafe.read_bytes()).hexdigest(),
        predecessor_archive_size_bytes=unsafe.stat().st_size,
    )
    with pytest.raises(ReleaseErratumError, match="non-regular"):
        snapshot_predecessor_archive(unsafe, contract=unsafe_contract)


def test_predecessor_archive_rejects_episode_file_outside_canonical_arm_path(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "extra.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        for path in sorted((campaign / "runs").glob("*/episodes.jsonl")):
            bundle.add(
                path,
                arcname=f"fixture_bundle/payload/runs/{path.parent.name}/episodes.jsonl",
            )
        data = b"{}\n"
        member = tarfile.TarInfo("fixture_bundle/payload/runs/arm/nested/episodes.jsonl")
        member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))
    contract = _contract(archive)

    with pytest.raises(ReleaseErratumError, match="outside payload/runs/<arm>"):
        snapshot_predecessor_archive(archive, contract=contract)


def test_predecessor_archive_rejects_event_ledger_source_mismatch(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    episodes = campaign / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["event_ledger"]["software_commit"] = "0" * 40
    episodes.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)

    with pytest.raises(ReleaseErratumError, match="event_ledger.software_commit"):
        snapshot_predecessor_archive(archive, contract=_contract(archive))


def _metadata() -> dict[str, Any]:
    return {
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
                    "identifier": ("https://github.com/ll7/robot_sf_ll7/releases/tag/" + NEW_TAG),
                    "relation": "isSupplementTo",
                    "scheme": "url",
                },
                {
                    "identifier": "10.5281/zenodo.22227035",
                    "relation": "isNewVersionOf",
                    "scheme": "doi",
                },
            ],
        }
    }


def test_erratum_contract_loads_exact_linked_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-release-erratum.v1",
                "correction_id": "september-2026-derived-metadata-erratum.1",
                "correction_scope": "derived_publication_metadata_only",
                "supersedes": {
                    "version_doi": "10.5281/zenodo.22227035",
                    "archive_sha256": "e" * 64,
                    "archive_size_bytes": 54219004,
                    "github_release_tag": OLD_TAG,
                    "old_publication_retained": True,
                },
                "scientific_identity": {
                    "source_sha": SOURCE_SHA,
                    "planner_arms": 14,
                    "scenario_count": 48,
                    "seed_count": 30,
                    "episode_rows": 20160,
                },
                "derivation": {
                    "builder_sha": BUILDER_SHA,
                    "validator_sha": BUILDER_SHA,
                    "orchestration_sha": ORCHESTRATION_SHA,
                    "simulation_rerun": False,
                },
                "successor": {
                    "concept_doi": "10.5281/zenodo.22227034",
                    "version_doi": "10.5281/zenodo.22229999",
                    "github_release_tag": NEW_TAG,
                    "metadata_path": "metadata.json",
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                },
                "corrected_verdict": {
                    "publication_preflight_status": "pass",
                    "publication_preflight_violations": [],
                    "release_status": "ok",
                    "ranking_claims_admitted": False,
                },
            }
        ),
        encoding="utf-8",
    )

    contract = load_erratum_contract(contract_path, repository_root=tmp_path)

    assert contract.source_sha == SOURCE_SHA
    assert contract.successor_github_release_tag == NEW_TAG
    assert contract.metadata_path == metadata


def test_erratum_contract_rejects_missing_predecessor_relation(tmp_path: Path) -> None:
    metadata_payload = _metadata()
    metadata_payload["metadata"]["related_identifiers"].pop()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(metadata_payload), encoding="utf-8")
    # Reuse the positive fixture writer, then replace only the metadata digest.
    positive = tmp_path / "positive"
    positive.mkdir()
    positive_metadata = positive / "metadata.json"
    positive_metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    # Build the contract through the same literal shape as the positive test.
    contract_payload = {
        "schema_version": "benchmark-release-erratum.v1",
        "correction_id": "september-2026-derived-metadata-erratum.1",
        "correction_scope": "derived_publication_metadata_only",
        "supersedes": {
            "version_doi": "10.5281/zenodo.22227035",
            "archive_sha256": "e" * 64,
            "archive_size_bytes": 54219004,
            "github_release_tag": OLD_TAG,
            "old_publication_retained": True,
        },
        "scientific_identity": {
            "source_sha": SOURCE_SHA,
            "planner_arms": 14,
            "scenario_count": 48,
            "seed_count": 30,
            "episode_rows": 20160,
        },
        "derivation": {
            "builder_sha": BUILDER_SHA,
            "validator_sha": BUILDER_SHA,
            "orchestration_sha": ORCHESTRATION_SHA,
            "simulation_rerun": False,
        },
        "successor": {
            "concept_doi": "10.5281/zenodo.22227034",
            "version_doi": "10.5281/zenodo.22229999",
            "github_release_tag": NEW_TAG,
            "metadata_path": "metadata.json",
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        },
        "corrected_verdict": {
            "publication_preflight_status": "pass",
            "publication_preflight_violations": [],
            "release_status": "ok",
            "ranking_claims_admitted": False,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="predecessor version DOI"):
        load_erratum_contract(contract_path, repository_root=tmp_path)
