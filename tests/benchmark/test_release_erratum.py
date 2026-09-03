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
    _MAX_IDENTITY_TRAVERSAL_DEPTH,
    SCIENTIFIC_CANONICALIZATION,
    SCIENTIFIC_CANONICALIZATION_POLICY,
    ErratumContract,
    PredecessorEvidence,
    ReleaseErratumError,
    _assert_predecessor_execution_aliases,
    _assert_publication_aliases,
    _assert_publication_url_aliases,
    build_erratum_receipt,
    compare_scientific_snapshots,
    load_erratum_contract,
    snapshot_campaign,
    snapshot_predecessor_archive,
    validate_erratum_contract_identity,
)
from robot_sf.benchmark.release_erratum import (
    validate_erratum_receipt_against_campaign as _validate_erratum_receipt,
)

SOURCE_SHA = "59577bad289dd692ba3580e1600c4a649ae27880"
BUILDER_SHA = "a4aaf1f06860cf632d0173c5a13e11ad855b6df2"
ORCHESTRATION_SHA = "b" * 40
OLD_TAG = f"paper-matrix-v2-h600-s30-2026-09-{SOURCE_SHA}"
NEW_TAG = f"{OLD_TAG}-erratum.1"
ARCHIVE_NAME = "fixture-publication-bundle.tar.gz"


def validate_erratum_receipt_against_campaign(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep legacy fixture calls explicit about their deterministic archive name."""
    kwargs.setdefault("archive_name", ARCHIVE_NAME)
    return _validate_erratum_receipt(*args, **kwargs)


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


def test_publication_aliases_accept_split_identity_and_reject_incomplete_shapes(
    tmp_path: Path,
) -> None:
    """Current coordinates may span provenance, but none may be absent or malformed."""
    archive = tmp_path / "predecessor.tar.gz"
    archive.write_bytes(b"fixture")
    contract = _contract(archive)
    split = {
        "release_tag": contract.successor_github_release_tag,
        "provenance": {
            "version_doi": contract.successor_version_doi,
            "concept_doi": contract.concept_doi,
        },
    }
    _assert_publication_aliases(split, contract=contract, label="split")

    invalid = (
        ({"release_tag": contract.successor_github_release_tag, "provenance": "bad"}, "object"),
        (
            {
                "provenance": {
                    "version_doi": contract.successor_version_doi,
                    "concept_doi": contract.concept_doi,
                }
            },
            "release-tag",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "provenance": {"concept_doi": contract.concept_doi},
            },
            "version-DOI",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "provenance": {"version_doi": contract.successor_version_doi},
            },
            "concept-DOI",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "provenance": {
                    "version_doi": contract.successor_version_doi,
                    "concept_doi": contract.concept_doi,
                    "provenance": {"version_doi": contract.predecessor_version_doi},
                },
            },
            "nested provenance",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "version_doi": contract.successor_version_doi,
                "concept_doi": contract.concept_doi,
                "publication": "malformed",
            },
            "publication must be an object",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "version_doi": contract.successor_version_doi,
                "concept_doi": contract.concept_doi,
                "publication": {
                    "release_tag": contract.predecessor_github_release_tag,
                    "version_doi": contract.successor_version_doi,
                    "concept_doi": contract.concept_doi,
                    "predecessor_version_doi": contract.predecessor_version_doi,
                },
            },
            "publication contains a stale release-tag alias",
        ),
        (
            {
                "release_tag": contract.successor_github_release_tag,
                "version_doi": contract.successor_version_doi,
                "concept_doi": contract.concept_doi,
                "publication": {
                    "release_tag": contract.successor_github_release_tag,
                    "version_doi": contract.successor_version_doi,
                    "concept_doi": contract.concept_doi,
                    "predecessor_version_doi": contract.predecessor_version_doi,
                    "provenance": {"provenance": {"version_doi": contract.predecessor_version_doi}},
                },
            },
            "publication.provenance contains unsupported nested provenance",
        ),
    )
    for payload, message in invalid:
        with pytest.raises(ReleaseErratumError, match=message):
            _assert_publication_aliases(payload, contract=contract, label="invalid")


def test_predecessor_aliases_accept_split_identity_and_reject_incomplete_shapes(
    tmp_path: Path,
) -> None:
    """Preserved execution coordinates remain complete across root and provenance."""
    archive = tmp_path / "predecessor.tar.gz"
    archive.write_bytes(b"fixture")
    contract = _contract(archive)
    split = {
        "release_tag": contract.predecessor_github_release_tag,
        "provenance": {
            "version_doi": contract.predecessor_version_doi,
            "concept_doi": contract.concept_doi,
        },
    }
    _assert_predecessor_execution_aliases(
        split,
        contract=contract,
        label="split",
        require_concept=True,
    )

    invalid = (
        (
            {"release_tag": contract.predecessor_github_release_tag, "provenance": "bad"},
            "object",
        ),
        (
            {
                "provenance": {
                    "version_doi": contract.predecessor_version_doi,
                    "concept_doi": contract.concept_doi,
                }
            },
            "tag alias",
        ),
        (
            {
                "release_tag": contract.predecessor_github_release_tag,
                "provenance": {"concept_doi": contract.concept_doi},
            },
            "DOI alias",
        ),
        (
            {
                "release_tag": contract.predecessor_github_release_tag,
                "provenance": {"version_doi": contract.predecessor_version_doi},
            },
            "concept DOI",
        ),
        (
            {
                "release_tag": contract.predecessor_github_release_tag,
                "provenance": {
                    "version_doi": contract.predecessor_version_doi,
                    "concept_doi": contract.concept_doi,
                    "provenance": {"version_doi": contract.successor_version_doi},
                },
            },
            "nested provenance",
        ),
    )
    for payload, message in invalid:
        with pytest.raises(ReleaseErratumError, match=message):
            _assert_predecessor_execution_aliases(
                payload,
                contract=contract,
                label="invalid",
                require_concept=True,
            )


def _predecessor_evidence(archive: Path, contract: ErratumContract) -> PredecessorEvidence:
    return PredecessorEvidence(
        archive_path=archive,
        version_doi=contract.predecessor_version_doi,
        concept_doi=contract.concept_doi,
        github_release_tag=contract.predecessor_github_release_tag,
        archive_sha256=contract.predecessor_archive_sha256,
        archive_size_bytes=contract.predecessor_archive_size_bytes,
    )


def _with_bundle_metadata(
    campaign: Path,
    contract: ErratumContract,
    *,
    archive_name: str = ARCHIVE_NAME,
) -> ErratumContract:
    metadata = campaign / "release/zenodo_metadata.erratum.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    copied_metadata = campaign / "release_metadata/zenodo_metadata.json"
    copied_metadata.parent.mkdir(parents=True, exist_ok=True)
    copied_metadata.write_bytes(metadata.read_bytes())
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
        "source_sha": updated.source_sha,
        "source_commit": updated.source_sha,
        "erratum_builder_sha": updated.builder_sha,
        "erratum_validator_sha": updated.validator_sha,
        "erratum_orchestration_sha": updated.orchestration_sha,
    }
    execution = {
        "release_tag": updated.predecessor_github_release_tag,
        "release_id": updated.predecessor_github_release_tag,
        "doi": updated.predecessor_version_doi,
        "version_doi": updated.predecessor_version_doi,
        "concept_doi": updated.concept_doi,
    }
    current = {
        "release_tag": updated.successor_github_release_tag,
        "release_id": updated.successor_github_release_tag,
        "doi": updated.successor_version_doi,
        "version_doi": updated.successor_version_doi,
        "concept_doi": updated.concept_doi,
        "source_sha": updated.source_sha,
        "source_commit": updated.source_sha,
        "scientific_source_sha": updated.source_sha,
        "provenance": provenance,
        "publication": {
            "release_tag": updated.successor_github_release_tag,
            "release_id": updated.successor_github_release_tag,
            "doi": updated.successor_version_doi,
            "source_sha": updated.source_sha,
            "source_commit": updated.source_sha,
            "scientific_source_sha": updated.source_sha,
            "concept_doi": updated.concept_doi,
            "version_doi": updated.successor_version_doi,
            "predecessor_version_doi": updated.predecessor_version_doi,
            "bundle_metadata_path": "release/zenodo_metadata.erratum.json",
            "metadata_sha256": updated.metadata_sha256,
            "correction_scope": "derived_publication_metadata_only",
        },
        "erratum": {
            "correction_id": updated.correction_id,
            "correction_scope": "derived_publication_metadata_only",
            "predecessor_version_doi": updated.predecessor_version_doi,
            "predecessor_github_release_tag": updated.predecessor_github_release_tag,
            "concept_doi": updated.concept_doi,
            "source_sha": updated.source_sha,
            "scientific_source_unchanged": True,
            "simulation_rerun": False,
        },
        "release_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/tag/"
            f"{updated.successor_github_release_tag}"
        ),
        "release_asset_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            f"{updated.successor_github_release_tag}/{archive_name}"
        ),
        "doi_url": f"https://doi.org/{updated.successor_version_doi}",
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
                "scientific_execution_benchmark_release": execution,
                "scientific_execution_resolved_manifest": execution,
                "derivation": {
                    "builder_sha": updated.builder_sha,
                    "validator_sha": updated.validator_sha,
                    "orchestration_sha": updated.orchestration_sha,
                    "scientific_source_sha": updated.source_sha,
                    "simulation_rerun": False,
                    "correction_id": updated.correction_id,
                    "predecessor_version_doi": updated.predecessor_version_doi,
                },
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
        json.dumps(
            {
                "benchmark_release": current,
                "campaign": {
                    **current,
                    "scientific_execution_release_identity": {
                        "release_tag": updated.predecessor_github_release_tag,
                        "doi": updated.predecessor_version_doi,
                        "source_sha": updated.source_sha,
                    },
                },
                "publication_erratum": {
                    "correction_id": updated.correction_id,
                    "correction_scope": "derived_publication_metadata_only",
                    "predecessor_version_doi": updated.predecessor_version_doi,
                    "predecessor_github_release_tag": updated.predecessor_github_release_tag,
                    "concept_doi": updated.concept_doi,
                    "source_sha": updated.source_sha,
                    "scientific_source_unchanged": True,
                    "simulation_rerun": False,
                },
            }
        ),
        encoding="utf-8",
    )
    derived_receipt = campaign / "provenance/derived_revalidation_receipt.json"
    derived_receipt.parent.mkdir(parents=True, exist_ok=True)
    derived_receipt.write_text(
        json.dumps(
            {
                "schema_version": "benchmark-derived-revalidation.v1",
                "source": {"execution_commit": updated.source_sha},
                "validator": {"commit": updated.validator_sha},
            }
        ),
        encoding="utf-8",
    )
    return updated


def _write_receipt(campaign: Path, contract: ErratumContract) -> Path:
    """Write a self-equality receipt for the campaign fixture."""
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(contract=contract, predecessor=snapshot, successor=snapshot)
        ),
        encoding="utf-8",
    )
    return receipt_path


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
    assert receipt["scientific_identity"]["episode_file_sha256"] == dict(
        successor.episode_file_sha256
    )
    assert receipt["scientific_equality"]["episode_file_bytes_equal"] is True
    assert receipt["derivation"] == {
        "builder_sha": BUILDER_SHA,
        "validator_sha": BUILDER_SHA,
        "orchestration_sha": ORCHESTRATION_SHA,
        "scientific_source_sha": SOURCE_SHA,
        "simulation_rerun": False,
    }
    assert receipt["corrected_verdict"]["ranking_claims_admitted"] is False
    assert receipt["scientific_canonicalization"]["schema"] == SCIENTIFIC_CANONICALIZATION
    assert receipt["scientific_canonicalization"] == dict(SCIENTIFIC_CANONICALIZATION_POLICY)


def test_cold_erratum_accepts_exact_root_and_nested_publication_url_aliases(
    tmp_path: Path,
) -> None:
    """Every exact current-publication URL alias may appear at any payload depth."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    archive_name = ARCHIVE_NAME
    contract = _with_bundle_metadata(
        campaign,
        _contract(predecessor_archive),
        archive_name=archive_name,
    )
    receipt_path = _write_receipt(campaign, contract)
    urls = {
        "release_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/tag/"
            f"{contract.successor_github_release_tag}"
        ),
        "release_asset_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            f"{contract.successor_github_release_tag}/{archive_name}"
        ),
        "doi_url": f"https://doi.org/{contract.successor_version_doi}",
    }

    result_path = campaign / "release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(urls)
    result["benchmark_release"]["publication_links"] = dict(urls)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary_path = campaign / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["publication_links"] = dict(urls)
    summary["publication_links"] = dict(urls)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    observed = validate_erratum_receipt_against_campaign(
        receipt_path,
        campaign_root=campaign,
        metadata_path=contract.metadata_path,
        predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
        archive_name=archive_name,
        expected_tag=NEW_TAG,
        expected_doi=contract.successor_version_doi,
    )

    assert observed["status"] == "pass"


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        (
            "root",
            "release_url",
            "https://github.com/attacker/robot_sf_ll7/releases/tag/decoy",
        ),
        (
            "nested",
            "release_url",
            "https://github.com/ll7/robot_sf_ll7/releases%2Ftag/decoy",
        ),
        (
            "root",
            "release_asset_url",
            "https://github.com/ll7/robot_sf_ll7/releases/download/decoy-bundle.tar.gz",
        ),
        (
            "nested",
            "release_asset_url",
            "https://user@github.com/ll7/robot_sf_ll7/releases/download/decoy/bundle.tar.gz",
        ),
        (
            "root",
            "doi_url",
            "https://doi.org/10.5281/zenodo.22229999?download=1",
        ),
        (
            "nested",
            "doi_url",
            "https://doi.org/10.5281/zenodo.22229999#fragment",
        ),
    ],
)
def test_cold_erratum_rejects_decoy_or_ambiguous_publication_url_alias(
    tmp_path: Path,
    location: str,
    key: str,
    value: str,
) -> None:
    """URL aliases must not accept decoys, prefixes, credentials, or URL syntax tricks."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    archive_name = ARCHIVE_NAME
    contract = _with_bundle_metadata(
        campaign,
        _contract(predecessor_archive),
        archive_name=archive_name,
    )
    receipt_path = _write_receipt(campaign, contract)
    urls = {
        "release_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/tag/"
            f"{contract.successor_github_release_tag}"
        ),
        "release_asset_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            f"{contract.successor_github_release_tag}/{archive_name}"
        ),
        "doi_url": f"https://doi.org/{contract.successor_version_doi}",
    }
    result_path = campaign / "release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(urls)
    result["nested"] = dict(urls)
    if location == "root":
        result[key] = value
    else:
        result["nested"][key] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=key):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            archive_name=archive_name,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_rejects_decoy_metadata_publication_url_alias(tmp_path: Path) -> None:
    """The canonical Zenodo metadata payload is covered by URL-alias validation too."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    metadata = json.loads(contract.metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"]["release_url"] = "https://github.com/attacker/decoy"
    contract.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    contract = replace(
        contract,
        metadata_sha256=hashlib.sha256(contract.metadata_path.read_bytes()).hexdigest(),
    )
    (campaign / "release_metadata/zenodo_metadata.json").write_bytes(
        contract.metadata_path.read_bytes()
    )
    for relative in (
        "release/release_manifest.resolved.json",
        "release/release_result.json",
        "reports/campaign_summary.json",
    ):
        path = campaign / relative
        payload = json.loads(path.read_text(encoding="utf-8"))

        def refresh_digest(value: Any) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key == "metadata_sha256":
                        value[key] = contract.metadata_sha256
                    else:
                        refresh_digest(nested)
            elif isinstance(value, list):
                for nested in value:
                    refresh_digest(nested)

        refresh_digest(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path = _write_receipt(campaign, contract)

    with pytest.raises(ReleaseErratumError, match="release_url"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            archive_name=ARCHIVE_NAME,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_direct_cold_erratum_requires_the_downloaded_archive_name(tmp_path: Path) -> None:
    """Direct callers cannot validate an asset URL without its exact filename."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)

    with pytest.raises(ReleaseErratumError, match="archive filename is required"):
        _validate_erratum_receipt(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_requires_all_successor_publication_url_evidence(tmp_path: Path) -> None:
    """A direct cold audit must observe all three current-publication URLs."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)
    summary_path = campaign / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("release_url", "release_asset_url", "doi_url"):
        summary["campaign"].pop(key)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="missing canonical publication URL evidence"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_rejects_tampered_release_metadata_copy(tmp_path: Path) -> None:
    """Inventory checksums cannot authorize a copied metadata document mismatch."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)
    copied_path = campaign / "release_metadata/zenodo_metadata.json"
    copied = json.loads(copied_path.read_text(encoding="utf-8"))
    copied["metadata"]["title"] = "attacker-controlled metadata"
    copied_path.write_text(json.dumps(copied), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="metadata copy differs"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_accepts_predecessor_publication_url_coordinates(tmp_path: Path) -> None:
    """Preserved execution URLs use predecessor coordinates, not successor ones."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)
    predecessor_urls = {
        "release_url": (
            f"https://github.com/ll7/robot_sf_ll7/releases/tag/"
            f"{contract.predecessor_github_release_tag}"
        ),
        "release_asset_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            f"{contract.predecessor_github_release_tag}/old-publication-bundle.tar.gz"
        ),
        "doi_url": f"https://doi.org/{contract.predecessor_version_doi}",
    }
    result_path = campaign / "release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for key in (
        "scientific_execution_benchmark_release",
        "scientific_execution_resolved_manifest",
    ):
        result[key].update(predecessor_urls)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary_path = campaign / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["scientific_execution_release_identity"].update(predecessor_urls)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    observed = validate_erratum_receipt_against_campaign(
        receipt_path,
        campaign_root=campaign,
        metadata_path=contract.metadata_path,
        predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
        expected_tag=NEW_TAG,
        expected_doi=contract.successor_version_doi,
    )

    assert observed["status"] == "pass"


def test_publication_url_scan_rechecks_shared_mapping_in_each_context(tmp_path: Path) -> None:
    """One Python mapping under both contexts must satisfy neither contract silently."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    shared = {
        "release_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/tag/"
            f"{contract.predecessor_github_release_tag}"
        ),
        "release_asset_url": (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            f"{contract.predecessor_github_release_tag}/old-publication-bundle.tar.gz"
        ),
        "doi_url": f"https://doi.org/{contract.predecessor_version_doi}",
    }
    payload = {
        "publication_links": shared,
        "scientific_execution_release_identity": shared,
    }

    with pytest.raises(ReleaseErratumError, match="requested release"):
        _assert_publication_url_aliases(
            payload,
            contract=contract,
            label="shared payload",
            archive_name=ARCHIVE_NAME,
        )


def test_publication_url_scan_rejects_cycles_with_typed_error(tmp_path: Path) -> None:
    """Direct Python callers receive a bounded typed failure for cyclic payloads."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    with pytest.raises(ReleaseErratumError, match="cyclic identity payload"):
        _assert_publication_url_aliases(
            cyclic,
            contract=contract,
            label="cyclic payload",
            archive_name=ARCHIVE_NAME,
        )


def test_publication_url_scan_rejects_excessive_depth_with_typed_error(
    tmp_path: Path,
) -> None:
    """Direct Python callers receive a bounded typed failure for deep payloads."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    payload: dict[str, Any] = {}
    nested = payload
    for _ in range(_MAX_IDENTITY_TRAVERSAL_DEPTH + 1):
        child: dict[str, Any] = {}
        nested["nested"] = child
        nested = child

    with pytest.raises(ReleaseErratumError, match="maximum identity traversal depth"):
        _assert_publication_url_aliases(
            payload,
            contract=contract,
            label="deep payload",
            archive_name=ARCHIVE_NAME,
        )


@pytest.mark.parametrize(
    "field",
    ("predecessor_version_doi", "concept_doi", "successor_version_doi"),
)
@pytest.mark.parametrize("record_id", ("0", "022227035"))
def test_erratum_contract_rejects_noncanonical_zenodo_record_ids(
    tmp_path: Path, field: str, record_id: str
) -> None:
    """Zenodo DOI record components are positive canonical decimal strings."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = replace(
        _contract(predecessor_archive),
        **{field: f"10.5281/zenodo.{record_id}"},
    )

    with pytest.raises(ReleaseErratumError, match="DOIs must be valid and distinct"):
        validate_erratum_contract_identity(contract)


@pytest.mark.parametrize("successor_doi", ("10.5281/zenodo.0", "10.5281/zenodo.022229999"))
def test_cold_erratum_rejects_noncanonical_successor_doi_coordinate(
    tmp_path: Path, successor_doi: str
) -> None:
    """Receipt contracts reject zero and leading-zero successor DOI spellings."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["successor"]["version_doi"] = successor_doi
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="DOI"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=successor_doi,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nonfinite_float_policy", "drop_nonfinite"),
        ("finite_float_policy", "decimal_string"),
        ("unexpected_policy_claim", "accepted"),
    ],
)
def test_cold_erratum_rejects_mutable_scientific_canonicalization_policy(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """A matching canonicalization schema cannot hide a changed policy claim."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    predecessor_archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, predecessor_archive)
    contract = _with_bundle_metadata(campaign, _contract(predecessor_archive))
    receipt_path = _write_receipt(campaign, contract)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scientific_canonicalization"][field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="canonicalization policy"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(predecessor_archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_scientific_equality_rejects_byte_different_episode_file_with_same_rows(
    tmp_path: Path,
) -> None:
    """Whitespace-only JSONL rewrites cannot pass a metadata-only erratum."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)
    predecessor = snapshot_predecessor_archive(archive, contract=contract)

    path = campaign / "runs" / "goal__differential_drive" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    successor = snapshot_campaign(campaign, contract=contract)

    assert predecessor.canonical_row_manifest_sha256 == successor.canonical_row_manifest_sha256
    assert predecessor.episode_file_sha256 != successor.episode_file_sha256
    with pytest.raises(ReleaseErratumError, match="scientific leaves differ"):
        compare_scientific_snapshots(predecessor, successor)


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


def test_scientific_snapshot_rejects_duplicate_matrix_cell_with_distinct_episode_id(
    tmp_path: Path,
) -> None:
    """A second episode ID cannot replace a missing arm/scenario/seed cell."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    episodes = campaign / "runs/goal__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    rows[0]["scenario_id"] = rows[1]["scenario_id"]
    rows[0]["seed"] = rows[1]["seed"]
    rows[0]["episode_id"] = f"{rows[1]['episode_id']}--duplicate-cell"
    episodes.write_text("".join(f"{_canonical_json(row)}\n" for row in rows), encoding="utf-8")
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _contract(archive)

    with pytest.raises(ReleaseErratumError, match="duplicate scientific arm/scenario/seed cell"):
        snapshot_campaign(campaign, contract=contract)
    with pytest.raises(ReleaseErratumError, match="duplicate scientific arm/scenario/seed cell"):
        snapshot_predecessor_archive(archive, contract=contract)


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
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    observed = validate_erratum_receipt_against_campaign(
        receipt_path,
        campaign_root=campaign,
        metadata_path=contract.metadata_path,
        predecessor_evidence=_predecessor_evidence(archive, contract),
        expected_tag=NEW_TAG,
        expected_doi=contract.successor_version_doi,
    )

    assert observed["status"] == "pass"
    assert observed["episode_rows"] == 8
    assert observed["canonical_row_manifest_sha256"] == snapshot.canonical_row_manifest_sha256
    assert observed["predecessor"] == {
        "version_doi": contract.predecessor_version_doi,
        "concept_doi": contract.concept_doi,
        "github_release_tag": contract.predecessor_github_release_tag,
        "archive_sha256": contract.predecessor_archive_sha256,
        "archive_size_bytes": contract.predecessor_archive_size_bytes,
        "scientific_identity": snapshot.public_dict(),
    }
    assert observed["successor"] == {
        "version_doi": contract.successor_version_doi,
        "concept_doi": contract.concept_doi,
        "github_release_tag": contract.successor_github_release_tag,
        "scientific_identity": snapshot.public_dict(),
    }
    assert observed["scientific_equality"] == receipt["scientific_equality"]
    assert "archive_path" not in json.dumps(observed)


def test_cold_erratum_requires_explicit_predecessor_evidence(tmp_path: Path) -> None:
    """A self-equality receipt cannot pass without a detached predecessor archive."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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

    with pytest.raises(ReleaseErratumError, match="predecessor evidence is required"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("version_doi", None, "version DOI"),
        ("version_doi", "10.5281/zenodo.999991", "version DOI"),
        ("concept_doi", None, "concept DOI"),
        ("concept_doi", "10.5281/zenodo.999992", "concept DOI"),
        ("github_release_tag", None, "GitHub release tag"),
        ("github_release_tag", "wrong-predecessor-tag", "GitHub release tag"),
        ("archive_sha256", None, "archive SHA-256"),
        ("archive_sha256", "0" * 64, "archive SHA-256"),
        ("archive_size_bytes", None, "archive size"),
        ("archive_size_bytes", 1, "archive size"),
    ],
)
def test_cold_erratum_rejects_predecessor_evidence_field_drift(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    """Every predecessor custody coordinate must agree with the embedded receipt."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
    evidence = replace(_predecessor_evidence(archive, contract), **{field: value})

    with pytest.raises(ReleaseErratumError, match=match):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=evidence,
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_rejects_mutated_predecessor_archive(tmp_path: Path) -> None:
    """Changing the detached predecessor after receipt creation fails custody first."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
    original = archive.read_bytes()
    archive.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(ReleaseErratumError, match="SHA-256"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_rejects_fabricated_self_equality(tmp_path: Path) -> None:
    """Receipt self-equality cannot hide different predecessor episode bytes."""
    campaign = tmp_path / "campaign"
    predecessor_campaign = tmp_path / "predecessor-campaign"
    _write_campaign(campaign)
    _write_campaign(predecessor_campaign)
    predecessor_episode = predecessor_campaign / "runs/goal__differential_drive/episodes.jsonl"
    rows = [
        json.loads(line) for line in predecessor_episode.read_text(encoding="utf-8").splitlines()
    ]
    predecessor_episode.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(predecessor_campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    successor = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(
                contract=contract,
                predecessor=successor,
                successor=successor,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseErratumError, match="scientific leaves differ"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_receipt_rejects_tampered_successor_row(tmp_path: Path) -> None:
    """A valid-looking receipt cannot mask a changed downloaded component leaf."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


def test_cold_erratum_receipt_rejects_byte_different_successor_episode_file(
    tmp_path: Path,
) -> None:
    """Cold intake rejects formatting drift even when canonical rows still match."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
    episodes.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseErratumError, match="differ from its receipt"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
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
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=match):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
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
    canonical_receipt.parent.mkdir(parents=True, exist_ok=True)
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
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )
    with pytest.raises(ReleaseErratumError, match="metadata is outside"):
        validate_erratum_receipt_against_campaign(
            canonical_receipt,
            campaign_root=campaign,
            metadata_path=external_metadata,
            predecessor_evidence=_predecessor_evidence(archive, contract),
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
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("nested_publication", "publication contains a stale version-DOI alias"),
        ("execution_provenance", "provenance contains a stale predecessor DOI"),
        ("derived_validator", "derived revalidation receipt identity is stale"),
        ("summary_source", "stale scientific source SHA"),
        ("optional_publication", "publication contains a stale release-tag alias"),
        ("optional_publication_null", "publication must be an object"),
    ],
)
def test_cold_erratum_receipt_rejects_nested_identity_drift(
    tmp_path: Path, fault: str, match: str
) -> None:
    """Cold intake checks nested publication, execution, and builder identities."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
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
    if fault in {"nested_publication", "execution_provenance"}:
        result_path = campaign / "release/release_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if fault == "nested_publication":
            result["benchmark_release"]["publication"]["version_doi"] = (
                contract.predecessor_version_doi
            )
        else:
            result["scientific_execution_benchmark_release"]["provenance"] = {
                "version_doi": contract.successor_version_doi
            }
        result_path.write_text(json.dumps(result), encoding="utf-8")
    elif fault == "derived_validator":
        derived_path = campaign / "provenance/derived_revalidation_receipt.json"
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        derived["validator"]["commit"] = "0" * 40
        derived_path.write_text(json.dumps(derived), encoding="utf-8")
    elif fault == "summary_source":
        summary_path = campaign / "reports/campaign_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["campaign"]["scientific_execution_release_identity"]["source_sha"] = "0" * 40
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        manifest = json.loads(
            (campaign / "release/release_manifest.resolved.json").read_text(encoding="utf-8")
        )
        result = json.loads((campaign / "release/release_result.json").read_text(encoding="utf-8"))
        optional = {
            **manifest,
            "benchmark_release": dict(manifest),
            "scientific_execution_benchmark_release": result[
                "scientific_execution_benchmark_release"
            ],
            "publication_erratum": manifest["erratum"],
        }
        if fault == "optional_publication":
            optional["benchmark_release"]["publication"]["release_tag"] = (
                contract.predecessor_github_release_tag
            )
        else:
            optional["publication"] = None
        (campaign / "run_meta.json").write_text(json.dumps(optional), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=match):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


@pytest.mark.parametrize("location", ["root", "provenance", "publication"])
def test_cold_erratum_receipt_rejects_stale_current_source_alias(
    tmp_path: Path, location: str
) -> None:
    """Every current-publication source alias remains bound to the source SHA."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(contract=contract, predecessor=snapshot, successor=snapshot)
        ),
        encoding="utf-8",
    )
    result_path = campaign / "release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if location == "root":
        result["source_commit"] = "0" * 40
    elif location == "provenance":
        result["benchmark_release"]["provenance"]["scientific_source_sha"] = "0" * 40
    else:
        result["resolved_manifest"]["publication"]["source_sha"] = "0" * 40
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="stale scientific source SHA"):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
            expected_tag=NEW_TAG,
            expected_doi=contract.successor_version_doi,
        )


@pytest.mark.parametrize(
    ("document", "field", "value", "match"),
    [
        ("manifest", "correction_scope", "simulation_rows_changed", "erratum.correction_scope"),
        ("summary", "concept_doi", "10.5281/zenodo.99999999", "publication_erratum.concept_doi"),
        ("summary", "simulation_rerun", True, "publication_erratum.simulation_rerun"),
    ],
)
def test_cold_erratum_receipt_rejects_invalid_erratum_identity_block(
    tmp_path: Path, document: str, field: str, value: object, match: str
) -> None:
    """Correction blocks cannot contradict the immutable erratum contract."""
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    archive = tmp_path / "old.tar.gz"
    _archive_campaign(campaign, archive)
    contract = _with_bundle_metadata(campaign, _contract(archive))
    snapshot = snapshot_campaign(campaign, contract=contract)
    receipt_path = campaign / "provenance/benchmark_release_erratum.json"
    receipt_path.write_text(
        json.dumps(
            build_erratum_receipt(contract=contract, predecessor=snapshot, successor=snapshot)
        ),
        encoding="utf-8",
    )
    if document == "manifest":
        path = campaign / "release/release_manifest.resolved.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["erratum"][field] = value
    else:
        path = campaign / "reports/campaign_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["publication_erratum"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=match):
        validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=campaign,
            metadata_path=contract.metadata_path,
            predecessor_evidence=_predecessor_evidence(archive, contract),
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


def _contract_document(
    metadata_path: Path,
    *,
    builder_sha: str = BUILDER_SHA,
    validator_sha: str = BUILDER_SHA,
) -> dict[str, Any]:
    """Return a checked-in contract payload for file-boundary tests."""
    return {
        "schema_version": "benchmark-release-erratum.v1",
        "correction_id": "september-2026-derived-metadata-erratum.1",
        "correction_scope": "derived_publication_metadata_only",
        "supersedes": {
            "version_doi": "10.5281/zenodo.22227035",
            "archive_sha256": "e" * 64,
            "archive_size_bytes": 54_219_004,
            "github_release_tag": OLD_TAG,
            "old_publication_retained": True,
        },
        "scientific_identity": {
            "source_sha": SOURCE_SHA,
            "planner_arms": 14,
            "scenario_count": 48,
            "seed_count": 30,
            "episode_rows": 20_160,
        },
        "derivation": {
            "builder_sha": builder_sha,
            "validator_sha": validator_sha,
            "orchestration_sha": ORCHESTRATION_SHA,
            "simulation_rerun": False,
        },
        "successor": {
            "concept_doi": "10.5281/zenodo.22227034",
            "version_doi": "10.5281/zenodo.22229999",
            "github_release_tag": NEW_TAG,
            "metadata_path": metadata_path.name,
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
        "corrected_verdict": {
            "publication_preflight_status": "pass",
            "publication_preflight_violations": [],
            "release_status": "ok",
            "ranking_claims_admitted": False,
        },
    }


@pytest.mark.parametrize(
    ("relation_index", "scheme", "match"),
    [
        (0, "doi", "successor GitHub tag"),
        (1, "url", "predecessor version DOI"),
    ],
)
def test_erratum_contract_requires_exact_relation_schemes(
    tmp_path: Path, relation_index: int, scheme: str, match: str
) -> None:
    metadata_payload = _metadata()
    metadata_payload["metadata"]["related_identifiers"][relation_index]["scheme"] = scheme
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(metadata_payload), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract_document(metadata)), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match=match):
        load_erratum_contract(contract_path, repository_root=tmp_path)


def test_erratum_contract_rejects_multiple_predecessor_relations(tmp_path: Path) -> None:
    metadata_payload = _metadata()
    predecessor = metadata_payload["metadata"]["related_identifiers"][1]
    metadata_payload["metadata"]["related_identifiers"].append(dict(predecessor))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(metadata_payload), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract_document(metadata)), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="exactly one predecessor version DOI"):
        load_erratum_contract(contract_path, repository_root=tmp_path)


def test_erratum_contract_requires_one_accepted_builder_validator_commit(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_contract_document(metadata, validator_sha="c" * 40)), encoding="utf-8"
    )

    with pytest.raises(ReleaseErratumError, match="same accepted commit"):
        load_erratum_contract(contract_path, repository_root=tmp_path)


def test_erratum_contract_rejects_contract_outside_repository_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract_document(metadata)), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="outside the repository root"):
        load_erratum_contract(contract_path, repository_root=repository_root)


def test_erratum_contract_rejects_symlinked_metadata(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_metadata = outside / "metadata.json"
    outside_metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    metadata_link = repository_root / "metadata.json"
    metadata_link.symlink_to(outside_metadata)
    contract_path = repository_root / "contract.json"
    contract_path.write_text(json.dumps(_contract_document(metadata_link)), encoding="utf-8")

    with pytest.raises(ReleaseErratumError, match="metadata_path contains a symlink"):
        load_erratum_contract(contract_path, repository_root=repository_root)


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
