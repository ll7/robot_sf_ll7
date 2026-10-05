"""D-087: authentic F2 publication identity, synthetic rows and offline readbacks."""
# robot-sf-test-lane: fast -- static JSON/archive proof; no simulation or HTTP

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from robot_sf.benchmark import release_erratum as erratum
from robot_sf.benchmark import zenodo_publisher as publisher
from robot_sf.benchmark.release_protocol import _replace_identity_tokens
from robot_sf.evidence.writers import write_json, write_review_sidecar
from tests.benchmark import test_release_erratum as fixtures
from tests.benchmark.test_zenodo_publisher_edge_cases import _Response, _Session

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / (
    "docs/context/evidence/2026-10-05_freeze008_box4_seed_admission/main/"
    "release_identity.resolved.json"
)
PLAN = ROOT / "docs/release/0.0.8/planned_main_zenodo_metadata_successor.json"
F2 = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
TOOLING = "f5782b7ea79d6e52ed68ad86b0d0a4318f1d1a67"
SYNTHETIC_DOI = "10.5281/zenodo.99999999"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preserved_bytes(path: Path, data: bytes) -> None:
    """Mark exact JSON/JSONL fixture bytes via the shared preserving sidecar writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    write_review_sidecar(path, repo_root=path.parent)


class _ReadOnlyZenodo(_Session):
    """Reuse the HTTP double, refusing mutations even if production code regresses."""

    def post(self, *_args, **_kwargs):
        raise AssertionError("offline erratum test permits GET only")

    def put(self, *_args, **_kwargs):
        raise AssertionError("offline erratum test permits GET only")

    def delete(self, *_args, **_kwargs):
        raise AssertionError("offline erratum test permits GET only")


@pytest.fixture(scope="module")
def correction(tmp_path_factory):
    """Build full-sized fabricated custody without constructing or stepping a planner."""
    root = tmp_path_factory.mktemp("f2-erratum")
    assert _sha(IDENTITY) == "527f9dc5e9ee3004e93444789472eb25a29438e64741c4c4b7b95a10f0db3e71"
    identity = json.loads(IDENTITY.read_bytes())
    manifest = identity["resolved_manifest"]
    assert manifest["schema_version"] == "benchmark-release-manifest.v0.2"
    assert identity["source_commit"] == F2
    predecessor = root / "predecessor"
    for planner in manifest["planners"]["keys"]:
        rows = [
            {
                "algo": planner,
                "scenario_id": f"synthetic-{scenario:02}",
                "seed": seed,  # Static artifact identity only; never used for an environment.
                "episode_id": f"synthetic-{scenario:02}-{seed}",
                "status": "success",
                "git_hash": F2,
                "provenance": {"git_hash": F2},
                "result_provenance": {"repo_commit": F2},
                "event_ledger": {"software_commit": F2},
                "metrics": {"collisions": 0, "snqi_v2": 0.5},
            }
            for scenario in range(48)
            for seed in manifest["release_contract"]["resolved_seeds"]
        ]
        _write_preserved_bytes(
            predecessor / "runs" / f"{planner}__differential_drive" / "episodes.jsonl",
            ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode(),
        )
    _write_preserved_bytes(
        predecessor / "release/release_manifest.resolved.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    archive = root / "synthetic-published-predecessor.tar.gz"
    fixtures._archive_campaign(predecessor, archive)
    write_review_sidecar(archive, repo_root=root)
    successor = root / "successor"
    shutil.copytree(predecessor, successor)
    tag = identity["release_tag"] + "-erratum.1"
    planned = json.loads(PLAN.read_bytes())
    assert planned["status"] == "planned-not-published"
    metadata = {
        "metadata": _replace_identity_tokens(
            planned["metadata"],
            {
                "source_sha": F2,
                "latest_main_base_commit": manifest["latest_main_base_commit"],
                "release_tag": tag,
                "concept_doi": identity["publication"]["concept_doi"],
                "version_doi": SYNTHETIC_DOI,
            },
        )
    }
    contract = replace(
        fixtures._contract(archive),
        correction_id="0.0.8-description-erratum.1",
        predecessor_version_doi=identity["publication"]["version_doi"],
        predecessor_github_release_tag=identity["release_tag"],
        source_sha=F2,
        scientific_release_id=manifest["release_id"],
        planner_arms=14,
        scenario_count=48,
        seed_count=30,
        episode_rows=20160,
        builder_sha=TOOLING,
        validator_sha=TOOLING,
        orchestration_sha=TOOLING,
        concept_doi=identity["publication"]["concept_doi"],
        successor_version_doi=SYNTHETIC_DOI,
        successor_github_release_tag=tag,
    )
    with patch.object(fixtures, "_metadata", lambda: copy.deepcopy(metadata)):
        contract = fixtures._with_bundle_metadata(successor, contract)
    before = erratum.snapshot_predecessor_archive(archive, contract=contract)
    after = erratum.snapshot_campaign(successor, contract=contract)
    receipt = erratum.build_erratum_receipt(contract=contract, predecessor=before, successor=after)
    path = successor / "provenance/benchmark_release_erratum.json"
    write_json(path, receipt)
    return root, identity, successor, archive, contract, path


def _cold_validate(correction):
    _root, _identity, successor, archive, contract, receipt = correction
    return erratum.validate_erratum_receipt_against_campaign(
        receipt,
        campaign_root=successor,
        metadata_path=contract.metadata_path,
        predecessor_evidence=fixtures._predecessor_evidence(archive, contract),
        expected_tag=contract.successor_github_release_tag,
        expected_doi=contract.successor_version_doi,
        expected_source_sha=F2,
        expected_concept_doi=contract.concept_doi,
        expected_predecessor_doi=contract.predecessor_version_doi,
        expected_predecessor_tag=contract.predecessor_github_release_tag,
        expected_predecessor_archive_sha256=_sha(archive),
        expected_predecessor_archive_size_bytes=archive.stat().st_size,
        expected_builder_sha=TOOLING,
        expected_validator_sha=TOOLING,
        expected_orchestration_sha=TOOLING,
        archive_name=fixtures.ARCHIVE_NAME,
    )


def test_f2_metadata_only_successor_accepts_real_manifest_and_offline_readback(correction):
    """The actual F2 archival schema/source/DOIs support unchanged-row derivation."""
    root, _identity, successor, _archive, contract, _receipt = correction
    cold = _cold_validate(correction)
    assert cold["status"] == "pass"
    assert cold["episode_rows"] == 20160
    assert cold["scientific_equality"]["episode_file_bytes_equal"] is True
    bundle = root / fixtures.ARCHIVE_NAME
    with tarfile.open(bundle, "w:gz") as stream:
        stream.add(successor, arcname="fixture_bundle/payload")
    write_review_sidecar(bundle, repo_root=root)
    metadata = json.loads(contract.metadata_path.read_bytes())["metadata"]
    binding = publisher.build_release_binding(
        {
            "metadata_path": contract.metadata_path,
            "metadata_sha256": contract.metadata_sha256,
            "release_tag": contract.successor_github_release_tag,
            "concept_doi": contract.concept_doi,
            "version_doi": contract.successor_version_doi,
        }
    )
    file = {
        "key": bundle.name,
        "size": bundle.stat().st_size,
        "links": {
            "self": f"https://zenodo.org/api/records/99999999/files/{bundle.name}/content",
        },
    }
    remote = {
        "id": 99999999,
        "record_id": 99999999,
        "conceptrecid": "23150471",
        "conceptdoi": contract.concept_doi,
        "doi": SYNTHETIC_DOI,
        "state": "done",
        "submitted": True,
        "metadata": metadata,
        "files": [file],
    }
    state = publisher._public_state(remote)
    state["files"] = [{"name": bundle.name, "size": bundle.stat().st_size, "sha256": _sha(bundle)}]
    session = _ReadOnlyZenodo()
    session.gets = [
        _Response(remote),
        _Response(
            {
                "id": 99999999,
                "conceptrecid": "23150471",
                "doi": SYNTHETIC_DOI,
                "status": "published",
                "files": [file],
            }
        ),
        _Response({}, content=bundle.read_bytes()),
    ]
    observed = publisher.verify(
        session, publisher._seal_state(state), metadata, release_binding=binding
    )
    assert observed["status"] == "pass", observed
    assert observed["problem_count"] == 0
    assert len(session.urls) == 3  # Only fake GETs; there is no session/token factory.


def test_f2_metadata_only_successor_refuses_changed_component(correction):
    """A metadata-only successor cannot silently change a scientific component."""
    _root, _identity, successor, _archive, _contract, _receipt = correction
    path = next((successor / "runs").glob("*/episodes.jsonl"))
    original = path.read_bytes()
    lines = original.decode().splitlines()
    row = json.loads(lines[0])
    row["metrics"]["snqi_v2"] = 0.75
    lines[0] = json.dumps(row, sort_keys=True)
    try:
        _write_preserved_bytes(path, ("\n".join(lines) + "\n").encode())
        with pytest.raises(erratum.ReleaseErratumError, match="scientific leaves differ"):
            _cold_validate(correction)
    finally:
        _write_preserved_bytes(path, original)
