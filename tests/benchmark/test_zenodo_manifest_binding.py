"""Tests for binding direct Zenodo operations to the benchmark release manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest
from robot_sf.benchmark.zenodo_publisher import (
    ZENODO_STATE_SCHEMA,
    ZenodoPublisherError,
    _seal_state,
    build_release_binding,
    load_dataset_metadata,
    publish,
    reserve,
    upload,
    verify,
)

_MANIFEST_PATH = Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")


class _Response:
    """Small requests-like response fixture."""

    def __init__(self, payload: dict[str, Any], *, content: bytes | None = None) -> None:
        self.payload = payload
        self.status_code = 200
        self.content = content if content is not None else json.dumps(payload).encode()

    def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self.payload

    def raise_for_status(self) -> None:
        """Implement the response protocol for successful fixtures."""

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield the configured body as one streamed chunk."""
        del chunk_size
        yield self.content


class _Session:
    """Queue-backed session fixture for all four publisher modes."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[_Response] = []
        self.gets: list[_Response] = []
        self.puts: list[_Response] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued POST response."""
        del url, kwargs
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued GET response."""
        del url, kwargs
        return self.gets.pop(0)

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued PUT response."""
        del url, kwargs
        return self.puts.pop(0)


def _binding_and_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the checked-in benchmark metadata and its release binding."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    binding = build_release_binding(manifest)
    metadata = load_dataset_metadata(
        binding["metadata_path"],
        expected_source_tag=binding["release_tag"],
        expected_metadata_sha256=binding["metadata_sha256"],
    )
    return binding, metadata


def _deposition_payload(binding: dict[str, Any], *, submitted: bool = False) -> dict[str, Any]:
    """Build a reserved or published deposition response."""
    return {
        "id": 22053133,
        "record_id": 22053133,
        "conceptrecid": "22053132",
        "doi": binding["version_doi"] if submitted else None,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": {"prereserve_doi": {"doi": binding["version_doi"]}},
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": [],
    }


def test_benchmark_manifest_loads_exact_zenodo_metadata_binding() -> None:
    """The v0.2 benchmark manifest exposes and validates metadata bytes."""
    manifest = load_release_manifest(_MANIFEST_PATH)

    assert manifest.metadata_path is not None
    assert manifest.metadata_path.is_file()
    assert (
        manifest.metadata_sha256 == hashlib.sha256(manifest.metadata_path.read_bytes()).hexdigest()
    )


def test_benchmark_manifest_rejects_metadata_checksum_drift(tmp_path: Path) -> None:
    """A changed or stale metadata digest blocks manifest loading."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    payload = yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["canonical_campaign_config"] = str(manifest.canonical_campaign_config_path)
    payload["scenario"]["matrix_path"] = str(manifest.scenario_matrix_path)
    payload["scenario"]["suite_policy_path"] = str(manifest.suite_policy_path)
    payload["scenario"]["route_certification_path"] = str(manifest.route_certification_path)
    payload["seed_policy"]["seed_sets_path"] = str(
        Path("configs/benchmarks/seed_sets_v1.yaml").resolve()
    )
    payload["metrics"]["snqi_weights_path"] = str(manifest.snqi_weights_path)
    payload["metrics"]["snqi_baseline_path"] = str(manifest.snqi_baseline_path)
    payload["citation_path"] = str(manifest.citation_path)
    payload["release_checklist_path"] = str(manifest.release_checklist_path)
    payload["publication"]["metadata_path"] = str(manifest.metadata_path)
    payload["publication"]["metadata_sha256"] = "0" * 64
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata_sha256 does not match"):
        load_release_manifest(path)


def test_release_validation_rechecks_metadata_digest() -> None:
    """Post-load manifest objects cannot bypass the metadata checksum gate."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    drifted = replace(manifest, metadata_sha256="0" * 64)

    report = validate_release_manifest(drifted)

    assert report["status"] == "invalid"
    assert (
        "publication.metadata_sha256 does not match publication.metadata_path" in report["problems"]
    )


def test_all_zenodo_modes_preserve_manifest_binding(tmp_path: Path) -> None:
    """Reserve, upload, verify, and publish carry the same release identity."""
    binding, metadata = _binding_and_metadata()
    session = _Session()
    session.posts = [
        _Response(_deposition_payload(binding)),
        _Response(_deposition_payload(binding, submitted=True)),
    ]

    state = reserve(session, metadata, release_binding=binding)
    assert state["release_binding"]["metadata_sha256"] == binding["metadata_sha256"]
    assert state["release_binding"]["concept_doi"] == binding["concept_doi"]
    assert state["release_binding"]["version_doi"] == binding["version_doi"]

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"manifest-bound bundle")
    session.gets = [_Response(_deposition_payload(binding))]
    session.puts = [_Response({"checksum": "md5:fixture"})]
    state = upload(session, state, [bundle], release_binding=binding)

    remote_draft = _deposition_payload(binding)
    remote_draft["metadata"] = {
        **metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
    }
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/22053133/files/bundle"},
        }
    ]
    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    report = verify(session, state, metadata, release_binding=binding)
    assert report["status"] == "pass", report
    assert state["verification_receipt"]["release_binding"] == state["release_binding"]
    assert state["verification_receipt"]["manifest_metadata_sha256"] == binding["metadata_sha256"]

    state = publish(session, state, metadata, release_binding=binding)
    assert state["submitted"] is True
    assert state["release_binding"]["version_doi"] == binding["version_doi"]


def test_bound_zenodo_operation_rejects_metadata_checksum_mismatch() -> None:
    """A binding with a stale checksum cannot reach the Zenodo API."""
    binding, metadata = _binding_and_metadata()
    binding["metadata_sha256"] = "0" * 64
    session = _Session()

    with pytest.raises(ZenodoPublisherError, match="metadata file SHA-256"):
        reserve(session, metadata, release_binding=binding)
    assert session.posts == []


def test_bound_reserve_rejects_concept_or_version_identity_drift() -> None:
    """Reserved DOI identity must match both manifest DOI fields exactly."""
    binding, metadata = _binding_and_metadata()
    session = _Session()
    response = _deposition_payload(binding)
    response["conceptrecid"] = "999999"
    session.posts = [_Response(response)]

    with pytest.raises(ZenodoPublisherError, match="concept DOI"):
        reserve(session, metadata, release_binding=binding)


def test_state_shape_for_binding_remains_credential_free() -> None:
    """Manifest binding state contains no token-shaped field or value."""
    state = _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": 1,
            "record_id": 2,
            "concept_record_id": "3",
            "doi": "10.5281/zenodo.4",
            "submitted": False,
            "files": [],
        }
    )
    assert "token" not in json.dumps(state).casefold()
