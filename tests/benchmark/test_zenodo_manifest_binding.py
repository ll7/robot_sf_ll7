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
    _verify_integrity,
    build_release_binding,
    load_dataset_metadata,
    load_state,
    publish,
    recover,
    reserve,
    upload,
    verify,
    write_state,
)

_MANIFEST_PATH = Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")


class _Response:
    """Small requests-like response fixture."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        content: bytes | None = None,
        status_code: int = 200,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
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
        self.deletes: list[_Response] = []

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

    def delete(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued DELETE response."""
        del url, kwargs
        return self.deletes.pop(0)


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
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    return {
        "id": version_record_id,
        "record_id": version_record_id,
        "conceptrecid": concept_record_id,
        "doi": binding["version_doi"] if submitted else None,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": {"prereserve_doi": {"doi": binding["version_doi"]}},
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": [],
    }


def _unbound_state(
    binding: dict[str, Any], *, files: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build a valid state that lets a bound operation adopt its binding."""
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    return _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": deposition_id,
            "record_id": deposition_id,
            "concept_record_id": concept_record_id,
            "doi": binding["version_doi"],
            "submitted": False,
            "state": "unsubmitted",
            "files": files or [],
        }
    )


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
    version_record_id = binding["version_doi"].rsplit(".", 1)[-1]
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
    remote_draft = _deposition_payload(binding)
    remote_draft["metadata"] = {
        **metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
    }
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": f"https://zenodo.org/api/records/{version_record_id}/files/bundle"
            },
        }
    ]
    session.gets = [
        _Response(_deposition_payload(binding)),
        _Response(remote_draft),
        _Response(remote_draft),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"})]
    state = upload(session, state, [bundle], release_binding=binding)

    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    report = verify(session, state, metadata, release_binding=binding)
    assert report["status"] == "pass", report
    assert state["verification_receipt"]["release_binding"] == state["release_binding"]
    assert state["verification_receipt"]["manifest_metadata_sha256"] == binding["metadata_sha256"]

    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    state = publish(session, state, metadata, release_binding=binding)
    assert state["submitted"] is True
    assert state["release_binding"]["version_doi"] == binding["version_doi"]


def test_recover_restores_manifest_bound_state_for_upload_and_verify(tmp_path: Path) -> None:
    """A read-only draft lookup restores the same state contract as reserve."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    draft = _deposition_payload(binding)
    draft["metadata"] = {**metadata, "prereserve_doi": {"doi": binding["version_doi"]}}
    session = _Session()
    session.gets = [_Response(draft)]

    state = recover(
        session,
        version_record_id,
        metadata,
        release_binding=binding,
    )

    assert session.posts == []
    assert state["deposition_id"] == version_record_id
    assert state["submitted"] is False
    assert state["files"] == []
    assert state["release_binding"]["metadata_sha256"] == binding["metadata_sha256"]
    state_path = tmp_path / "recovered-state.json"
    write_state(state_path, state)
    assert state_path.stat().st_mode & 0o777 == 0o600
    state = load_state(state_path)

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"recovered draft bundle")
    remote_draft = dict(draft)
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": f"https://zenodo.org/api/records/{version_record_id}/files/bundle"
            },
        }
    ]
    session.gets = [_Response(draft), _Response(remote_draft), _Response(remote_draft)]
    session.puts = [_Response({"checksum": "md5:fixture"})]
    state = upload(session, state, [bundle], release_binding=binding)

    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    report = verify(session, state, metadata, release_binding=binding)
    assert report["status"] == "pass", report


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("deposition", "requested deposition ID"),
        ("concept", "concept DOI"),
        ("version", "version DOI"),
        ("source", "metadata.related_identifiers"),
        ("metadata", "metadata.title"),
        ("missing_submitted", "submitted state"),
        ("invalid_submitted", "submitted state"),
        ("published", "unpublished draft"),
    ],
)
def test_recover_rejects_draft_identity_metadata_and_state_drift(drift: str, error: str) -> None:
    """Recovery fails closed before writing state when the remote draft drifts."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    draft = _deposition_payload(binding)
    draft["metadata"] = {**metadata, "prereserve_doi": {"doi": binding["version_doi"]}}
    if drift == "deposition":
        draft["id"] = version_record_id + 1
    elif drift == "concept":
        draft["conceptrecid"] = "999999"
    elif drift == "version":
        draft["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.999999"}
    elif drift == "source":
        draft["metadata"]["related_identifiers"] = [
            {
                "identifier": "https://github.com/ll7/robot_sf_ll7/releases/tag/other",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ]
    elif drift == "metadata":
        draft["metadata"]["title"] = "Different release"
    elif drift == "missing_submitted":
        draft.pop("submitted")
    elif drift == "invalid_submitted":
        draft["submitted"] = "false"
    else:
        draft["submitted"] = True
        draft["state"] = "done"
        draft["doi"] = binding["version_doi"]
    session = _Session()
    session.gets = [_Response(draft)]

    with pytest.raises(ZenodoPublisherError, match=error):
        recover(session, version_record_id, metadata, release_binding=binding)
    assert session.posts == []


def test_bound_zenodo_operation_rejects_metadata_checksum_mismatch() -> None:
    """A binding with a stale checksum cannot reach the Zenodo API."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    binding["metadata_sha256"] = "0" * 64
    session = _Session()

    with pytest.raises(ZenodoPublisherError, match="metadata file SHA-256"):
        reserve(session, metadata, release_binding=binding)
    with pytest.raises(ZenodoPublisherError, match="metadata file SHA-256"):
        recover(session, version_record_id, metadata, release_binding=binding)
    assert session.posts == []
    assert session.gets == []


def test_bound_reserve_rejects_concept_or_version_identity_drift() -> None:
    """Reserved DOI identity must match both manifest DOI fields exactly."""
    binding, metadata = _binding_and_metadata()
    session = _Session()
    response = _deposition_payload(binding)
    response["conceptrecid"] = "999999"
    session.posts = [_Response(response)]

    with pytest.raises(ZenodoPublisherError, match="concept DOI"):
        reserve(session, metadata, release_binding=binding)


@pytest.mark.parametrize("receipt_kind", ["missing", "stale"])
def test_publish_failure_preserves_bound_caller_state(
    receipt_kind: str,
) -> None:
    """Receipt admission failures cannot mutate or invalidate caller state."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    state_payload: dict[str, Any] = {
        "schema_version": ZENODO_STATE_SCHEMA,
        "deposition_id": deposition_id,
        "record_id": deposition_id,
        "concept_record_id": concept_record_id,
        "doi": binding["version_doi"],
        "submitted": False,
        "state": "unsubmitted",
        "files": [{"name": "bundle.tar.gz", "size": 1, "sha256": "0" * 64}],
    }
    if receipt_kind == "stale":
        state_payload["verification_receipt"] = {
            "status": "pass",
            "publication_state": "draft",
        }
    state = _seal_state(state_payload)
    state_before = json.loads(json.dumps(state, sort_keys=True))
    session = _Session()

    expected_error = "verification receipt" if receipt_kind == "missing" else "integrity"
    with pytest.raises(ZenodoPublisherError, match=expected_error):
        publish(session, state, metadata, release_binding=binding)

    assert state == state_before
    assert session.gets == []
    assert session.posts == []
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def test_bound_upload_failure_preserves_unbound_caller_state(tmp_path: Path) -> None:
    """Upload binding adoption is discarded when the remote bucket is rejected."""
    binding, _ = _binding_and_metadata()
    state = _unbound_state(binding)
    state_before = json.dumps(state, sort_keys=True)
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    draft = _deposition_payload(binding)
    draft["links"]["bucket"] = "http://zenodo.org/api/files/bucket"
    session = _Session()
    session.gets = [_Response(draft)]

    with pytest.raises(ZenodoPublisherError, match="secure upload bucket"):
        upload(session, state, [bundle], release_binding=binding)

    assert json.dumps(state, sort_keys=True) == state_before
    assert "release_binding" not in state
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def test_bound_verify_failure_preserves_unbound_caller_state() -> None:
    """Verify binding adoption is discarded when the remote lookup fails."""
    binding, metadata = _binding_and_metadata()
    state = _unbound_state(binding)
    state_before = json.dumps(state, sort_keys=True)
    session = _Session()
    session.gets = [_Response({}, status_code=503)]

    with pytest.raises(ZenodoPublisherError, match="verify request failed"):
        verify(session, state, metadata, release_binding=binding)

    assert json.dumps(state, sort_keys=True) == state_before
    assert "release_binding" not in state
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


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
            "state": "unsubmitted",
            "files": [],
        }
    )
    assert "token" not in json.dumps(state).casefold()
