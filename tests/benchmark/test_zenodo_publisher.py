"""Mocked API tests for the direct benchmark-dataset Zenodo publisher."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.zenodo_publisher import (
    _REMOTE_DOWNLOAD_CHUNK_SIZE,
    ZENODO_STATE_SCHEMA,
    ZenodoPublisherError,
    _seal_state,
    load_dataset_metadata,
    publish,
    read_token_file,
    reserve,
    upload,
    verify,
)


class _Response:
    """Minimal requests-like response fixture."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()

    def json(self) -> dict[str, Any]:
        """Return the fixture payload."""
        return self.payload

    def raise_for_status(self) -> None:
        """Raise for failure fixtures."""
        if self.status_code >= 400:
            raise RuntimeError("HTTP failure")

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield the configured body as a requests-like streamed chunk."""
        del chunk_size
        yield self.content


class _StreamingResponse:
    """Response fixture that forbids full-body access and exposes chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.status_code = 200
        self.chunks = chunks
        self.chunk_sizes: list[int] = []

    @property
    def content(self) -> bytes:
        """Fail if verification tries to materialize the response body."""
        raise AssertionError("streamed response content must not be accessed")

    def iter_content(self, *, chunk_size: int) -> Any:
        """Return bounded chunks and record the requested chunk size."""
        self.chunk_sizes.append(chunk_size)
        yield from self.chunks

    def json(self) -> dict[str, Any]:
        """Provide the response protocol's JSON method for type compatibility."""
        return {}

    def raise_for_status(self) -> None:
        """Provide the response protocol's status method for type compatibility."""


class _Session:
    """Queue-backed requests session fixture."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[_Response] = []
        self.gets: list[_Response] = []
        self.puts: list[_Response] = []
        self.deletes: list[_Response] = []
        self.urls: list[str] = []
        self.get_kwargs: list[dict[str, Any]] = []
        self.delete_kwargs: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Consume a POST response."""
        self.urls.append(url)
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Consume a GET response."""
        self.urls.append(url)
        self.get_kwargs.append(kwargs)
        return self.gets.pop(0)

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Consume a PUT response."""
        self.urls.append(url)
        return self.puts.pop(0)

    def delete(self, url: str, **kwargs: Any) -> _Response:
        """Consume a DELETE response."""
        self.urls.append(url)
        self.delete_kwargs.append(kwargs)
        return self.deletes.pop(0)


def _metadata() -> dict[str, Any]:
    """Return valid benchmark-dataset metadata."""
    return {
        "title": "Robot SF S30/H600 benchmark dataset",
        "upload_type": "dataset",
        "access_right": "open",
        "description": (
            "Release-bound raw and component benchmark results. SNQI is advisory only; "
            "this release makes no SNQI ranking claim."
        ),
        "license": "GPL-3.0-only",
        "creators": [{"name": "Luttkus, Lennart"}, {"name": "Tröster, Marco"}],
        "related_identifiers": [
            {
                "identifier": (
                    "https://github.com/ll7/robot_sf_ll7/releases/tag/"
                    "paper-matrix-v2-h600-s30-2026-08-abcdef123456"
                ),
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }


def _draft_payload(*, submitted: bool = False) -> dict[str, Any]:
    """Return a representative Zenodo deposition response."""
    return {
        "id": 123,
        "record_id": 123,
        "conceptrecid": "122",
        "doi": "10.5281/zenodo.123" if submitted else None,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": {
            **_metadata(),
            "prereserve_doi": {"doi": "10.5281/zenodo.123", "recid": 123},
        },
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": [],
    }


def _metadata_verification_fixture(
    tmp_path: Path,
) -> tuple[_Session, dict[str, Any], dict[str, Any]]:
    """Return a valid uploaded state and remote draft for metadata verification tests."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": 123,
            "record_id": 123,
            "concept_record_id": "122",
            "doi": "10.5281/zenodo.123",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    remote = _draft_payload()
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle/content"},
        }
    ]
    downloaded = _Response({})
    downloaded.content = bundle.read_bytes()
    session = _Session()
    session.gets = [_Response(remote), downloaded]
    return session, state, remote


def test_token_file_requires_private_permissions(tmp_path: Path) -> None:
    """Credentials cannot be read from a group/world-accessible file."""
    path = tmp_path / "zenodo.token"
    path.write_text("secret\n", encoding="utf-8")
    os.chmod(path, 0o644)
    with pytest.raises(ZenodoPublisherError, match="0600"):
        read_token_file(path)
    os.chmod(path, 0o600)
    assert read_token_file(path) == "secret"


def test_metadata_requires_dataset_license_creators_and_exact_tag(tmp_path: Path) -> None:
    """Metadata admission rejects mixed software uploads and unbound source identity."""
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"metadata": _metadata()}), encoding="utf-8")
    metadata = load_dataset_metadata(path)
    assert metadata["upload_type"] == "dataset"
    assert metadata["prereserve_doi"] is True
    metadata["upload_type"] = "software"
    path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    with pytest.raises(ZenodoPublisherError, match="upload_type=dataset"):
        load_dataset_metadata(path)


def test_reserve_upload_publish_and_verify_without_credentials_in_state(tmp_path: Path) -> None:
    """Publication is followed by mandatory published-record verification."""
    session = _Session()
    session.posts = [
        _Response(_draft_payload(), 201),
        _Response(_draft_payload(submitted=True), 202),
    ]
    state = reserve(session, _metadata())
    assert state["schema_version"] == ZENODO_STATE_SCHEMA
    assert state["doi"] == "10.5281/zenodo.123"
    assert "token" not in json.dumps(state).lower()

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    remote_draft = _draft_payload()
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle/content"},
        }
    ]
    session.gets = [
        _Response(_draft_payload()),
        _Response(remote_draft),
        _Response(remote_draft),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    state = upload(session, state, [bundle])
    assert state["files"][0]["name"] == bundle.name
    assert len(state["files"][0]["sha256"]) == 64

    downloaded = _Response({})
    downloaded.content = bundle.read_bytes()
    session.gets = [_Response(remote_draft), downloaded]
    report = verify(session, state, _metadata())
    assert report["status"] == "pass"
    assert report["publication_state"] == "draft"
    assert "verification_receipt" in state

    tampered = dict(state)
    tampered["verification_receipt"] = dict(state["verification_receipt"])
    tampered["verification_receipt"]["metadata_sha256"] = "0" * 64
    with pytest.raises(ZenodoPublisherError, match="integrity"):
        publish(session, tampered, _metadata())

    session.gets = [_Response(remote_draft), downloaded]
    state = publish(session, state, _metadata())
    assert state["submitted"] is True
    remote = _draft_payload(submitted=True)
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/draft/files/bundle/content"},
        }
    ]
    published_record = {
        "id": 123,
        "conceptrecid": "122",
        "doi": "10.5281/zenodo.123",
        "status": "published",
        "files": [
            {
                "key": bundle.name,
                "size": bundle.stat().st_size,
                "links": {"self": "https://zenodo.org/api/records/123/files/bundle/content"},
            }
        ],
    }
    downloaded = _Response({})
    downloaded.content = bundle.read_bytes()
    session.gets = [_Response(remote), _Response(published_record), downloaded]
    report = verify(session, state, _metadata())
    assert report["status"] == "pass"
    assert report["publication_state"] == "published"
    assert report["file_count"] == 1
    assert session.urls[-2] == "https://zenodo.org/api/records/123"
    assert session.urls[-1] == "https://zenodo.org/api/records/123/files/bundle/content"


def test_verify_accepts_zenodo_license_and_creator_normalization(tmp_path: Path) -> None:
    """Known Zenodo read-back aliases do not create false metadata mismatches."""
    session, state, remote = _metadata_verification_fixture(tmp_path)
    remote["metadata"]["license"] = "gpl-3.0"
    remote["metadata"]["creators"] = [
        {"name": "Luttkus, Lennart", "affiliation": None},
        {"name": "Tröster, Marco", "affiliation": None},
    ]

    report = verify(session, state, _metadata())

    assert report["status"] == "pass"
    assert not any(problem.startswith("metadata.") for problem in report["problems"])


def test_verify_accepts_draft_without_advertised_size_when_download_matches(
    tmp_path: Path,
) -> None:
    """Zenodo drafts may omit size; streamed bytes and SHA-256 remain authoritative."""
    session, state, remote = _metadata_verification_fixture(tmp_path)
    remote["files"][0].pop("size")

    report = verify(session, state, _metadata())

    assert report["status"] == "pass"
    assert report["publication_state"] == "draft"
    assert report["problem_count"] == 0


@pytest.mark.parametrize(
    ("remote_change", "match"),
    [
        ("metadata", "fresh draft verification failed"),
        ("modified", "changed since verification receipt"),
    ],
)
def test_publish_reverifies_fresh_draft_and_blocks_drift_without_post(
    tmp_path: Path, remote_change: str, match: str
) -> None:
    """Publication requires the exact verified draft immediately before its POST."""
    session, state, remote = _metadata_verification_fixture(tmp_path)
    initial_download = session.gets[1]
    assert verify(session, state, _metadata())["status"] == "pass"

    fresh_remote = json.loads(json.dumps(remote))
    if remote_change == "metadata":
        fresh_remote["metadata"]["title"] = "drifted title"
    else:
        fresh_remote["modified"] = "remote-version-2"
        remote["modified"] = "remote-version-1"
        # Rebuild the receipt with the first optimistic value before testing
        # the fresh readback with the second value.
        session.gets = [_Response(remote), initial_download]
        assert verify(session, state, _metadata())["status"] == "pass"
        fresh_remote["modified"] = "remote-version-2"

    publish_response = _Response(_draft_payload(submitted=True), 202)
    session.posts = [publish_response]
    session.gets = [_Response(fresh_remote), initial_download]
    state_before_publish = json.loads(json.dumps(state))

    with pytest.raises(ZenodoPublisherError, match=match):
        publish(session, state, _metadata())

    assert session.posts == [publish_response]
    assert not any(url.endswith("/actions/publish") for url in session.urls)
    assert state == state_before_publish


@pytest.mark.parametrize(
    ("field", "remote_value"),
    [
        ("license", "MIT"),
        ("creators", [{"name": "Different creator", "affiliation": None}]),
        ("creators", [{"name": "Luttkus, Lennart", "affiliation": "Different affiliation"}]),
        ("title", "Different title"),
        ("upload_type", "software"),
        ("related_identifiers", []),
    ],
)
def test_verify_rejects_true_metadata_mismatches(
    tmp_path: Path, field: str, remote_value: Any
) -> None:
    """Normalization must not hide real metadata, type, or identity mismatches."""
    session, state, remote = _metadata_verification_fixture(tmp_path)
    remote["metadata"][field] = remote_value

    report = verify(session, state, _metadata())

    assert report["status"] == "fail"
    assert f"metadata.{field} does not match requested metadata" in report["problems"]


def test_publish_requires_prior_verification_receipt(tmp_path: Path) -> None:
    """An uploaded draft cannot be published without a successful draft receipt."""
    session = _Session()
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = {
        "schema_version": ZENODO_STATE_SCHEMA,
        "deposition_id": 123,
        "record_id": 123,
        "concept_record_id": "122",
        "doi": "10.5281/zenodo.123",
        "submitted": False,
        "state": "unsubmitted",
        "files": [{"name": bundle.name, "size": 6, "sha256": "0" * 64}],
    }
    state = _seal_state(state)
    with pytest.raises(ZenodoPublisherError, match="verification receipt"):
        publish(session, state, _metadata())
    assert session.posts == []


def test_verify_streams_remote_bundle_in_bounded_chunks_without_content_access(
    tmp_path: Path,
) -> None:
    """Cold verification hashes chunks without materializing the remote bundle."""
    bundle = tmp_path / "bundle.tar.gz"
    chunks = [b"bundle ", b"chunk", b" payload"]
    bundle.write_bytes(b"".join(chunks))
    session = _Session()
    state = _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": 123,
            "record_id": 123,
            "concept_record_id": "122",
            "doi": "10.5281/zenodo.123",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    remote_draft = _draft_payload()
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle/content"},
        }
    ]
    streamed = _StreamingResponse(chunks)
    session.gets = [_Response(remote_draft), streamed]

    report = verify(session, state, _metadata())

    assert report["status"] == "pass"
    assert streamed.chunk_sizes == [_REMOTE_DOWNLOAD_CHUNK_SIZE]
    assert session.get_kwargs[-1] == {
        "stream": True,
        "timeout": 3600,
        "allow_redirects": False,
    }
