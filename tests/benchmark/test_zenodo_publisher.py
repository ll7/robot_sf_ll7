"""Mocked API tests for the direct benchmark-dataset Zenodo publisher."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.zenodo_publisher import (
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


class _Session:
    """Queue-backed requests session fixture."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[_Response] = []
        self.gets: list[_Response] = []
        self.puts: list[_Response] = []
        self.urls: list[str] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Consume a POST response."""
        self.urls.append(url)
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Consume a GET response."""
        self.urls.append(url)
        return self.gets.pop(0)

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Consume a PUT response."""
        self.urls.append(url)
        return self.puts.pop(0)


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
    """All publisher modes preserve identity and a credential-free state contract."""
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
    session.gets = [_Response(_draft_payload())]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    state = upload(session, state, [bundle])
    assert state["files"][0]["name"] == bundle.name
    assert len(state["files"][0]["sha256"]) == 64

    remote_draft = _draft_payload()
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle/content"},
        }
    ]
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

    state = publish(session, state, _metadata())
    assert state["submitted"] is True
    remote = _draft_payload(submitted=True)
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle/content"},
        }
    ]
    downloaded = _Response({})
    downloaded.content = bundle.read_bytes()
    session.gets = [_Response(remote), downloaded]
    report = verify(session, state, _metadata())
    assert report["status"] == "pass"
    assert report["publication_state"] == "published"
    assert report["file_count"] == 1


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
        "files": [{"name": bundle.name, "size": 6, "sha256": "0" * 64}],
    }
    state = _seal_state(state)
    with pytest.raises(ZenodoPublisherError, match="verification receipt"):
        publish(session, state, _metadata())
    assert session.posts == []
