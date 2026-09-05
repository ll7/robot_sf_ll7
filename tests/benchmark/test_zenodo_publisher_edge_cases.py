"""Fail-closed and credential-safe edge tests for the Zenodo publisher."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from robot_sf.benchmark import zenodo_publisher as publisher

if TYPE_CHECKING:
    from pathlib import Path

SOURCE_SHA = "5" * 40
PREDECESSOR_TAG = f"paper-matrix-v2-h600-s30-2026-09-{SOURCE_SHA}"
SUCCESSOR_TAG = f"{PREDECESSOR_TAG}-erratum.1"


class _Response:
    """Small requests-like response fixture."""

    def __init__(self, payload: object, status_code: int = 200, content: bytes | None = None):
        self.payload = payload
        self.status_code = status_code
        self.content = content if content is not None else json.dumps(payload).encode()

    def json(self) -> object:
        """Return the configured payload."""
        return self.payload

    def raise_for_status(self) -> None:
        """Raise for HTTP failures."""
        if self.status_code >= 400:
            raise RuntimeError("HTTP failure")

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield the configured body as a requests-like streamed chunk."""
        del chunk_size
        yield self.content


class _Session:
    """Queue-backed session fixture."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.gets: list[_Response] = []
        self.posts: list[_Response] = []
        self.puts: list[_Response] = []
        self.deletes: list[_Response] = []
        self.urls: list[str] = []
        self.delete_urls: list[str] = []
        self.put_kwargs: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Consume one GET fixture."""
        self.urls.append(url)
        return self.gets.pop(0)

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Consume one POST fixture."""
        self.urls.append(url)
        return self.posts.pop(0)

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Consume one PUT fixture."""
        self.urls.append(url)
        self.put_kwargs.append(kwargs)
        return self.puts.pop(0)

    def delete(self, url: str, **kwargs: Any) -> _Response:
        """Consume one DELETE fixture."""
        self.urls.append(url)
        self.delete_urls.append(url)
        return self.deletes.pop(0)


def _metadata() -> dict[str, Any]:
    """Return valid benchmark metadata."""
    return {
        "title": "Robot SF benchmark dataset",
        "upload_type": "dataset",
        "access_right": "open",
        "description": "SNQI is advisory only; this release makes no SNQI ranking claim.",
        "license": "GPL-3.0-only",
        "creators": [{"name": "Luttkus, Lennart"}],
        "related_identifiers": [
            {
                "identifier": "https://github.com/ll7/robot_sf_ll7/releases/tag/release",
                "relation": "isSupplementTo",
            }
        ],
    }


def _draft(*, submitted: bool = False, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one deposition API payload."""
    return {
        "id": 7,
        "record_id": 7,
        "conceptrecid": "6",
        "doi": "10.5281/zenodo.7" if submitted else None,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": metadata or {**_metadata(), "prereserve_doi": {"doi": "10.5281/zenodo.7"}},
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": [],
    }


def _draft_file(name: str, *, deposition_id: int = 8, file_id: str = "file-1") -> dict[str, Any]:
    """Return one legacy deposition-file resource with an exact self link."""
    return {
        "id": file_id,
        "filename": name,
        "links": {
            "self": (f"https://zenodo.org/api/deposit/depositions/{deposition_id}/files/{file_id}")
        },
    }


def _successor_state() -> dict[str, Any]:
    """Return sealed state carrying exact new-version predecessor provenance."""
    return publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 8,
            "record_id": 8,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.8",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
            "concept_doi": "10.5281/zenodo.6",
            "predecessor_deposition_id": 7,
            "predecessor_doi": "10.5281/zenodo.7",
            "predecessor": {"deposition_id": 7, "doi": "10.5281/zenodo.7"},
            "source_tag": ("https://github.com/ll7/robot_sf_ll7/releases/tag/" + SUCCESSOR_TAG),
        }
    )


def _published_record(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one published-record API payload."""
    return {
        "id": 7,
        "conceptrecid": "6",
        "doi": "10.5281/zenodo.7",
        "status": "published",
        "files": files,
    }


def _successor_draft(
    *,
    deposition_id: int = 8,
    record_id: int = 8,
    concept_record_id: str = "6",
    doi: str | None = "10.5281/zenodo.8",
    submitted: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a representative legacy-API successor draft."""
    return {
        "id": deposition_id,
        "record_id": record_id,
        "conceptrecid": concept_record_id,
        "conceptdoi": "10.5281/zenodo.6",
        "doi": doi,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": {
            **(metadata or _new_version_metadata()),
            "prereserve_doi": {"doi": doi or "10.5281/zenodo.8"},
        },
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": [],
    }


def _new_version_metadata() -> dict[str, Any]:
    """Return benchmark metadata explicitly related to the predecessor version."""
    metadata = _metadata()
    metadata["related_identifiers"] = [
        {
            "identifier": ("https://github.com/ll7/robot_sf_ll7/releases/tag/" + SUCCESSOR_TAG),
            "relation": "isSupplementTo",
            "scheme": "url",
        },
        {
            "identifier": "10.5281/zenodo.7",
            "relation": "isNewVersionOf",
            "scheme": "doi",
        },
    ]
    return metadata


def _successor_binding(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write metadata and return its exact manifest-style successor binding."""
    metadata = _new_version_metadata()
    metadata_path = tmp_path / "successor-metadata.json"
    metadata_path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    return (
        publisher.build_release_binding(
            {
                "metadata_path": metadata_path,
                "metadata_sha256": publisher._sha256_file(metadata_path),
                "release_tag": SUCCESSOR_TAG,
                "concept_doi": "10.5281/zenodo.6",
                "version_doi": "10.5281/zenodo.8",
            }
        ),
        metadata,
    )


def _predecessor_deposition(**updates: Any) -> dict[str, Any]:
    """Return the exact published predecessor required before mutation."""
    metadata = _metadata()
    metadata["related_identifiers"] = [
        {
            "identifier": ("https://github.com/ll7/robot_sf_ll7/releases/tag/" + PREDECESSOR_TAG),
            "relation": "isSupplementTo",
        }
    ]
    payload = {
        "id": 7,
        "record_id": 7,
        "conceptrecid": "6",
        "conceptdoi": "10.5281/zenodo.6",
        "doi": "10.5281/zenodo.7",
        "state": "done",
        "submitted": True,
        "metadata": metadata,
    }
    payload.update(updates)
    return payload


def _new_version_fixture(
    *,
    draft: dict[str, Any] | None = None,
    readback: dict[str, Any] | None = None,
) -> _Session:
    """Build a queued session for one successor reservation."""
    draft_payload = draft or _successor_draft()
    session = _Session()
    session.posts = [
        _Response(
            {
                "links": {
                    "latest_draft": (
                        f"https://zenodo.org/api/deposit/depositions/{draft_payload['id']}"
                    )
                }
            },
            status_code=201,
        )
    ]
    session.gets = [_Response(_predecessor_deposition()), _Response(draft_payload)]
    session.puts = [_Response(readback or draft_payload, status_code=200)]
    return session


def _new_version(session: _Session) -> dict[str, Any]:
    """Reserve a successor with the fixture's predecessor and concept identity."""
    return _new_version_with_metadata(session, _new_version_metadata())


def _new_version_with_metadata(session: _Session, metadata: dict[str, Any]) -> dict[str, Any]:
    """Reserve a successor with caller-supplied metadata and fixture identities."""
    return publisher.new_version(
        session,
        metadata,
        predecessor_deposition_id=7,
        expected_predecessor_doi="10.5281/zenodo.7",
        expected_concept_doi="10.5281/zenodo.6",
        expected_predecessor_tag=PREDECESSOR_TAG,
        expected_source_sha=SOURCE_SHA,
        expected_successor_tag=SUCCESSOR_TAG,
        api_base="https://zenodo.org/api",
    )


def test_new_version_replaces_metadata_and_seals_predecessor_identity() -> None:
    """A legacy new-version reservation returns only checked, credential-free state."""
    session = _new_version_fixture()

    state = _new_version(session)

    assert session.urls == [
        "https://zenodo.org/api/deposit/depositions/7",
        "https://zenodo.org/api/deposit/depositions/7/actions/newversion",
        "https://zenodo.org/api/deposit/depositions/8",
        "https://zenodo.org/api/deposit/depositions/8",
    ]
    assert session.put_kwargs == [
        {
            "json": {"metadata": _new_version_metadata()},
            "timeout": 60,
            "allow_redirects": False,
        }
    ]
    assert state["deposition_id"] == 8
    assert state["record_id"] == 8
    assert state["concept_record_id"] == "6"
    assert state["concept_doi"] == "10.5281/zenodo.6"
    assert state["doi"] == "10.5281/zenodo.8"
    assert state["submitted"] is False
    assert state["predecessor_deposition_id"] == 7
    assert state["predecessor_doi"] == "10.5281/zenodo.7"
    assert state["predecessor"] == {
        "deposition_id": 7,
        "doi": "10.5281/zenodo.7",
    }
    assert "token" not in json.dumps(state).casefold()
    publisher._verify_integrity(state, key="integrity", schema=publisher.ZENODO_STATE_SCHEMA)


def test_new_version_accepts_legacy_prereserved_version_doi() -> None:
    """Legacy drafts may expose the new DOI only under metadata.prereserve_doi."""
    session = _new_version_fixture(draft=_successor_draft(doi=None))

    state = _new_version(session)

    assert state["doi"] == "10.5281/zenodo.8"


def test_bound_new_version_rejects_returned_doi_before_metadata_put(tmp_path: Path) -> None:
    """A bound successor DOI mismatch is rejected before any metadata mutation."""
    binding, metadata = _successor_binding(tmp_path)
    mismatched_draft = _successor_draft(
        deposition_id=9,
        record_id=9,
        doi="10.5281/zenodo.9",
    )
    session = _new_version_fixture(draft=mismatched_draft)

    with pytest.raises(publisher.ZenodoPublisherError, match="version DOI"):
        publisher.new_version(
            session,
            metadata,
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag=SUCCESSOR_TAG,
            api_base="https://zenodo.org/api",
            release_binding=binding,
        )

    assert session.put_kwargs == []
    assert len(session.puts) == 1


def test_recover_restores_successor_lineage_for_inherited_file_cleanup(tmp_path: Path) -> None:
    """A recovered successor retains the provenance required to prune inherited files."""
    binding, metadata = _successor_binding(tmp_path)
    session = _Session()
    session.gets = [_Response(_successor_draft(metadata=metadata))]

    state = publisher.recover(session, 8, metadata, release_binding=binding)

    assert state["concept_doi"] == "10.5281/zenodo.6"
    assert state["predecessor_deposition_id"] == 7
    assert state["predecessor_doi"] == "10.5281/zenodo.7"
    assert state["predecessor"] == {
        "deposition_id": 7,
        "doi": "10.5281/zenodo.7",
    }
    assert state["source_tag"].endswith(SUCCESSOR_TAG)

    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft(metadata=metadata)
    remote["files"] = [inherited]
    post_upload = _successor_draft(metadata=metadata)
    post_upload["files"] = [inherited, uploaded]
    after_delete = _successor_draft(metadata=metadata)
    after_delete["files"] = [uploaded]
    session.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(after_delete),
        _Response(after_delete),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    updated = publisher.upload(session, state, [bundle], release_binding=binding)

    assert updated["files"][0]["name"] == bundle.name
    assert any(url.endswith("/deposit/depositions/8/files/inherited-file") for url in session.urls)
    assert updated["reconciliation_receipt"]["deleted_files"] == ["predecessor.tar.gz"]


@pytest.mark.parametrize(
    "latest_draft",
    [
        None,
        "http://zenodo.org/api/deposit/depositions/8",
        "https://other.test/api/deposit/depositions/8",
        "https://zenodo.org/not-the-api/deposit/depositions/8",
    ],
)
def test_new_version_rejects_malformed_or_cross_host_latest_draft(latest_draft: Any) -> None:
    """The new-version link cannot redirect the authenticated session elsewhere."""
    session = _Session()
    session.posts = [_Response({"links": {"latest_draft": latest_draft}}, status_code=201)]
    session.gets = [_Response(_predecessor_deposition())]

    with pytest.raises(publisher.ZenodoPublisherError, match="latest_draft"):
        _new_version(session)

    assert session.gets == []
    assert session.puts == []


@pytest.mark.parametrize(
    ("draft", "match"),
    [
        (_successor_draft(deposition_id=7, record_id=7, doi="10.5281/zenodo.7"), "reused"),
        (_successor_draft(doi="10.5281/zenodo.7"), "reused"),
        (_successor_draft(concept_record_id="99"), "concept ID"),
        (_successor_draft(submitted=True), "unpublished draft"),
    ],
)
def test_new_version_rejects_reused_identity_wrong_concept_or_submitted_draft(
    draft: dict[str, Any], match: str
) -> None:
    """Successor creation fails closed when the draft identity or state drifts."""
    session = _new_version_fixture(draft=draft)
    session.posts[0].payload["links"]["latest_draft"] = (
        f"https://zenodo.org/api/deposit/depositions/{draft['id']}"
    )

    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        _new_version(session)

    assert len(session.puts) == 1


def test_new_version_rejects_metadata_readback_mismatch() -> None:
    """A successful PUT transport response is insufficient when metadata changed."""
    readback = _successor_draft()
    readback["metadata"] = {**_metadata(), "title": "inherited title"}
    session = _new_version_fixture(readback=readback)

    with pytest.raises(publisher.ZenodoPublisherError, match="metadata readback mismatch"):
        _new_version(session)


@pytest.mark.parametrize(
    ("predecessor", "match"),
    [
        (_predecessor_deposition(doi="10.5281/zenodo.99"), "predecessor DOI"),
        (_predecessor_deposition(conceptrecid="99"), "concept ID"),
        (_predecessor_deposition(submitted=False, state="unsubmitted"), "published"),
        (_predecessor_deposition(record_id=9), "changed the requested identity"),
        (_predecessor_deposition(metadata=_metadata()), "source tag"),
    ],
)
def test_new_version_rejects_wrong_or_unpublished_predecessor(
    predecessor: dict[str, Any], match: str
) -> None:
    """No mutating action occurs until the predecessor identity is proven."""
    session = _new_version_fixture()
    session.gets[0] = _Response(predecessor)

    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        _new_version(session)

    assert session.urls == ["https://zenodo.org/api/deposit/depositions/7"]


def test_new_version_requires_exact_predecessor_relation_and_deposition_id() -> None:
    """A generic dataset deposition cannot silently become an erratum successor."""
    session = _new_version_fixture()
    metadata_without_predecessor = _new_version_metadata()
    metadata_without_predecessor["related_identifiers"] = [
        item
        for item in metadata_without_predecessor["related_identifiers"]
        if item.get("relation") != "isNewVersionOf"
    ]
    with pytest.raises(publisher.ZenodoPublisherError, match="isNewVersionOf"):
        publisher.new_version(
            session,
            metadata_without_predecessor,
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag=SUCCESSOR_TAG,
            api_base="https://zenodo.org/api",
        )
    assert session.urls == []

    with pytest.raises(publisher.ZenodoPublisherError, match="deposition ID"):
        publisher.new_version(
            session,
            _new_version_metadata(),
            predecessor_deposition_id=9,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag=SUCCESSOR_TAG,
            api_base="https://zenodo.org/api",
        )


@pytest.mark.parametrize(
    "relation_case",
    ["missing", "duplicate", "alternate", "wrong scheme"],
)
def test_new_version_rejects_non_unique_or_mismatched_predecessor_relation_before_remote_mutation(
    relation_case: str,
) -> None:
    """Every invalid predecessor relation fails before an authenticated request."""
    session = _new_version_fixture()
    metadata = _new_version_metadata()
    source_relation = next(
        item for item in metadata["related_identifiers"] if item["relation"] == "isSupplementTo"
    )
    predecessor_relation = {
        "identifier": "10.5281/zenodo.7",
        "relation": "isNewVersionOf",
        "scheme": "doi",
    }
    if relation_case == "missing":
        predecessor_relations: list[dict[str, str]] = []
    elif relation_case == "duplicate":
        predecessor_relations = [predecessor_relation, dict(predecessor_relation)]
    elif relation_case == "alternate":
        predecessor_relations = [
            predecessor_relation,
            {
                **predecessor_relation,
                "identifier": "10.5281/zenodo.99",
            },
        ]
    else:
        predecessor_relations = [
            {
                **predecessor_relation,
                "scheme": "url",
            }
        ]
    metadata["related_identifiers"] = [source_relation, *predecessor_relations]

    with pytest.raises(publisher.ZenodoPublisherError, match="isNewVersionOf"):
        _new_version_with_metadata(session, metadata)

    assert session.urls == []
    assert len(session.gets) == 2
    assert len(session.posts) == 1
    assert len(session.puts) == 1
    assert session.put_kwargs == []


@pytest.mark.parametrize("scheme", [None, "doi", "https"])
def test_new_version_rejects_non_url_successor_source_relation_before_remote_mutation(
    scheme: str | None,
) -> None:
    """A successor source relation must be an explicit URL before any API call."""
    session = _new_version_fixture()
    metadata = _new_version_metadata()
    source_relation = next(
        item for item in metadata["related_identifiers"] if item["relation"] == "isSupplementTo"
    )
    if scheme is None:
        source_relation.pop("scheme")
    else:
        source_relation["scheme"] = scheme

    with pytest.raises(publisher.ZenodoPublisherError, match="exact source tag"):
        publisher.new_version(
            session,
            metadata,
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag=SUCCESSOR_TAG,
            api_base="https://zenodo.org/api",
        )

    assert session.urls == []
    assert len(session.gets) == 2
    assert len(session.posts) == 1
    assert len(session.puts) == 1
    assert session.put_kwargs == []


def test_new_version_rejects_wrong_successor_tag_before_remote_mutation() -> None:
    """The exact successor GitHub tag is admitted before any authenticated request."""
    session = _new_version_fixture()

    with pytest.raises(publisher.ZenodoPublisherError, match="expected successor tag"):
        publisher.new_version(
            session,
            _new_version_metadata(),
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag="different-erratum.1",
            api_base="https://zenodo.org/api",
        )

    assert session.urls == []


@pytest.mark.parametrize(
    ("predecessor_tag", "source_sha", "successor_tag"),
    [
        (PREDECESSOR_TAG, "0" * 40, SUCCESSOR_TAG),
        ("semantic-release", SOURCE_SHA, "semantic-release-erratum.1"),
        (PREDECESSOR_TAG.upper(), SOURCE_SHA, f"{PREDECESSOR_TAG.upper()}-erratum.1"),
        (PREDECESSOR_TAG, SOURCE_SHA, f"{PREDECESSOR_TAG}-erratum.01"),
        (PREDECESSOR_TAG, SOURCE_SHA, f"{PREDECESSOR_TAG}-erratum.2"),
    ],
)
def test_new_version_rejects_noncanonical_lineage_before_remote_mutation(
    predecessor_tag: str, source_sha: str, successor_tag: str
) -> None:
    session = _new_version_fixture()

    with pytest.raises(
        publisher.ZenodoPublisherError, match="expected (predecessor|successor) tag"
    ):
        publisher.new_version(
            session,
            _new_version_metadata(),
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=predecessor_tag,
            expected_source_sha=source_sha,
            expected_successor_tag=successor_tag,
            api_base="https://zenodo.org/api",
        )

    assert session.urls == []


def test_token_file_missing_and_empty_are_rejected(tmp_path: Path) -> None:
    """Token input must exist and contain non-empty text."""
    with pytest.raises(publisher.ZenodoPublisherError, match="not found"):
        publisher.read_token_file(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    os.chmod(empty, 0o600)
    with pytest.raises(publisher.ZenodoPublisherError, match="empty"):
        publisher.read_token_file(empty)


def test_build_session_uses_header_only_token_and_requires_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Authentication is sent in a header and missing requests fails safely."""
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    os.chmod(token_file, 0o600)

    class Requests:
        """Minimal requests module fixture."""

        @staticmethod
        def Session() -> _Session:
            """Return an empty session."""
            return _Session()

    monkeypatch.setattr(publisher, "try_import", lambda name: Requests)
    session = publisher.build_session(token_file)
    assert session.headers == {"Authorization": "Bearer secret"}

    monkeypatch.setattr(publisher, "try_import", lambda name: None)
    with pytest.raises(publisher.ZenodoPublisherError, match="requests is required"):
        publisher.build_session(token_file)


@pytest.mark.parametrize(
    "api_base",
    [
        "http://zenodo.org/api",
        "https://zenodo.org/api?access_token=secret",
        "https://zenodo.org/api#fragment",
        "https://user:password@zenodo.org/api",
        "https://evil.example/api",
        "https://sandbox.zenodo.org/api",
        "https://zenodo.org/records",
        "https://zenodo.org/%61pi",
    ],
)
def test_authenticated_operations_reject_untrusted_api_base_before_network(
    api_base: str,
) -> None:
    """Every credential-bearing operation rejects a malicious API base first."""
    operation_calls = [
        lambda session: publisher.reserve(session, _metadata(), api_base=api_base),
        lambda session: publisher.upload(session, {}, [], api_base=api_base),
        lambda session: publisher.publish(session, {}, _metadata(), api_base=api_base),
        lambda session: publisher.verify(session, {}, _metadata(), api_base=api_base),
        lambda session: publisher.new_version(
            session,
            _new_version_metadata(),
            predecessor_deposition_id=7,
            expected_predecessor_doi="10.5281/zenodo.7",
            expected_concept_doi="10.5281/zenodo.6",
            expected_predecessor_tag=PREDECESSOR_TAG,
            expected_source_sha=SOURCE_SHA,
            expected_successor_tag=SUCCESSOR_TAG,
            api_base=api_base,
        ),
    ]

    for operation in operation_calls:
        session = _Session()
        with pytest.raises(publisher.ZenodoPublisherError, match="API base") as exc_info:
            operation(session)
        assert "secret" not in str(exc_info.value).casefold()
        assert session.urls == []


def test_api_base_normalizes_one_trailing_slash() -> None:
    """A canonical API base with a trailing slash never emits a double-slash endpoint."""
    session = _Session()
    session.posts = [_Response(_draft())]

    publisher.reserve(session, _metadata(), api_base="https://zenodo.org/api/")

    assert session.urls == ["https://zenodo.org/api/deposit/depositions"]


@pytest.mark.parametrize(
    "bucket",
    [
        "http://zenodo.org/api/files/bucket",
        "https://other.example/api/files/bucket",
        "https://zenodo.org/api/files/bucket?access_token=secret",
        "https://zenodo.org/records/7/files/bundle",
        "https://zenodo.org/apiary/files/bucket",
        "https://zenodo.org/api/deposit/depositions/7/files",
        "https://zenodo.org/api/files/bucket/extra",
        "https://zenodo.org/api/files/bucket%2Fextra",
        "https://zenodo.org/api/../records/7/files/bundle",
        "https://zenodo.org/api/%2e%2e/records/7/files/bundle",
    ],
)
def test_upload_rejects_cross_origin_or_credentialed_bucket_before_put(
    tmp_path: Path, bucket: str
) -> None:
    """A server-supplied bucket must stay on the approved API origin."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    draft = _draft()
    draft["links"]["bucket"] = bucket
    session = _Session()
    session.gets = [_Response(draft)]

    with pytest.raises(publisher.ZenodoPublisherError, match="secure upload bucket") as exc_info:
        publisher.upload(session, state, [bundle], api_base="https://zenodo.org/api")

    assert "secret" not in str(exc_info.value).casefold()
    assert session.urls == ["https://zenodo.org/api/deposit/depositions/7"]
    assert session.puts == []


def test_verify_rejects_cross_origin_download_without_authenticated_fetch(tmp_path: Path) -> None:
    """Verification reports an unapproved download URL without requesting it."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": publisher._sha256_file(bundle),
                }
            ],
        }
    )
    remote = _draft()
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": "https://evil.example/files/bundle?access_token=secret",
            },
        }
    ]
    session = _Session()
    session.gets = [_Response(remote)]

    report = publisher.verify(session, state, _metadata(), api_base="https://zenodo.org/api")

    assert report["status"] == "fail"
    assert any("secure download URL" in problem for problem in report["problems"])
    assert session.urls == ["https://zenodo.org/api/deposit/depositions/7"]
    assert "secret" not in json.dumps(report).casefold()


def test_verification_receipt_hashes_remote_version_without_echoing_server_value(
    tmp_path: Path,
) -> None:
    """Receipts bind a remote version while never persisting its raw server value."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": publisher._sha256_file(bundle),
                }
            ],
        }
    )
    remote = _draft()
    remote["modified"] = "Bearer secret-that-must-not-appear"
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/files/bundle/content"},
        }
    ]
    session = _Session()
    session.gets = [_Response(remote), _Response({}, content=bundle.read_bytes())]

    report = publisher.verify(session, state, _metadata(), api_base="https://zenodo.org/api")

    assert report["status"] == "pass"
    binding = report["receipt"]["remote_optimistic"]
    assert binding["field"] == "modified"
    assert len(binding["sha256"]) == 64
    serialized = json.dumps(report["receipt"])
    assert "Bearer" not in serialized
    assert "secret-that-must-not-appear" not in serialized


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"upload_type": "dataset"}, "top-level metadata"),
        ({"metadata": {"upload_type": "software"}}, "upload_type=dataset"),
        ({"metadata": {"upload_type": "dataset", "license": "MIT"}}, "license"),
        (
            {"metadata": {"upload_type": "dataset", "license": "GPL-3.0-only", "creators": []}},
            "at least one creator",
        ),
        (
            {
                "metadata": {
                    "upload_type": "dataset",
                    "license": "GPL-3.0-only",
                    "creators": [{"name": "Creator"}],
                    "related_identifiers": [],
                }
            },
            "exact source tag",
        ),
    ],
)
def test_metadata_contract_rejects_invalid_publication_metadata(
    tmp_path: Path, metadata: dict[str, Any], match: str
) -> None:
    """Dataset publication metadata is validated before any API call."""
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        publisher.load_dataset_metadata(path)


@pytest.mark.parametrize(
    ("response", "match"),
    [(_Response({}, status_code=500), "request failed"), (_Response([], 200), "not a JSON object")],
)
def test_json_object_requires_successful_mapping_response(response: _Response, match: str) -> None:
    """Zenodo responses must be successful JSON objects."""
    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        publisher._json_object(response, "fixture")


def test_reserve_rejects_incomplete_deposition_identity() -> None:
    """A reserved DOI cannot be accepted without complete deposition identity."""
    session = _Session()
    session.posts = [_Response({"id": 7, "conceptrecid": "6", "metadata": {}})]
    with pytest.raises(publisher.ZenodoPublisherError, match="response ID"):
        publisher.reserve(session, _metadata())


@pytest.mark.parametrize("lifecycle", ["inprogress", "error"])
def test_reserve_rejects_unstable_remote_lifecycle(lifecycle: str) -> None:
    """Reservation cannot seal a transient or failed remote deposition state."""
    response = _draft()
    response["state"] = lifecycle
    session = _Session()
    session.posts = [_Response(response)]

    with pytest.raises(publisher.ZenodoPublisherError, match="unpublished draft"):
        publisher.reserve(session, _metadata())


def test_public_state_normalizes_verified_identity() -> None:
    """State extraction normalizes only the supported Zenodo identity fields."""
    payload = _draft()
    payload["conceptrecid"] = "006"

    state = publisher._public_state(payload)

    assert state["deposition_id"] == 7
    assert state["record_id"] == 7
    assert state["concept_record_id"] == "6"
    assert state["doi"] == "10.5281/zenodo.7"
    assert state["state"] == "unsubmitted"
    assert state["submitted"] is False
    assert state["files"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "7?access_token=query-injected-id"),
        ("record_id", "7?access_token=query-injected-id"),
        ("conceptrecid", "7?access_token=query-injected-id"),
        ("doi", "Bearer secret-reflection"),
        ("submitted", "false"),
        ("state", "Bearer secret-reflection"),
    ],
)
def test_public_state_rejects_untrusted_identity_and_state_values(field: str, value: Any) -> None:
    """Server response values cannot become arbitrary persisted state fields."""
    payload = _draft()
    payload[field] = value

    with pytest.raises(publisher.ZenodoPublisherError) as exc_info:
        publisher._public_state(payload)

    assert "query-injected-id" not in str(exc_info.value)
    assert "secret-reflection" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deposition_id", "7?access_token=query-injected-id"),
        ("record_id", "7?access_token=query-injected-id"),
        ("concept_record_id", "6?access_token=query-injected-id"),
        ("doi", "10.5281/zenodo.999"),
        ("submitted", "false"),
        ("state", "inprogress"),
        ("state", "done"),
    ],
)
def test_operation_rejects_self_hashed_untrusted_state_before_network(
    field: str, value: Any
) -> None:
    """A valid self-hash cannot authorize unsafe state values for an operation."""
    payload: dict[str, Any] = {
        "schema_version": publisher.ZENODO_STATE_SCHEMA,
        "deposition_id": 7,
        "record_id": 7,
        "concept_record_id": "6",
        "doi": "10.5281/zenodo.7",
        "submitted": False,
        "state": "unsubmitted",
        "files": [],
    }
    payload[field] = value
    state = publisher._seal_state(payload)
    session = _Session()

    with pytest.raises(publisher.ZenodoPublisherError):
        publisher.upload(session, state, [])

    assert session.urls == []


def test_upload_does_not_persist_server_checksum_or_reflected_secret(tmp_path: Path) -> None:
    """Only the local SHA-256 belongs in credential-free upload state."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    session = _Session()
    uploaded = _draft_file(bundle.name, deposition_id=7, file_id="uploaded-file")
    post_upload = _draft()
    post_upload["files"] = [uploaded]
    session.gets = [_Response(_draft()), _Response(post_upload)]
    session.puts = [_Response({"checksum": "Bearer secret-reflection"})]

    updated = publisher.upload(session, state, [bundle])
    serialized = json.dumps(updated, sort_keys=True)

    assert "secret-reflection" not in serialized
    assert "zenodo_checksum" not in updated["files"][0]
    assert updated["files"][0]["sha256"] == publisher._sha256_file(bundle)


def test_upload_requires_exact_post_upload_inventory_without_extras(tmp_path: Path) -> None:
    """Even an initially clean draft is reread and must expose every intended upload."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    session = _Session()
    missing_upload = _successor_draft()
    session.gets = [_Response(_successor_draft()), _Response(missing_upload)]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    with pytest.raises(publisher.ZenodoPublisherError, match="changed unexpectedly"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.deletes == []
    assert session.urls[-1] == "https://zenodo.org/api/deposit/depositions/8"


def test_upload_reconciles_inherited_successor_files(tmp_path: Path) -> None:
    """A bound successor uploads first, then removes only its inherited extra files."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    after_delete = _successor_draft()
    after_delete["files"] = [uploaded]
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(after_delete),
        _Response(after_delete),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    updated = publisher.upload(session, _successor_state(), [bundle])

    delete_urls = session.delete_urls
    assert delete_urls == ["https://zenodo.org/api/deposit/depositions/8/files/inherited-file"]
    assert all("/deposit/depositions/7/" not in url for url in delete_urls)
    assert session.urls.count("https://zenodo.org/api/deposit/depositions/8") == 5
    assert not any(url.endswith("/deposit/depositions/8/files") for url in session.urls)
    fresh_file_url = "https://zenodo.org/api/deposit/depositions/8/files/inherited-file"
    fresh_file_index = session.urls.index(fresh_file_url)
    assert session.urls[fresh_file_index + 1] == delete_urls[0]
    assert session.urls.index("https://zenodo.org/api/files/bucket/successor.tar.gz") < (
        fresh_file_index
    )
    assert updated["files"] == [
        {
            "name": bundle.name,
            "size": bundle.stat().st_size,
            "sha256": publisher._sha256_file(bundle),
        }
    ]
    reconciliation = updated["reconciliation_receipt"]
    assert reconciliation["intended_inventory_sha256"] == publisher._inventory_sha256(
        updated["files"]
    )
    assert reconciliation["deleted_files"] == ["predecessor.tar.gz"]
    assert reconciliation["deleted_file_count"] == 1
    assert reconciliation["final_remote_state"] == "unsubmitted"
    assert reconciliation["final_remote_submitted"] is False
    assert reconciliation["final_remote_revision"]["field"] == "snapshot"
    publisher._verify_integrity(
        reconciliation,
        key="integrity",
        schema=publisher.ZENODO_RECONCILIATION_SCHEMA,
    )


def test_reconciliation_receipt_rejects_tampered_inventory_binding(tmp_path: Path) -> None:
    """Receipt self-integrity and intended-inventory binding are both enforced."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    files = [
        {
            "name": bundle.name,
            "size": bundle.stat().st_size,
            "sha256": publisher._sha256_file(bundle),
        }
    ]
    deposition = _successor_draft()
    receipt = publisher._reconciliation_receipt(
        deposition,
        {bundle.name: "successor-file"},
        files,
        [],
    )
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 8,
            "record_id": 8,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.8",
            "submitted": False,
            "state": "unsubmitted",
            "files": files,
            "reconciliation_receipt": receipt,
        }
    )
    tampered_receipt = dict(receipt)
    tampered_receipt["intended_inventory_sha256"] = "0" * 64
    tampered_receipt = publisher._seal_payload(
        tampered_receipt,
        "integrity",
        publisher.ZENODO_RECONCILIATION_SCHEMA,
    )
    tampered_state = dict(state)
    tampered_state["reconciliation_receipt"] = tampered_receipt
    tampered_state = publisher._seal_state(tampered_state)

    with pytest.raises(publisher.ZenodoPublisherError, match="inventory digest"):
        publisher._validate_state_for_operation(tampered_state)


def test_reconciliation_receipt_rejects_credential_shaped_deleted_filename() -> None:
    """A remote filename that looks like a credential cannot enter a receipt."""
    files = [{"name": "successor.tar.gz", "size": 1, "sha256": "0" * 64}]

    with pytest.raises(publisher.ZenodoPublisherError, match="unsafe deleted filename") as exc_info:
        publisher._reconciliation_receipt(
            _successor_draft(),
            {"successor.tar.gz": "successor-file"},
            files,
            ["Bearer secret-reflection"],
        )

    assert "secret-reflection" not in str(exc_info.value)


@pytest.mark.parametrize(
    "file_resource",
    [
        _draft_file("predecessor.tar.gz", file_id="changed-file"),
        {
            **_draft_file("predecessor.tar.gz"),
            "links": {
                "self": ("https://zenodo.org/api/deposit/depositions/7/files/inherited-file")
            },
        },
    ],
)
def test_upload_rejects_successor_file_identity_drift_before_delete(
    tmp_path: Path,
    file_resource: dict[str, Any],
) -> None:
    """A file-level readback must bind both identity and URL before DELETE."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    pre_delete = _successor_draft()
    pre_delete["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(pre_delete),
        _Response(file_resource),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    with pytest.raises(publisher.ZenodoPublisherError, match="identity|deposition"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.delete_urls == []


@pytest.mark.parametrize(
    "delete_status",
    [
        404,
        403,
        500,
        200,
    ],
)
def test_upload_conditionally_accepts_classified_delete_with_stable_absence(
    tmp_path: Path,
    delete_status: int,
) -> None:
    """Non-success DELETE results are idempotent only after exact stable readback."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    after_delete = _successor_draft()
    after_delete["files"] = [uploaded]
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(after_delete),
        _Response(after_delete),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, delete_status)]

    updated = publisher.upload(session, _successor_state(), [bundle])

    assert updated["reconciliation_receipt"]["deleted_files"] == ["predecessor.tar.gz"]
    assert session.delete_urls == [
        "https://zenodo.org/api/deposit/depositions/8/files/inherited-file"
    ]


def test_upload_conditionally_accepts_network_delete_after_stable_absence(tmp_path: Path) -> None:
    """A network DELETE failure is accepted only when the exact readback proves absence."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    after_delete = _successor_draft()
    after_delete["files"] = [uploaded]
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(after_delete),
        _Response(after_delete),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    def fail_delete(url: str, **kwargs: Any) -> _Response:
        """Simulate an interrupted DELETE request."""
        del url, kwargs
        raise OSError("connection reset")

    with patch.object(session, "delete", side_effect=fail_delete):
        updated = publisher.upload(session, _successor_state(), [bundle])

    assert updated["reconciliation_receipt"]["deleted_files"] == ["predecessor.tar.gz"]


def test_upload_requires_bounded_stable_remote_readback(tmp_path: Path) -> None:
    """A changing remote revision never becomes a successful cleanup receipt."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    readbacks: list[dict[str, Any]] = []
    for revision in ("one", "two", "three"):
        readback = _successor_draft()
        readback["files"] = [uploaded]
        readback["modified"] = revision
        readbacks.append(readback)
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        *[_Response(readback) for readback in readbacks],
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    with pytest.raises(publisher.ZenodoPublisherError, match="stable readback"):
        publisher.upload(session, _successor_state(), [bundle])

    assert len(session.gets) == 0
    assert len(session.urls) == 9


def test_upload_rejects_lifecycle_drift_during_pre_delete_readback(tmp_path: Path) -> None:
    """A lifecycle change in the exact pre-delete response blocks DELETE."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    published = _successor_draft(submitted=True)
    published["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [_Response(initial), _Response(post_upload), _Response(published)]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    with pytest.raises(publisher.ZenodoPublisherError, match="unpublished draft"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.delete_urls == []


@pytest.mark.parametrize(
    ("delete_status", "outcome"),
    [
        (404, "not_found"),
        (403, "forbidden"),
        (500, "server_error"),
        (200, "unexpected"),
    ],
)
def test_upload_fails_closed_when_delete_readback_cannot_confirm_absence(
    tmp_path: Path,
    delete_status: int,
    outcome: str,
) -> None:
    """A classified DELETE result cannot authorize cleanup while the target remains."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    still_present = _successor_draft()
    still_present["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, delete_status)]
    state = _successor_state()

    with pytest.raises(publisher.ZenodoPublisherError, match=f"DELETE {outcome}"):
        publisher.upload(session, state, [bundle])

    assert state["files"] == []
    assert len(session.urls) == 9


def test_upload_fails_closed_when_network_delete_readback_cannot_confirm_absence(
    tmp_path: Path,
) -> None:
    """A network DELETE failure stays blocked when the target remains present."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    initial = _successor_draft()
    initial["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    still_present = _successor_draft()
    still_present["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [
        _Response(initial),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    state = _successor_state()

    def fail_delete(url: str, **kwargs: Any) -> _Response:
        """Simulate a network error before the server reports DELETE status."""
        del url, kwargs
        raise OSError("connection reset")

    with (
        patch.object(session, "delete", side_effect=fail_delete),
        pytest.raises(publisher.ZenodoPublisherError, match="DELETE network"),
    ):
        publisher.upload(session, state, [bundle])

    assert state["files"] == []
    assert len(session.urls) == 8


def test_upload_rejects_unbound_draft_with_unexpected_files_before_put(tmp_path: Path) -> None:
    """A fresh reservation cannot silently prune an unrelated draft file."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    remote = _draft()
    remote["files"] = [_draft_file("unrelated.txt", deposition_id=7)]
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(publisher.ZenodoPublisherError, match="predecessor deposition ID"):
        publisher.upload(session, state, [bundle])

    assert session.puts == []
    assert session.deletes == []


@pytest.mark.parametrize(
    "files",
    [
        "not-a-list",
        ["not-an-object"],
        [_draft_file("../unsafe", file_id="safe-id")],
        [_draft_file("duplicate"), _draft_file("duplicate", file_id="file-2")],
        [_draft_file("bad-id", file_id="unsafe/id")],
        [_draft_file("one", file_id="same-id"), _draft_file("two", file_id="same-id")],
        [{**_draft_file("filename-alias"), "key": "different-alias"}],
        [{**_draft_file("id-alias"), "file_id": "different-id"}],
        [
            {
                **_draft_file("wrong-link", file_id="file-1"),
                "links": {"self": "https://zenodo.org/api/deposit/depositions/7/files/file-1"},
            }
        ],
    ],
)
def test_upload_rejects_malformed_or_ambiguous_remote_inventory_before_put(
    tmp_path: Path,
    files: object,
) -> None:
    """Malformed remote file identities never reach upload or deletion."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    remote = _successor_draft()
    remote["files"] = files
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(publisher.ZenodoPublisherError):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.puts == []
    assert session.deletes == []


def test_upload_blocks_inventory_race_without_deleting_new_file(tmp_path: Path) -> None:
    """A new unknown file after PUT blocks cleanup instead of being deleted."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    raced = _draft_file("raced.txt", file_id="raced-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    raced_after_upload = _successor_draft()
    raced_after_upload["files"] = [inherited, uploaded, raced]
    session = _Session()
    session.gets = [_Response(remote), _Response(raced_after_upload)]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    with pytest.raises(publisher.ZenodoPublisherError, match="changed unexpectedly"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.deletes == []


def test_upload_requires_204_when_deleting_inherited_file(tmp_path: Path) -> None:
    """A non-204 deletion response leaves caller state unchanged and blocks completion."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    still_present = _successor_draft()
    still_present["files"] = [inherited, uploaded]
    state = _successor_state()
    state_before = json.dumps(state, sort_keys=True)
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 200)]

    with pytest.raises(publisher.ZenodoPublisherError, match="DELETE unexpected"):
        publisher.upload(session, state, [bundle])

    assert json.dumps(state, sort_keys=True) == state_before


def test_upload_partial_delete_is_retryable(tmp_path: Path) -> None:
    """A retry safely resumes after one of multiple inherited-file deletions fails."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited_a = _draft_file("a-predecessor.tar.gz", file_id="inherited-a")
    inherited_b = _draft_file("b-predecessor.tar.gz", file_id="inherited-b")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited_a, inherited_b]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited_a, inherited_b, uploaded]
    after_first_delete = _successor_draft()
    after_first_delete["files"] = [inherited_b, uploaded]
    still_present = _successor_draft()
    still_present["files"] = [inherited_b, uploaded]
    state = _successor_state()
    state_before = json.dumps(state, sort_keys=True)
    first = _Session()
    first.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited_a),
        _Response(after_first_delete),
        _Response(after_first_delete),
        _Response(still_present),
        _Response(inherited_b),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
    ]
    first.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    first.deletes = [_Response({}, 204), _Response({}, 500)]

    with pytest.raises(publisher.ZenodoPublisherError, match="DELETE server_error"):
        publisher.upload(first, state, [bundle])
    assert json.dumps(state, sort_keys=True) == state_before

    retry_remote = _successor_draft()
    retry_remote["files"] = [inherited_b, uploaded]
    retry_post_upload = _successor_draft()
    retry_post_upload["files"] = [inherited_b, uploaded]
    retry_after_delete = _successor_draft()
    retry_after_delete["files"] = [uploaded]
    retry = _Session()
    retry.gets = [
        _Response(retry_remote),
        _Response(retry_post_upload),
        _Response(retry_post_upload),
        _Response(inherited_b),
        _Response(retry_after_delete),
        _Response(retry_after_delete),
    ]
    retry.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    retry.deletes = [_Response({}, 204)]

    updated = publisher.upload(retry, state, [bundle])
    assert updated["files"][0]["name"] == bundle.name
    assert any(url.endswith("/files/inherited-b") for url in retry.urls)


def test_upload_rejects_inherited_identity_drift_before_delete(tmp_path: Path) -> None:
    """A same-name inherited file with a changed ID is never deleted by stale identity."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-old")
    changed = _draft_file("predecessor.tar.gz", file_id="inherited-new")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    changed_after_upload = _successor_draft()
    changed_after_upload["files"] = [changed, uploaded]
    session = _Session()
    session.gets = [_Response(remote), _Response(changed_after_upload)]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    with pytest.raises(publisher.ZenodoPublisherError, match="identity changed"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.deletes == []


def test_upload_rejects_fresh_successor_file_identity_drift_before_delete(tmp_path: Path) -> None:
    """A fresh file read with a changed ID is never deleted by stale identity."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-old")
    changed = _draft_file("predecessor.tar.gz", file_id="inherited-new")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(changed),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    with pytest.raises(publisher.ZenodoPublisherError, match="identity changed"):
        publisher.upload(session, _successor_state(), [bundle])

    assert session.deletes == []


def test_upload_requires_exact_inventory_after_cleanup(tmp_path: Path) -> None:
    """A stale or newly added extra after deletion prevents state acceptance."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    inherited = _draft_file("predecessor.tar.gz", file_id="inherited-file")
    uploaded = _draft_file(bundle.name, file_id="successor-file")
    remote = _successor_draft()
    remote["files"] = [inherited]
    post_upload = _successor_draft()
    post_upload["files"] = [inherited, uploaded]
    still_present = _successor_draft()
    still_present["files"] = [inherited, uploaded]
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response(post_upload),
        _Response(post_upload),
        _Response(inherited),
        _Response(still_present),
        _Response(still_present),
        _Response(still_present),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]
    session.deletes = [_Response({}, 204)]

    with pytest.raises(publisher.ZenodoPublisherError, match="DELETE success"):
        publisher.upload(session, _successor_state(), [bundle])


def test_upload_rejects_duplicate_local_basenames_before_network(tmp_path: Path) -> None:
    """Two local paths cannot silently overwrite one remote filename."""
    first = tmp_path / "one" / "bundle.tar.gz"
    second = tmp_path / "two" / "bundle.tar.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    session = _Session()

    with pytest.raises(publisher.ZenodoPublisherError, match="duplicate filename"):
        publisher.upload(session, _successor_state(), [first, second])

    assert session.urls == []


@pytest.mark.parametrize(
    "filename",
    ["query?name.tar", "fragment#name.tar", "encoded%2Fname.tar", "control\x01name.tar"],
)
def test_upload_rejects_unsafe_local_filenames_before_network(
    tmp_path: Path,
    filename: str,
) -> None:
    """Local names cannot smuggle URL delimiters, encodings, or controls."""
    path = tmp_path / filename
    path.write_bytes(b"unsafe-name")
    session = _Session()

    with pytest.raises(publisher.ZenodoPublisherError, match="filename"):
        publisher.upload(session, _successor_state(), [path])

    assert session.urls == []


def test_upload_rejects_symlink_input_before_network(tmp_path: Path) -> None:
    """A symlink input cannot change its target between validation and PUT."""
    target = tmp_path / "target.tar.gz"
    target.write_bytes(b"target")
    link = tmp_path / "bundle.tar.gz"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not support symlinks")
    session = _Session()

    with pytest.raises(publisher.ZenodoPublisherError, match="symlink"):
        publisher.upload(session, _successor_state(), [link])

    assert session.urls == []


def test_upload_orders_local_files_by_canonical_filename(tmp_path: Path) -> None:
    """Upload order is deterministic even when callers provide paths reversed."""
    first = tmp_path / "z-last.tar.gz"
    second = tmp_path / "a-first.tar.gz"
    first.write_bytes(b"last")
    second.write_bytes(b"first")
    remote = _successor_draft()
    remote["files"] = [
        _draft_file(second.name, file_id="a-file"),
        _draft_file(first.name, file_id="z-file"),
    ]
    session = _Session()
    session.gets = [_Response(_successor_draft()), _Response(remote)]
    session.puts = [
        _Response({"checksum": "md5:first"}, 201),
        _Response({"checksum": "md5:last"}, 201),
    ]

    updated = publisher.upload(session, _successor_state(), [first, second])

    assert [
        url for url in session.urls if url.startswith("https://zenodo.org/api/files/bucket/")
    ] == [
        "https://zenodo.org/api/files/bucket/a-first.tar.gz",
        "https://zenodo.org/api/files/bucket/z-last.tar.gz",
    ]
    assert [file["name"] for file in updated["files"]] == ["a-first.tar.gz", "z-last.tar.gz"]


def test_upload_rechecks_each_file_immediately_before_its_put(tmp_path: Path) -> None:
    """A mutation after one PUT blocks the next PUT before it starts."""
    first = tmp_path / "a-first.tar.gz"
    second = tmp_path / "b-second.tar.gz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    session = _Session()
    session.gets = [_Response(_successor_draft())]
    session.puts = [
        _Response({"checksum": "md5:first"}, 201),
        _Response({"checksum": "md5:second"}, 201),
    ]
    original_put = session.put

    def put_and_change_second(url: str, **kwargs: Any) -> _Response:
        """Mutate the next input after the first PUT response returns."""
        response = original_put(url, **kwargs)
        if url.endswith(first.name):
            second.write_bytes(b"changed before second PUT")
        return response

    with (
        patch.object(session, "put", side_effect=put_and_change_second),
        pytest.raises(publisher.ZenodoPublisherError, match="immediately before transfer"),
    ):
        publisher.upload(session, _successor_state(), [first, second])

    assert len(session.puts) == 1
    assert session.deletes == []


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_upload_rejects_missing_or_empty_local_inventory_before_network(
    tmp_path: Path,
    kind: str,
) -> None:
    """A missing inventory or empty file cannot begin an authenticated read."""
    files: list[Path] = []
    if kind == "empty":
        empty = tmp_path / "empty.tar.gz"
        empty.touch()
        files = [empty]
    session = _Session()

    with pytest.raises(publisher.ZenodoPublisherError, match="at least one|empty"):
        publisher.upload(session, _successor_state(), files)

    assert session.urls == []


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {
                "predecessor_deposition_id": 8,
                "predecessor_doi": "10.5281/zenodo.8",
                "predecessor": {"deposition_id": 8, "doi": "10.5281/zenodo.8"},
            },
            "reuses",
        ),
        (
            {"predecessor": {"deposition_id": 7, "doi": "10.5281/zenodo.9"}},
            "predecessor",
        ),
        ({"concept_doi": "10.5281/zenodo.9"}, "concept DOI"),
    ],
)
def test_successor_cleanup_rejects_incoherent_lineage(
    updates: dict[str, object],
    match: str,
) -> None:
    """All successor/predecessor/concept identities must agree before cleanup."""
    state = _successor_state()
    state.update(updates)

    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        publisher._validate_successor_cleanup_state(state)


def test_post_upload_inventory_http_failure_blocks_cleanup() -> None:
    """A failed inventory refresh is not interpreted as an empty draft."""
    session = _Session()
    session.gets = [_Response({}, 500)]

    with pytest.raises(publisher.ZenodoPublisherError, match="inventory request failed"):
        publisher._list_draft_files(
            session,
            deposition_id=8,
            api_base="https://zenodo.org/api",
            operation="post-upload draft file inventory",
        )


def test_upload_rejects_local_file_change_during_transfer(tmp_path: Path) -> None:
    """The sealed local digest must still match after the streamed PUT completes."""
    bundle = tmp_path / "successor.tar.gz"
    bundle.write_bytes(b"successor")
    session = _Session()
    session.gets = [_Response(_successor_draft())]
    session.puts = [_Response({"checksum": "md5:fixture"}, 201)]

    with (
        patch.object(
            publisher,
            "_sha256_file",
            side_effect=["a" * 64, "a" * 64, "b" * 64],
        ),
        pytest.raises(publisher.ZenodoPublisherError, match="changed during transfer"),
    ):
        publisher.upload(session, _successor_state(), [bundle])


@pytest.mark.parametrize("field", ["id", "record_id", "conceptrecid", "doi"])
def test_upload_binds_authenticated_draft_identity_before_put(field: str, tmp_path: Path) -> None:
    """Upload rejects a draft response whose identity is not the sealed deposition."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    remote = _draft()
    if field == "id":
        remote["id"] = 8
    elif field == "record_id":
        remote["record_id"] = 8
        remote["doi"] = "10.5281/zenodo.8"
        remote["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.8"}
    elif field == "conceptrecid":
        remote["conceptrecid"] = "8"
    else:
        remote["doi"] = "10.5072/zenodo.7"
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(publisher.ZenodoPublisherError):
        publisher.upload(session, state, [bundle])

    assert session.puts == []


@pytest.mark.parametrize("lifecycle", ["inprogress", "error"])
def test_upload_rejects_unstable_remote_draft_before_put(lifecycle: str, tmp_path: Path) -> None:
    """Upload accepts only a stable unpublished draft response."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    remote = _draft()
    remote["state"] = lifecycle
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(publisher.ZenodoPublisherError, match="unpublished draft"):
        publisher.upload(session, state, [bundle])

    assert session.puts == []


def test_verify_rejects_reflected_doi_before_sealing_a_receipt(tmp_path: Path) -> None:
    """A reflected secret cannot enter a serialized verification receipt."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [{"name": bundle.name, "size": 6, "sha256": publisher._sha256_file(bundle)}],
        }
    )
    remote = _draft()
    remote["doi"] = "Bearer secret-reflection"
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/records/7/files/bundle/content"},
        }
    ]
    session = _Session()
    session.gets = [_Response(remote), _Response({}, content=bundle.read_bytes())]
    state_before = json.dumps(state, sort_keys=True)

    with pytest.raises(publisher.ZenodoPublisherError, match="DOI"):
        publisher.verify(session, state, _metadata())

    assert json.dumps(state, sort_keys=True) == state_before
    assert "secret-reflection" not in json.dumps(state, sort_keys=True)


@pytest.mark.parametrize("lifecycle", ["inprogress", "error"])
def test_verify_rejects_unstable_remote_lifecycle_before_download(lifecycle: str) -> None:
    """Verification does not treat transient or failed drafts as usable records."""
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    state_before = json.dumps(state, sort_keys=True)
    remote = _draft()
    remote["state"] = lifecycle
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(publisher.ZenodoPublisherError, match="lifecycle state"):
        publisher.verify(session, state, _metadata())

    assert session.urls == ["https://zenodo.org/api/deposit/depositions/7"]
    assert json.dumps(state, sort_keys=True) == state_before


@pytest.mark.parametrize(
    ("state", "draft", "match"),
    [
        ({}, _draft(), "no deposition_id"),
        ({"deposition_id": 7}, {"links": {"bucket": "http://insecure"}}, "invalid Zenodo"),
        ({"deposition_id": 7}, {"links": {}}, "invalid Zenodo"),
    ],
)
def test_upload_rejects_missing_identity_or_insecure_bucket(
    state: dict[str, Any], draft: dict[str, Any], match: str
) -> None:
    """Upload requires a valid draft and HTTPS bucket."""
    session = _Session()
    session.gets = [_Response(draft)] if state.get("deposition_id") else []
    with pytest.raises(publisher.ZenodoPublisherError, match=match):
        publisher.upload(session, state, [])


def test_upload_rejects_missing_file_and_publish_rejects_unsubmitted_response(
    tmp_path: Path,
) -> None:
    """Missing local files and incomplete publish responses fail closed."""
    session = _Session()
    session.gets = [_Response(_draft())]
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    with pytest.raises(publisher.ZenodoPublisherError, match="file not found"):
        publisher.upload(session, state, [tmp_path / "missing.tar"])

    state_for_publish = publisher._seal_state(
        {
            **{key: value for key, value in state.items() if key != "integrity"},
            "files": [{"name": "bundle.tar", "size": 1, "sha256": "0" * 64}],
        }
    )
    session.posts = [_Response(_draft(submitted=False))]
    with pytest.raises(publisher.ZenodoPublisherError, match="verification receipt"):
        publisher.publish(session, state_for_publish, _metadata())
    assert len(session.posts) == 1
    with pytest.raises(publisher.ZenodoPublisherError, match="no deposition_id"):
        publisher.publish(session, {}, _metadata())


def test_verify_reports_inventory_transport_and_checksum_mismatches(tmp_path: Path) -> None:
    """Verification reports each remote mismatch without treating it as success."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"expected")
    state = {
        "schema_version": publisher.ZENODO_STATE_SCHEMA,
        "deposition_id": 7,
        "record_id": 7,
        "doi": "10.5281/zenodo.7",
        "concept_record_id": "6",
        "submitted": True,
        "state": "done",
        "files": [{"name": bundle.name, "size": bundle.stat().st_size, "sha256": "0" * 64}],
    }
    state = publisher._seal_state(state)
    remote = _draft(submitted=True)
    remote["metadata"] = {**_metadata(), "title": "different"}
    published_files = [
        {
            "key": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"self": "http://insecure"},
        },
        {"key": "unexpected.tar", "links": {}},
    ]
    session = _Session()
    session.gets = [_Response(remote), _Response(_published_record(published_files))]
    report = publisher.verify(session, state, _metadata())
    assert report["status"] == "fail"
    assert report["problem_count"] >= 3

    published_files = [
        {
            "key": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"self": "https://zenodo.org/api/files/bundle/content"},
        }
    ]
    session.gets = [
        _Response(remote),
        _Response(_published_record(published_files)),
        _Response({}, status_code=503),
    ]
    report = publisher.verify(session, state, _metadata())
    assert any("download failed" in problem for problem in report["problems"])

    session.gets = [
        _Response(remote),
        _Response(_published_record(published_files)),
        _Response({}, content=b"wrong"),
    ]
    report = publisher.verify(session, state, _metadata())
    assert any("SHA-256" in problem for problem in report["problems"])

    session = _Session()
    session.gets = [_Response(remote), _Response(_published_record([]))]
    report = publisher.verify(session, state, _metadata())
    assert report["status"] == "fail"
    assert any("inventory is empty" in problem for problem in report["problems"])


def test_verify_does_not_reflect_secret_shaped_remote_filename(tmp_path: Path) -> None:
    """Remote duplicate filenames are reported without echoing server strings."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "doi": "10.5281/zenodo.7",
            "concept_record_id": "6",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": publisher._sha256_file(bundle),
                }
            ],
        }
    )
    secret_name = "Bearer secret-reflection"
    remote = _draft()
    remote["files"] = [
        {"filename": secret_name, "size": bundle.stat().st_size, "links": {}},
        {"filename": secret_name, "size": bundle.stat().st_size, "links": {}},
    ]
    session = _Session()
    session.gets = [_Response(remote)]

    report = publisher.verify(session, state, _metadata())

    assert report["status"] == "fail"
    serialized = json.dumps(report, sort_keys=True)
    assert secret_name not in serialized
    assert "invalid or conflicting filename at index 0" in serialized


@pytest.mark.parametrize(
    "updates",
    [
        {"key": "different-name"},
        {"id": "file-a", "file_id": "file-b"},
    ],
)
def test_verify_rejects_conflicting_remote_aliases_without_download(
    tmp_path: Path,
    updates: dict[str, str],
) -> None:
    """Verification never downloads a file selected from conflicting aliases."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "doi": "10.5281/zenodo.7",
            "concept_record_id": "6",
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": publisher._sha256_file(bundle),
                }
            ],
        }
    )
    remote = _draft()
    remote["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {"download": "https://zenodo.org/api/files/bundle/content"},
            **updates,
        }
    ]
    session = _Session()
    session.gets = [_Response(remote)]

    report = publisher.verify(session, state, _metadata())

    assert report["status"] == "fail"
    assert any("conflicting" in problem for problem in report["problems"])
    assert session.gets == []
    assert session.urls == ["https://zenodo.org/api/deposit/depositions/7"]


@pytest.mark.parametrize(
    ("field", "value", "problem"),
    [
        ("id", 8, "published record id does not match reserved state"),
        ("conceptrecid", "8", "published record concept id does not match reserved state"),
        ("doi", "10.5281/zenodo.8", "published record DOI does not match reserved state"),
        ("status", "draft", "published record status is not published"),
    ],
)
def test_verify_rejects_published_record_identity_drift(
    field: str, value: Any, problem: str
) -> None:
    """Published-file verification remains bound to the reserved record identity."""
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "doi": "10.5281/zenodo.7",
            "concept_record_id": "6",
            "submitted": True,
            "state": "done",
            "files": [],
        }
    )
    published_record = _published_record([])
    published_record[field] = value
    session = _Session()
    session.gets = [_Response(_draft(submitted=True)), _Response(published_record)]

    report = publisher.verify(session, state, _metadata())

    assert report["status"] == "fail"
    assert problem in report["problems"]


@pytest.mark.parametrize(
    ("remote_size", "problem"),
    [
        (None, "has an invalid size"),
        ("8", "has an invalid size"),
        ({"value": 8}, "has an invalid size"),
        (True, "has an invalid size"),
        (0, "is empty"),
        (9, "size does not match uploaded bytes"),
    ],
)
def test_verify_rejects_invalid_published_file_size(
    tmp_path: Path, remote_size: Any, problem: str
) -> None:
    """Published file sizes are required positive integers bound to uploaded bytes."""
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"expected")
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "doi": "10.5281/zenodo.7",
            "concept_record_id": "6",
            "submitted": True,
            "state": "done",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": publisher._sha256_file(bundle),
                }
            ],
        }
    )
    public_file: dict[str, Any] = {
        "key": bundle.name,
        "links": {"self": "https://zenodo.org/api/records/7/files/bundle/content"},
    }
    if remote_size is not None:
        public_file["size"] = remote_size
    session = _Session()
    session.gets = [
        _Response(_draft(submitted=True)),
        _Response(_published_record([public_file])),
        _Response({}, content=bundle.read_bytes()),
    ]

    report = publisher.verify(session, state, _metadata())

    assert report["status"] == "fail"
    assert any(problem in item for item in report["problems"])


def test_verify_rejects_missing_state_identity() -> None:
    """Verification requires a deposition identifier before contacting Zenodo."""
    with pytest.raises(publisher.ZenodoPublisherError, match="no deposition_id"):
        publisher.verify(_Session(), {}, _metadata())


def test_state_load_and_write_are_schema_checked_and_non_destructive(tmp_path: Path) -> None:
    """State persistence preserves the deposition identity and private permissions."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(publisher.ZenodoPublisherError, match="invalid"):
        publisher.load_state(invalid)

    state_path = tmp_path / "nested" / "state.json"
    state = publisher._seal_state(
        {
            "schema_version": publisher.ZENODO_STATE_SCHEMA,
            "deposition_id": 7,
            "record_id": 7,
            "concept_record_id": "6",
            "doi": "10.5281/zenodo.7",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    publisher.write_state(state_path, state)
    loaded = publisher.load_state(state_path)
    assert loaded["deposition_id"] == state["deposition_id"]
    assert "integrity" in loaded
    assert state_path.stat().st_mode & 0o077 == 0
    with pytest.raises(publisher.ZenodoPublisherError, match="different"):
        publisher.write_state(
            state_path,
            publisher._seal_state(
                {
                    **{key: value for key, value in state.items() if key != "integrity"},
                    "deposition_id": 8,
                }
            ),
        )


def test_state_integrity_rejects_manual_edit(tmp_path: Path) -> None:
    """A manually edited state file cannot be admitted to publication."""
    state_path = tmp_path / "state.json"
    publisher.write_state(
        state_path,
        publisher._seal_state(
            {
                "schema_version": publisher.ZENODO_STATE_SCHEMA,
                "deposition_id": 7,
                "record_id": 7,
                "concept_record_id": "6",
                "doi": "10.5281/zenodo.7",
                "submitted": False,
                "state": "unsubmitted",
                "files": [],
            }
        ),
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["deposition_id"] = 8
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(publisher.ZenodoPublisherError, match="integrity"):
        publisher.load_state(state_path)


@pytest.mark.parametrize("access_right", [None, "restricted"])
def test_metadata_requires_open_access_and_claim_boundary(
    tmp_path: Path, access_right: str | None
) -> None:
    """Benchmark metadata must expose public access and the SNQI claim boundary."""
    metadata = _metadata()
    if access_right is None:
        metadata.pop("access_right")
    else:
        metadata["access_right"] = access_right
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    with pytest.raises(publisher.ZenodoPublisherError, match="access_right"):
        publisher.load_dataset_metadata(path)

    metadata = _metadata()
    metadata["description"] = "Benchmark dataset without a claim boundary."
    path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    with pytest.raises(publisher.ZenodoPublisherError, match="claim boundary"):
        publisher.load_dataset_metadata(path)
