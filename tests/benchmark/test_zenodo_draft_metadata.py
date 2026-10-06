"""Offline witnesses for one reservation followed by DOI-bound draft metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from robot_sf import release_cli
from robot_sf.benchmark import zenodo_publisher as publisher
from robot_sf.evidence.writers import write_json, write_text
from tests.benchmark.test_zenodo_publisher import _draft_payload, _metadata, _Response, _Session


class _API(_Session):
    """Record mock mutations; no HTTP client or credential file is used."""

    def __init__(self) -> None:
        super().__init__()
        self.mutations: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.mutations.append(("POST", url, deepcopy(kwargs)))
        return super().post(url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> _Response:
        self.mutations.append(("PUT", url, deepcopy(kwargs)))
        return super().put(url, **kwargs)


def _args(tmp_path: Path, mode: str = "update-draft-metadata") -> argparse.Namespace:
    return argparse.Namespace(
        release_cmd="zenodo",
        zenodo_mode=mode,
        token_file=tmp_path / "unused-fake-token",
        state=tmp_path / "state.json",
        metadata=tmp_path / "metadata.json",
        manifest=tmp_path / "identity.json" if mode != "reserve" else None,
        api_base=publisher.ZENODO_API_BASE,
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[argparse.Namespace, _API, dict[str, Any], dict[str, Any]]:
    args = _args(tmp_path)
    metadata = _metadata()
    metadata["description"] += " Concept DOI 10.5281/zenodo.122; version DOI 10.5281/zenodo.123."
    write_json(args.metadata, {"metadata": metadata})
    binding = publisher.build_release_binding(
        {
            "metadata_path": args.metadata,
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "release_tag": metadata["related_identifiers"][0]["identifier"],
            "concept_doi": "10.5281/zenodo.122",
            "version_doi": "10.5281/zenodo.123",
        }
    )
    # Manifest loading has separate coverage; real file/hash/DOI binding remains active.
    monkeypatch.setattr(release_cli, "_load_release_binding", lambda _: (object(), binding))
    api = _API()
    monkeypatch.setattr(publisher, "build_session", lambda _: api)
    state = publisher._seal_state(publisher._public_state(_draft_payload()))
    publisher.write_state(args.state, state)
    return args, api, state, metadata


def test_one_reservation_updates_concrete_metadata_then_verifies_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One POST, bound PUT/GET, exact description, and unchanged publish contract."""
    args, api, _, concrete = _fixture(tmp_path, monkeypatch)
    binding = release_cli._load_release_binding(args)[1]
    args.state.unlink()
    reserve_args = _args(tmp_path, "reserve")
    reserve_args.metadata = tmp_path / "reservation.json"
    write_json(reserve_args.metadata, {"metadata": _metadata()})
    monkeypatch.setattr(
        release_cli,
        "_load_release_binding",
        lambda value: None if value.zenodo_mode == "reserve" else (object(), binding),
    )
    api.posts = [_Response(_draft_payload())]
    assert release_cli.handle(reserve_args) == 0
    reserved = publisher.load_state(args.state)
    assert reserved["doi"] == "10.5281/zenodo.123"
    updated = _draft_payload()
    updated["metadata"].update(concrete)
    updated["metadata"]["license"] = "gpl-3.0"
    updated["metadata"]["creators"] = [
        {**creator, "affiliation": None} for creator in concrete["creators"]
    ]
    reserved["verification_receipt"] = {"status": "stale"}
    publisher.write_state(args.state, publisher._seal_state(reserved))
    api.gets = [_Response(_draft_payload()), _Response(updated)]
    api.puts = [_Response(updated)]
    assert release_cli.handle(args) == 0
    assert api.urls[-3:] == [f"{publisher.ZENODO_API_BASE}/deposit/depositions/123"] * 3
    assert [method for method, _, _ in api.mutations] == ["POST", "PUT"]
    assert api.mutations[-1][2]["json"]["metadata"]["description"] == concrete["description"]
    state = publisher.load_state(args.state)
    assert state["release_binding"]["metadata_sha256"] == binding["metadata_sha256"]
    assert state["files"] == []
    assert "verification_receipt" not in state
    api.gets = [_Response(updated), _Response(updated)]
    api.puts = [_Response(updated)]
    assert release_cli.handle(args) == 0
    assert publisher.load_state(args.state) == state
    # Use a real cold-read body, independently pinned, rather than mocking verify/publish.
    state["files"] = [
        {"name": "bundle.tar.gz", "size": 6, "sha256": hashlib.sha256(b"bundle").hexdigest()}
    ]
    state = publisher._seal_state(state)
    updated["files"] = [
        {
            "filename": "bundle.tar.gz",
            "size": 6,
            "links": {"download": "https://zenodo.org/api/records/123/files/bundle.tar.gz/content"},
        }
    ]
    body = _Response({})
    body.content = b"bundle"
    api.gets = [_Response(updated), body]
    report = publisher.verify(api, state, concrete, release_binding=binding)
    assert report["status"] == "pass", report
    api.gets = [_Response(updated), body]
    published = deepcopy(updated)
    published.update(submitted=True, state="done", doi="10.5281/zenodo.123")
    api.posts = [_Response(published)]
    result = publisher.publish(api, state, concrete, release_binding=binding)
    assert result["submitted"] is True
    record = {
        "id": 123,
        "conceptrecid": "122",
        "doi": result["doi"],
        "status": "published",
        "files": [
            {
                "key": "bundle.tar.gz",
                "size": 6,
                "links": {"self": "https://zenodo.org/api/records/123/files/bundle.tar.gz/content"},
            }
        ],
    }
    api.gets = [_Response(published), _Response(record), body]
    assert publisher.verify(api, result, concrete, release_binding=binding)["status"] == "pass"
    assert (
        len(
            [
                url
                for method, url, _ in api.mutations
                if method == "POST" and url.endswith("/depositions")
            ]
        )
        == 1
    )


@pytest.mark.parametrize("existing", ["reserved", "malformed", "directory", "dangling-symlink"])
def test_reserve_existing_state_refuses_before_session_or_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    existing: str,
) -> None:
    args, api, _, _ = _fixture(tmp_path, monkeypatch)
    args.zenodo_mode = "reserve"
    args.manifest = None
    monkeypatch.setattr(release_cli, "_load_release_binding", lambda _: None)
    if existing != "reserved":
        args.state.unlink()
    if existing == "malformed":
        write_text(args.state, "not JSON", issue_ref="zenodraft")
    elif existing == "directory":
        args.state.mkdir()
    elif existing == "dangling-symlink":
        args.state.symlink_to(tmp_path / "missing-state")
    sessions: list[Path] = []
    monkeypatch.setattr(publisher, "build_session", lambda path: sessions.append(path) or api)
    api.posts = [_Response(_draft_payload())]
    assert release_cli.handle(args) == 2
    assert sessions == [], "reserve must refuse before constructing an authenticated session"
    assert api.mutations == []
    assert "state path already exists" in json.loads(capsys.readouterr().out)["reason"]


@pytest.mark.parametrize("published_where", ["local", "remote", "put", "readback"])
def test_update_refuses_published_deposition_without_accepting_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    published_where: str,
) -> None:
    args, api, state, metadata = _fixture(tmp_path, monkeypatch)
    remote = _draft_payload()
    updated = deepcopy(remote)
    updated["metadata"].update(metadata)
    if published_where == "local":
        state.update(submitted=True, state="done")
        publisher.write_state(args.state, publisher._seal_state(state))
    elif published_where == "remote":
        remote.update(submitted=True, state="done", doi=state["doi"])
    put = deepcopy(updated)
    if published_where in {"put", "readback"}:
        target = put if published_where == "put" else updated
        target.update(submitted=True, state="done", doi=state["doi"])
    before = args.state.read_bytes()
    api.gets = [_Response(remote), _Response(updated)]
    api.puts = [_Response(put)]
    assert release_cli.handle(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked", report
    assert "update-draft-metadata" in report["reason"]
    assert args.state.read_bytes() == before
    assert [item[0] for item in api.mutations] == (
        [] if published_where in {"local", "remote"} else ["PUT"]
    )


@pytest.mark.parametrize("phase", ["before-put", "put", "readback"])
@pytest.mark.parametrize("field", ["id", "record_id", "conceptrecid", "doi"])
def test_update_refuses_changed_deposition_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    field: str,
) -> None:
    args, api, _, metadata = _fixture(tmp_path, monkeypatch)
    remote = _draft_payload()
    updated = deepcopy(remote)
    updated["metadata"].update(metadata)
    put = deepcopy(updated)
    target = {"before-put": remote, "put": put, "readback": updated}[phase]
    target[field] = {
        "id": 124,
        "record_id": 124,
        "conceptrecid": "124",
        "doi": "10.5281/zenodo.124",
    }[field]
    if field == "record_id":
        target["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.124", "recid": 124}
    elif field == "doi":
        target["record_id"] = 124
    before = args.state.read_bytes()
    api.gets = [_Response(remote), _Response(updated)]
    api.puts = [_Response(put)]
    assert release_cli.handle(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked", report
    assert "does not match reserved state" in report["reason"]
    assert args.state.read_bytes() == before
    assert [item[0] for item in api.mutations] == ([] if phase == "before-put" else ["PUT"])


@pytest.mark.parametrize("phase", ["put", "readback"])
@pytest.mark.parametrize("field", ["description", "related_identifiers", "missing-metadata"])
def test_update_requires_exact_metadata_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    field: str,
) -> None:
    args, api, _, metadata = _fixture(tmp_path, monkeypatch)
    updated = _draft_payload()
    updated["metadata"].update(metadata)
    put = deepcopy(updated)
    target = put if phase == "put" else updated
    if field == "missing-metadata":
        target.pop("metadata")
        target["doi"] = "10.5281/zenodo.123"
    elif field == "description":
        target["metadata"][field] += " "
    else:
        target["metadata"][field][0]["identifier"] += "-wrong"
    before = args.state.read_bytes()
    api.gets = [_Response(_draft_payload()), _Response(updated)]
    api.puts = [_Response(put)]
    assert release_cli.handle(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked", report
    reason = report["reason"]
    assert "update-draft-metadata" in reason and "metadata" in reason
    assert "readback" in reason
    assert args.state.read_bytes() == before
    assert [item[0] for item in api.mutations] == ["PUT"]


def test_update_requires_manifest_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(tmp_path)
    args.manifest = None
    monkeypatch.setattr(
        publisher, "build_session", lambda _: pytest.fail("unbound update constructed session")
    )
    assert release_cli.handle(args) == 2
    assert "requires a validated release manifest" in json.loads(capsys.readouterr().out)["reason"]


def test_reserve_rechecks_state_path_immediately_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A state appearing after CLI preflight must still prevent the reservation POST."""
    args, api, state, _ = _fixture(tmp_path, monkeypatch)
    args.state.unlink()
    args.zenodo_mode = "reserve"
    args.manifest = None
    monkeypatch.setattr(release_cli, "_load_release_binding", lambda _: None)

    def build_fake_session(_: Path) -> _API:
        publisher.write_state(args.state, state)
        return api

    monkeypatch.setattr(publisher, "build_session", build_fake_session)
    api.posts = [_Response(_draft_payload())]
    assert release_cli.handle(args) == 2
    assert api.mutations == []
    assert publisher.load_state(args.state) == state
    assert "state path already exists" in json.loads(capsys.readouterr().out)["reason"]


def test_cli_exposes_bound_draft_update_command(tmp_path: Path) -> None:
    """The public command accepts state, metadata and manifest together."""
    parser = argparse.ArgumentParser()
    release_cli.build_subparser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "release",
            "zenodo",
            "update-draft-metadata",
            "--token-file",
            str(tmp_path / "unused-fake-token"),
            "--state",
            str(tmp_path / "state.json"),
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--manifest",
            str(tmp_path / "identity.json"),
        ]
    )
    assert args.zenodo_mode == "update-draft-metadata"
    assert args.manifest == tmp_path / "identity.json"


@pytest.mark.parametrize("failure", ["response", "write-state"])
def test_reserve_failed_attempt_blocks_another_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A minted draft without persisted state must never admit another reservation."""
    args = _args(tmp_path, "reserve")
    write_json(args.metadata, {"metadata": _metadata()})
    api = _API()
    monkeypatch.setattr(publisher, "build_session", lambda _: api)
    invalid = _draft_payload()
    invalid["record_id"] = 124  # POST succeeded, but its DOI does not match its record.
    api.posts = [_Response(invalid if failure == "response" else _draft_payload())]
    if failure == "write-state":

        def fail_write(*_args: Any) -> None:
            raise OSError("fake state replacement failure")

        monkeypatch.setattr(publisher.os, "replace", fail_write)
    assert release_cli.handle(args) == 2
    assert not args.state.exists()
    assert len(api.mutations) == 1
    api.posts = [_Response(_draft_payload())]
    assert release_cli.handle(args) == 2
    assert len(api.mutations) == 1, "rerun minted a second DOI after a failed reservation attempt"
    assert args.state.with_name(args.state.name + ".reserve-attempt").exists()


@pytest.mark.parametrize("marker_kind", ["file", "dangling-symlink"])
def test_reserve_existing_attempt_refuses_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_kind: str
) -> None:
    """Even an interrupted pre-POST attempt requires operator recovery."""
    args = _args(tmp_path, "reserve")
    write_json(args.metadata, {"metadata": _metadata()})
    marker = args.state.with_name(args.state.name + ".reserve-attempt")
    if marker_kind == "file":
        write_text(marker, "", issue_ref="zenodraft")
    else:
        marker.symlink_to(tmp_path / "missing-attempt")
    sessions = []
    api = _API()
    api.posts = [_Response(_draft_payload())]
    monkeypatch.setattr(publisher, "build_session", lambda path: sessions.append(path) or api)
    assert release_cli.handle(args) == 2
    assert sessions == [], "reservation attempt marker did not block authentication"
    assert api.mutations == []


def test_reserve_attempt_cleared_only_after_persisted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exclusive attempt exists throughout POST and state persistence."""
    args = _args(tmp_path, "reserve")
    write_json(args.metadata, {"metadata": _metadata()})
    marker = args.state.with_name(args.state.name + ".reserve-attempt")
    api = _API()
    api.posts = [_Response(_draft_payload())]
    original_post = api.post
    original_write = publisher.write_state

    def checked_post(url: str, **kwargs: Any) -> _Response:
        assert marker.is_file(), "POST has no exclusive reservation attempt"
        return original_post(url, **kwargs)

    def checked_write(path: Path, state: dict[str, Any]) -> None:
        assert marker.is_file(), "attempt cleared before state persistence"
        original_write(path, state)

    monkeypatch.setattr(api, "post", checked_post)
    monkeypatch.setattr(publisher, "write_state", checked_write)
    monkeypatch.setattr(publisher, "build_session", lambda _: api)
    assert release_cli.handle(args) == 0
    assert publisher.load_state(args.state)["doi"] == "10.5281/zenodo.123"
    assert not marker.exists()


@pytest.mark.parametrize("phase", ["put", "readback"])
def test_update_refuses_extra_remote_metadata_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str
) -> None:
    """Remote residue outside the exact caller contract must block acceptance."""
    args, api, _, metadata = _fixture(tmp_path, monkeypatch)
    updated = _draft_payload()
    updated["metadata"].update(metadata)
    put = deepcopy(updated)
    (put if phase == "put" else updated)["metadata"]["notes"] = "stale draft residue"
    before = args.state.read_bytes()
    api.gets = [_Response(_draft_payload()), _Response(updated)]
    api.puts = [_Response(put)]
    assert release_cli.handle(args) == 2
    assert "unexpected fields" in json.loads(capsys.readouterr().out)["reason"]
    assert args.state.read_bytes() == before


def test_generated_identity_drives_real_cli_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real resolver bytes reach the CLI loader and bind the same reserved draft."""
    from robot_sf.benchmark import release_protocol
    from tests.benchmark.test_release_resolved_identity import (
        _identity_inputs,
        _release_template_repository,
    )
    from tests.benchmark.test_sealed_source_pins import bind_runtime_sources

    repo, template, source = _release_template_repository(tmp_path)
    bind_runtime_sources(repo, monkeypatch)
    monkeypatch.setattr(release_cli, "get_repository_root", lambda: repo)
    monkeypatch.setattr(release_protocol, "get_repository_root", lambda: repo)
    inputs = _identity_inputs(repo, template, source)
    inputs.update(concept_doi="10.5281/zenodo.122", version_doi="10.5281/zenodo.123")
    output = repo / "output/release/release_identity.resolved.json"
    release_protocol.write_resolved_release_identity(output_path=output, **inputs)
    args = _args(output.parent)
    args.manifest = output
    args.metadata = output.parent / "zenodo_metadata.resolved.json"
    metadata = publisher.load_dataset_metadata(args.metadata)
    assert args.state.parent.is_relative_to(repo)
    assert "{{" not in metadata["description"]
    assert "10.5281/zenodo.123" in metadata["description"]
    api = _API()
    monkeypatch.setattr(publisher, "build_session", lambda _: api)
    publisher.write_state(
        args.state, publisher._seal_state(publisher._public_state(_draft_payload()))
    )
    updated = _draft_payload()
    updated["metadata"].update(publisher._metadata_contract(metadata))
    api.gets = [_Response(_draft_payload()), _Response(updated)]
    api.puts = [_Response(updated)]
    assert release_cli.handle(args) == 0
    state = publisher.load_state(args.state)
    manifest = json.loads(output.read_text())
    assert (
        state["release_binding"]["metadata_sha256"]
        == hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    )
    assert state["release_binding"]["metadata_sha256"] == manifest["publication"]["metadata_sha256"]
    assert [method for method, _, _ in api.mutations] == ["PUT"]
