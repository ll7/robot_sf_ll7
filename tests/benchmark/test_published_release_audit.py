"""Tests for the published-release audit (issue #7936)."""

from __future__ import annotations

import copy
import hashlib
import io
import json
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


def test_path_escape_fails_closed(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(ValueError, match="path escape"):
        _extract_members(evil, tmp_path / "dest")


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
    script = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "published_release_audit.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
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
    )
    assert proc.returncode == 0
    receipt = json.loads(proc.stdout)
    assert receipt["ok"] is True


def test_cli_main_missing_channel_returns_one(tmp_path: Path) -> None:
    github = tmp_path / "github"
    github.mkdir()
    script = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "benchmark"
        / "published_release_audit.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
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
) -> tuple[_PublicSession, bytes, str, str, str]:
    """Build a complete mocked GitHub/Zenodo public response set."""
    del tmp_path
    github_base = "https://github.test"
    zenodo_base = "https://zenodo.test/api"
    tag = "paper-matrix-v2-h600-s30"
    source_sha = "b" * 40
    bundle_buffer = io.BytesIO()
    with zipfile.ZipFile(bundle_buffer, "w") as archive:
        archive.writestr("manifest.json", b"network-fixture")
    bundle = bundle_buffer.getvalue()
    digest = hashlib.sha256(bundle).hexdigest()
    github_release_url = f"{github_base}/repos/ll7/robot_sf_ll7/releases/tags/{tag}"
    github_ref_url = f"{github_base}/repos/ll7/robot_sf_ll7/git/ref/tags/{tag}"
    github_asset_url = f"https://cdn.github.test/{tag}/bundle.zip"
    zenodo_record_url = f"{zenodo_base}/records/1234567"
    zenodo_asset_url = "https://zenodo.test/api/records/1234567/files/bundle.zip/content"
    source_tag_url = f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}"
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
                    "related_identifiers": [
                        {"identifier": source_tag_url, "relation": "isSupplementTo"}
                    ],
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
