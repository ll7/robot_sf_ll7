"""Tests for the camera-ready release publisher's draft-creation contract."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.tools import publish_camera_ready_release as publisher

if TYPE_CHECKING:
    from pathlib import Path

_SOURCE_SHA = "a" * 40
_TAG = "v0.0.1"


def _summary_payload(*, tag: str = "v0.0.1") -> dict[str, object]:
    """Return a minimal accepted campaign summary."""
    return {
        "campaign": {
            "release_id": "test_release_01",
            "release_tag": tag,
            "git_hash": _SOURCE_SHA,
            "repository_url": "https://github.com/ll7/robot_sf_ll7",
            "doi": "10.5281/zenodo.0000001",
        },
        "publication_bundle": {
            "archive_path": "test_release_01_publication_bundle.tar.gz",
            "checksums_path": "publication_bundle/checksums.sha256",
            "manifest_path": "publication_bundle/manifest.json",
        },
    }


def _run(tmp_path: Path, *extra: str, summary: dict[str, object] | None = None) -> str:
    """Run the publisher with prerequisites satisfied by a mocked summary."""
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    payload = summary or _summary_payload()
    with (
        patch(
            "scripts.tools.publish_camera_ready_release._validate_prerequisites",
            return_value=(
                campaign_root / "archive.tar.gz",
                campaign_root / "checksums.sha256",
                campaign_root / "manifest.json",
                payload,
            ),
        ),
        patch(
            "scripts.tools.publish_camera_ready_release.get_repository_root",
            return_value=tmp_path,
        ),
    ):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = publisher.main(
                [
                    "--campaign-root",
                    str(campaign_root),
                    "--tag",
                    "v0.0.1",
                    *extra,
                ]
            )
        assert exit_code == 0
        return buffer.getvalue()


def test_create_draft_requires_expected_source_sha(tmp_path: Path) -> None:
    """--create-draft without --expected-source-sha fails argument validation."""
    with pytest.raises(SystemExit):
        _run(tmp_path, "--create-draft")


def test_expected_source_sha_must_be_40_chars(tmp_path: Path) -> None:
    """A non-40-character expected SHA is rejected."""
    with pytest.raises(SystemExit):
        _run(tmp_path, "--create-draft", "--expected-source-sha", "abc")


def test_dry_run_plans_draft_create_before_upload(tmp_path: Path) -> None:
    """Dry-run reports the draft-create command without executing anything."""
    with patch("subprocess.run", side_effect=AssertionError("dry-run must not execute")) as run:
        payload = json.loads(_run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA))
        run.assert_not_called()
    assert payload["draft_create_command"][:3] == ["gh", "release", "create"]
    assert payload["draft_create_command"][0:6] == [
        "gh",
        "release",
        "create",
        "v0.0.1",
        "--repo",
        "ll7/robot_sf_ll7",
    ]
    assert "--draft" in payload["draft_create_command"]
    assert "--target" in payload["draft_create_command"]
    assert payload["expected_source_sha"] == _SOURCE_SHA
    assert payload["release_title"] == "Benchmark data release test_release_01"


def test_create_draft_then_upload_order(tmp_path: Path) -> None:
    """With --execute-upload, the draft is created before the upload runs."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
    upload_calls = [c for c in calls if c[:3] == ["gh", "release", "upload"]]
    assert len(create_calls) == 1
    assert len(upload_calls) == 1
    assert calls.index(create_calls[0]) < calls.index(upload_calls[0])


def test_collision_with_existing_release_on_different_sha_fails_closed(tmp_path: Path) -> None:
    """An existing release at a different target SHA blocks creation and upload."""
    existing = json.dumps(
        [[{"tag_name": _TAG, "draft": True, "target_commitish": "b" * 40}]],
    )

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "already exists at target" in str(exc_info.value)


def test_collision_with_public_release_fails_closed(tmp_path: Path) -> None:
    """A non-draft existing release is never mutated."""
    existing = json.dumps(
        [[{"tag_name": _TAG, "draft": False, "target_commitish": _SOURCE_SHA}]],
    )

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "not a draft" in str(exc_info.value)


def test_existing_exact_sha_draft_allows_upload(tmp_path: Path) -> None:
    """An exact-SHA draft is not a blocker; upload proceeds without creation."""
    existing = json.dumps(
        [[{"tag_name": _TAG, "draft": True, "target_commitish": _SOURCE_SHA}]],
    )
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
    upload_calls = [c for c in calls if c[:3] == ["gh", "release", "upload"]]
    assert create_calls == []
    assert len(upload_calls) == 1


@pytest.mark.parametrize(
    ("release", "message"),
    [
        ({"tag_name": _TAG, "draft": True}, "exact target commit"),
        (
            {"tag_name": _TAG, "draft": "true", "target_commitish": _SOURCE_SHA},
            "malformed draft flag",
        ),
    ],
)
def test_malformed_existing_draft_blocks_upload(
    tmp_path: Path,
    release: dict[str, object],
    message: str,
) -> None:
    """Draft reuse requires strict typed state bound to the exact source SHA."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([[release]]), stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert message in str(exc_info.value)


def test_missing_release_creates_draft(tmp_path: Path) -> None:
    """When the REST release lookup finds nothing, draft creation is planned."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
    upload_calls = [c for c in calls if c[:3] == ["gh", "release", "upload"]]
    assert len(create_calls) == 1
    assert len(upload_calls) == 1


def test_existing_git_tag_blocks_draft_creation(tmp_path: Path) -> None:
    """A Git tag collision is never silently adopted by draft creation."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"object": {"sha": _SOURCE_SHA}}),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "tag" in str(exc_info.value)
    assert "already exists" in str(exc_info.value)


def test_ambiguous_release_lookup_blocks_draft_creation(tmp_path: Path) -> None:
    """An authentication/transport error is not interpreted as release absence."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="authentication failed")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "cannot determine whether release" in str(exc_info.value)


@pytest.mark.parametrize("payload", ("{}", "[[null]]", "not-json"))
def test_malformed_release_listing_blocks_draft_creation(tmp_path: Path, payload: str) -> None:
    """Malformed paginated REST output is never interpreted as release absence."""

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "release lookup" in str(exc_info.value)


def test_duplicate_release_listing_blocks_draft_creation(tmp_path: Path) -> None:
    """Multiple release records for one tag are ambiguous and fail closed."""
    release = {"tag_name": _TAG, "draft": True, "target_commitish": _SOURCE_SHA}

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps([[release], [release]]), stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit) as exc_info:
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert "multiple releases" in str(exc_info.value)


def test_release_identity_derives_title_and_notes() -> None:
    """The derived title/notes bind the campaign identity deterministically."""
    summary = {
        "campaign": {
            "release_id": "paper_v1",
            "repository_url": "https://github.com/ll7/robot_sf_ll7",
            "doi": "10.5281/zenodo.42",
        }
    }
    title, notes = publisher._resolve_release_identity(
        summary, tag="v0.1.0", release_title=None, release_notes=None
    )
    assert title == "Benchmark data release paper_v1"
    assert "paper_v1" in notes
    assert "10.5281/zenodo.42" in notes


def test_upload_without_create_draft_is_unchanged(tmp_path: Path) -> None:
    """Existing upload-only behavior is preserved when --create-draft is absent."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        payload = json.loads(_run(tmp_path, "--execute-upload"))
    assert "draft_create_command" not in payload
    assert len(calls) == 1
    assert calls[0][:3] == ["gh", "release", "upload"]


def test_publisher_rejects_provisional_sha_in_requested_tag() -> None:
    """Publication cannot use a planning/base SHA as the tag identity."""
    summary = _summary_payload()
    with pytest.raises(ValueError, match="tag SHA component"):
        publisher._validate_source_identity(
            summary,
            tag=f"paper-matrix-future-{'b' * 40}",
            expected_source_sha=_SOURCE_SHA,
        )


def test_publisher_rejects_immutable_historical_tag_upload() -> None:
    """The historical stale-suffix release cannot be mutated by the publisher."""
    summary = _summary_payload()
    with pytest.raises(ValueError, match="tag abbreviation"):
        publisher._validate_source_identity(
            summary,
            tag="paper-matrix-v2-h600-s30-2026-08-cd831d7582c1",
            expected_source_sha=_SOURCE_SHA,
        )
