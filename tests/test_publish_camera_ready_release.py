"""Tests for the camera-ready release publisher's draft-creation contract."""

from __future__ import annotations

import contextlib
import hashlib
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
    for name, contents in (
        ("archive.tar.gz", b"archive"),
        ("checksums.sha256", b"checksums\n"),
        ("manifest.json", b"{}\n"),
    ):
        (campaign_root / name).write_bytes(contents)
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
    """A newly created draft uploads before GitHub creates its tag ref."""
    calls: list[list[str]] = []
    created = False

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            release = {
                "tag_name": _TAG,
                "draft": True,
                "target_commitish": _SOURCE_SHA,
                "assets": [],
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps([[release]] if created else [[]]), stderr=""
            )
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        if cmd[:3] == ["gh", "release", "create"]:
            created = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
    upload_calls = [c for c in calls if c[:3] == ["gh", "release", "upload"]]
    assert len(create_calls) == 1
    assert len(upload_calls) == 1
    assert calls.index(create_calls[0]) < calls.index(upload_calls[0])


def test_create_draft_retries_eventually_consistent_release_readback(tmp_path: Path) -> None:
    """A newly created draft may take a bounded interval to appear in the release listing."""
    calls: list[list[str]] = []
    created = False
    missing_reads = 2

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created, missing_reads
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            if not created or missing_reads > 0:
                if created:
                    missing_reads -= 1
                return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps([[_release_record()]]), stderr=""
            )
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        if cmd[:3] == ["gh", "release", "create"]:
            created = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch.object(publisher.time, "sleep") as sleep,
    ):
        _run(
            tmp_path,
            "--create-draft",
            "--expected-source-sha",
            _SOURCE_SHA,
            "--execute-upload",
        )
    assert sleep.call_count == 2
    assert len([call for call in calls if call[:3] == ["gh", "release", "upload"]]) == 1


def test_create_draft_does_not_retry_release_readback_api_failure(tmp_path: Path) -> None:
    """Only a successful listing with no exact draft is retryable; API failures block."""
    calls: list[list[str]] = []
    created = False

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            if created:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="HTTP 503: Service Unavailable"
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        if cmd[:3] == ["gh", "release", "create"]:
            created = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch.object(publisher.time, "sleep") as sleep,
        pytest.raises(SystemExit, match="cannot determine whether release"),
    ):
        _run(
            tmp_path,
            "--create-draft",
            "--expected-source-sha",
            _SOURCE_SHA,
            "--execute-upload",
        )
    sleep.assert_not_called()
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_create_draft_admits_when_tag_ref_lookup_has_unparseable_404(tmp_path: Path) -> None:
    """A draft is admitted when the tag-ref lookup exits non-zero with an unparseable 404.

    Regression test for issue #8355: ``gh api`` can exit non-zero with an
    unparseable status line while the error body still quotes a 404.  The draft
    readback must treat this as an absent tag ref rather than a transport error.
    """
    existing = json.dumps([[_release_record()]])
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        if cmd[:2] == ["gh", "api"]:
            # Simulate a 404 response with an unparseable status line but
            # the error body still quotes a 404.
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(
            tmp_path,
            "--expected-source-sha",
            _SOURCE_SHA,
            "--execute-upload",
        )
    assert len([call for call in calls if call[:3] == ["gh", "release", "upload"]]) == 1


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
        [
            [
                {
                    "tag_name": _TAG,
                    "draft": True,
                    "target_commitish": _SOURCE_SHA,
                    "assets": [],
                }
            ]
        ],
    )
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {"ref": f"refs/tags/{_TAG}", "object": {"sha": _SOURCE_SHA, "type": "commit"}}
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--create-draft", "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    create_calls = [c for c in calls if c[:3] == ["gh", "release", "create"]]
    upload_calls = [c for c in calls if c[:3] == ["gh", "release", "upload"]]
    assert create_calls == []
    assert len(upload_calls) == 1


def test_existing_exact_sha_draft_without_tag_ref_allows_upload(tmp_path: Path) -> None:
    """An unpublished exact-target draft is valid before GitHub creates its tag ref."""
    existing = json.dumps([[_release_record()]])
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert len([call for call in calls if call[:3] == ["gh", "release", "upload"]]) == 1


def test_existing_exact_sha_draft_rejects_non_404_tag_lookup_failure(tmp_path: Path) -> None:
    """A transport/server failure is not interpreted as an absent draft tag ref."""
    existing = json.dumps([[_release_record()]])
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=existing, stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="HTTP 500: Internal Server Error"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="cannot resolve tag"),
    ):
        _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


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
    created = False

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            release = {
                "tag_name": _TAG,
                "draft": True,
                "target_commitish": _SOURCE_SHA,
                "assets": [],
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps([[release]] if created else [[]]), stderr=""
            )
        if cmd[:2] == ["gh", "api"]:
            if not created:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {"ref": f"refs/tags/{_TAG}", "object": {"sha": _SOURCE_SHA, "type": "commit"}}
                ),
                stderr="",
            )
        if cmd[:3] == ["gh", "release", "create"]:
            created = True
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
                stdout=json.dumps(
                    {"ref": f"refs/tags/{_TAG}", "object": {"sha": _SOURCE_SHA, "type": "commit"}}
                ),
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


def test_upload_without_create_draft_requires_expected_source_sha(tmp_path: Path) -> None:
    """Every mutating upload requires an explicit expected source SHA."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(SystemExit):
        _run(tmp_path, "--execute-upload")
    assert calls == []


def _asset_record(name: str, contents: bytes) -> dict[str, object]:
    """Build the REST asset record for one local fixture file."""
    return {
        "name": name,
        "state": "uploaded",
        "size": len(contents),
        "digest": f"sha256:{hashlib.sha256(contents).hexdigest()}",
    }


def _release_record(
    *,
    draft: bool = True,
    target: str = _SOURCE_SHA,
    assets: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build an exact-tag release record for mocked REST responses."""
    return {
        "tag_name": _TAG,
        "draft": draft,
        "target_commitish": target,
        "assets": assets or [],
    }


def _tag_response(*, sha: str = _SOURCE_SHA, object_type: str = "commit") -> str:
    """Build a lightweight or annotated tag-ref response."""
    return json.dumps(
        {
            "ref": f"refs/tags/{_TAG}",
            "object": {"sha": sha, "type": object_type},
        }
    )


def _run_existing_draft(
    tmp_path: Path,
    *,
    release: dict[str, object],
    tag_outputs: list[tuple[int, str, str]] | None = None,
    expect_failure: bool = False,
) -> tuple[list[list[str]], str]:
    """Run upload-only against a deterministic mocked exact draft."""
    calls: list[list[str]] = []
    tag_results = list(tag_outputs or [(0, _tag_response(), ""), (0, _tag_response(), "")])

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([[release]]), stderr="")
        if cmd[:2] == ["gh", "api"]:
            if not tag_results:
                raise AssertionError("unexpected extra tag lookup")
            returncode, stdout, stderr = tag_results.pop(0)
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        try:
            output = _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
        except SystemExit:
            if not expect_failure:
                raise
            output = ""
    return calls, output


def test_upload_only_rejects_published_release_before_tag_or_upload(tmp_path: Path) -> None:
    """Upload-only mode cannot mutate a published release."""
    calls = []
    release = _release_record(draft=False)

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([[release]]), stderr="")
        raise AssertionError(f"unexpected command after published-release blocker: {cmd}")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="not a draft"),
    ):
        _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_upload_only_rejects_wrong_target_release(tmp_path: Path) -> None:
    """Upload-only mode rejects an exact tag whose draft target is different."""
    calls = []
    release = _release_record(target="b" * 40)

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([[release]]), stderr="")
        raise AssertionError(f"unexpected command after target blocker: {cmd}")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="already exists at target"),
    ):
        _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_tag_collision_requires_explicit_rest_404(tmp_path: Path) -> None:
    """A hostile 'not found' string without an HTTP status is not tag absence."""
    calls: list[list[str]] = []
    created = False

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="[[]]", stderr="")
        if cmd[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Not Found")
        created = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="cannot resolve tag"),
    ):
        _run(
            tmp_path,
            "--create-draft",
            "--expected-source-sha",
            _SOURCE_SHA,
            "--execute-upload",
        )
    assert not created
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_create_draft_rejects_post_create_tag_drift(tmp_path: Path) -> None:
    """A tag created at the wrong target blocks before the first upload."""
    calls: list[list[str]] = []
    created = False

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal created
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            release = {
                "tag_name": _TAG,
                "draft": True,
                "target_commitish": _SOURCE_SHA,
                "assets": [],
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps([[release]] if created else [[]]), stderr=""
            )
        if cmd[:2] == ["gh", "api"]:
            if not created:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="HTTP 404: Not Found")
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_tag_response(sha="b" * 40), stderr=""
            )
        if cmd[:3] == ["gh", "release", "create"]:
            created = True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="readback blocked after creation"),
    ):
        _run(
            tmp_path,
            "--create-draft",
            "--expected-source-sha",
            _SOURCE_SHA,
            "--execute-upload",
        )
    assert created
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_existing_draft_rejects_extra_asset_before_upload(tmp_path: Path) -> None:
    """An unexpected remote asset blocks retry-safe draft reuse."""
    release = _release_record(
        assets=[_asset_record("archive.tar.gz", b"archive"), _asset_record("stale.txt", b"stale")]
    )
    calls, _ = _run_existing_draft(
        tmp_path, release=release, tag_outputs=[(0, _tag_response(), "")], expect_failure=True
    )
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_existing_draft_rejects_duplicate_asset_before_upload(tmp_path: Path) -> None:
    """Duplicate remote asset names are ambiguous and block mutation."""
    archive = _asset_record("archive.tar.gz", b"archive")
    release = _release_record(assets=[archive, dict(archive)])
    calls, _ = _run_existing_draft(
        tmp_path, release=release, tag_outputs=[(0, _tag_response(), "")], expect_failure=True
    )
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_existing_draft_rejects_mismatched_asset_digest_before_upload(tmp_path: Path) -> None:
    """A same-name remote asset with stale bytes cannot be clobbered."""
    stale = _asset_record("archive.tar.gz", b"different-bytes")
    release = _release_record(assets=[stale])
    calls, _ = _run_existing_draft(
        tmp_path, release=release, tag_outputs=[(0, _tag_response(), "")], expect_failure=True
    )
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_existing_draft_uploads_only_missing_assets_without_clobber(tmp_path: Path) -> None:
    """A partial exact draft uploads only absent assets and never uses clobber."""
    release = _release_record(assets=[_asset_record("archive.tar.gz", b"archive")])
    calls, output = _run_existing_draft(tmp_path, release=release)
    upload_calls = [call for call in calls if call[:3] == ["gh", "release", "upload"]]
    assert len(upload_calls) == 1
    assert "archive.tar.gz" not in upload_calls[0]
    assert any(argument.endswith("checksums.sha256") for argument in upload_calls[0])
    assert any(argument.endswith("manifest.json") for argument in upload_calls[0])
    assert "--clobber" not in upload_calls[0]
    payload = json.loads(output)
    assert payload["missing_upload_assets"] == ["checksums.sha256", "manifest.json"]


def test_existing_draft_skips_upload_when_all_assets_match(tmp_path: Path) -> None:
    """An exactly complete draft is a safe idempotent no-op."""
    release = _release_record(
        assets=[
            _asset_record("archive.tar.gz", b"archive"),
            _asset_record("checksums.sha256", b"checksums\n"),
            _asset_record("manifest.json", b"{}\n"),
        ]
    )
    calls, output = _run_existing_draft(tmp_path, release=release)
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)
    payload = json.loads(output)
    assert payload["upload_skipped"] is True


def test_annotated_tag_is_peeled_to_commit(tmp_path: Path) -> None:
    """Tag admission resolves annotated refs instead of trusting tag-object SHA."""
    tag_object = "c" * 40
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if "git/ref/tags" in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_tag_response(sha=tag_object, object_type="tag"), stderr=""
            )
        if "git/tags/" in " ".join(cmd):
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"object": {"sha": _SOURCE_SHA, "type": "commit"}}),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run):
        target, blocker = publisher._resolve_tag_ref_target(
            repo="ll7/robot_sf_ll7", tag=_TAG, allow_absent=False
        )
    assert blocker is None
    assert target == _SOURCE_SHA
    assert len(calls) == 2


def test_tag_ref_without_exact_ref_identity_is_rejected() -> None:
    """A tag-object payload without its requested ref is ambiguous."""
    result = subprocess.CompletedProcess(
        ["gh", "api"],
        0,
        stdout=json.dumps({"object": {"sha": _SOURCE_SHA, "type": "commit"}}),
        stderr="",
    )
    with patch.object(publisher, "_run_tag_api", return_value=result):
        target, blocker = publisher._resolve_tag_ref_target(
            repo="ll7/robot_sf_ll7", tag=_TAG, allow_absent=False
        )
    assert target is None
    assert blocker is not None
    assert "different ref" in blocker


def test_existing_draft_rejects_stale_tag_ref(tmp_path: Path) -> None:
    """A release target alone is insufficient when the actual tag moved."""
    release = _release_record()
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "api"] and "--paginate" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([[release]]), stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=_tag_response(sha="b" * 40),
            stderr="",
        )

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(SystemExit, match="refusing to mutate"),
    ):
        _run(tmp_path, "--expected-source-sha", _SOURCE_SHA, "--execute-upload")
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


def test_existing_draft_rejects_tag_created_at_wrong_target_between_reads(tmp_path: Path) -> None:
    """A tag that appears after admission is rechecked and cannot redirect the upload."""
    release = _release_record()
    calls, _ = _run_existing_draft(
        tmp_path,
        release=release,
        tag_outputs=[
            (1, "", "HTTP 404: Not Found"),
            (0, _tag_response(sha="b" * 40), ""),
        ],
        expect_failure=True,
    )
    assert not any(call[:3] == ["gh", "release", "upload"] for call in calls)


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
