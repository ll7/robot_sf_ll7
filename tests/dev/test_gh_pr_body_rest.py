"""Offline tests for the REST-only PR body updater (issue #5221)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.gh_pr_body_rest import main, reconcile_pr_metadata, update_pr_body
from scripts.dev.pr_metadata import metadata_digest

if TYPE_CHECKING:
    from pathlib import Path


def _proc(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake ``subprocess.CompletedProcess`` for ``gh api``."""
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


def test_update_pr_body_patches_rest_endpoint_and_verifies_body(tmp_path: Path) -> None:
    """The helper must send JSON through PATCH and verify GitHub's returned body."""
    body_file = tmp_path / "body.md"
    body_file.write_text("## Summary\n\nUpdated body\n", encoding="utf-8")
    response = {
        "body": body_file.read_text(encoding="utf-8"),
        "html_url": "https://example/pr/5220",
    }
    with patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run:
        mock_run.return_value = _proc(stdout=json.dumps(response))
        result = update_pr_body(5220, body_file, repo="ll7/robot_sf_ll7")

    assert result == {
        "status": "ok",
        "number": 5220,
        "repo": "ll7/robot_sf_ll7",
        "url": "https://example/pr/5220",
    }
    assert mock_run.call_args.args[0] == [
        "gh",
        "api",
        "--method",
        "PATCH",
        "repos/ll7/robot_sf_ll7/pulls/5220",
        "--input",
        "-",
    ]
    assert json.loads(mock_run.call_args.kwargs["input"]) == {"body": response["body"]}


def test_update_pr_body_fails_closed_when_response_body_differs(tmp_path: Path) -> None:
    """A successful HTTP response is insufficient when it reports a different body."""
    body_file = tmp_path / "body.md"
    body_file.write_text("expected", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(stdout=json.dumps({"body": "different"}))
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "did not preserve" in result["error"]


def test_update_pr_body_fails_closed_for_missing_file(tmp_path: Path) -> None:
    """A missing body file must never issue a partial REST update."""
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        result = update_pr_body(5220, tmp_path / "missing.md")

    assert result["status"] == "error"
    assert "could not read body file" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_fails_closed_on_api_error(tmp_path: Path) -> None:
    """Authentication and API failures must be visible to the caller."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "Bad credentials" in result["error"]


def test_update_pr_body_fails_closed_on_timeout(tmp_path: Path) -> None:
    """A timeout must remain a structured, unverified error rather than escaping."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=30)
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert "not verified" in result["error"]


def test_cli_prints_compact_success_json(tmp_path: Path, capsys) -> None:
    """The command-line contract is a single machine-readable success result."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch:
        mock_patch.return_value = _proc(
            stdout=json.dumps({"body": "body", "html_url": "https://example/pr/5220"})
        )
        rc = main(["5220", "--repo", "ll7/robot_sf_ll7", "--body-file", str(body_file)])

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["url"] == "https://example/pr/5220"


def test_reconcile_pr_metadata_patches_title_and_body_atomically(tmp_path: Path) -> None:
    """A changed final state uses one PATCH carrying both metadata fields."""
    body_file = tmp_path / "body.md"
    body_file.write_text("final body", encoding="utf-8")
    desired_title = "fix: final title"
    response = {
        "title": desired_title,
        "body": "final body",
        "html_url": "https://example/pr/5220",
    }
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.side_effect = [
            _proc(stdout=json.dumps({"title": "old title", "body": "old body"})),
            _proc(stdout=json.dumps(response)),
        ]
        mock_patch.return_value = _proc(stdout=json.dumps(response))
        result = reconcile_pr_metadata(
            5220,
            desired_title,
            body_file,
            repo="ll7/robot_sf_ll7",
        )

    assert result == {
        "status": "ok",
        "number": 5220,
        "repo": "ll7/robot_sf_ll7",
        "url": "https://example/pr/5220",
        "metadata_digest": metadata_digest(desired_title, "final body"),
        "previous_metadata_digest": metadata_digest("old title", "old body"),
        "changed_fields": ["title", "body"],
        "changed": True,
    }
    mock_patch.assert_called_once_with(
        "repos/ll7/robot_sf_ll7/pulls/5220",
        {"title": desired_title, "body": "final body"},
    )
    assert mock_get.call_count == 2


def test_reconcile_pr_metadata_fails_closed_when_remote_changes_after_patch(
    tmp_path: Path,
) -> None:
    """A concurrent external writer must not be reported as a successful reconciliation."""
    body_file = tmp_path / "body.md"
    body_file.write_text("final body", encoding="utf-8")
    response = {"title": "final title", "body": "final body", "html_url": "https://example/pr/5220"}
    drifted = {"title": "newer title", "body": "newer body", "html_url": response["html_url"]}
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.side_effect = [
            _proc(stdout=json.dumps({"title": "old title", "body": "old body"})),
            _proc(stdout=json.dumps(drifted)),
        ]
        mock_patch.return_value = _proc(stdout=json.dumps(response))
        result = reconcile_pr_metadata(5220, "final title", body_file)

    assert result["status"] == "conflict"
    assert "changed during reconciliation" in result["error"]
    assert result["observed_metadata_digest"] == metadata_digest("newer title", "newer body")


def test_reconcile_pr_metadata_is_an_explicit_noop_when_current_state_matches(
    tmp_path: Path,
) -> None:
    """An already-current title/body must not create a remote mutation."""
    body_file = tmp_path / "body.md"
    body_file.write_text("final body", encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(
            stdout=json.dumps(
                {
                    "title": "final title",
                    "body": "final body",
                    "html_url": "https://example/pr/5220",
                }
            )
        )
        result = reconcile_pr_metadata(5220, "final title", body_file)

    assert result["status"] == "unchanged"
    assert result["changed"] is False
    assert result["changed_fields"] == []
    assert result["metadata_digest"] == metadata_digest("final title", "final body")
    mock_patch.assert_not_called()


def test_reconcile_pr_metadata_fails_closed_on_malformed_current_response(
    tmp_path: Path,
) -> None:
    """A missing current title cannot be treated as a safe no-op or update."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(stdout=json.dumps({"body": "body"}))
        result = reconcile_pr_metadata(5220, "title", body_file)

    assert result["status"] == "error"
    assert "malformed title" in result["error"]
    mock_patch.assert_not_called()


def test_reconcile_pr_metadata_rejects_invalid_title_before_reading_remote(
    tmp_path: Path,
) -> None:
    """Invalid title input must fail before any GET or PATCH request."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get:
        result = reconcile_pr_metadata(5220, "bad\ntitle", body_file)

    assert result["status"] == "error"
    assert "single line" in result["error"]
    mock_get.assert_not_called()


def test_reconcile_cli_requires_title(capsys, tmp_path: Path) -> None:
    """The atomic CLI mode must not silently fall back to body-only updates."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["5220", "--reconcile", "--body-file", str(body_file)])

    assert "--reconcile requires --title" in capsys.readouterr().err


def test_reconcile_cli_treats_noop_as_success(tmp_path: Path, capsys) -> None:
    """An idempotent reconciliation is successful at the shell boundary."""
    body_file = tmp_path / "body.md"
    body_file.write_text("body", encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(stdout=json.dumps({"title": "title", "body": "body"}))
        rc = main(
            [
                "5220",
                "--reconcile",
                "--title",
                "title",
                "--body-file",
                str(body_file),
            ]
        )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unchanged"
    mock_patch.assert_not_called()


LIVE_HEAD = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"
OTHER_HEAD = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
FABRICATED = f"{LIVE_HEAD[:9]}{'0' * 31}"


def _pull_response(*, title: str, body: str, head_sha: str = LIVE_HEAD) -> str:
    return json.dumps(
        {
            "title": title,
            "body": body,
            "html_url": "https://example/pr/5220",
            "head": {"sha": head_sha, "ref": "fix/example"},
        }
    )


def test_update_pr_body_accepts_live_head_carrier(tmp_path: Path) -> None:
    """A carrier equal to the resolved live head passes the guard and PATCHes."""
    body = f"## Summary\n\n- gate-verdict: accepted @ {LIVE_HEAD}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(
            stdout=_pull_response(title="t", body="old", head_sha=LIVE_HEAD)
        )
        mock_patch.return_value = _proc(
            stdout=json.dumps({"body": body, "html_url": "https://example/pr/5220"})
        )
        result = update_pr_body(5220, body_file)

    assert result["status"] == "ok"
    mock_patch.assert_called_once()
    assert mock_get.call_count == 1


def test_update_pr_body_rejects_fabricated_sha_before_patch(tmp_path: Path) -> None:
    """A fabricated carrier with a shared prefix must fail closed before any PATCH."""
    body = f"## Summary\n\n- gate-verdict: accepted @ {FABRICATED}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
        patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run,
    ):
        mock_get.return_value = _proc(stdout=_pull_response(title="t", body="old"))
        mock_run.return_value = _proc(stdout="missing", returncode=128)
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "do not match the live head" in result["error"]
    assert FABRICATED in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_rejects_stale_real_commit(tmp_path: Path) -> None:
    """A full SHA naming a different real commit must fail closed."""
    body = f"## Summary\n\n- Exact head: {OTHER_HEAD}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
        patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run,
    ):
        mock_get.return_value = _proc(stdout=_pull_response(title="t", body="old"))
        mock_run.return_value = _proc(stdout="commit\n")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert OTHER_HEAD in result["error"]
    assert "commit" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_rejects_abbreviated_sha(tmp_path: Path) -> None:
    """An abbreviated carrier cannot be verified and must fail closed."""
    body = f"## Summary\n\n- Exact head: {LIVE_HEAD[:12]}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(stdout=_pull_response(title="t", body="old"))
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "(abbreviated)" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_rejects_quoted_historical_evidence(tmp_path: Path) -> None:
    """Quoted historical carriers for a previous head must fail closed."""
    body = (
        "## Summary\n\n> Previous exact-head review:\n>\n"
        f"> `gate-verdict: accepted @ {OTHER_HEAD}`\n"
    )
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
        patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run,
    ):
        mock_get.return_value = _proc(stdout=_pull_response(title="t", body="old"))
        mock_run.return_value = _proc(stdout="commit\n")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert OTHER_HEAD in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_rejects_concurrent_head_movement(tmp_path: Path) -> None:
    """The resolved live head is the admission reference; a moved head is stale."""
    body = f"## Summary\n\n- gate-verdict: accepted @ {LIVE_HEAD}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
        patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run,
    ):
        mock_get.return_value = _proc(
            stdout=_pull_response(title="t", body="old", head_sha=OTHER_HEAD)
        )
        mock_run.return_value = _proc(stdout="commit\n")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "do not match the live head" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_fails_closed_when_head_resolution_fails(tmp_path: Path) -> None:
    """A carrier-bearing body cannot be written when the live head is unresolvable."""
    body = f"## Summary\n\n- gate-verdict: accepted @ {LIVE_HEAD}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.return_value = _proc(returncode=1, stderr="HTTP 401: Bad credentials")
        result = update_pr_body(5220, body_file)

    assert result["status"] == "error"
    assert "live-head read failed" in result["error"]
    mock_patch.assert_not_called()


def test_update_pr_body_skips_guard_without_carriers(tmp_path: Path) -> None:
    """Bodies without carriers must not require a live-head read before PATCH."""
    body_file = tmp_path / "body.md"
    body_file.write_text("## Summary\n\nplain body\n", encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_patch.return_value = _proc(
            stdout=json.dumps(
                {"body": "## Summary\n\nplain body\n", "html_url": "https://example/pr/5220"}
            )
        )
        result = update_pr_body(5220, body_file)

    assert result["status"] == "ok"
    mock_get.assert_not_called()


def test_reconcile_pr_metadata_rejects_invalid_carrier_before_patch(tmp_path: Path) -> None:
    """Reconcile must fail closed on a fabricated carrier before any PATCH."""
    body = f"## Summary\n\n- gate-verdict: accepted @ {FABRICATED}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
        patch("scripts.dev.gh_pr_body_rest.subprocess.run") as mock_run,
    ):
        mock_get.return_value = _proc(stdout=_pull_response(title="old title", body="old body"))
        mock_run.return_value = _proc(stdout="missing", returncode=128)
        result = reconcile_pr_metadata(5220, "fix: final", body_file)

    assert result["status"] == "error"
    assert "do not match the live head" in result["error"]
    mock_patch.assert_not_called()
    assert mock_get.call_count == 1


def test_reconcile_pr_metadata_accepts_live_head_carrier(tmp_path: Path) -> None:
    """Reconcile accepts a body whose carriers match the resolved live head."""
    body = f"## Summary\n\n- Exact head: {LIVE_HEAD}\n"
    body_file = tmp_path / "body.md"
    body_file.write_text(body, encoding="utf-8")
    response = {
        "title": "fix: final",
        "body": body,
        "html_url": "https://example/pr/5220",
        "head": {"sha": LIVE_HEAD, "ref": "fix/example"},
    }
    with (
        patch("scripts.dev.gh_pr_body_rest._gh_api_get") as mock_get,
        patch("scripts.dev.gh_pr_body_rest._gh_api_patch") as mock_patch,
    ):
        mock_get.side_effect = [
            _proc(stdout=_pull_response(title="old title", body="old body")),
            _proc(stdout=json.dumps(response)),
        ]
        mock_patch.return_value = _proc(stdout=json.dumps(response))
        result = reconcile_pr_metadata(5220, "fix: final", body_file)

    assert result["status"] == "ok"
    mock_patch.assert_called_once()
