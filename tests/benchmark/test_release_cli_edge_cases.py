"""Behavior tests for release CLI dispatch and sanitized exit codes."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from robot_sf import release_cli

if TYPE_CHECKING:
    from pathlib import Path


def _args(mode: str, tmp_path: Path) -> argparse.Namespace:
    """Build the common namespace consumed by ``release_cli.handle``."""
    return argparse.Namespace(
        release_cmd="zenodo",
        zenodo_mode=mode,
        token_file=tmp_path / "token",
        state=tmp_path / "state.json",
        metadata=tmp_path / "metadata.json",
        api_base="https://example.test/api",
        files=[tmp_path / "bundle.tar.gz"],
    )


@pytest.mark.parametrize("mode", ["reserve", "upload", "publish", "verify"])
def test_release_cli_dispatches_each_zenodo_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    """Each mode delegates to the publisher and emits only JSON state/report."""
    calls: list[tuple[str, object]] = []
    state = {"schema_version": "robot-sf-zenodo-deposition.v1", "deposition_id": 7}
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path: {"upload_type": "dataset"},
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_state",
        lambda path: state,
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "reserve",
        lambda session, metadata, api_base: calls.append(("reserve", api_base)) or state,
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "upload",
        lambda session, loaded, files, api_base: calls.append(("upload", api_base)) or state,
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "publish",
        lambda session, loaded, metadata, api_base: calls.append(("publish", api_base)) or state,
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "verify",
        lambda session, loaded, metadata, api_base: (
            calls.append(("verify", api_base)) or {"status": "pass", "file_count": 0}
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "write_state",
        lambda path, value: calls.append(("write_state", value)),
    )
    result = release_cli.handle(_args(mode, tmp_path))
    assert result == 0
    assert calls[0][0] == mode
    assert "secret" not in capsys.readouterr().out


def test_release_cli_returns_blocked_for_verification_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed remote verification is reported with a non-zero status."""
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(release_cli.zenodo_publisher, "load_state", lambda path: {})
    monkeypatch.setattr(release_cli.zenodo_publisher, "load_dataset_metadata", lambda path: {})
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "verify",
        lambda *args, **kwargs: {"status": "fail", "problems": ["mismatch"]},
    )
    assert release_cli.handle(_args("verify", tmp_path)) == 2


def test_release_cli_sanitizes_publisher_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Publisher errors become a bounded blocked JSON response."""
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "build_session",
        lambda path: (_ for _ in ()).throw(
            release_cli.zenodo_publisher.ZenodoPublisherError("private credential detail")
        ),
    )
    assert release_cli.handle(_args("reserve", tmp_path)) == 2
    output = capsys.readouterr().out
    assert "blocked" in output
    assert "private credential detail" in output


@pytest.mark.parametrize("status", ["pass", "blocked"])
def test_release_cli_doctor_propagates_report_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Doctor status controls the CLI exit code and report output."""
    args = argparse.Namespace(
        release_cmd="doctor",
        repo=tmp_path,
        manifest=tmp_path / "manifest.yaml",
        expected_release_sha="a" * 40,
        expected_base_sha="b" * 40,
        tag="release",
        checkpoint_receipt=None,
        private_launch_packet=None,
        dissertation=None,
        token_file=None,
        expected_cells=20160,
        minimum_free_gib=100.0,
        require_zenodo_webhook_disabled=False,
    )
    monkeypatch.setattr(
        release_cli,
        "collect_release_doctor_report",
        lambda **kwargs: {"status": status, "checks": []},
    )
    assert release_cli.handle(args) == (0 if status == "pass" else 2)
    assert status in capsys.readouterr().out
