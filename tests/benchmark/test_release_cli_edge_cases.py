"""Behavior tests for release CLI dispatch and sanitized exit codes."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
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
        manifest=None,
        deposition_id=7,
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


def test_release_cli_dispatches_new_version_without_loading_existing_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The successor mode reserves a fresh state file from explicit identities."""
    calls: list[tuple[str, object]] = []
    state = {
        "schema_version": "robot-sf-zenodo-deposition.v1",
        "deposition_id": 8,
        "predecessor_deposition_id": 7,
    }
    args = _args("new-version", tmp_path)
    args.predecessor_deposition_id = 7
    args.expected_predecessor_doi = "10.5281/zenodo.7"
    args.expected_concept_doi = "10.5281/zenodo.6"
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path: {"upload_type": "dataset"},
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_state",
        lambda path: (_ for _ in ()).throw(AssertionError("new-version must not load state")),
    )

    def new_version(session, metadata, **kwargs):
        calls.append(("new-version", kwargs))
        return state

    monkeypatch.setattr(release_cli.zenodo_publisher, "new_version", new_version)
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "write_state",
        lambda path, value: calls.append(("write_state", value)),
    )

    assert release_cli.handle(args) == 0
    assert calls[0] == (
        "new-version",
        {
            "predecessor_deposition_id": 7,
            "expected_predecessor_doi": "10.5281/zenodo.7",
            "expected_concept_doi": "10.5281/zenodo.6",
            "api_base": "https://example.test/api",
        },
    )
    assert calls[1] == ("write_state", state)
    assert "secret" not in capsys.readouterr().out


def test_release_cli_parser_exposes_new_version_identity_arguments() -> None:
    """The public release parser exposes the successor arguments and optional manifest."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    release_cli.build_subparser(subparsers)

    args = parser.parse_args(
        [
            "release",
            "zenodo",
            "new-version",
            "--token-file",
            "token",
            "--state",
            "state.json",
            "--metadata",
            "metadata.json",
            "--predecessor-deposition-id",
            "7",
            "--expected-predecessor-doi",
            "10.5281/zenodo.7",
            "--expected-concept-doi",
            "10.5281/zenodo.6",
        ]
    )

    assert args.zenodo_mode == "new-version"
    assert args.predecessor_deposition_id == 7
    assert args.expected_predecessor_doi == "10.5281/zenodo.7"
    assert args.expected_concept_doi == "10.5281/zenodo.6"
    assert args.manifest is None


def test_release_cli_recovers_without_loading_state_or_reserving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The recovery mode performs one bound read and writes sanitized state."""
    calls: list[tuple[str, object]] = []
    state = {"schema_version": "robot-sf-zenodo-deposition.v1", "deposition_id": 7}
    manifest = SimpleNamespace(release_tag="v1", metadata_sha256="a" * 64)
    binding = {"release_tag": "v1"}
    args = _args("recover", tmp_path)
    args.manifest = tmp_path / "release.yaml"
    monkeypatch.setattr(release_cli, "_load_release_binding", lambda value: (manifest, binding))
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path, **kwargs: {"upload_type": "dataset"},
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_state",
        lambda path: (_ for _ in ()).throw(AssertionError("recovery must not load missing state")),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "reserve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not reserve a deposition")
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "recover",
        lambda session, deposition_id, metadata, **kwargs: (
            calls.append(("recover", deposition_id)) or state
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "write_state",
        lambda path, value: calls.append(("write_state", value)),
    )

    assert release_cli.handle(args) == 0
    assert calls == [("recover", 7), ("write_state", state)]
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
        expected_campaign_id="campaign-1",
        checkpoint_receipt=None,
        private_launch_packet=None,
        dissertation=None,
        token_file=None,
        expected_cells=20160,
        minimum_free_gib=100.0,
        require_zenodo_webhook_disabled=False,
    )
    captured: dict[str, object] = {}

    def collect(**kwargs):
        captured.update(kwargs)
        return {"status": status, "checks": []}

    monkeypatch.setattr(release_cli, "collect_release_doctor_report", collect)
    assert release_cli.handle(args) == (0 if status == "pass" else 2)
    assert captured["expected_campaign_id"] == "campaign-1"
    assert status in capsys.readouterr().out
