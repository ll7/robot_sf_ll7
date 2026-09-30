"""Behavior tests for release CLI dispatch and sanitized exit codes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf import release_cli

SOURCE_SHA = "5" * 40
PREDECESSOR_TAG = f"paper-matrix-v2-h600-s30-2026-09-{SOURCE_SHA}"
SUCCESSOR_TAG = f"{PREDECESSOR_TAG}-erratum.1"
ERRATUM_CONTRACT_PATH = Path(
    "configs/benchmarks/releases/benchmark_data_release_s30_h600_2026_09_erratum_1.json"
)
RELEASE_MANIFEST_PATH = Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")


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
    metadata_calls: list[dict[str, object]] = []
    expected_binding: dict[str, str] | None = None
    state = {"schema_version": "robot-sf-zenodo-deposition.v1", "deposition_id": 7}
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path, **kwargs: metadata_calls.append(kwargs) or {"upload_type": "dataset"},
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_state",
        lambda path: state,
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "reserve",
        lambda session, metadata, api_base, **kwargs: (
            calls.append(("reserve", {"api_base": api_base, **kwargs})) or state
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "upload",
        lambda session, loaded, files, api_base, **kwargs: (
            calls.append(("upload", {"api_base": api_base, **kwargs})) or state
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "publish",
        lambda session, loaded, metadata, api_base, **kwargs: (
            calls.append(("publish", {"api_base": api_base, **kwargs})) or state
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "verify",
        lambda session, loaded, metadata, api_base, **kwargs: (
            calls.append(("verify", {"api_base": api_base, **kwargs}))
            or {"status": "pass", "file_count": 0}
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "write_state",
        lambda path, value: calls.append(("write_state", value)),
    )
    args = _args(mode, tmp_path)
    if mode in {"upload", "publish", "verify"}:
        args.manifest = tmp_path / "manifest.yaml"
        expected_binding = {"release_tag": "v1", "metadata_sha256": "a" * 64}
        monkeypatch.setattr(
            release_cli,
            "_load_release_binding",
            lambda value: (
                SimpleNamespace(release_tag="v1", metadata_sha256="a" * 64),
                expected_binding,
            ),
        )
    result = release_cli.handle(args)
    assert result == 0
    assert calls[0][0] == mode
    operation_kwargs = calls[0][1]
    assert isinstance(operation_kwargs, dict)
    if expected_binding is None:
        assert "release_binding" not in operation_kwargs
    else:
        assert operation_kwargs["release_binding"] is expected_binding
    if mode in {"publish", "verify"}:
        assert metadata_calls == [
            {"expected_source_tag": "v1", "expected_metadata_sha256": "a" * 64}
        ]
    elif mode == "reserve":
        assert metadata_calls == [{}]
    else:
        assert metadata_calls == []
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
    args.expected_predecessor_tag = PREDECESSOR_TAG
    args.expected_source_sha = SOURCE_SHA
    args.expected_successor_tag = SUCCESSOR_TAG
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path, **kwargs: calls.append(("metadata", kwargs)) or {"upload_type": "dataset"},
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
    assert calls[0] == ("metadata", {"expected_source_tag": SUCCESSOR_TAG})
    assert calls[1] == (
        "new-version",
        {
            "predecessor_deposition_id": 7,
            "expected_predecessor_doi": "10.5281/zenodo.7",
            "expected_concept_doi": "10.5281/zenodo.6",
            "expected_predecessor_tag": PREDECESSOR_TAG,
            "expected_source_sha": SOURCE_SHA,
            "expected_successor_tag": SUCCESSOR_TAG,
            "api_base": "https://example.test/api",
        },
    )
    assert calls[2] == ("write_state", state)
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
            "--expected-predecessor-tag",
            PREDECESSOR_TAG,
            "--expected-source-sha",
            SOURCE_SHA,
            "--expected-successor-tag",
            SUCCESSOR_TAG,
        ]
    )

    assert args.zenodo_mode == "new-version"
    assert args.predecessor_deposition_id == 7
    assert args.expected_predecessor_doi == "10.5281/zenodo.7"
    assert args.expected_concept_doi == "10.5281/zenodo.6"
    assert args.expected_predecessor_tag == PREDECESSOR_TAG
    assert args.expected_source_sha == SOURCE_SHA
    assert args.expected_successor_tag == SUCCESSOR_TAG
    assert args.manifest is None


@pytest.mark.parametrize("mode", ["upload", "verify", "publish"])
def test_release_cli_parser_requires_manifest_after_reservation(mode: str) -> None:
    """Post-reservation modes cannot be invoked without a release manifest."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    release_cli.build_subparser(subparsers)

    arguments = [
        "release",
        "zenodo",
        mode,
        "--token-file",
        "token",
        "--state",
        "state.json",
    ]
    if mode in {"verify", "publish"}:
        arguments.extend(["--metadata", "metadata.json"])
    else:
        arguments.append("bundle.tar.gz")

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_release_cli_loads_checked_in_erratum_contract_for_post_reservation() -> None:
    """The reviewed successor contract is a valid exact Zenodo release binding."""
    payload = json.loads(ERRATUM_CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata_path = Path(payload["successor"]["metadata_path"])
    args = argparse.Namespace(manifest=ERRATUM_CONTRACT_PATH, metadata=metadata_path)

    release_context = release_cli._load_release_binding(args)

    assert release_context is not None
    contract, binding = release_context
    assert contract.source_sha == "59577bad289dd692ba3580e1600c4a649ae27880"
    assert contract.scientific_release_id == (
        "paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880"
    )
    assert binding == {
        "metadata_path": metadata_path.resolve(),
        "metadata_sha256": "0a86f188c44d83949715e2d457c3b6154a511fbf041527a3e669d4995bdc6b2b",
        "release_tag": (
            "paper-matrix-v2-h600-s30-2026-09-59577bad289dd692ba3580e1600c4a649ae27880-erratum.1"
        ),
        "source_tag": (
            "https://github.com/ll7/robot_sf_ll7/releases/tag/"
            "paper-matrix-v2-h600-s30-2026-09-"
            "59577bad289dd692ba3580e1600c4a649ae27880-erratum.1"
        ),
        "concept_doi": "10.5281/zenodo.22227034",
        "version_doi": "10.5281/zenodo.22265925",
    }


def test_release_cli_keeps_legacy_v02_manifest_binding() -> None:
    """Schema dispatch preserves the existing benchmark-release manifest route."""
    args = argparse.Namespace(manifest=RELEASE_MANIFEST_PATH, metadata=None)

    release_context = release_cli._load_release_binding(args)

    assert release_context is not None
    manifest, binding = release_context
    assert manifest.schema_version == "benchmark-release-manifest.v0.2"
    assert binding["release_tag"] == manifest.release_tag
    assert binding["metadata_sha256"] == manifest.metadata_sha256
    assert binding["concept_doi"] == manifest.concept_doi
    assert binding["version_doi"] == manifest.version_doi


@pytest.mark.parametrize(
    "field",
    [
        "predecessor_deposition_id",
        "expected_predecessor_doi",
        "expected_concept_doi",
        "expected_predecessor_tag",
        "expected_source_sha",
        "expected_successor_tag",
    ],
)
def test_release_cli_rejects_erratum_new_version_argument_conflicts_before_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    """Explicit successor inputs cannot contradict a checked-in erratum contract."""
    payload = json.loads(ERRATUM_CONTRACT_PATH.read_text(encoding="utf-8"))
    supersedes = payload["supersedes"]
    scientific = payload["scientific_identity"]
    successor = payload["successor"]
    args = _args("new-version", tmp_path)
    args.manifest = ERRATUM_CONTRACT_PATH
    args.metadata = Path(successor["metadata_path"])
    args.predecessor_deposition_id = int(supersedes["version_doi"].rsplit(".", 1)[-1])
    args.expected_predecessor_doi = supersedes["version_doi"]
    args.expected_concept_doi = successor["concept_doi"]
    args.expected_predecessor_tag = supersedes["github_release_tag"]
    args.expected_source_sha = scientific["source_sha"]
    args.expected_successor_tag = successor["github_release_tag"]
    setattr(args, field, -1 if field == "predecessor_deposition_id" else "conflict")
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "build_session",
        lambda path: (_ for _ in ()).throw(
            AssertionError("conflicting erratum arguments must fail before session construction")
        ),
    )

    assert release_cli.handle(args) == 2
    output = capsys.readouterr().out
    assert "conflict" in output
    assert field in output


def test_release_cli_rejects_tampered_erratum_before_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid successor contract cannot construct an authenticated client."""
    payload = json.loads(ERRATUM_CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata_relative = Path(payload["successor"]["metadata_path"])
    metadata_source = metadata_relative.resolve()
    metadata_target = tmp_path / metadata_relative
    metadata_target.parent.mkdir(parents=True)
    metadata_target.write_bytes(metadata_source.read_bytes())
    payload["successor"]["metadata_sha256"] = "0" * 64
    contract_path = tmp_path / "erratum.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    args = _args("verify", tmp_path)
    args.manifest = contract_path
    args.metadata = metadata_target

    monkeypatch.setattr(release_cli, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "build_session",
        lambda path: (_ for _ in ()).throw(
            AssertionError("invalid erratum must fail before session construction")
        ),
    )

    assert release_cli.handle(args) == 2
    output = capsys.readouterr().out
    assert "metadata" in output
    assert "blocked" in output


@pytest.mark.parametrize("mode", ["recover", "upload", "verify", "publish"])
def test_release_cli_rejects_unbound_post_reservation_before_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    """Hand-built CLI namespaces also fail before constructing an HTTP session."""
    args = _args(mode, tmp_path)
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "build_session",
        lambda path: (_ for _ in ()).throw(AssertionError("unbound mode must not build session")),
    )

    assert release_cli.handle(args) == 2
    output = capsys.readouterr().out
    assert "validated release manifest" in output


def test_release_cli_recovers_without_loading_state_or_reserving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The recovery mode performs one bound read and writes sanitized state."""
    calls: list[tuple[str, object]] = []
    state = {"schema_version": "robot-sf-zenodo-deposition.v1", "deposition_id": 7}
    manifest = SimpleNamespace(release_tag="v1", metadata_sha256="a" * 64)
    binding = {"release_tag": "v1", "metadata_sha256": "a" * 64}
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
            calls.append(("recover", {"deposition_id": deposition_id, **kwargs})) or state
        ),
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "write_state",
        lambda path, value: calls.append(("write_state", value)),
    )

    assert release_cli.handle(args) == 0
    assert calls == [
        (
            "recover",
            {
                "deposition_id": 7,
                "api_base": "https://example.test/api",
                "release_binding": binding,
            },
        ),
        ("write_state", state),
    ]
    assert "secret" not in capsys.readouterr().out


def test_release_cli_returns_blocked_for_verification_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed remote verification is reported with a non-zero status."""
    args = _args("verify", tmp_path)
    args.manifest = tmp_path / "manifest.yaml"
    monkeypatch.setattr(
        release_cli,
        "_load_release_binding",
        lambda value: (
            SimpleNamespace(release_tag="v1", metadata_sha256="a" * 64),
            {"release_tag": "v1", "metadata_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(release_cli.zenodo_publisher, "build_session", lambda path: object())
    monkeypatch.setattr(release_cli.zenodo_publisher, "load_state", lambda path: {})
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "load_dataset_metadata",
        lambda path, **kwargs: {},
    )
    monkeypatch.setattr(
        release_cli.zenodo_publisher,
        "verify",
        lambda *args, **kwargs: {"status": "fail", "problems": ["mismatch"]},
    )
    assert release_cli.handle(args) == 2


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
