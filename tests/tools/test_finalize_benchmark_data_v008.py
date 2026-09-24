"""Fail-closed checks for the deferred 0.0.8 publication step."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.artifact_publication import PublicationPreflightError
from scripts.tools import finalize_benchmark_data_v008 as finalizer

SOURCE_SHA = "a" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _producer(root: Path) -> dict:
    result = {
        "release_benchmark_success": True,
        "release_status": "ok",
        "release_exit_code": 0,
        "release_acceptance": {"status": "valid"},
        "publication_requested": False,
        "publication_preflight_status": "not_requested",
        "benchmark_release": {
            "source_commit": SOURCE_SHA,
            "release_tag": "benchmark-data-0.0.8",
            "version_doi": "10.5281/zenodo.123456",
        },
    }
    _write_json(root / "release" / "release_result.json", result)
    _write_json(
        root / "release" / "release_manifest.resolved.json",
        {
            "release_tag": "benchmark-data-0.0.8",
            "provenance": {"source_sha": SOURCE_SHA, "version_doi": "10.5281/zenodo.123456"},
        },
    )
    _write_json(root / "campaign_manifest.json", {"git_hash": SOURCE_SHA})
    return result


def test_finalize_copies_accepted_producer_before_postrun_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    original = _producer(producer)
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "predecessor.tar.gz"
    baseline.write_bytes(b"frozen predecessor fixture")
    identity = tmp_path / "identity.json"
    identity.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "BASELINE_ARCHIVE_SHA256", finalizer._sha256(baseline))
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    manifest = SimpleNamespace(
        source_sha=SOURCE_SHA,
        release_tag="benchmark-data-0.0.8",
        version_doi="10.5281/zenodo.123456",
    )
    monkeypatch.setattr(finalizer, "verify_resolved_release_identity", lambda _: manifest)
    monkeypatch.setattr(finalizer, "load_release_campaign_config", lambda _: object())
    calls: list[str] = []

    def gate(command: list[str], log_path: Path) -> None:
        calls.append(Path(command[1]).name)
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": "pass"})
        log_path.write_text("gate passed\n", encoding="utf-8")

    def acceptance(root: Path, **kwargs: object) -> dict:
        assert root == candidate
        assert calls == [
            "check_release_metric_equivalence.py",
            "issue_9668_robot_force_validation.py",
        ]
        return {"status": "valid"}

    def publish(root: Path, result: dict, checked_manifest: object) -> Path:
        assert root == candidate
        assert result == original
        assert checked_manifest is manifest
        archive = tmp_path / "candidate.tar.gz"
        archive.write_bytes(b"candidate bundle fixture")
        return archive

    monkeypatch.setattr(finalizer, "_run_gate", gate)
    monkeypatch.setattr(finalizer, "validate_full_benchmark_release_acceptance", acceptance)
    monkeypatch.setattr(finalizer, "_publish_copy", publish)
    result = finalizer.finalize(
        producer_root=producer,
        candidate_root=candidate,
        resolved_identity=identity,
        baseline_archive=baseline,
        expected_source_sha=SOURCE_SHA,
    )
    assert result["publication_archive_sha256"] == finalizer._sha256(tmp_path / "candidate.tar.gz")
    assert json.loads((producer / "release" / "release_result.json").read_text()) == original
    assert (
        json.loads((candidate / "release" / "producer_release_result.json").read_text()) == original
    )
    assert (
        json.loads((candidate / "release" / "release_result.json").read_text())[
            "finalization_status"
        ]
        == "fail"
    )  # Publication is mocked; the real helper replaces this provisional state.
    assert (tmp_path / "candidate.finalization_receipt.json").is_file()


def test_producer_rejects_changed_source_or_publication_identity(tmp_path: Path) -> None:
    root = tmp_path / "producer"
    _producer(root)
    with pytest.raises(ValueError, match="exact-source"):
        finalizer._require_producer(root, "b" * 40)
    campaign = root / "campaign_manifest.json"
    _write_json(campaign, {"git_hash": "b" * 40})
    with pytest.raises(ValueError, match="exact-source"):
        finalizer._require_producer(root, SOURCE_SHA)


def test_postrun_gate_failure_invalidates_only_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    original = _producer(producer)
    baseline = tmp_path / "predecessor.tar.gz"
    baseline.write_bytes(b"frozen predecessor fixture")
    monkeypatch.setattr(finalizer, "BASELINE_ARCHIVE_SHA256", finalizer._sha256(baseline))
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        finalizer,
        "verify_resolved_release_identity",
        lambda _: SimpleNamespace(
            source_sha=SOURCE_SHA,
            release_tag="benchmark-data-0.0.8",
            version_doi="10.5281/zenodo.123456",
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_run_gate",
        lambda command, log: (_ for _ in ()).throw(ValueError("equivalence mismatch")),
    )
    candidate = tmp_path / "candidate"
    with pytest.raises(ValueError, match="equivalence mismatch"):
        finalizer.finalize(
            producer_root=producer,
            candidate_root=candidate,
            resolved_identity=tmp_path / "identity.json",
            baseline_archive=baseline,
            expected_source_sha=SOURCE_SHA,
        )
    assert json.loads((producer / "release" / "release_result.json").read_text()) == original
    rejected = json.loads((candidate / "release" / "release_result.json").read_text())
    assert rejected["release_benchmark_success"] is False
    assert rejected["finalization_failed_stage"] == "metric_equivalence"


def test_publication_export_failure_is_recorded_in_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    original = _producer(candidate)
    _write_json(candidate / "release" / "producer_release_result.json", original)

    def fail_export(**kwargs: object) -> dict:
        pending = json.loads((candidate / "release" / "release_result.json").read_text())
        assert pending["finalization_status"] == "pass"
        raise PublicationPreflightError("required report missing")

    monkeypatch.setattr(
        finalizer,
        "_build_publication_payload",
        fail_export,
    )
    with pytest.raises(PublicationPreflightError, match="required report missing"):
        finalizer._publish_copy(
            candidate,
            original,
            SimpleNamespace(
                release_tag="benchmark-data-0.0.8",
                doi="10.5281/zenodo.123456",
                repository_url="https://example.org/repository",
            ),
        )
    result = json.loads((candidate / "release" / "release_result.json").read_text())
    assert result["release_benchmark_success"] is False
    assert result["publication_preflight_status"] == "fail"
    assert result["publication_preflight_violations"] == ["required report missing"]
    assert (
        json.loads((candidate / "release" / "producer_release_result.json").read_text()) == original
    )


def test_publication_path_rejects_external_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="not a repository path"):
        finalizer._publication_path({"bundle_dir": "<external>/bundle"}, "bundle_dir")
    with pytest.raises(ValueError, match="leaves the source checkout"):
        finalizer._publication_path({"bundle_dir": "../bundle"}, "bundle_dir")


def test_publication_does_not_accept_missing_export_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    original = _producer(candidate)
    _write_json(candidate / "release" / "producer_release_result.json", original)
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        finalizer,
        "_build_publication_payload",
        lambda **kwargs: {
            "bundle_dir": "publication/missing_bundle",
            "archive_path": "publication/missing_bundle.tar.gz",
        },
    )
    monkeypatch.setattr(finalizer, "_record_publication_payload", lambda *args: None)
    with pytest.raises(ValueError, match="did not leave a bundle"):
        finalizer._publish_copy(
            candidate,
            original,
            SimpleNamespace(
                release_tag="benchmark-data-0.0.8",
                doi="10.5281/zenodo.123456",
                repository_url="https://example.org/repository",
            ),
        )
    rejected = json.loads((candidate / "release" / "release_result.json").read_text())
    assert rejected["publication_preflight_status"] == "fail"
