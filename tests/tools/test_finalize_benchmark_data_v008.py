"""Fail-closed checks for the deferred 0.0.8 publication step."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.artifact_publication import PublicationPreflightError
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig
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
            "source_sha": SOURCE_SHA,
            "release_tag": "benchmark-data-0.0.8",
            "provenance": {"source_sha": SOURCE_SHA, "version_doi": "10.5281/zenodo.123456"},
        },
    )
    _write_json(root / "campaign_manifest.json", {"git_hash": SOURCE_SHA})
    return result


def test_pre_doi_candidate_custody_rejects_changed_raw_bytes(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    baseline = tmp_path / "predecessor.tar.gz"
    baseline.write_bytes(b"synthetic archive only")
    monkeypatch.setattr(finalizer, "BASELINE_ARCHIVE_SHA256", finalizer._sha256(baseline))
    registry = tmp_path / "model/registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("fixture: true\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    for index in range(14):
        path = producer / f"runs/arm{index}__differential_drive/episodes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'{{"arm":{index}}}\n'.encode())
    sidecar_names = [
        "campaign_manifest.json",
        "manifest.json",
        "run_meta.json",
        "reports/campaign_summary.json",
        "reports/campaign_integrity.json",
    ]
    sidecar_names.extend(
        name
        for index in range(14)
        for name in (
            f"runs/arm{index}__differential_drive/episodes.jsonl.provenance.json",
            f"runs/arm{index}__differential_drive/summary.json",
        )
    )
    for name in sidecar_names:
        payload = {"source": SOURCE_SHA}
        if name == "reports/campaign_summary.json":
            payload = {"campaign": {"status": "benchmark_success", "benchmark_success": True}}
        elif name == "campaign_manifest.json":
            payload = {"config_hash": "c" * 64}
        _write_json(producer / name, payload)
    science = {
        "provenance": {"source_sha": SOURCE_SHA},
        **{
            key: {}
            for key in ("matrix", "scenario", "seed_policy", "planners", "kinematics", "metrics")
        },
    }
    identity = {
        "schema_version": "benchmark-scientific-candidate.v1",
        "baseline_archive_sha256": finalizer.BASELINE_ARCHIVE_SHA256,
        "source_sha": SOURCE_SHA,
        "campaign_template_path": "configs/benchmarks/fixture.yaml",
        "campaign_template_sha256": "b" * 64,
        "scientific_config_hash_schema": "camera-ready-publication-free.v1",
        "scientific_config_hash": "c" * 64,
        "model_registry_sha256": finalizer._sha256(registry),
        "scientific_manifest": science,
        "raw_episode_sha256": finalizer._raw_episode_hashes(producer),
        "producer_sidecar_sha256": {
            name: finalizer._sha256(producer / name) for name in sidecar_names
        },
    }
    canonical = (
        json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    identity["scientific_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    identity_path = producer / "release/scientific_candidate.json"
    _write_json(identity_path, identity)
    reports = {
        "full_acceptance_sha256": (
            "reports/scientific_candidate_acceptance.json",
            {"status": "valid"},
        ),
        "metric_equivalence_sha256": ("reports/metric_equivalence.json", {"status": "pass"}),
        "robot_force_validation_sha256": (
            "reports/robot_force_validation.json",
            {"classification": "release_robot_force_validation", "episodes": 20160},
        ),
    }
    result = {
        "schema_version": "benchmark-scientific-candidate-result.v1",
        "baseline_archive_sha256": finalizer.BASELINE_ARCHIVE_SHA256,
        "status": "accepted_pre_publication",
        "source_sha": SOURCE_SHA,
        "identity_file_sha256": finalizer._sha256(identity_path),
        "scientific_identity_sha256": identity["scientific_identity_sha256"],
    }
    for key, (name, report) in reports.items():
        _write_json(producer / name, report)
        result[key] = finalizer._sha256(producer / name)
    for key, name in (
        ("metric_equivalence_log_sha256", "reports/scientific_candidate_equivalence.log"),
        ("robot_force_log_sha256", "reports/scientific_candidate_force.log"),
    ):
        log = producer / name
        log.write_text("synthetic gate passed\n", encoding="utf-8")
        result[key] = finalizer._sha256(log)
    _write_json(producer / "release/scientific_candidate_result.json", result)
    manifest = SimpleNamespace(
        source_sha=SOURCE_SHA,
        resolved_manifest_payload={
            **science,
            "canonical_campaign_config": "configs/benchmarks/fixture.yaml",
            "canonical_campaign_config_sha256": "b" * 64,
        },
    )
    assert finalizer._require_scientific_candidate(producer, SOURCE_SHA, manifest)[0] == identity
    result["baseline_archive_sha256"] = "b" * 64
    _write_json(producer / "release/scientific_candidate_result.json", result)
    with pytest.raises(ValueError, match="accepted exact-source scientific candidate"):
        finalizer._require_scientific_candidate(producer, SOURCE_SHA, manifest)
    result["baseline_archive_sha256"] = finalizer.BASELINE_ARCHIVE_SHA256
    _write_json(producer / "release/scientific_candidate_result.json", result)
    resolved_identity = tmp_path / "resolved_identity.json"
    resolved_identity.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(finalizer, "verify_resolved_release_identity", lambda _: manifest)
    monkeypatch.setattr(finalizer, "_read_scientific_candidate_manifest", lambda *_: science)
    monkeypatch.setattr(
        finalizer, "build_release_provenance", lambda *_, **__: {"source_sha": SOURCE_SHA}
    )
    monkeypatch.setattr(finalizer, "_merge_release_provenance", lambda *_, **__: None)
    monkeypatch.setattr(
        finalizer,
        "load_release_campaign_config",
        lambda _: CampaignConfig(name="synthetic", scenario_matrix_path=Path("s"), planners=()),
    )
    monkeypatch.setattr(
        finalizer,
        "validate_full_benchmark_release_acceptance",
        lambda *_, **__: {"status": "valid"},
    )

    def publication_stub(candidate: Path, _result: dict, _manifest: object) -> Path:
        assert finalizer._raw_episode_hashes(candidate) == identity["raw_episode_sha256"]
        archive = tmp_path / "derivative.tar.gz"
        archive.write_bytes(b"synthetic derivative")
        return archive

    monkeypatch.setattr(finalizer, "_publish_copy", publication_stub)
    receipt = finalizer.finalize_pre_doi_candidate(
        producer_root=producer,
        candidate_root=tmp_path / "derivative",
        resolved_identity=resolved_identity,
        baseline_archive=baseline,
        expected_source_sha=SOURCE_SHA,
    )
    assert receipt["raw_episode_sha256"] == finalizer._raw_episode_hashes(producer)
    assert receipt["raw_episode_sha256"] == finalizer._raw_episode_hashes(tmp_path / "derivative")
    raw_path = producer / "runs/arm0__differential_drive/episodes.jsonl"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw episode bytes changed"):
        finalizer._require_scientific_candidate(producer, SOURCE_SHA, manifest)
    raw_path.write_bytes(b'{"arm":0}\n')
    sidecar = producer / "run_meta.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar bytes changed"):
        finalizer._require_scientific_candidate(producer, SOURCE_SHA, manifest)


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
        resolved_manifest_payload=finalizer._read_mapping(
            producer / "release" / "release_manifest.resolved.json"
        ),
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
    _write_json(campaign, {"git_hash": SOURCE_SHA})
    resolved_path = root / "release" / "release_manifest.resolved.json"
    resolved = finalizer._read_mapping(resolved_path)
    resolved["source_sha"] = "b" * 40
    _write_json(resolved_path, resolved)
    with pytest.raises(ValueError, match="exact-source"):
        finalizer._require_producer(root, SOURCE_SHA)


def test_finalizer_rejects_scientific_manifest_drift_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    _producer(producer)
    baseline = tmp_path / "predecessor.tar.gz"
    baseline.write_bytes(b"frozen predecessor fixture")
    monkeypatch.setattr(finalizer, "BASELINE_ARCHIVE_SHA256", finalizer._sha256(baseline))
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    expected = finalizer._read_mapping(producer / "release" / "release_manifest.resolved.json")
    expected["canonical_campaign_config_sha256"] = "b" * 64
    monkeypatch.setattr(
        finalizer,
        "verify_resolved_release_identity",
        lambda _: SimpleNamespace(
            source_sha=SOURCE_SHA,
            release_tag="benchmark-data-0.0.8",
            version_doi="10.5281/zenodo.123456",
            resolved_manifest_payload=expected,
        ),
    )
    candidate = tmp_path / "candidate"
    with pytest.raises(ValueError, match="resolved manifest differs"):
        finalizer.finalize(
            producer_root=producer,
            candidate_root=candidate,
            resolved_identity=tmp_path / "identity.json",
            baseline_archive=baseline,
            expected_source_sha=SOURCE_SHA,
        )
    assert not candidate.exists()


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
            resolved_manifest_payload=finalizer._read_mapping(
                producer / "release" / "release_manifest.resolved.json"
            ),
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


def test_interrupted_copy_never_exposes_accepted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    _producer(producer)
    candidate = tmp_path / "candidate"
    original_copy2 = finalizer.shutil.copy2

    def interrupt_snapshot(source: Path, target: Path) -> None:
        if source == producer / "release" / "release_result.json":
            raise KeyboardInterrupt
        original_copy2(source, target)

    monkeypatch.setattr(finalizer.shutil, "copy2", interrupt_snapshot)
    with pytest.raises(KeyboardInterrupt):
        finalizer._copy_producer(producer, candidate)
    partial = tmp_path / "candidate.copying"
    assert partial.is_dir()
    assert not (partial / "release" / "release_result.json").exists()
    assert not candidate.exists()


def test_failed_preflight_removes_owned_publication_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    original = _producer(candidate)
    _write_json(candidate / "release" / "producer_release_result.json", original)
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    output = finalizer._publication_output(candidate)

    def export(**kwargs: object) -> dict:
        assert kwargs["output_dir"] == output
        bundle = output / "candidate_publication_bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        archive = output / "candidate_publication_bundle.tar.gz"
        archive.write_bytes(b"unverified bundle")
        return {
            "bundle_dir": str(bundle.relative_to(tmp_path)),
            "archive_path": str(archive.relative_to(tmp_path)),
        }

    monkeypatch.setattr(finalizer, "_build_publication_payload", export)
    monkeypatch.setattr(finalizer, "_record_publication_payload", lambda *args: None)
    monkeypatch.setattr(finalizer, "_assert_no_historical_release_identity", lambda *args: None)
    monkeypatch.setattr(
        finalizer,
        "_run_publication_preflight",
        lambda *args: (_ for _ in ()).throw(PublicationPreflightError("force drift")),
    )
    manifest = SimpleNamespace(
        release_tag="benchmark-data-0.0.8",
        doi="10.5281/zenodo.123456",
        repository_url="https://example.org/repository",
    )
    with pytest.raises(PublicationPreflightError, match="force drift"):
        finalizer._publish_copy(candidate, original, manifest)
    assert not output.exists()
    assert (
        finalizer._read_mapping(candidate / "release" / "release_result.json")[
            "release_benchmark_success"
        ]
        is False
    )

    output.mkdir()
    sentinel = output / "prior.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        finalizer._publish_copy(candidate, original, manifest)
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    sentinel.unlink()
    output.rmdir()
    external = tmp_path / "external_missing"
    output.symlink_to(external, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        finalizer._publish_copy(candidate, original, manifest)
    assert output.is_symlink()
    assert not external.exists()


@pytest.mark.parametrize("mark_failure", [False, True])
def test_receipt_write_failure_invalidates_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mark_failure: bool
) -> None:
    producer = tmp_path / "producer"
    _producer(producer)
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline.tar.gz"
    baseline.write_bytes(b"predecessor")
    (tmp_path / "identity.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(finalizer, "BASELINE_ARCHIVE_SHA256", finalizer._sha256(baseline))
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(finalizer, "load_release_campaign_config", lambda _: object())
    monkeypatch.setattr(
        finalizer,
        "verify_resolved_release_identity",
        lambda _: SimpleNamespace(
            source_sha=SOURCE_SHA,
            release_tag="benchmark-data-0.0.8",
            version_doi="10.5281/zenodo.123456",
            resolved_manifest_payload=finalizer._read_mapping(
                producer / "release" / "release_manifest.resolved.json"
            ),
        ),
    )

    def gate(command: list[str], log_path: Path) -> None:
        _write_json(Path(command[command.index("--output") + 1]), {"status": "pass"})
        log_path.write_text("pass\n", encoding="utf-8")

    def publish(root: Path, result: dict, manifest: object) -> Path:
        output = finalizer._publication_output(root)
        output.mkdir()
        archive = output / "bundle.tar.gz"
        archive.write_bytes(b"candidate archive")
        return archive

    real_write = finalizer._write_json

    def fail_receipt(path: Path, payload: dict) -> None:
        if path.name.endswith(".finalization_receipt.pending.json"):
            raise OSError("receipt storage failed")
        real_write(path, payload)

    monkeypatch.setattr(finalizer, "_run_gate", gate)
    monkeypatch.setattr(
        finalizer,
        "validate_full_benchmark_release_acceptance",
        lambda *args, **kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(finalizer, "_publish_copy", publish)
    monkeypatch.setattr(finalizer, "_write_json", fail_receipt)
    if mark_failure:
        monkeypatch.setattr(
            finalizer,
            "_mark_candidate_failure",
            lambda *args: (_ for _ in ()).throw(OSError("mark storage failed")),
        )
    expected_error = "mark storage failed" if mark_failure else "receipt storage failed"
    with pytest.raises(OSError, match=expected_error):
        finalizer.finalize(
            producer_root=producer,
            candidate_root=candidate,
            resolved_identity=tmp_path / "identity.json",
            baseline_archive=baseline,
            expected_source_sha=SOURCE_SHA,
        )
    if not mark_failure:
        result = finalizer._read_mapping(candidate / "release" / "release_result.json")
        assert result["release_benchmark_success"] is False
        assert result["finalization_failed_stage"] == "finalization_receipt"
    assert not finalizer._publication_output(candidate).exists()
    assert not any(path.exists() for path in finalizer._receipt_paths(candidate))


def test_interrupted_export_invalidates_candidate_and_removes_partial_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    original = _producer(candidate)
    _write_json(candidate / "release" / "producer_release_result.json", original)
    output = finalizer._publication_output(candidate)

    def interrupt_export(**kwargs: object) -> dict:
        output.mkdir()
        (output / "partial.tar.gz").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(finalizer, "_build_publication_payload", interrupt_export)
    with pytest.raises(KeyboardInterrupt):
        finalizer._publish_copy(
            candidate,
            original,
            SimpleNamespace(
                release_tag="benchmark-data-0.0.8",
                doi="10.5281/zenodo.123456",
                repository_url="https://example.org/repository",
            ),
        )
    assert not output.exists()
    result = finalizer._read_mapping(candidate / "release" / "release_result.json")
    assert result["release_benchmark_success"] is False
    assert result["publication_preflight_violations"] == ["KeyboardInterrupt"]


def test_prepare_candidate_preserves_preexisting_dangling_output_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = tmp_path / "producer"
    _producer(producer)
    candidate = tmp_path / "candidate"
    outside = tmp_path / "missing_elsewhere"
    output = finalizer._publication_output(candidate)
    output.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(finalizer, "get_repository_root", lambda: tmp_path)
    manifest = SimpleNamespace(
        release_tag="benchmark-data-0.0.8",
        version_doi="10.5281/zenodo.123456",
        resolved_manifest_payload=finalizer._read_mapping(
            producer / "release" / "release_manifest.resolved.json"
        ),
    )
    with pytest.raises(FileExistsError, match="publication output already exists"):
        finalizer._prepare_candidate(producer, candidate, SOURCE_SHA, manifest)
    assert output.is_symlink()
    assert not outside.exists()
    assert not candidate.exists()
