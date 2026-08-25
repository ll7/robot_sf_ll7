"""Focused tests for preserved post-execution release acceptance."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path  # noqa: TC003
from types import SimpleNamespace

import pytest
import yaml

from robot_sf import cli as robot_sf_cli
from robot_sf import release_cli
from robot_sf.benchmark import post_execution_release_doctor as doctor


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _acceptance() -> dict[str, object]:
    return {
        "status": "valid",
        "benchmark_success": True,
        "blockers": [],
        "expected_planner_arms": 14,
        "expected_scenario_count": 48,
        "expected_seed_count": 30,
        "expected_episode_cells": 20_160,
        "observed_episode_rows": 20_160,
        "unique_episode_identities": 20_160,
        "successful_planner_arms": 14,
        "missing_episode_identities": 0,
        "unexpected_episode_identities": 0,
        "forbidden_status_counts": {},
        "source_commits": [doctor.FROZEN_SOURCE_SHA],
    }


def _synthetic_contract(tmp_path: Path) -> dict[str, Path]:
    """Build the complete small post-execution contract without episode payloads."""
    bundle = tmp_path / "bundle"
    payload = bundle / "payload"
    archive = tmp_path / "bundle.tar.gz"
    release_result = {
        "benchmark_release": {
            "release_tag": doctor.EXPECTED_RELEASE_TAG,
            "latest_main_base_commit": doctor.EXPECTED_BASE_SHA,
        },
        "campaign_id": doctor.EXPECTED_CAMPAIGN_ID,
        "release_status": "ok",
        "release_exit_code": 0,
        "publication_preflight_status": "pass",
        "full_release_acceptance": _acceptance(),
        "release_acceptance": _acceptance(),
        "benchmark_success": True,
        "release_benchmark_success": True,
    }
    result_path = _write_json(payload / "release" / "release_result.json", release_result)
    _write_json(
        payload / "release" / "release_manifest.resolved.json",
        {"provenance": {"latest_main_base_commit": doctor.EXPECTED_BASE_SHA}},
    )
    packet = {
        "schema": "robot-sf-launch-packet.v1",
        "queue_id": "s30-h600-release-b1d5ab6de708-20260825",
        "campaign_id": doctor.EXPECTED_CAMPAIGN_ID,
        "campaign": doctor.EXPECTED_CAMPAIGN_ID,
        "status": "admitted_frozen",
        "identity": {
            "public_source_commit": doctor.FROZEN_SOURCE_SHA,
            "release_tag": doctor.EXPECTED_RELEASE_TAG,
            "latest_main_base_commit": doctor.EXPECTED_BASE_SHA,
            "checkpoint_receipt_path": "checkpoint_staging_receipt.json",
            "checkpoint_receipt_sha256": "0" * 64,
        },
        "accepted_runtime_smoke": {
            "status": "accepted_preserved_verified",
            "public_source_commit": doctor.FROZEN_SOURCE_SHA,
            "expected_episode_cells": 14,
            "fallback_or_degraded_rows": 0,
            "exit_code": "0:0",
            "derived_exit_code": "0:0",
        },
        "accepted_hybrid_stress": {
            "status": "accepted_preserved_verified",
            "public_source_commit": doctor.FROZEN_SOURCE_SHA,
            "expected_episode_cells": 70,
            "gate_receipt_sha256": "1" * 64,
            "release_result_sha256": "2" * 64,
            "preservation_manifest_sha256": "3" * 64,
        },
    }
    checkpoint = {
        "schema_version": "campaign-checkpoint-staging-receipt.v1",
        "status": "ok",
        "mode": "enforced_staged",
        "submit_safe": True,
        "checked": 5,
        "resolved": 5,
    }
    checkpoint_path = _write_json(payload / "checkpoint_staging_receipt.json", checkpoint)
    packet["identity"]["checkpoint_receipt_sha256"] = doctor._sha256(checkpoint_path)
    packet_path = _write_yaml(tmp_path / "packet.yaml", packet)
    _write_yaml(payload / "launch_packet.yaml", packet)
    _write_json(
        payload / "runtime_smoke_release_result.json",
        {"benchmark_success": True, "public_source_commit": doctor.FROZEN_SOURCE_SHA},
    )
    _write_json(
        payload / "accepted_hybrid_stress_gate.json",
        {"status": "accepted_preserved_verified", "public_source_commit": doctor.FROZEN_SOURCE_SHA},
    )
    receipt = {
        "schema_version": "benchmark-derived-revalidation.v1",
        "mode": "preserved_rows_corrected_validator",
        "source": {"execution_commit": doctor.FROZEN_SOURCE_SHA},
        "acceptance": _acceptance(),
        "projection_acceptance": _acceptance(),
        "source_acceptance": _acceptance(),
        "validator": {
            "commit": doctor.EXPECTED_VALIDATOR_SHA,
            "expected_reviewed_commit": doctor.EXPECTED_VALIDATOR_SHA,
            "file_sha256": "4" * 64,
        },
        "snqi": {"status": "advisory", "ranking_authority": False},
        "publication_reconciliation": {
            "scientific_execution_changed": False,
            "simulation_rerun": False,
            "sidecar_path_binding": {"row_count": 20_160},
            "goal_timeout_boundary": {"timing_evidence_fabricated": False},
        },
        "publication_inputs": {"manifest": "manifest.json"},
        "credentials": "not_recorded",
        "cross_root_binding": {
            "retrieved_file_map": {
                "launch_packet.yaml": {"sha256": doctor._sha256(packet_path)},
                "checkpoint_staging_receipt.json": {"sha256": doctor._sha256(checkpoint_path)},
            }
        },
    }
    receipt_path = _write_json(tmp_path / "derived_receipt.json", receipt)
    _write_json(payload / "provenance" / "derived_revalidation_receipt.json", receipt)
    _write_json(bundle / "publication_manifest.json", {"schema_version": "publication-bundle.v1"})
    (bundle / "checksums.sha256").write_text("synthetic\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    preservation_campaign = "derived-synthetic"
    preservation_receipt = {
        "status": "verified",
        "campaign_id": preservation_campaign,
        "receipt_digest": "sha256:" + "5" * 64,
        "manifest_digest": "sha256:" + "6" * 64,
        "two_copy_policy": {
            "satisfied": True,
            "verified_copies": 2,
            "distinct_failure_domains": ["local", "remote"],
            "distinct_backend_classes": ["disk", "wandb"],
        },
    }
    preservation_path = _write_json(tmp_path / "preservation.json", preservation_receipt)
    evaluation = {
        "schema": "robot-sf-derived-evaluation-receipt.v1",
        "evaluation_status": "complete",
        "evidence_valid": True,
        "scientific_outcome": "not_applicable",
        "source": {"public_commit": doctor.FROZEN_SOURCE_SHA},
        "producer_campaign_id": doctor.EXPECTED_CAMPAIGN_ID,
        "job_id": doctor.EXPECTED_JOB_ID,
        "execution": {
            "simulation_rerun": False,
            "scientific_execution_status": "completed",
            "scheduler_state": "FAILED",
            "exit_code": "2:0",
            "execution_status": "failed",
            "completion_status": "failed",
            "no_rerun_authorized": True,
        },
        "acceptance": {
            "status": "valid",
            "expected_planner_arms": 14,
            "successful_planner_arms": 14,
            "expected_episode_cells": 20_160,
            "observed_episode_rows": 20_160,
            "unique_episode_identities": 20_160,
            "missing_episode_identities": 0,
            "unexpected_episode_identities": 0,
            "duplicate_episode_identities": 0,
            "blockers": [],
            "forbidden_status_counts": {},
            "source_commits": [doctor.FROZEN_SOURCE_SHA],
        },
        "snqi": {
            "calibration_status": "fail",
            "policy": "warn",
            "status": "advisory",
            "ranking_authority": False,
            "ranking_claims_admitted": False,
        },
        "derived_campaign_id": preservation_campaign,
        "derived_bundle": {
            "archive": {
                "path": str(archive),
                "sha256": doctor._sha256(archive),
                "bytes": archive.stat().st_size,
            },
            "revalidation_receipt": {
                "path": str(receipt_path),
                "sha256": doctor._sha256(receipt_path),
            },
            "release_result": {
                "path": str(result_path),
                "sha256": doctor._sha256(result_path),
            },
        },
        "preservation": {
            "status": "verified",
            "receipt_path": str(preservation_path),
            "receipt_digest": preservation_receipt["receipt_digest"],
            "manifest_digest": preservation_receipt["manifest_digest"],
            "two_copy_satisfied": True,
            "remote_state": "COMMITTED",
            "readback_match": True,
            "digest_mismatches": 0,
        },
    }
    evaluation_path = _write_json(tmp_path / "evaluation.json", evaluation)
    queue, jobs = _private_ledgers(tmp_path)
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    queue_payload[0]["artifact_manifest"] = (
        f"ops/jobs/launch_packets/{packet_path.name} sha256:{doctor._sha256(packet_path)}"
    )
    _write_yaml(queue, queue_payload)
    _write_json(
        tmp_path / "preflight.json",
        {
            "schema_version": "publication-preflight.v1",
            "status": "pass",
            "violation_count": 0,
            "violations": [],
            "bundle_dir": str(bundle),
        },
    )
    return {
        "archive": archive,
        "bundle": bundle,
        "evaluation": evaluation_path,
        "jobs": jobs,
        "packet": packet_path,
        "preflight": tmp_path / "preflight.json",
        "queue": queue,
        "receipt": receipt_path,
    }


def _private_ledgers(tmp_path: Path) -> tuple[Path, Path]:
    queue = {
        "queue_id": "s30-h600-release-b1d5ab6de708-20260825",
        "campaign": doctor.EXPECTED_CAMPAIGN_ID,
        "state": "failed",
        "go": "false",
        "expected_public_commit": doctor.FROZEN_SOURCE_SHA,
        "attempts": "1",
        "max_attempts": "1",
        "go_reason": "Job 14890 terminal publication validator rejected a provenance model defect",
    }
    job = {
        "job_id": doctor.EXPECTED_JOB_ID,
        "campaign": doctor.EXPECTED_CAMPAIGN_ID,
        "public_commit": doctor.FROZEN_SOURCE_SHA,
        "state": "analyzed",
        "slurm_state": "FAILED",
        "exit_code": "2:0",
        "artifact_status": "verified",
        "terminal_triage_reason": "Job 14890 validator-only publication gate failure after complete execution",
    }
    return _write_yaml(tmp_path / "queue.yaml", [queue]), _write_yaml(tmp_path / "jobs.yaml", [job])


def test_terminal_failed_queue_requires_derived_receipt(tmp_path: Path) -> None:
    """Validator-only scheduler text cannot replace reviewed derived evidence."""
    queue_path, jobs_path = _private_ledgers(tmp_path)
    check = doctor._private_execution_check(
        queue_path,
        jobs_path,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        job_id=doctor.EXPECTED_JOB_ID,
        derived_evaluation_receipt=None,
        public_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
    )
    assert check.status == "fail"
    assert "private derived evaluation receipt is required" in check.summary


def test_terminal_queue_rejects_second_attempt(tmp_path: Path) -> None:
    """The post-run exception cannot authorize a retry of the same identity."""
    queue_path, jobs_path = _private_ledgers(tmp_path)
    queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    queue[0]["attempts"] = "2"
    _write_yaml(queue_path, queue)
    check = doctor._private_execution_check(
        queue_path,
        jobs_path,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        job_id=doctor.EXPECTED_JOB_ID,
        derived_evaluation_receipt=None,
        public_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
    )
    assert check.status == "fail"
    assert "one consumed attempt" in check.summary


def test_terminal_queue_rejects_malformed_forbidden_marker(tmp_path: Path) -> None:
    """A structured or nonnumeric marker cannot masquerade as a zero count."""
    queue_path, jobs_path = _private_ledgers(tmp_path)
    queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    queue[0]["fallback_rows"] = ["malformed-positive-marker"]
    _write_yaml(queue_path, queue)
    check = doctor._private_execution_check(
        queue_path,
        jobs_path,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        job_id=doctor.EXPECTED_JOB_ID,
        derived_evaluation_receipt=None,
        public_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
    )
    assert check.status == "fail"
    assert "positive forbidden runtime marker" in check.summary


def test_post_execution_receipt_rejects_ranking_snqi(tmp_path: Path) -> None:
    """A derived receipt cannot turn failed SNQI calibration into ranking evidence."""
    acceptance = {
        "status": "valid",
        "benchmark_success": True,
        "blockers": [],
        "expected_planner_arms": 14,
        "expected_scenario_count": 48,
        "expected_seed_count": 30,
        "expected_episode_cells": 20_160,
        "observed_episode_rows": 20_160,
        "unique_episode_identities": 20_160,
        "successful_planner_arms": 14,
        "missing_episode_identities": 0,
        "unexpected_episode_identities": 0,
        "forbidden_status_counts": {},
        "source_commits": [doctor.FROZEN_SOURCE_SHA],
    }
    receipt = {
        "schema_version": "benchmark-derived-revalidation.v1",
        "mode": "preserved_rows_corrected_validator",
        "source": {"execution_commit": doctor.FROZEN_SOURCE_SHA},
        "acceptance": acceptance,
        "projection_acceptance": acceptance,
        "source_acceptance": acceptance,
        "validator": {
            "commit": "a" * 40,
            "expected_reviewed_commit": "a" * 40,
            "file_sha256": "b" * 64,
        },
        "snqi": {"status": "advisory", "ranking_authority": True},
        "publication_reconciliation": {
            "scientific_execution_changed": False,
            "simulation_rerun": False,
            "sidecar_path_binding": {"row_count": 20_160},
        },
        "publication_inputs": {"manifest": "manifest.json"},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    check, _ = doctor._receipt_check(
        receipt_path,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        expected_validator_sha=None,
    )
    assert check.status == "fail"
    assert "ranking authority" in check.summary


def test_portable_checkpoint_copy_remains_bound_to_exact_producer_digest() -> None:
    """Portable path rewriting cannot replace the admitted producer receipt."""
    digest = "a" * 64
    receipt = {
        "cross_root_binding": {
            "retrieved_file_map": {
                "checkpoint_staging_receipt.json": {"sha256": digest, "bytes": 123}
            }
        }
    }
    assert doctor._producer_file_digest(receipt, "checkpoint_staging_receipt.json") == digest
    receipt["cross_root_binding"]["retrieved_file_map"]["checkpoint_staging_receipt.json"][
        "sha256"
    ] = "not-a-digest"
    assert doctor._producer_file_digest(receipt, "checkpoint_staging_receipt.json") is None


def test_malformed_two_copy_identities_fail_without_exception() -> None:
    """Unhashable or duplicate preservation identities fail closed."""
    assert not doctor._two_distinct_strings([{"domain": "a"}, {"domain": "b"}])
    assert not doctor._two_distinct_strings(["same", "same"])
    assert doctor._two_distinct_strings(["local", "remote"])


def test_credential_scan_rejects_recorded_or_payload_credentials(tmp_path: Path) -> None:
    """Credential-shaped publication content is rejected without echoing it."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "safe.json").write_text('{"credentials":"not_recorded"}', encoding="utf-8")
    safe_archive = tmp_path / "safe.tar.gz"
    with tarfile.open(safe_archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
    assert (
        doctor._credential_scan_problems(bundle, {"credentials": "not_recorded"}, safe_archive)
        == []
    )
    (bundle / "unsafe.txt").write_text("Authorization: Bearer redacted", encoding="utf-8")
    problems = doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, safe_archive
    )
    assert problems == ["publication payload contains credential-shaped content"]
    assert "redacted" not in " ".join(problems)
    (bundle / "unsafe.txt").unlink()
    unsafe_member = tmp_path / "unsafe-member.txt"
    unsafe_member.write_text("Authorization: Bearer archive-redacted", encoding="utf-8")
    unsafe_archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe_archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
        handle.add(unsafe_member, arcname=f"{bundle.name}/unsafe.txt")
    problems = doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, unsafe_archive
    )
    assert problems == ["publication archive contains credential-shaped content"]
    assert "archive-redacted" not in " ".join(problems)
    appended_archive = tmp_path / "appended.tar.gz"
    appended_archive.write_bytes(safe_archive.read_bytes() + b"Authorization: Bearer trailer")
    assert doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, appended_archive
    ) == ["publication archive contains credential-shaped raw bytes"]
    named_archive = tmp_path / "named.tar.gz"
    with tarfile.open(named_archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
        handle.add(
            unsafe_member,
            arcname=f"{bundle.name}/authorization: bearer member",
        )
    assert doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, named_archive
    ) == ["publication archive contains credential-shaped member metadata"]
    extra_archive = tmp_path / "extra.tar.gz"
    extra = _write_json(tmp_path / "extra.json", {"safe": "extra"})
    with tarfile.open(extra_archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
        handle.add(extra, arcname=f"{bundle.name}/extra.json")
    assert doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, extra_archive
    ) == ["publication archive file set differs from supplied bundle"]
    traversal_directory_archive = tmp_path / "traversal-directory.tar.gz"
    with tarfile.open(traversal_directory_archive, "w:gz") as handle:
        handle.add(bundle, arcname=bundle.name)
        traversal = tarfile.TarInfo(f"{bundle.name}/../escape")
        traversal.type = tarfile.DIRTYPE
        handle.addfile(traversal)
    assert doctor._credential_scan_problems(
        bundle, {"credentials": "not_recorded"}, traversal_directory_archive
    ) == ["publication archive member path differs from bundle layout"]


def test_post_execution_identity_is_fixed_to_frozen_candidate(tmp_path: Path) -> None:
    """Caller overrides cannot retarget the post-execution evidence contract."""
    report = doctor.collect_post_execution_release_doctor_report(
        repo=tmp_path,
        derived_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
        publication_preflight=None,
        private_queue=None,
        private_jobs=None,
        expected_source_sha="a" * 40,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        tag=doctor.EXPECTED_RELEASE_TAG,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        expected_job_id=doctor.EXPECTED_JOB_ID,
    )
    assert report["status"] == "blocked"
    source = next(check for check in report["checks"] if check["name"] == "source_identity")
    assert source["status"] == "fail"
    assert "fixed to the frozen b1d5 source" in source["summary"]


def test_post_execution_identity_rejects_unreviewed_validator(tmp_path: Path) -> None:
    """A self-consistent but unreviewed validator SHA cannot retarget acceptance."""
    report = doctor.collect_post_execution_release_doctor_report(
        repo=tmp_path,
        derived_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
        publication_preflight=None,
        private_queue=None,
        private_jobs=None,
        expected_source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        tag=doctor.EXPECTED_RELEASE_TAG,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        expected_job_id=doctor.EXPECTED_JOB_ID,
        expected_validator_sha="a" * 40,
    )
    assert report["status"] == "blocked"
    identity = next(check for check in report["checks"] if check["name"] == "release_identity")
    assert identity["status"] == "fail"
    assert "validator SHA" in identity["summary"]


def test_post_execution_cli_routes_all_evidence_paths(monkeypatch, tmp_path: Path) -> None:
    """The public CLI anchors relative post-execution paths to ``--repo``."""
    args = robot_sf_cli._build_parser().parse_args(
        [
            "release",
            "doctor",
            "--post-execution",
            "--repo",
            str(tmp_path),
            "--manifest",
            "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml",
            "--expected-release-sha",
            doctor.FROZEN_SOURCE_SHA,
            "--expected-base-sha",
            doctor.EXPECTED_BASE_SHA,
            "--tag",
            doctor.EXPECTED_RELEASE_TAG,
            "--expected-campaign-id",
            doctor.EXPECTED_CAMPAIGN_ID,
            "--derived-revalidation-receipt",
            "derived/receipt.json",
            "--publication-bundle",
            "derived/bundle",
            "--publication-archive",
            "derived/bundle.tar.gz",
            "--publication-preflight",
            "derived/preflight.json",
            "--private-launch-packet",
            "private/packet.yaml",
            "--private-queue",
            "private/queue.yaml",
            "--private-jobs",
            "private/jobs.yaml",
            "--publication-mode",
            "final",
        ]
    )
    captured: dict[str, object] = {}

    def fake_report(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(release_cli, "collect_post_execution_release_doctor_report", fake_report)
    assert release_cli.handle(args) == 0
    assert captured["repo"] == tmp_path.resolve()
    for key, relative in {
        "derived_revalidation_receipt": "derived/receipt.json",
        "publication_bundle": "derived/bundle",
        "publication_archive": "derived/bundle.tar.gz",
        "publication_preflight": "derived/preflight.json",
        "private_launch_packet": "private/packet.yaml",
        "private_queue": "private/queue.yaml",
        "private_jobs": "private/jobs.yaml",
    }.items():
        assert captured[key] == (tmp_path / relative).resolve()
    assert captured["expected_validator_sha"] == doctor.EXPECTED_VALIDATOR_SHA
    assert captured["require_zenodo_webhook_disabled"] is True


def test_complete_synthetic_post_execution_contract_passes(monkeypatch, tmp_path: Path) -> None:
    """Drive every joined evidence owner through one complete synthetic contract."""
    paths = _synthetic_contract(tmp_path)
    monkeypatch.setattr(
        doctor,
        "verify_publication_bundle_preflight",
        lambda _bundle: {"status": "pass"},
    )
    receipt_check, receipt = doctor._receipt_check(
        paths["receipt"],
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        expected_validator_sha=doctor.EXPECTED_VALIDATOR_SHA,
    )
    assert receipt_check.status == "pass"
    bundle_check, result = doctor._bundle_check(
        paths["bundle"],
        receipt_path=paths["receipt"],
        receipt=receipt,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
    )
    assert bundle_check.status == "pass"
    assert result is not None
    assert doctor._publication_preflight_check(paths["preflight"], paths["bundle"]).status == "pass"
    assert (
        doctor._private_execution_check(
            paths["queue"],
            paths["jobs"],
            source_sha=doctor.FROZEN_SOURCE_SHA,
            campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
            job_id=doctor.EXPECTED_JOB_ID,
            derived_evaluation_receipt=paths["evaluation"],
            public_revalidation_receipt=paths["receipt"],
            publication_bundle=paths["bundle"],
            publication_archive=paths["archive"],
        ).status
        == "pass"
    )
    assert (
        doctor._historical_provenance_check(
            paths["bundle"],
            paths["packet"],
            paths["queue"],
            receipt,
            source_sha=doctor.FROZEN_SOURCE_SHA,
            expected_tag=doctor.EXPECTED_RELEASE_TAG,
            expected_base_sha=doctor.EXPECTED_BASE_SHA,
            campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        ).status
        == "pass"
    )
    assert doctor._credential_scan_problems(paths["bundle"], receipt, paths["archive"]) == []

    queue_rows = yaml.safe_load(paths["queue"].read_text(encoding="utf-8"))
    original_manifest = queue_rows[0]["artifact_manifest"]
    packet_digest = original_manifest.partition(" sha256:")[2]
    queue_rows[0]["artifact_manifest"] = f"unrelated.txt sha256:{packet_digest}"
    _write_yaml(paths["queue"], queue_rows)
    rejected = doctor._historical_provenance_check(
        paths["bundle"],
        paths["packet"],
        paths["queue"],
        receipt,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
    )
    assert rejected.status == "fail"
    assert "artifact manifest does not match packet" in rejected.summary
    queue_rows[0]["artifact_manifest"] = original_manifest
    _write_yaml(paths["queue"], queue_rows)
    missing_packet = doctor._historical_provenance_check(
        paths["bundle"],
        None,
        paths["queue"],
        receipt,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
    )
    assert missing_packet.status == "fail"
    assert "missing producer launch packet" in missing_packet.summary

    shared = doctor.PostExecutionReleaseDoctorCheck("shared", "pass", "ok")
    monkeypatch.setattr(doctor, "_git_check", lambda *_args: shared)
    monkeypatch.setattr(doctor, "_ci_check", lambda *_args: shared)
    monkeypatch.setattr(doctor, "_tag_check", lambda *_args: shared)
    monkeypatch.setattr(doctor, "_zenodo_check", lambda *_args, **_kwargs: [shared, shared])
    monkeypatch.setattr(doctor, "_dissertation_check", lambda *_args: shared)
    monkeypatch.setattr(
        doctor,
        "_source_manifest_check",
        lambda *_args, **_kwargs: doctor.PostExecutionReleaseDoctorCheck("manifest", "pass", "ok"),
    )
    report = doctor.collect_post_execution_release_doctor_report(
        repo=tmp_path,
        manifest_path=tmp_path / "manifest.yaml",
        derived_revalidation_receipt=paths["receipt"],
        publication_bundle=paths["bundle"],
        publication_archive=paths["archive"],
        publication_preflight=paths["preflight"],
        private_queue=paths["queue"],
        private_jobs=paths["jobs"],
        private_launch_packet=paths["packet"],
        private_evaluation_receipt=paths["evaluation"],
        minimum_free_gib=0,
    )
    assert report["status"] == "pass"
    assert report["failed_checks"] == []


def test_complete_manifest_check_uses_frozen_validator_subprocess(
    monkeypatch, tmp_path: Path
) -> None:
    """Complete manifest identity and all three local pins are required together."""
    repo = tmp_path / "repo"
    manifest_dir = repo / "configs" / "benchmarks" / "releases"
    manifest_dir.mkdir(parents=True)
    pinned: dict[str, tuple[str, str]] = {}
    for name in ("campaign.yaml", "scenario.yaml", "metadata.json"):
        path = manifest_dir / name
        path.write_text(name, encoding="utf-8")
        pinned[name] = (name, doctor._sha256(path))
    manifest = {
        "schema_version": "benchmark-release-manifest.v0.2",
        "latest_main_base_commit": doctor.EXPECTED_BASE_SHA,
        "release_tag": doctor.EXPECTED_RELEASE_TAG,
        "matrix": {
            "planner_arms": 14,
            "scenarios": 48,
            "seeds": 30,
            "expected_episode_cells": 20_160,
            "horizon_steps": 600,
        },
        "canonical_campaign_config": pinned["campaign.yaml"][0],
        "campaign_config_sha256": pinned["campaign.yaml"][1],
        "scenario": {
            "matrix_path": pinned["scenario.yaml"][0],
            "matrix_sha256": pinned["scenario.yaml"][1],
        },
        "publication": {
            "metadata_path": pinned["metadata.json"][0],
            "metadata_sha256": pinned["metadata.json"][1],
        },
    }
    manifest_path = _write_yaml(manifest_dir / "manifest.yaml", manifest)
    complete = {
        "report": {"status": "valid"},
        "identity": {
            "release_tag": doctor.EXPECTED_RELEASE_TAG,
            "latest_main_base_commit": doctor.EXPECTED_BASE_SHA,
            "expected_episode_cells": 20_160,
            "expected_horizon_steps": 600,
            "publication_channel": "direct_zenodo_benchmark_dataset",
            "concept_doi": "10.5281/zenodo.22077447",
            "version_doi": "10.5281/zenodo.22077448",
            "snqi_claim_policy": "advisory_no_ranking",
            "planner_count": 14,
            "seed_count": 30,
        },
    }
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=doctor._MANIFEST_SENTINEL + json.dumps(complete) + "\n",
        ),
    )
    check = doctor._source_manifest_check(
        manifest_path,
        repo=repo,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        base_sha=doctor.EXPECTED_BASE_SHA,
        tag=doctor.EXPECTED_RELEASE_TAG,
    )
    assert check.status == "pass"


def test_low_level_inputs_fail_closed_without_leaking_payload(tmp_path: Path) -> None:
    """Malformed primitives and missing structured owners return sanitized failures."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid structured artifact"):
        doctor._read_mapping(invalid)
    assert doctor._digest("no") is None
    assert doctor._digest("sha256:" + "A" * 64) == "a" * 64
    assert doctor._int(True) is None
    assert doctor._int(" 12 ") == 12
    assert doctor._int("1.2") is None
    assert doctor._positive_forbidden_markers({"nested": [{"fallback_count": 1}]})
    assert doctor._positive_forbidden_markers({"fallback_rows": ["malformed"]})
    assert doctor._positive_forbidden_markers({"fallback_count": "not-a-number"})
    assert not doctor._positive_forbidden_markers({"fallback_count": 0})
    with pytest.raises(ValueError, match="ledger is missing"):
        doctor._load_rows(None)
    repo = tmp_path / "repo"
    inside = _write_json(repo / "inside.json", {})
    outside = _write_json(tmp_path / "outside.json", {})
    assert doctor._contained_file(inside, repo)
    assert not doctor._contained_file(outside, repo)
    assert doctor._disk_check(repo, float("inf")).status == "fail"
    assert (
        doctor._source_manifest_check(
            None,
            repo=repo,
            source_sha=doctor.FROZEN_SOURCE_SHA,
            base_sha=doctor.EXPECTED_BASE_SHA,
            tag=doctor.EXPECTED_RELEASE_TAG,
        ).status
        == "fail"
    )
    assert doctor._publication_preflight_check(None, None).status == "fail"
    malformed_preflight = tmp_path / "malformed-preflight.json"
    malformed_preflight.write_text("[", encoding="utf-8")
    assert doctor._publication_preflight_check(malformed_preflight, repo).status == "fail"
    drifted_preflight = _write_json(
        tmp_path / "drifted-preflight.json",
        {
            "schema_version": "wrong",
            "status": "fail",
            "violation_count": 1,
            "violations": ["blocked"],
            "bundle_dir": "other",
        },
    )
    assert doctor._publication_preflight_check(drifted_preflight, repo).status == "fail"
    wrapped_rows = _write_yaml(tmp_path / "wrapped.yaml", {"jobs": [{"job_id": "1"}]})
    assert doctor._load_rows(wrapped_rows) == [{"job_id": "1"}]
    invalid_rows = _write_yaml(tmp_path / "invalid-rows.yaml", {"jobs": "bad"})
    with pytest.raises(ValueError, match="must be a list"):
        doctor._load_rows(invalid_rows)
    empty_bundle = tmp_path / "empty-bundle"
    empty_bundle.mkdir()
    assert "archive is unavailable" in " ".join(
        doctor._credential_scan_problems(empty_bundle, {"credentials": "not_recorded"})
    )
    invalid_archive = tmp_path / "invalid.tar.gz"
    invalid_archive.write_text("not an archive", encoding="utf-8")
    assert "archive could not be scanned" in " ".join(
        doctor._credential_scan_problems(
            empty_bundle, {"credentials": "not_recorded"}, invalid_archive
        )
    )
    evaluation_problems, accepted = doctor._derived_evaluation_problems(
        invalid,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        job_id=doctor.EXPECTED_JOB_ID,
        public_revalidation_receipt=None,
        publication_bundle=None,
        publication_archive=None,
    )
    assert not accepted and evaluation_problems
    assert (
        doctor._historical_provenance_check(
            None,
            None,
            None,
            None,
            source_sha=doctor.FROZEN_SOURCE_SHA,
            expected_tag=doctor.EXPECTED_RELEASE_TAG,
            expected_base_sha=doctor.EXPECTED_BASE_SHA,
            campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        ).status
        == "fail"
    )
    assert (
        doctor._bundle_check(
            None,
            receipt_path=None,
            receipt=None,
            source_sha=doctor.FROZEN_SOURCE_SHA,
            expected_tag=doctor.EXPECTED_RELEASE_TAG,
            expected_base_sha=doctor.EXPECTED_BASE_SHA,
            expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        )[0].status
        == "fail"
    )


def test_receipt_and_bundle_semantic_drifts_are_reported(monkeypatch, tmp_path: Path) -> None:
    """Independent receipt and bundle contradictions cannot cancel each other out."""
    paths = _synthetic_contract(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt.update(
        {
            "schema_version": "wrong",
            "mode": "rerun",
            "source": {},
            "validator": {
                "commit": "bad",
                "expected_reviewed_commit": "different",
                "file_sha256": "bad",
            },
            "snqi": {"status": "ranked", "ranking_authority": True},
            "publication_reconciliation": {
                "scientific_execution_changed": True,
                "simulation_rerun": True,
                "sidecar_path_binding": {"row_count": 1},
                "goal_timeout_boundary": {"timing_evidence_fabricated": True},
            },
            "publication_inputs": None,
            "fallback_count": 1,
        }
    )
    for key in ("acceptance", "projection_acceptance", "source_acceptance"):
        receipt[key] = {
            "status": "invalid",
            "benchmark_success": False,
            "blockers": ["blocked"],
            "expected_planner_arms": 1,
            "expected_scenario_count": 1,
            "expected_seed_count": 1,
            "expected_episode_cells": 1,
            "observed_episode_rows": 1,
            "unique_episode_identities": 1,
            "successful_planner_arms": 1,
            "missing_episode_identities": 1,
            "unexpected_episode_identities": 1,
            "forbidden_status_counts": {"fallback": 1},
            "source_commits": ["wrong"],
        }
    _write_json(paths["receipt"], receipt)
    receipt_check, _ = doctor._receipt_check(
        paths["receipt"],
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        expected_validator_sha=doctor.EXPECTED_VALIDATOR_SHA,
    )
    assert receipt_check.status == "fail"
    assert "scientific execution change" in receipt_check.summary

    valid_receipt = json.loads(
        (paths["bundle"] / "payload/provenance/derived_revalidation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    result_path = paths["bundle"] / "payload/release/release_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "benchmark_release": {
                "release_tag": "wrong",
                "latest_main_base_commit": "wrong",
            },
            "campaign_id": "wrong",
            "release_status": "failed",
            "release_exit_code": 2,
            "publication_preflight_status": "fail",
            "benchmark_success": False,
            "release_benchmark_success": False,
        }
    )
    result["full_release_acceptance"] = None
    result["release_acceptance"] = None
    _write_json(result_path, result)
    _write_json(
        paths["bundle"] / "payload/release/release_manifest.resolved.json",
        {"provenance": {"latest_main_base_commit": "wrong"}},
    )
    _write_json(
        paths["bundle"] / "publication_manifest.json",
        {
            "schema_version": "publication-bundle.v1",
            "publication_channels": {"release_tag": "wrong"},
        },
    )
    monkeypatch.setattr(
        doctor, "verify_publication_bundle_preflight", lambda _bundle: {"status": "pass"}
    )
    bundle_check, _ = doctor._bundle_check(
        paths["bundle"],
        receipt_path=paths["receipt"],
        receipt=valid_receipt,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        expected_campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
    )
    assert bundle_check.status == "fail"
    assert "release result tag" in bundle_check.summary


def test_private_evaluation_drift_reports_every_boundary(tmp_path: Path) -> None:
    """Malformed private acceptance, preservation, and source boundaries fail closed."""
    paths = _synthetic_contract(tmp_path)
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    evaluation.update(
        {
            "schema": "wrong",
            "evaluation_status": "partial",
            "evidence_valid": False,
            "scientific_outcome": "success",
            "source": {},
            "producer_campaign_id": "wrong",
            "job_id": "wrong",
            "fallback_count": 1,
        }
    )
    evaluation["execution"] = {}
    evaluation["acceptance"] = {
        "status": "invalid",
        "blockers": ["blocked"],
        "forbidden_status_counts": {"fallback": 1},
        "source_commits": ["wrong"],
    }
    evaluation["snqi"] = {}
    evaluation["derived_bundle"] = {
        key: {"path": "relative", "sha256": "bad", "bytes": 0}
        for key in ("archive", "revalidation_receipt", "release_result")
    }
    evaluation["preservation"] = {
        "status": "failed",
        "receipt_path": "relative",
        "two_copy_satisfied": False,
        "remote_state": "FAILED",
        "readback_match": False,
        "digest_mismatches": 1,
    }
    _write_json(paths["evaluation"], evaluation)
    problems, accepted = doctor._derived_evaluation_problems(
        paths["evaluation"],
        source_sha=doctor.FROZEN_SOURCE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
        job_id=doctor.EXPECTED_JOB_ID,
        public_revalidation_receipt=paths["receipt"],
        publication_bundle=paths["bundle"],
        publication_archive=paths["archive"],
    )
    assert not accepted
    assert len(problems) >= 20
    assert "derived evaluation receipt schema does not match" in problems


def test_historical_packet_drift_is_rejected(tmp_path: Path) -> None:
    """Packet, queue, smoke, stress, and portable evidence drift are all visible."""
    paths = _synthetic_contract(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    packet = yaml.safe_load(paths["packet"].read_text(encoding="utf-8"))
    packet.update(
        {
            "schema": "wrong",
            "status": "draft",
            "campaign_id": "wrong",
            "campaign": "wrong",
            "identity": {
                "public_source_commit": "wrong",
                "release_tag": "wrong",
                "latest_main_base_commit": "wrong",
            },
            "accepted_runtime_smoke": {
                "status": "failed",
                "public_source_commit": "wrong",
                "expected_episode_cells": 1,
                "fallback_or_degraded_rows": 1,
                "exit_code": "2:0",
                "derived_exit_code": "2:0",
            },
            "accepted_hybrid_stress": {
                "status": "failed",
                "public_source_commit": "wrong",
                "expected_episode_cells": 1,
            },
        }
    )
    _write_yaml(paths["packet"], packet)
    check = doctor._historical_provenance_check(
        paths["bundle"],
        paths["packet"],
        paths["queue"],
        receipt,
        source_sha=doctor.FROZEN_SOURCE_SHA,
        expected_tag=doctor.EXPECTED_RELEASE_TAG,
        expected_base_sha=doctor.EXPECTED_BASE_SHA,
        campaign_id=doctor.EXPECTED_CAMPAIGN_ID,
    )
    assert check.status == "fail"
    assert "not admitted_frozen" in check.summary
    assert "hybrid stress" in check.summary
