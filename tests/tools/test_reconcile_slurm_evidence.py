"""Tests for the SLURM/evidence reconciliation CLI helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools import reconcile_slurm_evidence

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, payload: dict) -> None:
    """Write YAML fixture data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write CSV fixture data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(row.get(key, "") for key in header) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    """Write one compact JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _queue_payload() -> dict:
    """Return a minimal queue payload used by most tests."""
    return {
        "schema_version": "slurm-submission-queue.v1",
        "entries": [
            {
                "id": "issue-2656-example",
                "status": "planned",
                "issue": 2656,
                "objective": "compact integration check",
                "target": {"cluster": "auxme"},
                "resources": {},
                "seeds": [101, 102],
                "output_root": "output/slurm/issue2656-example",
                "log_path": "output/slurm/%j-issue2656-example.out",
                "priority_reason": "test",
                "auto_submit": True,
            }
        ],
    }


def _manifest_payload(
    *, queue_id: str, status: str, seeds: list[int], slurm_job_id: int | str, exp_id: str
) -> dict:
    """Build a minimal manifest payload."""
    return {
        "schema_version": "slurm-submission-manifest.v1",
        "jobs": [
            {
                "queue_id": queue_id,
                "status": status,
                "slurm_job_id": str(slurm_job_id),
                "experiment_id": exp_id,
                "seeds": seeds,
            }
        ],
    }


def test_duplicate_manifest_experiment_ids_are_reported(tmp_path: Path) -> None:
    """Duplicate manifest experiment IDs should be surfaced in report errors."""
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, _queue_payload())

    manifest_a = tmp_path / "manifests" / "one.yaml"
    manifest_b = tmp_path / "manifests" / "two.yaml"
    _write_yaml(
        manifest_a,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="submitted",
            seeds=[101],
            slurm_job_id=111,
            exp_id="dup-exp-001",
        ),
    )
    _write_yaml(
        manifest_b,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="submitted",
            seeds=[102],
            slurm_job_id=112,
            exp_id="dup-exp-001",
        ),
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_a, manifest_b],
        evidence_root=tmp_path / "evidence",
    )

    assert any(
        "duplicate experiment_id across manifests: dup-exp-001" in item for item in report["errors"]
    )


def test_missing_wandb_link_for_completed_seed_is_reported(tmp_path: Path) -> None:
    """Completed seeds should fail preservation checks when no durable link exists."""
    queue_payload = _queue_payload()
    queue_payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, queue_payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="completed",
            seeds=[101],
            slurm_job_id=120,
            exp_id="exp-001",
        ),
    )

    evidence = tmp_path / "evidence" / "seed_summary.csv"
    _write_csv(
        evidence,
        ["queue_id", "seed", "job_id", "run_summary_sha256", "claim_boundary"],
        [
            {
                "queue_id": "issue-2656-example",
                "seed": "101",
                "job_id": "120",
                "run_summary_sha256": "abc",
                "claim_boundary": "diagnostic_only",
            }
        ],
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    row = report["observations"][0]
    assert row["queue_id"] == "issue-2656-example"
    assert row["seed"] == 101
    assert row["status"] == "completed"
    assert any("missing wandb link or durable pointer" in note for note in row["notes"])


def test_completed_seed_without_compact_evidence_is_flagged(tmp_path: Path) -> None:
    """Completed seeds without matching compact evidence should be flagged."""
    queue_payload = _queue_payload()
    queue_payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, queue_payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="COMPLETED",
            seeds=[101],
            slurm_job_id=121,
            exp_id="exp-002",
        ),
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    row = report["observations"][0]
    assert row["status"] == "completed"
    assert any("completed but not preserved" in note for note in row["notes"])


def test_conflicting_evidence_job_id_does_not_preserve_completed_seed(tmp_path: Path) -> None:
    """Evidence from another job ID should not preserve the current completed job."""
    queue_payload = _queue_payload()
    queue_payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, queue_payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="completed",
            seeds=[101],
            slurm_job_id=121,
            exp_id="exp-002",
        ),
    )

    evidence = tmp_path / "evidence" / "seed_summary.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "queue_id": "issue-2656-example",
                        "seed": 101,
                        "job_id": "999",
                        "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/stale",
                        "claim_boundary": "compact",
                        "run_summary_sha256": "0123456789abcdef0123456789abcdef",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    row = report["observations"][0]
    assert row["status"] == "completed"
    assert any("completed but not preserved" in note for note in row["notes"])


def test_duplicate_observations_do_not_confuse_missing_seed_with_seed_zero(
    tmp_path: Path,
) -> None:
    """A job with unspecified seeds should not collide with an actual seed 0 row."""
    queue_payload = _queue_payload()
    queue_payload["entries"][0]["seeds"] = [0]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, queue_payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    manifest = _manifest_payload(
        queue_id="issue-2656-example",
        status="submitted",
        seeds=[0],
        slurm_job_id=121,
        exp_id="exp-002",
    )
    manifest["jobs"].append(
        {
            "queue_id": "issue-2656-example",
            "status": "submitted",
            "slurm_job_id": "122",
            "experiment_id": "exp-003",
        }
    )
    _write_yaml(manifest_path, manifest)

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    assert "issue-2656-example::0" not in report["duplicate_ids"]["queue_seed_observations"]
    assert not any("issue-2656-example::0" in warning for warning in report["warnings"])


def test_explicitly_excluded_seed_is_excluded(tmp_path: Path) -> None:
    """Excluded queue seeds should resolve to excluded status regardless of submission rows."""
    payload = _queue_payload()
    payload["entries"][0]["seeds"] = [101, 102]
    payload["entries"][0]["excluded_seeds"] = [102]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="submitted",
            seeds=[101, 102],
            slurm_job_id=122,
            exp_id="exp-003",
        ),
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    rows_by_seed = {row["seed"]: row for row in report["observations"]}
    assert rows_by_seed[102]["status"] == "excluded"


def test_evidence_row_can_explicitly_exclude_seed(tmp_path: Path) -> None:
    """Compact evidence can explicitly mark a seed as excluded."""
    payload = _queue_payload()
    payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    evidence = tmp_path / "evidence" / "seed_summary.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "queue_id": "issue-2656-example",
                        "seed": 101,
                        "status": "excluded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[],
        evidence_root=tmp_path / "evidence",
    )

    assert report["observations"][0]["status"] == "excluded"


def test_scheduler_completion_overrides_submitted_manifest_status(tmp_path: Path) -> None:
    """SLURM scheduler state should advance a submitted manifest to completed."""
    payload = _queue_payload()
    payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    manifest = _manifest_payload(
        queue_id="issue-2656-example",
        status="submitted",
        seeds=[101],
        slurm_job_id=124,
        exp_id="exp-005",
    )
    manifest["jobs"][0]["scheduler_state"] = "COMPLETED"
    _write_yaml(manifest_path, manifest)

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    assert report["observations"][0]["status"] == "completed"


def test_happy_path_json_shape_is_stable(tmp_path: Path) -> None:
    """JSON report should provide stable keys and preserve compact status mappings."""
    payload = _queue_payload()
    payload["entries"][0]["seeds"] = [101]
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="completed",
            seeds=[101],
            slurm_job_id=123,
            exp_id="exp-004",
        ),
    )

    evidence = tmp_path / "evidence" / "seed_summary.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "queue_id": "issue-2656-example",
                        "seed": 101,
                        "job_id": "123",
                        "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/demo",
                        "claim_boundary": "compact",
                        "run_summary_sha256": "0123456789abcdef0123456789abcdef",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
    )

    assert {"schema_version", "generated_at", "observations", "errors", "warnings"}.issubset(
        report.keys()
    )
    row = report["observations"][0]
    assert row["status"] == "evidence_preserved"
    assert row["seed"] == 101
    assert row["queue_id"] == "issue-2656-example"


def test_finalizer_manifest_bridge_captures_output_pointer_and_transition(tmp_path: Path) -> None:
    """Successful finalizers should surface output pointers and issue transition fields."""
    payload = _queue_payload()
    payload["entries"][0]["id"] = "issue-2656-example"
    payload["entries"][0]["seeds"] = [201]
    payload["entries"][0]["issue"] = 2656
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="completed",
            seeds=[201],
            slurm_job_id=901,
            exp_id="exp-finalizer",
        ),
    )

    finalizer = tmp_path / "evidence" / "finalization_901.json"
    _write_json(
        finalizer,
        {
            "schema_version": "robot-sf-slurm-job-finalization.v1",
            "issue_number": 2656,
            "job_id": "901",
            "job_state": "COMPLETED",
            "classification": "success",
            "artifact_status": "all_required_present",
            "artifacts": [
                {
                    "path": "output/slurm/job-901/oracle_candidate_trace_manifest.json",
                    "artifact_uri": "wandb-artifact://robot-sf/finalizer/job-901:latest",
                    "exists": True,
                    "required": True,
                    "kind": "file",
                    "sha256": "deadbeef",
                    "size_bytes": 1,
                }
            ],
            "claim_boundary": "compact local",
            "claim_decision": "keep diagnostic",
        },
    )

    evidence = tmp_path / "evidence" / "seed_summary.csv"
    _write_csv(
        evidence,
        ["queue_id", "seed", "job_id", "wandb_url", "claim_boundary", "run_summary_sha256"],
        [
            {
                "queue_id": "issue-2656-example",
                "seed": "201",
                "job_id": "901",
                "wandb_url": "https://wandb.ai/example/run/901",
                "claim_boundary": "compact local",
                "run_summary_sha256": "0123456789abcdef0123456789abcdef",
            }
        ],
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
        finalizer_manifests=[finalizer],
        generated_at="2026-06-19T00:00:00+00:00",
    )

    assert report["generated_at"] == "2026-06-19T00:00:00+00:00"
    assert report["finalizer_bridge"]["schema_version"] == "slurm-job-finalizer-bridge.v1"
    bridge_row = report["finalizer_bridge"]["rows"][0]
    assert bridge_row["job_id"] == "901"
    assert bridge_row["issue_transition"]["to"] == "success"
    assert bridge_row["durable_pointer"] == "wandb-artifact://robot-sf/finalizer/job-901:latest"
    assert bridge_row["output_pointers"]
    assert bridge_row["artifact_status"] == "all_required_present"
    assert bridge_row["claim_decision"] == "keep_diagnostic"
    assert bridge_row["source_path"] == str(finalizer)


def test_finalizer_bridge_accepts_public_source_manifest_linkage(tmp_path: Path) -> None:
    """Source manifests can link finalized diagnostic jobs without queue routing state."""
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, _queue_payload())
    evidence_root = tmp_path / "evidence"
    finalizer = evidence_root / "finalization_13268.json"
    _write_json(
        finalizer,
        {
            "schema_version": "robot-sf-slurm-job-finalization.v1",
            "issue_number": 4243,
            "job_id": "13268",
            "classification": "success",
            "artifact_status": "all_required_present",
            "durable_uri": "https://github.com/ll7/robot_sf_ll7/tree/main/docs/context/evidence/h600",
            "claim_boundary": "workflow trace only; no benchmark claim",
            "artifacts": [
                {
                    "path": "docs/context/evidence/h600/source_manifest.json",
                    "exists": True,
                    "required": True,
                    "kind": "file",
                    "sha256": "abc123",
                    "size_bytes": 10,
                }
            ],
        },
    )
    source_manifest = evidence_root / "source_manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": "issue_4195_h600_aggregation.v1.source_manifest",
            "runs": [
                {
                    "job_id": "13268",
                    "run_label": "confirm",
                    "campaign": {"campaign_id": "issue3810_h600_confirm"},
                }
            ],
        },
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[],
        source_manifests=[source_manifest],
        evidence_root=evidence_root,
        finalizer_manifests=[finalizer],
        generated_at="2026-07-03T09:20:00+00:00",
    )

    assert report["errors"] == []
    assert report["source_manifests"] == [str(source_manifest)]
    bridge_row = report["finalizer_bridge"]["rows"][0]
    assert bridge_row["job_id"] == "13268"
    assert bridge_row["claim_decision"] == "keep_diagnostic"
    assert bridge_row["source_manifest"] == [
        {
            "campaign_id": "issue3810_h600_confirm",
            "run_label": "confirm",
            "source_path": str(source_manifest),
        }
    ]
    assert bridge_row["issue_transition"] == {
        "from": "source_manifest",
        "to": "success",
    }


def test_finalizer_missing_durable_pointer_and_queue_linkage_flags_error(tmp_path: Path) -> None:
    """Missing durable pointers and queue mapping should be flagged by the checker."""
    payload = _queue_payload()
    payload["entries"][0]["id"] = "issue-2656-example"
    payload["entries"][0]["seeds"] = [202]
    payload["entries"][0]["issue"] = 2656
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    manifest_path = tmp_path / "manifests" / "one.yaml"
    _write_yaml(
        manifest_path,
        _manifest_payload(
            queue_id="issue-2656-example",
            status="completed",
            seeds=[202],
            slurm_job_id=902,
            exp_id="exp-finalizer",
        ),
    )

    finalizer = tmp_path / "evidence" / "finalization_902.json"
    _write_json(
        finalizer,
        {
            "schema_version": "robot-sf-slurm-job-finalization.v1",
            "issue_number": 2656,
            "job_id": "902",
            "classification": "success",
            "artifact_status": "all_required_present",
            "artifacts": [
                {
                    "path": "output/slurm/job-902/oracle_candidate_trace_manifest.json",
                    "exists": True,
                    "required": True,
                    "kind": "file",
                    "sha256": "cafebabe",
                    "size_bytes": 1,
                }
            ],
            "claim_boundary": "compact local",
        },
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[manifest_path],
        evidence_root=tmp_path / "evidence",
        finalizer_manifests=[finalizer],
    )

    assert any(
        "missing durable_pointer for successful output" in error for error in report["errors"]
    )
    assert any("completed artifacts are not preserved" in error for error in report["errors"])


def test_missing_finalizer_manifest_mapping_is_reported(tmp_path: Path) -> None:
    """Finalizer rows that cannot be traced to a manifest must be flagged."""
    payload = _queue_payload()
    payload["entries"][0]["issue"] = 2656
    queue_path = tmp_path / "experiments" / "submission_queue.yaml"
    _write_yaml(queue_path, payload)

    finalizer = tmp_path / "evidence" / "finalization_903.json"
    _write_json(
        finalizer,
        {
            "schema_version": "robot-sf-slurm-job-finalization.v1",
            "issue_number": 2656,
            "job_id": "903",
            "classification": "success",
            "artifact_status": "required_missing",
            "artifacts": [
                {
                    "path": "output/slurm/job-903/oracle_candidate_trace_manifest.json",
                    "exists": False,
                    "required": True,
                    "kind": "file",
                }
            ],
        },
    )

    report = reconcile_slurm_evidence.reconcile(
        queue_path=queue_path,
        submission_manifests=[],
        evidence_root=tmp_path / "evidence",
        finalizer_manifests=[finalizer],
    )

    assert any(
        "no manifest row for queue/seed linkage" in error for error in report["errors"]
    ) or any("issue 2656 not in queue" in error for error in report["warnings"])
