"""Tests for the SLURM job finalization helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import slurm_job_finalize

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: bytes | str) -> None:
    """Write a tiny fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def test_successful_job_records_checksums_for_expected_artifacts(tmp_path: Path) -> None:
    """Completed jobs with required artifacts should finalize as success."""
    artifact = tmp_path / "output" / "slurm" / "job-123" / "summary.json"
    _write(artifact, '{"ok": true}\n')

    report = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="123",
        job_state="COMPLETED",
        expected_artifacts=["output/slurm/job-123/summary.json"],
        repo_root=tmp_path,
    )

    assert report["classification"] == "success"
    assert report["artifact_status"] == "all_required_present"
    assert report["artifacts"][0]["path"] == "output/slurm/job-123/summary.json"
    assert report["artifacts"][0]["sha256"]
    assert "not durable benchmark evidence" in report["claim_boundary"]
    assert (
        "| `output/slurm/job-123/summary.json` | true | present |"
        in report["issue_update_markdown"]
    )


def test_failed_job_classifies_failed_even_when_artifact_exists(tmp_path: Path) -> None:
    """Failed job states should not become successful because a file exists."""
    artifact = tmp_path / "output" / "slurm" / "job-124" / "partial.log"
    _write(artifact, "traceback\n")

    report = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="124",
        job_state="FAILED",
        expected_artifacts=[artifact],
        repo_root=tmp_path,
    )

    assert report["classification"] == "failed"
    assert report["artifact_status"] == "all_required_present"
    assert "rerun versus revise" in report["next_action"]


def test_missing_required_artifact_fails_closed(tmp_path: Path) -> None:
    """A completed job with missing required artifacts should not finalize as success."""
    report = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="125",
        job_state="COMPLETED",
        expected_artifacts=["output/slurm/job-125/summary.json"],
        repo_root=tmp_path,
    )

    assert report["classification"] == "missing_artifacts"
    assert report["artifact_status"] == "required_missing"
    assert report["artifacts"][0]["exists"] is False
    assert "Do not close as successful" in report["next_action"]


def test_incomplete_and_not_available_states_are_explicit(tmp_path: Path) -> None:
    """Running and unavailable jobs should have distinct classifications."""
    running = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="126",
        job_state="RUNNING",
        expected_artifacts=["output/slurm/job-126/summary.json"],
        repo_root=tmp_path,
    )
    unavailable = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="127",
        job_state="not-available",
        expected_artifacts=["output/slurm/job-127/summary.json"],
        repo_root=tmp_path,
    )

    assert running["classification"] == "incomplete"
    assert unavailable["classification"] == "not_available"


def test_manual_decision_when_no_required_artifacts_are_declared(tmp_path: Path) -> None:
    """No expected artifacts means the helper cannot claim completion."""
    report = slurm_job_finalize.build_finalization_report(
        issue_number=1894,
        job_id="128",
        job_state="COMPLETED",
        expected_artifacts=[],
        repo_root=tmp_path,
    )

    assert report["classification"] == "manual_decision_required"
    assert report["artifact_status"] == "no_required_artifacts_declared"


def test_directory_artifact_digest_is_deterministic(tmp_path: Path) -> None:
    """Directory records should summarize contents without copying raw files."""
    artifact_dir = tmp_path / "output" / "slurm" / "job-129" / "reports"
    _write(artifact_dir / "b.txt", "b")
    _write(artifact_dir / "a.txt", "a")

    first = slurm_job_finalize.artifact_record(
        "output/slurm/job-129/reports",
        repo_root=tmp_path,
    )
    second = slurm_job_finalize.artifact_record(
        "output/slurm/job-129/reports",
        repo_root=tmp_path,
    )

    assert first.kind == "directory"
    assert first.size_bytes == 2
    assert first.sha256 == second.sha256


def test_cli_writes_json_and_markdown(tmp_path: Path, capsys) -> None:
    """CLI should emit compact JSON plus optional Markdown."""
    artifact = tmp_path / "output" / "slurm" / "job-130" / "summary.json"
    _write(artifact, "{}\n")
    output = tmp_path / "docs" / "context" / "evidence" / "slurm_job_130.json"
    markdown = tmp_path / "docs" / "context" / "evidence" / "slurm_job_130.md"

    exit_code = slurm_job_finalize.main(
        [
            "--repo-root",
            str(tmp_path),
            "--issue",
            "1894",
            "--job-id",
            "130",
            "--job-state",
            "COMPLETED",
            "--expected-artifact",
            "output/slurm/job-130/summary.json",
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot-sf-slurm-job-finalization.v1"
    assert payload["classification"] == "success"
    assert markdown.read_text(encoding="utf-8").startswith("SLURM finalization")
    assert "classification: `success`" in capsys.readouterr().out


def test_cli_returns_nonzero_for_missing_artifacts(tmp_path: Path) -> None:
    """Non-success classifications should be visible to shell callers."""
    output = tmp_path / "missing.json"

    exit_code = slurm_job_finalize.main(
        [
            "--repo-root",
            str(tmp_path),
            "--issue",
            "1894",
            "--job-id",
            "131",
            "--job-state",
            "COMPLETED",
            "--expected-artifact",
            "output/slurm/job-131/summary.json",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == "missing_artifacts"


def test_cli_rejects_ambiguous_artifact_sources(tmp_path: Path) -> None:
    """Callers should choose manual artifacts or the control-plane run root, not both."""
    output = tmp_path / "ambiguous.json"

    with pytest.raises(SystemExit):
        slurm_job_finalize.main(
            [
                "--repo-root",
                str(tmp_path),
                "--issue",
                "3075",
                "--job-id",
                "132",
                "--job-state",
                "COMPLETED",
                "--expected-artifact",
                "output/slurm/job-132/summary.json",
                "--control-plane-run-root",
                "output/slurm/job-132",
                "--output",
                str(output),
            ]
        )


def test_control_plane_contract_requires_full_run_artifact_set(tmp_path: Path) -> None:
    """The sprint finalizer contract should fail closed on missing canonical files."""
    run_root = tmp_path / "output" / "slurm" / "job-132"
    for name in slurm_job_finalize.CONTROL_PLANE_RUN_ARTIFACTS:
        if name == "stderr.log":
            continue
        _write(run_root / name, f"{name}\n")

    report = slurm_job_finalize.build_control_plane_finalization_report(
        issue_number=3075,
        job_id="132",
        job_state="COMPLETED",
        run_root="output/slurm/job-132",
        repo_root=tmp_path,
    )

    assert report["classification"] == "missing_artifacts"
    missing = [artifact for artifact in report["artifacts"] if not artifact["exists"]]
    assert [artifact["path"] for artifact in missing] == ["output/slurm/job-132/stderr.log"]


def test_control_plane_contract_success_when_required_files_exist(tmp_path: Path) -> None:
    """A complete canonical run root should finalize as success."""
    run_root = tmp_path / "output" / "slurm" / "job-133"
    for name in slurm_job_finalize.CONTROL_PLANE_RUN_ARTIFACTS:
        _write(run_root / name, f"{name}\n")

    report = slurm_job_finalize.build_control_plane_finalization_report(
        issue_number=3075,
        job_id="133",
        job_state="COMPLETED",
        run_root=run_root,
        repo_root=tmp_path,
    )

    assert report["classification"] == "success"
    assert report["artifact_status"] == "all_required_present"
