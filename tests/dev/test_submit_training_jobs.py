"""Tests for the safe SLURM training submission queue helper."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from scripts.dev import submit_training_jobs

if TYPE_CHECKING:
    from pathlib import Path


def _write_minimal_repo(tmp_path: Path, *, allow_slurm: bool = False) -> Path:
    repo = tmp_path
    (repo / "configs" / "training").mkdir(parents=True)
    (repo / "configs" / "training" / "example.yaml").write_text("training: {}\n")
    (repo / "SLURM" / "Auxme").mkdir(parents=True)
    (repo / "SLURM" / "Auxme" / "example.sl").write_text("#!/usr/bin/env bash\n")
    (repo / "scripts" / "dev").mkdir(parents=True)
    wrapper = repo / "scripts" / "dev" / "sbatch_use_max_time.sh"
    wrapper.write_text("#!/usr/bin/env bash\necho Submitted batch job 4242\n")
    wrapper.chmod(0o755)
    (repo / "local.machine.md").write_text(
        f"allow_slurm_submission: {'true' if allow_slurm else 'false'}\n",
        encoding="utf-8",
    )
    return repo


def _queue_payload(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "issue-999-example",
        "status": "ready_to_submit",
        "issue": 999,
        "objective": "Run the example training smoke on a compute node.",
        "config": "configs/training/example.yaml",
        "launcher": "SLURM/Auxme/example.sl",
        "target": {"cluster": "auxme", "host": "licca"},
        "resources": {"gpus": 1, "cpus": 4, "memory": "16G", "walltime": "02:00:00"},
        "seeds": [0, 1],
        "output_root": "output/slurm/issue999-example",
        "log_path": "output/slurm/%j-issue999-example.out",
        "priority_reason": "Covers the current training follow-up.",
        "auto_submit": True,
        "job_name": "gse-999-example",
        "wrapper": "scripts/dev/sbatch_use_max_time.sh",
    }
    entry.update(overrides)
    return {"schema_version": "slurm-submission-queue.v1", "entries": [entry]}


def _write_queue(repo: Path, payload: dict[str, Any]) -> Path:
    queue_path = repo / "experiments" / "submission_queue.yaml"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return queue_path


def test_load_queue_rejects_missing_required_field(tmp_path: Path) -> None:
    """Queue entries must not become runnable when required provenance is absent."""
    repo = _write_minimal_repo(tmp_path)
    payload = _queue_payload()
    del payload["entries"][0]["priority_reason"]
    queue_path = _write_queue(repo, payload)

    with pytest.raises(submit_training_jobs.QueueValidationError, match="priority_reason"):
        submit_training_jobs.load_queue(queue_path, repo_root=repo)


def test_load_queue_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Each queue id should identify exactly one planned submission."""
    repo = _write_minimal_repo(tmp_path)
    first = _queue_payload()["entries"][0]
    payload = {"schema_version": "slurm-submission-queue.v1", "entries": [first, dict(first)]}
    queue_path = _write_queue(repo, payload)

    with pytest.raises(submit_training_jobs.QueueValidationError, match="duplicate id"):
        submit_training_jobs.load_queue(queue_path, repo_root=repo)


def test_load_queue_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """Queue files should declare the supported schema version."""
    repo = _write_minimal_repo(tmp_path)
    payload = _queue_payload()
    payload["schema_version"] = "old-schema"
    queue_path = _write_queue(repo, payload)

    with pytest.raises(submit_training_jobs.QueueValidationError, match="schema_version"):
        submit_training_jobs.load_queue(queue_path, repo_root=repo)


def test_load_queue_rejects_missing_config_path(tmp_path: Path) -> None:
    """Runnable entries should not reference absent config paths."""
    repo = _write_minimal_repo(tmp_path)
    queue_path = _write_queue(repo, _queue_payload(config="configs/training/missing.yaml"))

    with pytest.raises(submit_training_jobs.QueueValidationError, match="config not found"):
        submit_training_jobs.load_queue(queue_path, repo_root=repo)


def test_dry_run_writes_manifest_without_calling_sbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry runs should produce reviewable manifests but never submit jobs."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=False)
    queue_path = _write_queue(repo, _queue_payload())
    calls: list[list[str]] = []
    monkeypatch.setattr(submit_training_jobs.shutil, "which", lambda name: None)

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["git", "branch"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="feature\n", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command in dry run: {cmd}")

    monkeypatch.setattr(submit_training_jobs.subprocess, "run", fake_run)

    result = submit_training_jobs.plan_submissions(
        queue_path=queue_path,
        repo_root=repo,
        submit=False,
        now="2026-06-11T10-30-00",
    )

    assert result.manifest_path.exists()
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry-run"
    assert manifest["jobs"][0]["queue_id"] == "issue-999-example"
    assert manifest["jobs"][0]["status"] == "dry_run"
    assert "--dry-run" in manifest["jobs"][0]["command"]
    assert not any("sbatch_use_max_time.sh" in " ".join(call) for call in calls)


def test_submit_requires_slurm_enabled_machine(tmp_path: Path) -> None:
    """Auto-submit must honor local.machine.md and refuse non-SLURM hosts."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=False)
    queue_path = _write_queue(repo, _queue_payload())

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="allow_slurm_submission"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_submit_requires_explicit_local_machine_allow_when_file_is_absent(tmp_path: Path) -> None:
    """Submit mode should fail closed when local machine policy is absent."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    (repo / "local.machine.md").unlink()
    queue_path = _write_queue(repo, _queue_payload(status="planned"))

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="allow_slurm_submission"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_submit_rejects_bulleted_local_machine_slurm_policy(tmp_path: Path) -> None:
    """The real local.machine.md style uses Markdown bullets and must still block submit."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    (repo / "local.machine.md").write_text("- allow_slurm_submission: false\n", encoding="utf-8")
    queue_path = _write_queue(repo, _queue_payload(status="planned"))

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="allow_slurm_submission"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_submit_invokes_wrapper_and_captures_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible submit mode should call the existing wrapper and record the scheduler id."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    queue_path = _write_queue(repo, _queue_payload())
    monkeypatch.setattr(submit_training_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "branch"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="feature\n", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0].endswith("sbatch_use_max_time.sh"):
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Submitted batch job 4242\n", stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(submit_training_jobs.subprocess, "run", fake_run)

    result = submit_training_jobs.plan_submissions(
        queue_path=queue_path,
        repo_root=repo,
        submit=True,
        now="2026-06-11T10-30-00",
    )

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert result.jobs[0]["status"] == "submitted"
    assert manifest["jobs"][0]["slurm_job_id"] == "4242"
    assert "--dry-run" not in manifest["jobs"][0]["command"]


def test_submit_blocks_when_auto_submit_is_false(tmp_path: Path) -> None:
    """Submit mode should require explicit queue-level auto-submit consent."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    queue_path = _write_queue(repo, _queue_payload(auto_submit=False))

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="auto_submit is false"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_submit_blocks_when_wrapper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed wrapper should block submission and expose the exit status."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    queue_path = _write_queue(repo, _queue_payload())
    monkeypatch.setattr(submit_training_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["git", "branch"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="feature\n", stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0].endswith("sbatch_use_max_time.sh"):
            return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="bad wrapper")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(submit_training_jobs.subprocess, "run", fake_run)

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="wrapper failed"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_output_root_duplicate_blocks_submission(tmp_path: Path) -> None:
    """An existing output root should prevent equivalent auto-submit attempts."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    (repo / "output" / "slurm" / "issue999-example").mkdir(parents=True)
    queue_path = _write_queue(repo, _queue_payload())

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="output_root exists"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_recent_manifest_duplicate_blocks_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior submitted manifest for the same queue id should block auto-submit."""
    repo = _write_minimal_repo(tmp_path, allow_slurm=True)
    queue_path = _write_queue(repo, _queue_payload())
    manifest_dir = repo / "output" / "slurm" / "submissions"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "2026-06-10T00-00-00Z_training_submission_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "slurm-submission-manifest.v1",
                "jobs": [
                    {
                        "queue_id": "issue-999-example",
                        "status": "submitted",
                        "experiment_id": "issue-999-example_seed0-1_abc1234_auxme",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(submit_training_jobs.shutil, "which", lambda name: f"/usr/bin/{name}")

    with pytest.raises(submit_training_jobs.SubmissionBlockedError, match="prior manifest"):
        submit_training_jobs.plan_submissions(
            queue_path=queue_path,
            repo_root=repo,
            submit=True,
            now="2026-06-11T10-30-00",
        )


def test_experiment_id_is_deterministic(tmp_path: Path) -> None:
    """Experiment ids should be stable for duplicate detection and reports."""
    repo = _write_minimal_repo(tmp_path)
    entry = submit_training_jobs.QueueEntry.from_mapping(_queue_payload()["entries"][0], repo)

    assert (
        submit_training_jobs.deterministic_experiment_id(entry, commit="abcdef123456")
        == "issue-999-example_seed0-1_abcdef1_auxme"
    )
