"""Contract and behavior tests for bounded CodeQL SARIF upload recovery."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "codeql_sarif_upload_retry.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"
CODEQL_ACTION = "github/codeql-action/analyze@5595ccaf912efad79be6eef63a5619ff05969be3"
UPLOAD_ACTION = "github/codeql-action/upload-sarif@5595ccaf912efad79be6eef63a5619ff05969be3"


def _run_helper(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the shell helper with an isolated summary file."""
    bash = shutil.which("bash")
    assert bash, "bash is required by the CodeQL SARIF retry helper"
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
            "CODEQL_ANALYSIS_OUTCOME": "success",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME": "failure",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_2_OUTCOME": "success",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [bash, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )


def test_codeql_sarif_upload_retry_shell_syntax_and_executable() -> None:
    """The helper must be executable and pass Bash syntax validation."""
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111
    result = subprocess.run(["bash", "-n", str(SCRIPT)], check=False, timeout=30)
    assert result.returncode == 0


def test_wait_uses_bounded_backoff_and_records_retry(tmp_path: Path) -> None:
    """The retry wait is bounded and observable without sleeping in the test."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sleep_log = tmp_path / "sleep.log"
    sleep = fake_bin / "sleep"
    sleep.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$1\" > '{sleep_log}'\n")
    sleep.chmod(0o755)

    result = _run_helper(
        tmp_path,
        "wait",
        "2",
        extra_env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CODEQL_SARIF_UPLOAD_BACKOFF_BASE": "8",
            "CODEQL_SARIF_UPLOAD_BACKOFF_CAP": "5",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "backoff_seconds=5" in result.stdout
    assert sleep_log.read_text().strip() == "5"
    assert "retry_scheduled: true" in (tmp_path / "summary.md").read_text()


def test_finalize_accepts_first_upload_and_records_effective_status(tmp_path: Path) -> None:
    """A successful first upload is the effective result with no retry."""
    result = _run_helper(
        tmp_path,
        "finalize",
        extra_env={
            "CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME": "success",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_2_OUTCOME": "skipped",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "effective_status=uploaded" in result.stdout
    summary = (tmp_path / "summary.md").read_text()
    assert "effective_status: uploaded" in summary
    assert "retry_count: 0" in summary


def test_finalize_accepts_recovered_upload_after_retry(tmp_path: Path) -> None:
    """A second-attempt success is explicit recovery, not an unqualified first pass."""
    result = _run_helper(tmp_path, "finalize")

    assert result.returncode == 0, result.stderr
    assert "effective_status=uploaded_after_retry" in result.stdout
    summary = (tmp_path / "summary.md").read_text()
    assert "effective_attempt: 2" in summary
    assert "retry_count: 1" in summary


def test_finalize_fails_closed_when_upload_budget_is_exhausted(tmp_path: Path) -> None:
    """Persistent upload failure must remain a failed CodeQL job."""
    result = _run_helper(
        tmp_path,
        "finalize",
        extra_env={
            "CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME": "failure",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_2_OUTCOME": "failure",
        },
    )

    assert result.returncode == 1
    assert "effective_status=upload_failed" in result.stdout
    assert "SARIF upload failed on all bounded attempts" in result.stderr


def test_finalize_fails_closed_when_analysis_fails(tmp_path: Path) -> None:
    """A retry cannot turn failed or skipped analysis into CodeQL success."""
    result = _run_helper(
        tmp_path,
        "finalize",
        extra_env={
            "CODEQL_ANALYSIS_OUTCOME": "failure",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME": "skipped",
            "CODEQL_SARIF_UPLOAD_ATTEMPT_2_OUTCOME": "skipped",
        },
    )

    assert result.returncode == 1
    assert "effective_status=analysis_failed" in result.stdout
    assert "no upload is admitted" in result.stderr


def test_codeql_workflow_separates_analysis_and_bounded_upload_retry() -> None:
    """The workflow retries only the upload action and keeps final status fail-closed."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["analyze"]["steps"]

    analysis = next(step for step in steps if step.get("id") == "codeql-analysis")
    assert analysis["uses"] == CODEQL_ACTION
    assert analysis["with"]["upload"] == "never"

    uploads = [step for step in steps if step.get("uses") == UPLOAD_ACTION]
    assert len(uploads) == 2
    for upload in uploads:
        assert upload["continue-on-error"] is True
        assert upload["with"]["category"] == "/language:python"
        assert upload["with"]["sarif_file"] == "${{ steps.codeql-analysis.outputs.sarif-output }}"

    wait = next(step for step in steps if step.get("name") == "Back off before CodeQL SARIF retry")
    assert wait["run"] == "scripts/dev/codeql_sarif_upload_retry.sh wait 2"
    assert "upload-codeql-sarif-attempt-1.outcome == 'failure'" in wait["if"]

    finalizer = next(
        step for step in steps if step.get("name") == "Finalize CodeQL SARIF upload status"
    )
    assert finalizer["if"] == "always()"
    assert finalizer["run"] == "scripts/dev/codeql_sarif_upload_retry.sh finalize"
    assert finalizer["env"]["CODEQL_ANALYSIS_OUTCOME"] == "${{ steps.codeql-analysis.outcome }}"
    assert finalizer["env"]["CODEQL_SARIF_UPLOAD_ATTEMPT_1_OUTCOME"] == (
        "${{ steps.upload-codeql-sarif-attempt-1.outcome }}"
    )
