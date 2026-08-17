"""Tests for the fail-closed CodeQL SARIF upload retry helper."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/ci/upload_codeql_sarif_retry.sh"


def _fake_gh(bin_dir: Path) -> None:
    """Install a deterministic fake ``gh`` command for upload-path tests."""
    (bin_dir / "gh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
state_file=${FAKE_GH_STATE:?}
count=0
if [[ -f "$state_file" ]]; then
  count=$(<"$state_file")
fi
count=$((count + 1))
printf '%s' "$count" > "$state_file"
if ((count <= FAKE_GH_FAILURES)); then
  printf 'gh: transient upload failure (HTTP %s)\\n' "$FAKE_GH_STATUS" >&2
  exit 1
fi
cat > "${FAKE_GH_CAPTURE:?}"
printf '{"id":"fake-sarif-id"}\\n'
""",
        encoding="utf-8",
    )
    (bin_dir / "gh").chmod(0o755)


def _run_helper(tmp_path: Path, *, failures: int, status: int) -> subprocess.CompletedProcess[str]:
    """Run the helper with a fake GitHub CLI and no backoff delay."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_gh(bin_dir)
    sarif_dir = tmp_path / "sarif"
    sarif_dir.mkdir()
    sarif = '{"version":"2.1.0","runs":[]}'
    (sarif_dir / "python.sarif").write_text(sarif, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "GH_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "ll7/robot_sf_ll7",
            "GITHUB_SHA": "abc123",
            "GITHUB_REF": "refs/pull/7373/merge",
            "CODEQL_UPLOAD_MAX_ATTEMPTS": "3",
            "CODEQL_UPLOAD_BACKOFF_BASE_SECONDS": "0",
            "CODEQL_UPLOAD_BACKOFF_CAP_SECONDS": "0",
            "FAKE_GH_FAILURES": str(failures),
            "FAKE_GH_STATUS": str(status),
            "FAKE_GH_STATE": str(tmp_path / "gh-state"),
            "FAKE_GH_CAPTURE": str(tmp_path / "gh-capture.json"),
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), str(sarif_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_codeql_upload_retries_transient_failure_and_records_success(tmp_path: Path) -> None:
    """A transient 503 is retried and the final upload is proven successful."""
    result = _run_helper(tmp_path, failures=2, status=503)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "attempt=3/3" in result.stdout
    assert "final_status=success" in result.stdout
    payload = json.loads((tmp_path / "gh-capture.json").read_text(encoding="utf-8"))
    assert payload["commit_sha"] == "abc123"
    assert payload["ref"] == "refs/pull/7373/merge"
    assert base64.b64decode(payload["sarif"]).decode() == '{"version":"2.1.0","runs":[]}'


def test_codeql_upload_fails_closed_on_non_transient_error(tmp_path: Path) -> None:
    """A permission/validation error is not retried or reported as success."""
    result = _run_helper(tmp_path, failures=3, status=403)

    assert result.returncode != 0
    assert "final_status=failure" in result.stderr
    assert "retryable=false" in result.stderr
    assert (tmp_path / "gh-state").read_text(encoding="utf-8") == "1"


def test_codeql_upload_reports_exhausted_transient_failure(tmp_path: Path) -> None:
    """An exhausted transient budget remains an explicit failed result."""
    result = _run_helper(tmp_path, failures=5, status=503)

    assert result.returncode != 0
    assert "status=503" in result.stderr
    assert "final_status=failure" in result.stderr
    assert "retryable=true" in result.stderr
    assert "attempts=3/3" in result.stderr
    assert (tmp_path / "gh-state").read_text(encoding="utf-8") == "3"


def test_codeql_workflow_uses_local_fail_closed_upload_helper() -> None:
    """The workflow preserves analysis and delegates upload status to the helper."""
    workflow = (REPO_ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")

    assert "upload: never" in workflow
    assert "output: codeql-results" in workflow
    assert "scripts/ci/upload_codeql_sarif_retry.sh codeql-results" in workflow
    assert "security-events: write" in workflow
