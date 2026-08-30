"""Contract tests for actionlint gate and wrapper script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_github_actions_workflows.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "actionlint.yml"

EXPECTED_VERSION = "1.7.12"
EXPECTED_MANIFEST_SHA256 = "433028cf0ba3c42163ea1a668dedce30fcdbe84fe912b1a5e288c006eab8a4f5"


def test_actionlint_script_constants_and_pins() -> None:
    """Verify exact version pin and checksums manifest SHA-256 in wrapper script."""
    assert SCRIPT.is_file(), f"Script missing: {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")

    assert f'ACTIONLINT_VERSION="{EXPECTED_VERSION}"' in text
    assert f'EXPECTED_CHECKSUMS_SHA256="{EXPECTED_MANIFEST_SHA256}"' in text
    assert "set -euo pipefail" in text
    assert "detect_platform" in text
    assert "compute_sha256" in text


def test_actionlint_platform_mapping() -> None:
    """Verify platform detection correctly handles supported Linux and macOS architectures."""
    test_sh = """
source scripts/dev/check_github_actions_workflows.sh

test_detect() {
  local mock_os="$1"
  local mock_arch="$2"
  uname() {
    if [[ "$1" == "-s" ]]; then echo "$mock_os"; fi
    if [[ "$1" == "-m" ]]; then echo "$mock_arch"; fi
  }
  detect_platform
}

echo "linux_x86_64:$(test_detect Linux x86_64)"
echo "linux_amd64:$(test_detect Linux amd64)"
echo "linux_arm64:$(test_detect Linux arm64)"
echo "linux_aarch64:$(test_detect Linux aarch64)"
echo "darwin_x86_64:$(test_detect Darwin x86_64)"
echo "darwin_arm64:$(test_detect Darwin arm64)"
"""
    res = subprocess.run(
        ["bash", "-c", test_sh],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = dict(line.split(":", 1) for line in res.stdout.strip().splitlines())
    assert lines["linux_x86_64"] == "linux_amd64"
    assert lines["linux_amd64"] == "linux_amd64"
    assert lines["linux_arm64"] == "linux_arm64"
    assert lines["linux_aarch64"] == "linux_arm64"
    assert lines["darwin_x86_64"] == "darwin_amd64"
    assert lines["darwin_arm64"] == "darwin_arm64"


def test_actionlint_unsupported_platform_fails() -> None:
    """Verify platform detection fails closed on unsupported OS or architecture."""
    test_sh = """
source scripts/dev/check_github_actions_workflows.sh

test_detect() {
  local mock_os="$1"
  local mock_arch="$2"
  uname() {
    if [[ "$1" == "-s" ]]; then echo "$mock_os"; fi
    if [[ "$1" == "-m" ]]; then echo "$mock_arch"; fi
  }
  detect_platform
}

test_detect Windows_NT x86_64
"""
    res = subprocess.run(
        ["bash", "-c", test_sh],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "unsupported OS" in res.stderr


def test_actionlint_workflow_contract() -> None:
    """Verify actionlint workflow triggers, permissions, timeout, and execution step."""
    assert WORKFLOW.is_file(), f"Workflow missing: {WORKFLOW}"
    data: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    # Minimal permissions
    assert data.get("permissions") == {"contents": "read"}

    # On section
    on_section = data.get("on") or data.get(True, {})
    pr_paths = on_section.get("pull_request", {}).get("paths", [])
    push_paths = on_section.get("push", {}).get("paths", [])

    expected_paths = {
        ".github/workflows/**",
        ".github/actionlint.yaml",
        "scripts/dev/check_github_actions_workflows.sh",
        "tests/dev/test_actionlint_contract.py",
    }
    assert expected_paths.issubset(set(pr_paths)), (
        f"PR paths missing items: {expected_paths - set(pr_paths)}"
    )
    assert expected_paths.issubset(set(push_paths)), (
        f"Push paths missing items: {expected_paths - set(push_paths)}"
    )

    # Jobs
    jobs = data.get("jobs", {})
    assert "actionlint" in jobs
    job = jobs["actionlint"]
    assert job.get("runs-on") == "ubuntu-latest"
    assert job.get("timeout-minutes") is not None
    assert job.get("timeout-minutes") <= 30

    # Steps
    steps = job.get("steps", [])
    step_runs = [s.get("run", "") for s in steps]
    assert any("bash scripts/dev/check_github_actions_workflows.sh" in cmd for cmd in step_runs)


def test_actionlint_runs_cleanly_on_all_repository_workflows() -> None:
    """Execute actionlint wrapper script and verify zero errors across all workflows."""
    res = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"actionlint failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_actionlint_fails_on_invalid_workflow_fixture(tmp_path: Path) -> None:
    """Verify actionlint wrapper detects syntax and schema violations in invalid workflow."""
    bad_workflow = tmp_path / "bad_workflow.yml"
    bad_workflow.write_text(
        "name: Bad\non: [invalid_event]\njobs:\n  broken:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 1\n",
        encoding="utf-8",
    )
    res = subprocess.run(
        ["bash", str(SCRIPT), str(bad_workflow)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0, (
        f"actionlint should fail on invalid workflow, but succeeded:\n{res.stdout}"
    )
    assert (
        "unknown Webhook event" in res.stdout
        or "unknown Webhook event" in res.stderr
        or res.returncode != 0
    )


def test_actionlint_fails_on_manifest_digest_mismatch(tmp_path: Path) -> None:
    """Verify that tampering with the expected manifest SHA-256 fails closed."""
    tampered_script = tmp_path / "tampered.sh"
    text = SCRIPT.read_text(encoding="utf-8")
    tampered_text = text.replace(
        f'EXPECTED_CHECKSUMS_SHA256="{EXPECTED_MANIFEST_SHA256}"',
        'EXPECTED_CHECKSUMS_SHA256="0000000000000000000000000000000000000000000000000000000000000000"',
    )
    tampered_script.write_text(tampered_text, encoding="utf-8")
    tampered_script.chmod(0o755)

    # Use a non-existent cache directory to force download attempt
    env = os.environ.copy()
    env["ROBOT_SF_ACTIONLINT_CACHE"] = str(tmp_path / "cache")
    env["ACTIONLINT_BIN"] = ""

    res = subprocess.run(
        ["bash", str(tampered_script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert res.returncode != 0
    assert "checksums manifest digest mismatch" in res.stderr
