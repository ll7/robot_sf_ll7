"""Contract tests for actionlint gate and wrapper script."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "check_github_actions_workflows.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "actionlint.yml"

EXPECTED_VERSION = "1.7.12"
EXPECTED_MANIFEST_SHA256 = "433028cf0ba3c42163ea1a668dedce30fcdbe84fe912b1a5e288c006eab8a4f5"
EXPECTED_BINARY_SHA256 = {
    "linux_amd64": "c872d6db8c6bf83a8eaa704fc93999f027d55dffbc63b8a6abdccb47df5f4cd4",
    "linux_arm64": "ac0323433c2853ec3fb978c611430c5b3dc5d43c58d1a1ec031b00ab572beb60",
    "darwin_amd64": "d1f7cee75ae2873609bd9567b4600bebc5315a5e733e73202987a44fafdd53b2",
    "darwin_arm64": "8db11704dc296f096216db4db65d86cd7f0ebfdf4c38453a1da276b137b88388",
}


def _write_invalid_workflow(path: Path) -> Path:
    path.write_text(
        "name: Bad\non: [invalid_event]\njobs:\n"
        "  broken:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo 1\n",
        encoding="utf-8",
    )
    return path


def _write_fake_actionlint(
    path: Path,
    *,
    version: str,
    mark_invocation: bool = False,
) -> Path:
    marker_command = ""
    if mark_invocation:
        marker_command = "printf '%s\\n' invoked > \"${BASH_SOURCE[0]}.invoked\"\n"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"{marker_command}"
        'if [[ "${1:-}" == "-version" ]]; then\n'
        f"  printf '%s\\n' {version!r}\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_binding_probe_actionlint(
    path: Path,
    *,
    reject_invalid_workflow: bool,
) -> Path:
    invalid_workflow_command = ""
    if reject_invalid_workflow:
        invalid_workflow_command = "if grep -q 'invalid_event' \"${1:-}\"; then\n  exit 23\nfi\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-version" ]]; then\n'
        "  printf '%s\\n' validated > \"${BASH_SOURCE[0]}.validated\"\n"
        f"  printf '%s\\n' {EXPECTED_VERSION!r}\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' invoked > \"${BASH_SOURCE[0]}.invoked\"\n"
        'if [[ ! -f "${BASH_SOURCE[0]}.validated" ]]; then\n'
        "  exit 24\n"
        "fi\n"
        f"{invalid_workflow_command}"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _assert_relative_candidate_is_bound(
    tmp_path: Path,
    relative_candidate: Path,
    *,
    candidate_source: str,
) -> None:
    caller_dir = tmp_path / "caller"
    repo_root = tmp_path / "repo"
    trusted_candidate = _write_binding_probe_actionlint(
        caller_dir / relative_candidate,
        reject_invalid_workflow=True,
    )
    shadow_candidate = _write_binding_probe_actionlint(
        repo_root / relative_candidate,
        reject_invalid_workflow=False,
    )
    trusted_source_marker = Path(f"{trusted_candidate}.invoked")
    shadow_marker = Path(f"{shadow_candidate}.invoked")
    wrapper = repo_root / "scripts" / "dev" / SCRIPT.name
    wrapper.parent.mkdir(parents=True)
    _script_trusting_test_binary(wrapper, trusted_candidate)
    bad_workflow = _write_invalid_workflow(tmp_path / "bad-relative-candidate.yml")

    env = os.environ.copy()
    env["ACTIONLINT_BIN"] = ""
    env["ROBOT_SF_ACTIONLINT_CACHE"] = str(tmp_path / "unused-cache")
    if candidate_source == "override":
        env["ACTIONLINT_BIN"] = str(relative_candidate)
    elif candidate_source == "cache":
        env["ROBOT_SF_ACTIONLINT_CACHE"] = str(relative_candidate.parents[1])
    elif candidate_source == "path":
        env["PATH"] = f"{relative_candidate.parent}:{env['PATH']}"
    else:
        raise AssertionError(f"unexpected candidate source: {candidate_source}")

    res = subprocess.run(
        ["bash", str(wrapper), str(bad_workflow)],
        cwd=caller_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert res.returncode == 23, "the authenticated candidate must reject the invalid workflow"
    assert not trusted_source_marker.exists(), "the source is copied instead of executed in place"
    assert not shadow_marker.exists(), "the same-relative repository shadow must not execute"


def _run_wrapper(
    workflow: Path,
    *,
    tmp_path: Path,
    actionlint_bin: Path | None = None,
    path_prefix: Path | None = None,
    script: Path = SCRIPT,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ROBOT_SF_ACTIONLINT_CACHE"] = str(tmp_path / "cache")
    env["ACTIONLINT_BIN"] = str(actionlint_bin) if actionlint_bin is not None else ""
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(script), str(workflow)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _script_trusting_test_binary(path: Path, binary: Path) -> Path:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "amd64" if machine in {"x86_64", "amd64"} else "arm64"
    platform_key = f"{system}_{arch}"
    old_digest = EXPECTED_BINARY_SHA256[platform_key]
    test_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    text = SCRIPT.read_text(encoding="utf-8")
    trusted_text = text.replace(
        f'EXPECTED_BINARY_SHA256_{platform_key.upper()}="{old_digest}"',
        f'EXPECTED_BINARY_SHA256_{platform_key.upper()}="{test_digest}"',
    )
    assert trusted_text != text
    path.write_text(trusted_text, encoding="utf-8")
    return path


def test_actionlint_script_constants_and_pins() -> None:
    """Verify exact version pin and checksums manifest SHA-256 in wrapper script."""
    assert SCRIPT.is_file(), f"Script missing: {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")

    assert f'ACTIONLINT_VERSION="{EXPECTED_VERSION}"' in text
    assert f'EXPECTED_CHECKSUMS_SHA256="{EXPECTED_MANIFEST_SHA256}"' in text
    for platform_key, digest in EXPECTED_BINARY_SHA256.items():
        assert f'EXPECTED_BINARY_SHA256_{platform_key.upper()}="{digest}"' in text
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
        ".github/actions/**",
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
    step_runs = [s.get("run", "").strip() for s in steps if s.get("run")]
    assert step_runs == ["bash scripts/dev/check_github_actions_workflows.sh"]
    assert all("|| true" not in command for command in step_runs)


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
    bad_workflow = _write_invalid_workflow(tmp_path / "bad_workflow.yml")
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
    assert "unknown Webhook event" in f"{res.stdout}\n{res.stderr}"


def test_actionlint_rejects_untrusted_explicit_override(tmp_path: Path) -> None:
    """An executable override must not bypass actionlint with unrelated bytes."""
    bad_workflow = _write_invalid_workflow(tmp_path / "bad_override.yml")
    res = _run_wrapper(
        bad_workflow,
        tmp_path=tmp_path,
        actionlint_bin=Path("/bin/true"),
    )

    assert res.returncode != 0
    diagnostics = f"{res.stdout}\n{res.stderr}"
    assert "binary digest mismatch" in diagnostics
    assert "unknown Webhook event" not in diagnostics


def test_actionlint_binds_relative_explicit_override_before_repository_chdir(
    tmp_path: Path,
) -> None:
    """The exact relative override authenticated from the caller directory is executed."""
    _assert_relative_candidate_is_bound(
        tmp_path,
        Path("relative-bin") / "actionlint",
        candidate_source="override",
    )


def test_actionlint_binds_relative_cache_candidate_before_repository_chdir(
    tmp_path: Path,
) -> None:
    """The exact cached candidate authenticated from the caller directory is executed."""
    _assert_relative_candidate_is_bound(
        tmp_path,
        Path("relative-cache") / EXPECTED_VERSION / "actionlint",
        candidate_source="cache",
    )


def test_actionlint_rejects_poisoned_cached_binary(tmp_path: Path) -> None:
    """A cached executable is revalidated from its bytes before every use."""
    cached_bin = tmp_path / "cache" / EXPECTED_VERSION / "actionlint"
    cached_bin.parent.mkdir(parents=True)
    cached_bin.symlink_to("/bin/true")
    bad_workflow = _write_invalid_workflow(tmp_path / "bad_cache.yml")

    res = _run_wrapper(bad_workflow, tmp_path=tmp_path)

    assert res.returncode != 0
    diagnostics = f"{res.stdout}\n{res.stderr}"
    assert "binary digest mismatch" in diagnostics
    assert "unknown Webhook event" not in diagnostics


def test_actionlint_rejects_path_binary_with_untrusted_bytes(tmp_path: Path) -> None:
    """A PATH candidate claiming the pinned version must also match trusted release bytes."""
    fake_bin_dir = tmp_path / "bin"
    fake_bin_dir.mkdir()
    fake_actionlint = _write_fake_actionlint(
        fake_bin_dir / "actionlint",
        version=EXPECTED_VERSION,
        mark_invocation=True,
    )
    invocation_marker = Path(f"{fake_actionlint}.invoked")
    bad_workflow = _write_invalid_workflow(tmp_path / "bad_path.yml")

    res = _run_wrapper(bad_workflow, tmp_path=tmp_path, path_prefix=fake_bin_dir)

    assert res.returncode != 0
    assert "binary digest mismatch" in f"{res.stdout}\n{res.stderr}"
    assert not invocation_marker.exists(), "untrusted candidate must not be executed"


def test_actionlint_binds_relative_path_candidate_before_repository_chdir(
    tmp_path: Path,
) -> None:
    """The exact PATH candidate authenticated from the caller directory is executed."""
    _assert_relative_candidate_is_bound(
        tmp_path,
        Path("relative-bin") / "actionlint",
        candidate_source="path",
    )


@pytest.mark.parametrize("reported_version", ["1.7.11", "1.7.120"])
def test_actionlint_rejects_wrong_or_fuzzy_version(tmp_path: Path, reported_version: str) -> None:
    """Candidate version output must equal the pinned release, not contain it."""
    fake_actionlint = _write_fake_actionlint(
        tmp_path / f"actionlint-{reported_version}",
        version=reported_version,
    )
    trusted_test_script = _script_trusting_test_binary(
        tmp_path / f"wrapper-{reported_version}.sh",
        fake_actionlint,
    )
    bad_workflow = _write_invalid_workflow(tmp_path / f"bad-{reported_version}.yml")

    res = _run_wrapper(
        bad_workflow,
        tmp_path=tmp_path,
        actionlint_bin=fake_actionlint,
        script=trusted_test_script,
    )

    assert res.returncode != 0
    assert "version mismatch" in f"{res.stdout}\n{res.stderr}"


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
