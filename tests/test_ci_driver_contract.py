"""Guard the shared CI phase contract across the shell driver and workflow."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_DRIVER = ROOT / "scripts" / "dev" / "ci_driver.sh"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PHASE_PATTERN = re.compile(
    r"(?:^|\s)(?:\./)?scripts/dev/ci_driver\.sh(?P<args>(?:\s+--?[a-z0-9_-]+|\s+[a-z0-9_-]+)*)",
    re.MULTILINE,
)
USES_PATTERN = re.compile(r"^\s*uses:\s+(?P<value>\S+)(?:\s+#\s*(?P<comment>\S+))?\s*$")
PINNED_ACTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
READABLE_ACTION_TAG_PATTERN = re.compile(r"^v[0-9][A-Za-z0-9_.-]*$")


def _driver_phases() -> set[str]:
    """Return phases advertised by the shell CI driver."""
    result = subprocess.run(
        [str(CI_DRIVER), "--list-phases"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _extract_workflow_phases(run_text: str) -> set[str]:
    """Return all ci_driver phase arguments referenced in a workflow run block."""

    normalized = re.sub(r"\\\s*\n", " ", run_text)
    referenced_phases: set[str] = set()
    for match in PHASE_PATTERN.finditer(normalized):
        args = match.group("args")
        if not args or not args.strip():
            continue
        for token in shlex.split(args, comments=True, posix=True):
            if token and not token.startswith("-"):
                referenced_phases.add(token)
    return referenced_phases


def _workflow_phases() -> set[str]:
    """Return CI driver phases referenced by the GitHub Actions workflow."""
    workflow = yaml.safe_load(_workflow_text())

    referenced_phases: set[str] = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            run_text = step.get("run")
            if not run_text:
                continue
            referenced_phases.update(_extract_workflow_phases(run_text))

    return referenced_phases


def _workflow_job_phases(job_name: str) -> set[str]:
    """Return CI driver phases referenced by a specific workflow job."""
    workflow = yaml.safe_load(_workflow_text())
    steps = workflow["jobs"][job_name].get("steps", [])

    referenced_phases: set[str] = set()
    for step in steps:
        run_text = step.get("run")
        if not run_text:
            continue
        referenced_phases.update(_extract_workflow_phases(run_text))
    return referenced_phases


def _workflow_text() -> str:
    """Return the raw CI workflow text."""
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _workflow_files() -> list[Path]:
    """Return tracked GitHub Actions workflow YAML files."""
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def test_extract_workflow_phases_handles_multiple_phase_args() -> None:
    """Capture all phase arguments from one workflow run stanza."""

    run_text = """
    scripts/dev/ci_driver.sh lint   test
    ./scripts/dev/ci_driver.sh   --verbose smoke artifact-policy
    """

    assert _extract_workflow_phases(run_text) == {
        "lint",
        "test",
        "smoke",
        "artifact-policy",
    }


def test_extract_workflow_phases_handles_line_continuations() -> None:
    """Capture phases when a workflow command uses shell line continuations."""

    run_text = """
    scripts/dev/ci_driver.sh \
      lint \
      typecheck
    """

    assert _extract_workflow_phases(run_text) == {"lint", "typecheck"}


def test_ci_workflow_only_references_known_driver_phases() -> None:
    """Fail if the workflow drifts to a non-existent shared CI phase.

    This matters because issue 533 makes the shell driver the canonical source
    for CI validation phase names, so the workflow must only reference phases
    the driver advertises.
    """

    driver_phases = _driver_phases()
    workflow_phases = _workflow_phases()

    assert workflow_phases, "workflow no longer references scripts/dev/ci_driver.sh"
    assert workflow_phases <= driver_phases, (
        f"unknown workflow phases: {sorted(workflow_phases - driver_phases)}"
    )


def test_ci_workflow_splits_fast_feedback_from_smoke_artifacts() -> None:
    """Keep fast PR feedback independent from the heavier smoke/artifact lane."""
    workflow = yaml.safe_load(_workflow_text())

    assert {"fast-feedback", "smoke-artifacts"} <= set(workflow["jobs"])
    assert _workflow_job_phases("fast-feedback") == {"lint", "typecheck", "test"}
    assert _workflow_job_phases("smoke-artifacts") == {"smoke", "artifact-policy"}
    assert "needs" not in workflow["jobs"]["fast-feedback"]


def test_ci_workflow_does_not_download_apt_fast_at_runtime() -> None:
    """Avoid bespoke apt-fast bootstrap in hosted CI."""
    workflow_text = _workflow_text()

    assert "apt-fast" not in workflow_text
    assert "raw.githubusercontent.com/ilikenwf/apt-fast" not in workflow_text


def test_workflow_action_refs_are_pinned_with_readable_version_comments() -> None:
    """Reject mutable action refs while keeping the intended upstream version visible."""
    failures: list[str] = []

    for workflow_file in _workflow_files():
        for line_number, line in enumerate(
            workflow_file.read_text(encoding="utf-8").splitlines(), 1
        ):
            if "uses:" not in line:
                continue
            match = USES_PATTERN.match(line)
            if not match:
                failures.append(
                    f"{workflow_file.relative_to(ROOT)}:{line_number}: malformed uses line"
                )
                continue

            value = match.group("value")
            action, separator, ref = value.partition("@")
            if not separator:
                failures.append(
                    f"{workflow_file.relative_to(ROOT)}:{line_number}: missing action ref"
                )
                continue
            if not PINNED_ACTION_SHA_PATTERN.fullmatch(ref):
                failures.append(
                    f"{workflow_file.relative_to(ROOT)}:{line_number}: {action}@{ref} is not pinned"
                )

            comment = match.group("comment")
            if comment is None or not READABLE_ACTION_TAG_PATTERN.fullmatch(comment):
                failures.append(
                    f"{workflow_file.relative_to(ROOT)}:{line_number}: missing readable version comment"
                )

    assert not failures, "\n".join(failures)
