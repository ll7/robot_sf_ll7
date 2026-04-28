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
PHASE_PATTERN = re.compile(
    r"(?:^|\s)(?:\./)?scripts/dev/ci_driver\.sh(?P<args>(?:\s+--?[a-z0-9_-]+|\s+[a-z0-9_-]+)*)",
    re.MULTILINE,
)


def _driver_phases() -> set[str]:
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
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ci"]["steps"]

    referenced_phases: set[str] = set()
    for step in steps:
        run_text = step.get("run")
        if not run_text:
            continue
        referenced_phases.update(_extract_workflow_phases(run_text))

    return referenced_phases


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
