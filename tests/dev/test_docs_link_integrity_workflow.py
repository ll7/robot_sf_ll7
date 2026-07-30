"""Contract tests for the full-repository Markdown-link workflow."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/docs-link-integrity.yml"


def test_docs_link_workflow_runs_when_its_implementation_changes() -> None:
    """Docs, checker, and workflow edits must all trigger the full scan."""
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    paths = set(workflow["on"]["pull_request"]["paths"])

    assert {
        "docs/**",
        "scripts/dev/check_docs_evidence_integrity.py",
        ".github/workflows/docs-link-integrity.yml",
    } <= paths


def test_docs_link_workflow_invokes_full_scan() -> None:
    """The workflow must exercise the checker's deterministic full-link mode."""
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    steps = workflow["jobs"]["docs-link-integrity"]["steps"]
    run_commands = [step["run"] for step in steps if "run" in step]

    assert any(
        "python scripts/dev/check_docs_evidence_integrity.py --full" in command
        for command in run_commands
    )
