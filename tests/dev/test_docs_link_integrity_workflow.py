"""Contract tests for the full-repository Markdown-link workflow."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/docs-evidence-integrity.yml"


def test_docs_link_workflow_runs_when_its_implementation_changes() -> None:
    """Docs, checker, and workflow edits must all trigger the full scan."""
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    paths = set(workflow["on"]["pull_request"]["paths"])

    assert {
        "docs/**",
        "scripts/dev/check_docs_evidence_integrity.py",
        ".github/workflows/docs-evidence-integrity.yml",
        "tests/dev/test_docs_link_integrity_workflow.py",
    } <= paths


def test_docs_link_workflow_invokes_full_scan() -> None:
    """The workflow must exercise the checker's deterministic full-link mode."""
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    steps = workflow["jobs"]["docs-evidence-integrity"]["steps"]
    run_commands = [step["run"] for step in steps if "run" in step]

    assert any(
        "python scripts/dev/check_docs_evidence_integrity.py --full" in command
        for command in run_commands
    )

    # A failed full scan must propagate to the preserved check identity.
    passthrough = workflow["jobs"]["docs-link-integrity"]
    assert passthrough["needs"] == "docs-evidence-integrity"
    assert "always()" in passthrough["if"]
    assert any(
        'test "$INTEGRITY_RESULT" = success' in s.get("run", "") for s in passthrough["steps"]
    )
    assert all("uses" not in step for step in passthrough["steps"])
