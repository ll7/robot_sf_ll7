"""Regression contract for plain-Python standalone scripts (issue #9248)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

EXEMPTED_SCRIPTS = [
    Path("scripts/dev/check_docs_evidence_integrity.py"),
    Path("scripts/dev/pr_gate_lease.py"),
]

WORKFLOW_INVOCATIONS = [
    (
        Path(".github/workflows/docs-link-integrity.yml"),
        "scripts/dev/check_docs_evidence_integrity.py",
    ),
    (
        Path(".github/workflows/docs-evidence-integrity.yml"),
        "scripts/dev/check_docs_evidence_integrity.py",
    ),
]

EXEMPTION_DOC = Path("docs/dev/standalone_script_contract.md")

FORBIDDEN_TOP_LEVELS = ("scripts", "robot_sf")


def _imported_top_levels(tree: ast.AST) -> set[str]:
    """Collect top-level package names from module-level import statements.

    Function-level conditional imports (such as the dual-mode
    ``if __package__`` guard in ``pr_gate_lease.py`` that falls back to a
    sibling-file import under direct execution) stay importable under plain
    hosted Python, so only module-level imports can break the standalone
    contract at import time.
    """
    found: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("script", EXEMPTED_SCRIPTS, ids=[str(p) for p in EXEMPTED_SCRIPTS])
def test_exempted_script_has_no_repo_package_imports(script: Path) -> None:
    """Exempted scripts must stay importable under plain hosted Python."""
    path = REPO_ROOT / script
    assert path.exists(), f"exempted script missing: {script}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = _imported_top_levels(tree)
    violations = imported & set(FORBIDDEN_TOP_LEVELS)
    assert not violations, (
        f"{script} must not import {sorted(violations)}; "
        "see docs/dev/standalone_script_contract.md for the exemption rule"
    )


@pytest.mark.parametrize(
    ("workflow", "script"),
    WORKFLOW_INVOCATIONS,
    ids=[f"{w}:{s}" for w, s in WORKFLOW_INVOCATIONS],
)
def test_workflow_uses_plain_python_invocation(workflow: Path, script: str) -> None:
    """Workflows must invoke exempted scripts with plain python, not uv run."""
    path = REPO_ROOT / workflow
    assert path.exists(), f"workflow missing: {workflow}"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    runs: list[str] = []
    for job in payload.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if script in run:
                runs.append(run)
    assert runs, f"{workflow} no longer invokes {script}"
    for run in runs:
        assert "uv run" not in run, (
            f"{workflow} must keep plain-python invocation for {script}; "
            "a uv run migration must retire the exemption in the same PR"
        )
        assert "python scripts/dev/" in run or "python3 scripts/dev/" in run, (
            f"{workflow} must use the sanctioned plain-python form for {script}"
        )


def test_exemption_doc_names_all_scripts() -> None:
    """Reuse authors must find every exemption in one documented list."""
    path = REPO_ROOT / EXEMPTION_DOC
    assert path.exists(), f"exemption doc missing: {EXEMPTION_DOC}"
    text = path.read_text(encoding="utf-8")
    for script in EXEMPTED_SCRIPTS:
        assert str(script) in text, f"{EXEMPTION_DOC} must name {script}"
