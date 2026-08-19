"""Tests for release legal-input ownership and trigger coverage."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import check_release_legal_input_coverage as coverage


def test_release_legal_input_coverage_passes_for_current_tree() -> None:
    """The checked-in ownership table is executable against the current workflows."""
    root = Path(__file__).resolve().parents[2]
    report = coverage.validate_workflow_coverage(
        repo_root=root,
        ownership_path=root / "scripts/validation/release_legal_inputs.v1.json",
    )
    assert report["status"] == "passed", report["issues"]
    assert report["read_only"] is True
    assert set(report["surfaces_checked"]) == {
        "package-archive",
        "dependency-metadata",
        "model-release",
        "coverage-contract",
    }


def test_pattern_covers_subtrees_and_rejects_unrelated_paths() -> None:
    """Coverage matching follows the repository path semantics used by GitHub filters."""
    assert coverage.pattern_covers(
        "third_party/socnavbench/**", "third_party/socnavbench/agents/agent.py"
    )
    assert coverage.pattern_covers("model/**", "model/registry.yaml")
    assert coverage.pattern_covers("**", "anywhere/legal.txt")
    assert not coverage.pattern_covers("third_party/python-rvo2/**", "model/registry.yaml")


def test_missing_trigger_and_unowned_input_fail_closed(tmp_path: Path) -> None:
    """Removing a path trigger or ownership entry cannot produce a passing verdict."""
    root = Path(__file__).resolve().parents[2]
    ownership_path = tmp_path / "ownership.json"
    ownership = json.loads(
        (root / "scripts/validation/release_legal_inputs.v1.json").read_text(encoding="utf-8")
    )
    ownership["surfaces"][2]["direct_globs"].remove("model/**")
    ownership["surfaces"][2]["workflow"] = ".github/workflows/packaging-extras.yml"
    ownership_path.write_text(json.dumps(ownership), encoding="utf-8")

    report = coverage.validate_workflow_coverage(repo_root=root, ownership_path=ownership_path)

    assert report["status"] == "blocked"
    assert any("validator is not invoked" in issue for issue in report["issues"])
    assert any("required legal input is unowned: model/**" in issue for issue in report["issues"])
