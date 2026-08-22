"""Regression contracts for the selected-issue skill consolidation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SKILLS_ROOT = ROOT / ".agents" / "skills"
REGISTRY_PATH = SKILLS_ROOT / "skills.yaml"
TARGET_SKILL = SKILLS_ROOT / "goal-issue-implementation" / "SKILL.md"
LEGACY_SKILL = SKILLS_ROOT / "gh-issue-autopilot"
ROUTING_CASES = SKILLS_ROOT / "tests" / "routing_cases.yaml"


def _load_check_skills_module():
    """Load the production checker without requiring scripts to be a package."""
    module_path = ROOT / "scripts/dev/check_skills.py"
    spec = importlib.util.spec_from_file_location("check_skills_consolidation", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry() -> dict[str, object]:
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_legacy_names_preflight_to_canonical_selected_issue_mode(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All retired issue-to-PR names resolve through the production preflight resolver."""
    check_skills = _load_check_skills_module()
    monkeypatch.setattr(check_skills, "_check_command", lambda *_args: (True, "available"))

    for alias in ("gh-issue-autopilot", "issue-to-pr", "gh-issue-to-pr"):
        assert check_skills._preflight(alias) == 0
        assert "Preflight check for skill: goal-issue-implementation" in capsys.readouterr().out


def test_selected_issue_mode_removes_duplicate_and_reciprocal_delegation() -> None:
    """The canonical registry owns the mode and has no compatibility back-edge."""
    skills = _registry()["skills"]
    assert isinstance(skills, dict)
    assert "gh-issue-autopilot" not in skills

    target = skills["goal-issue-implementation"]
    assert isinstance(target, dict)
    assert {"gh-issue-autopilot", "issue-to-pr", "gh-issue-to-pr"} <= set(target["aliases"])
    assert "gh-issue-autopilot" not in target["delegates_to"]
    assert not LEGACY_SKILL.exists()


def test_selected_issue_identity_and_preserved_guardrails_are_fail_closed() -> None:
    """Routing cannot silently select work without an issue identity or delivery guardrails."""
    text = TARGET_SKILL.read_text(encoding="utf-8")
    assert "## Selected-Issue Mode" in text
    assert "exactly one issue number" in text
    assert "fails closed before claim, branch, worktree, or GitHub mutation" in text

    required_guardrails = (
        "gh_issue_rest.py thread",
        "goal_issue_admission.py",
        "Exact merged-fix stale-evidence guard",
        "scripts/dev/pr_ready_check.sh",
        "issue_claim.py release",
        "origin/main",
    )
    for marker in required_guardrails:
        assert marker in text, f"selected-issue guardrail missing: {marker}"

    cases = yaml.safe_load(ROUTING_CASES.read_text(encoding="utf-8"))["cases"]
    missing_identity = next(
        case
        for case in cases
        if case["intent"] == "Execute issue-to-PR without a selected issue identity"
    )
    assert missing_identity["primary"] == "issue-contract-maintainer"
    assert "goal-issue-implementation" in missing_identity["negative"]
