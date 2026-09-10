"""Tests for the machine-readable execution-profile contract (issue #8934).

The manifest is the single machine-readable owner of the profile and route mapping. These fixtures
assert the risk contract (low-risk profiles do not inherit release-grade ceremony; evidence-critical
work keeps every gate) and that the schema fails closed on malformed mappings.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import yaml

from scripts.dev.check_instruction_references import (
    PROFILE_BOOL_FIELDS,
    REPO_ROOT,
    check_task_scope_manifest,
    load_task_scope_manifest,
    run_checks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict:
    """Return a mutable copy of the live manifest."""
    data = load_task_scope_manifest(REPO_ROOT)
    assert isinstance(data, dict)
    return copy.deepcopy(data)


def test_manifest_is_valid() -> None:
    """The live manifest passes schema validation and contributes no check errors."""
    assert check_task_scope_manifest(REPO_ROOT) == []
    assert run_checks(REPO_ROOT)["task_scope_errors"] == []


def test_low_risk_profiles_forbid_release_ceremony() -> None:
    """Observe and Local require no plan, environment, worktree, PR, or evidence gates."""
    manifest = _manifest()
    for profile_id in ("observe", "local"):
        profile = manifest["profiles"][profile_id]
        assert all(profile[field] is False for field in PROFILE_BOOL_FIELDS), profile_id
        assert ".agents/PLANS.md" not in profile["required_context"], profile_id
        assert "execution plan" in profile["forbidden_ceremony"], profile_id
        assert "worktree creation" in profile["forbidden_ceremony"], profile_id


def test_coordinated_requires_plan_and_isolation_without_release_gates() -> None:
    """Coordinated work plans and isolates, but does not open release-grade evidence gates."""
    profile = _manifest()["profiles"]["coordinated"]
    assert profile["plan_required"] is True
    assert profile["worktree_required"] is True
    assert profile["pr_required"] is True
    assert profile["evidence_gates_required"] is False


def test_evidence_critical_keeps_every_gate() -> None:
    """Benchmark, research, release, and publication work keeps all evidence gates."""
    profile = _manifest()["profiles"]["evidence_critical"]
    assert all(profile[field] is True for field in PROFILE_BOOL_FIELDS)
    assert "docs/benchmark_governance.md" in profile["required_context"]


def test_route_defaults_cover_the_five_canonical_routes() -> None:
    """Every canonical route maps to one declared profile."""
    manifest = _manifest()
    routes = manifest["routes"]
    assert len(routes) == 5
    for route, entry in routes.items():
        assert entry["default_profile"] in manifest["profiles"], route


def test_agents_router_names_profiles_and_manifest() -> None:
    """The root router names every profile and points at the machine-readable mapping."""
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert ".agents/task_scope_manifest.yaml" in text
    for label in ("Observe", "Local", "Coordinated", "Evidence-critical"):
        assert label in text
    assert "escalate" in text.lower()


def test_missing_profile_field_fails() -> None:
    """A missing boolean field fails the schema check."""
    manifest = _manifest()
    del manifest["profiles"]["local"]["plan_required"]

    errors = check_task_scope_manifest(REPO_ROOT, manifest=manifest)

    assert any("plan_required" in error for error in errors)


def test_missing_forbidden_ceremony_field_fails() -> None:
    """Every profile declares which ceremony it forbids, including an intentionally empty list."""
    manifest = _manifest()
    del manifest["profiles"]["local"]["forbidden_ceremony"]

    errors = check_task_scope_manifest(REPO_ROOT, manifest=manifest)

    assert any("forbidden_ceremony" in error for error in errors)


def test_unknown_manifest_field_fails() -> None:
    """A schema-versioned manifest rejects a silently ignored top-level field."""
    manifest = _manifest()
    manifest["unexpected"] = True

    errors = check_task_scope_manifest(REPO_ROOT, manifest=manifest)

    assert any("unknown field" in error for error in errors)


def test_required_context_must_stay_inside_repository(tmp_path: Path) -> None:
    """Required context cannot use absolute paths or escape the repository root."""
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    manifest = _manifest()
    for profile in manifest["profiles"].values():
        profile["required_context"] = [str(outside)]

    errors = check_task_scope_manifest(root, manifest=manifest)

    assert any("relative repository paths" in error for error in errors)

    for profile in manifest["profiles"].values():
        profile["required_context"] = ["../outside.md"]
    errors = check_task_scope_manifest(root, manifest=manifest)

    assert any("within the repository root" in error for error in errors)


def test_unknown_profile_fails() -> None:
    """An undeclared profile fails the schema check."""
    manifest = _manifest()
    manifest["profiles"]["fast"] = dict(manifest["profiles"]["local"])

    errors = check_task_scope_manifest(REPO_ROOT, manifest=manifest)

    assert any("profiles must be exactly" in error for error in errors)


def test_unresolved_required_context_fails(tmp_path: Path) -> None:
    """Every required-context path is a required repository-local reference."""
    manifest = _manifest()

    errors = check_task_scope_manifest(tmp_path, manifest=manifest)

    assert any("required_context path does not exist" in error for error in errors)


def test_route_default_profile_must_exist() -> None:
    """A route cannot point at an undeclared profile."""
    manifest = _manifest()
    manifest["routes"]["documentation-only-edit"]["default_profile"] = "unknown"

    errors = check_task_scope_manifest(REPO_ROOT, manifest=manifest)

    assert any("default_profile" in error for error in errors)


def test_manifest_is_yaml_mapping() -> None:
    """The manifest stays a plain YAML mapping, not a generated or templated document."""
    text = (REPO_ROOT / ".agents" / "task_scope_manifest.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    assert data["version"] == 1
