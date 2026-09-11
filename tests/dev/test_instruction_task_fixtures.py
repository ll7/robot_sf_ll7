"""Behavior fixtures for the instruction contract (issue #8942).

Each fixture names a representative task, its expected execution profile, the owners it must
discover, and the ceremony it must not inherit. The canonical graph assertion fails when a
mandatory instruction surface is added without an explicit decision and fixture coverage.
"""

from __future__ import annotations

from scripts.dev.check_instruction_references import (
    CANONICAL_INSTRUCTION_GRAPH,
    REPO_ROOT,
    load_task_scope_manifest,
)

EXPECTED_CANONICAL_GRAPH = (
    "AGENTS.md",
    ".agents/README.md",
    ".agents/PLANS.md",
    "docs/maintainer_values.md",
    "docs/ai/agent_workflow_entrypoints.md",
    "docs/dev/agents/relocated-agents-guidance.md",
    "SLURM/AGENTS.md",
    ".claude/CLAUDE.md",
    ".github/copilot-instructions.md",
    ".specify/memory/constitution.md",
)

LOW_RISK_PROFILES = frozenset({"observe", "local"})
LOW_RISK_FLAGS = (
    "plan_required",
    "environment_required",
    "worktree_required",
    "pr_required",
    "evidence_gates_required",
)

# (fixture, profile, required owners, forbidden startup dependencies)
TASK_FIXTURES = (
    (
        "read-only repository orientation",
        "observe",
        ("docs/code_review.md",),
        (".agents/PLANS.md",),
    ),
    ("issue triage", "observe", ("docs/ai/agent_workflow_entrypoints.md",), (".agents/PLANS.md",)),
    ("one-file markdown correction", "local", ("docs/glossary.md",), (".agents/PLANS.md",)),
    ("localized Python bug fix", "local", ("AGENTS.md",), (".agents/PLANS.md",)),
    ("provider-adapter update", "local", ("AGENTS.md",), (".agents/PLANS.md",)),
    ("public API change", "coordinated", (".agents/PLANS.md", "docs/code_review.md"), ()),
    (
        "multi-module migration",
        "coordinated",
        (".agents/PLANS.md", "docs/dev/worktree_lifecycle.md"),
        (),
    ),
    ("destructive cleanup/recovery", "coordinated", ("docs/dev/worktree_lifecycle.md",), ()),
    (
        "benchmark metric change",
        "evidence_critical",
        ("docs/benchmark_governance.md", ".agents/PLANS.md"),
        (),
    ),
    ("research claim update", "evidence_critical", ("docs/benchmark_governance.md",), ()),
    ("SLURM campaign submission", "evidence_critical", (".agents/PLANS.md",), ()),
    ("release preparation", "evidence_critical", (".agents/PLANS.md", "docs/code_review.md"), ()),
)


def _manifest() -> dict:
    """Load the live task-scope manifest as a mapping."""
    manifest = load_task_scope_manifest(REPO_ROOT)
    assert isinstance(manifest, dict)
    return manifest


def test_every_fixture_has_a_declared_profile() -> None:
    """Each representative task maps to a profile that exists in the manifest."""
    profiles = _manifest()["profiles"]
    for fixture, profile, _required, _forbidden in TASK_FIXTURES:
        assert profile in profiles, (fixture, profile)


def test_low_risk_fixtures_avoid_unrelated_ceremony() -> None:
    """Observe and Local fixtures inherit no plan, environment, worktree, PR, or release ceremony."""
    profiles = _manifest()["profiles"]
    for fixture, profile, _required, forbidden in TASK_FIXTURES:
        if profile not in LOW_RISK_PROFILES:
            continue
        entry = profiles[profile]
        assert all(entry[flag] is False for flag in LOW_RISK_FLAGS), fixture
        assert ".agents/PLANS.md" not in entry["required_context"], fixture
        for dependency in forbidden:
            assert dependency not in entry["required_context"], fixture


def test_high_risk_fixtures_reach_required_gates() -> None:
    """Coordinated and Evidence-critical fixtures load their plan, review, and evidence owners."""
    profiles = _manifest()["profiles"]
    for fixture, profile, required, _forbidden in TASK_FIXTURES:
        if profile in LOW_RISK_PROFILES:
            continue
        entry = profiles[profile]
        assert entry["plan_required"] is True, fixture
        for owner in required:
            assert owner in entry["required_context"], (fixture, owner)
        if profile == "evidence_critical":
            assert entry["evidence_gates_required"] is True, fixture
            assert entry["worktree_required"] is True, fixture
            assert "docs/benchmark_governance.md" in entry["required_context"], fixture


def test_mandatory_surfaces_are_explicit() -> None:
    """Adding a mandatory surface requires updating this fixture gate and its coverage."""
    assert CANONICAL_INSTRUCTION_GRAPH == EXPECTED_CANONICAL_GRAPH


def test_fixture_set_covers_the_ten_task_classes() -> None:
    """The fixture matrix covers orientation, docs, code, API, benchmark, research, SLURM,
    release, adapter, and cleanup task classes."""
    covered = {fixture for fixture, _profile, _required, _forbidden in TASK_FIXTURES}
    required_classes = (
        "read-only repository orientation",
        "one-file markdown correction",
        "localized Python bug fix",
        "public API change",
        "benchmark metric change",
        "research claim update",
        "SLURM campaign submission",
        "release preparation",
        "provider-adapter update",
        "destructive cleanup/recovery",
    )
    for fixture in required_classes:
        assert fixture in covered, fixture
