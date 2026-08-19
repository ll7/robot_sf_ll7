"""Contract tests for the stack-ancestry gate (issue #7515).

Covers the pure declaration parser, the deterministic ancestry classifier, the
#7308/#7309 (contaminated) vs #7389/#7390 (clean replacement) regression
fixtures, and the remediation diagnostic output.  All fixtures use synthetic
inputs to the pure functions so the tests are deterministic and fast.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev.stack_ancestry import (
    BLOCKING_STATES,
    NOT_INDEPENDENTLY_MERGEABLE_STATES,
    StackDeclaration,
    ancestry_state,
    collect_ancestry_facts,
    parse_stack_declaration,
    remediation_command,
    render_diagnostics,
)

MAIN_TIP = "m" * 40
PARENT_HEAD = "a" * 40
CHILD_HEAD = "c" * 40
OTHER_HEAD = "f" * 40

# The contaminated #7308/#7309 pattern: the branch carries commits from another
# issue's branch (merged/rebased in) plus its own intended commits, and the live
# merge base is an older commit than current main.
CONTAMINATED_COMMITS = [
    f"{'9' * 40} refactor: share PySF state slice constants (#7283)",
    f"{'8' * 40} fix: unrelated benchmark pin",
    f"{'7' * 40} refactor: share Bresenham rasterizer (#7282)",
    f"{'6' * 40} feat: intended issue work",
]

# The clean #7389/#7390 replacement pattern: only the intended commits, created
# from current main.
CLEAN_COMMITS = [f"{'5' * 40} refactor: share PySF state slice constants (#7389)"]


def _declaration(parent_pr: int = 7308, parent_head: str = PARENT_HEAD) -> StackDeclaration:
    return StackDeclaration(parent_pr=parent_pr, parent_head=parent_head)


def _classify(**overrides: Any) -> dict[str, Any]:
    """Run the classifier with canonical stacked-branch inputs."""
    defaults: dict[str, Any] = {
        "head_sha": CHILD_HEAD,
        "base_ref": "fix/child",
        "main_tip_sha": MAIN_TIP,
        "merge_base_sha": PARENT_HEAD,
        "commits": CONTAMINATED_COMMITS,
    }
    defaults.update(overrides)
    return ancestry_state(**defaults)


# ---------------------------------------------------------------------------
# Declaration parser
# ---------------------------------------------------------------------------


def test_parse_stack_declaration_binds_parent_pr_and_head() -> None:
    text = f"## Stack Declaration\nparent_pr: #7308\nparent_head: {PARENT_HEAD}\n"
    declaration, error = parse_stack_declaration(text)

    assert error is None
    assert declaration == StackDeclaration(parent_pr=7308, parent_head=PARENT_HEAD)


def test_parse_stack_declaration_accepts_bare_pr_number_and_uppercase_sha() -> None:
    text = f"## Stack Declaration\nparent_pr: 7308\nparent_head: {PARENT_HEAD.upper()}\n"
    declaration, error = parse_stack_declaration(text)

    assert error is None
    assert declaration is not None
    assert declaration.parent_pr == 7308
    assert declaration.parent_head == PARENT_HEAD


def test_parse_stack_declaration_absent_section_is_not_an_error() -> None:
    declaration, error = parse_stack_declaration("No stack declaration here.")

    assert declaration is None
    assert error is None


def test_parse_stack_declaration_rejects_partial_declaration() -> None:
    declaration, error = parse_stack_declaration("## Stack Declaration\nparent_pr: #7308\n")

    assert declaration is None
    assert "requires both parent_pr and parent_head" in (error or "")


def test_parse_stack_declaration_rejects_short_parent_head() -> None:
    text = "## Stack Declaration\nparent_pr: #7308\nparent_head: deadbeef\n"
    declaration, error = parse_stack_declaration(text)

    assert declaration is None
    assert "requires both parent_pr and parent_head" in (error or "")


def test_parse_stack_declaration_rejects_non_positive_pr() -> None:
    text = f"## Stack Declaration\nparent_pr: #0\nparent_head: {PARENT_HEAD}\n"
    declaration, error = parse_stack_declaration(text)

    assert declaration is None
    assert "parent_pr must be positive" in (error or "")


def test_parse_stack_declaration_ignores_other_sections() -> None:
    text = (
        "## Summary\nwork here\n"
        f"## Stack Declaration\nparent_pr: #7389\nparent_head: {PARENT_HEAD}\n"
    )
    declaration, error = parse_stack_declaration(text)

    assert error is None
    assert declaration is not None
    assert declaration.parent_pr == 7389


def test_parse_stack_declaration_does_not_read_fields_from_later_sections() -> None:
    """Fields after the canonical section cannot manufacture a declaration."""
    text = f"## Stack Declaration\nparent_pr: #7308\n## Summary\nparent_head: {PARENT_HEAD}\n"

    declaration, error = parse_stack_declaration(text)

    assert declaration is None
    assert error == "stack declaration requires both parent_pr and parent_head"


# ---------------------------------------------------------------------------
# Classification: clean
# ---------------------------------------------------------------------------


def test_branch_from_current_main_with_intended_commits_is_clean() -> None:
    """Acceptance: a branch created from current main with only its commits passes."""
    result = _classify(
        base_ref="main",
        merge_base_sha=MAIN_TIP,
        commits=CLEAN_COMMITS,
    )

    assert result["state"] == "clean"
    assert result["unexpected_commits"] == []
    assert result["declared_parent"] is None


def test_head_without_commits_beyond_main_is_clean() -> None:
    """A head equal to (or an ancestor of) current main has nothing unexpected."""
    result = _classify(base_ref="main", merge_base_sha=MAIN_TIP, commits=[])

    assert result["state"] == "clean"


def test_stale_declaration_on_clean_branch_is_not_a_blocker() -> None:
    """A leftover declaration on an already-clean branch is harmless cleanup."""
    result = _classify(
        base_ref="main",
        merge_base_sha=MAIN_TIP,
        commits=CLEAN_COMMITS,
        declaration=_declaration(parent_pr=7389),
    )

    assert result["state"] == "clean"


# ---------------------------------------------------------------------------
# Classification: undeclared stack
# ---------------------------------------------------------------------------


def test_contaminated_branch_without_declaration_fails_undeclared() -> None:
    """Acceptance: the #7308 contaminated pattern fails closed with no declaration."""
    result = _classify(declaration=None)

    assert result["state"] == "undeclared_stack"
    assert result["state"] in BLOCKING_STATES
    assert result["unexpected_commits"] == CONTAMINATED_COMMITS
    assert "## Stack Declaration" in result["remediation"]


def test_clean_replacement_without_declaration_passes() -> None:
    """Acceptance: the #7389/#7390 clean replacement pattern passes."""
    result = _classify(base_ref="main", merge_base_sha=MAIN_TIP, commits=CLEAN_COMMITS)

    assert result["state"] == "clean"


# ---------------------------------------------------------------------------
# Classification: stacked (valid declaration)
# ---------------------------------------------------------------------------


def test_declared_stack_is_classified_stacked() -> None:
    """Acceptance: an exact parent PR + head declaration yields a valid stack."""
    result = _classify(declaration=_declaration(parent_pr=7308), parent_state="open")

    assert result["state"] == "stacked"
    assert result["state"] in NOT_INDEPENDENTLY_MERGEABLE_STATES
    assert result["declared_parent"] == 7308
    assert result["declared_parent_head"] == PARENT_HEAD


def test_stacked_is_never_independently_mergeable_contract() -> None:
    """The stacked state belongs to the not-independently-mergeable contract."""
    assert "stacked" in NOT_INDEPENDENTLY_MERGEABLE_STATES
    assert BLOCKING_STATES.isdisjoint(NOT_INDEPENDENTLY_MERGEABLE_STATES)


# ---------------------------------------------------------------------------
# Classification: mismatched declaration
# ---------------------------------------------------------------------------


def test_declared_parent_head_not_in_ancestry_fails_mismatched() -> None:
    """Acceptance: a declaration not matching the actual ancestry fails closed."""
    result = _classify(
        declaration=_declaration(parent_pr=7308, parent_head=OTHER_HEAD),
        parent_state="open",
    )

    assert result["state"] == "mismatched_declaration"
    assert result["state"] in BLOCKING_STATES


def test_declared_head_is_authoritative_over_pr_number() -> None:
    """The parent head is the cryptographic anchor of a stack declaration.

    The parent PR number is bound through the coordinator's parent lookup: a
    number naming a different PR whose live head differs from the declared head
    is caught as ``parent_invalidated`` (head changed / rewritten).  Inside the
    pure classifier the declared head matching the actual ancestry is the
    authoritative signal, so a wrong number with the *correct* head stays a
    valid stack.
    """
    result = _classify(
        declaration=_declaration(parent_pr=9999, parent_head=PARENT_HEAD),
        parent_state="open",
    )

    assert result["state"] == "stacked"


def test_mismatched_head_with_referenced_pr_distinguishes_head_mismatch() -> None:
    """A head mismatch on the right PR surfaces the head-specific remediation."""
    commits = [f"{'9' * 40} refactor: share slices (#7308)"]
    result = _classify(
        commits=commits,
        declaration=_declaration(parent_pr=7308, parent_head=OTHER_HEAD),
        parent_state="open",
    )

    assert result["state"] == "mismatched_declaration"
    assert "exact head SHA" in result["remediation"]


# ---------------------------------------------------------------------------
# Classification: parent lifecycle
# ---------------------------------------------------------------------------


def test_parent_closed_unmerged_invalidates_declaration() -> None:
    """Acceptance: a closed-unmerged parent invalidates the prior declaration."""
    result = _classify(declaration=_declaration(parent_pr=7308), parent_state="closed")

    assert result["state"] == "parent_invalidated"
    assert result["state"] in BLOCKING_STATES


def test_parent_rewritten_head_changed_invalidates_declaration() -> None:
    """Acceptance: a rewritten (head-changed) parent invalidates the declaration."""
    result = _classify(
        declaration=_declaration(parent_pr=7308),
        parent_state="open",
        parent_head_changed=True,
    )

    assert result["state"] == "parent_invalidated"


def test_unverifiable_parent_invalidates_declaration() -> None:
    """A parent PR that cannot be verified fails closed as invalidated."""
    result = _classify(declaration=_declaration(parent_pr=7308), parent_state="unknown")

    assert result["state"] == "parent_invalidated"
    assert result["parent_state"] == "unknown"


def test_parent_merged_requires_child_re_evaluation() -> None:
    """Acceptance: a merged parent moves the child to parent_merged (re-evaluate)."""
    result = _classify(
        declaration=_declaration(parent_pr=7308),
        parent_state="closed",
        parent_merged=True,
    )

    assert result["state"] == "parent_merged"
    assert result["state"] in NOT_INDEPENDENTLY_MERGEABLE_STATES
    assert "re-evaluate" in result["remediation"]


def test_parent_merged_then_child_rebased_to_main_becomes_clean() -> None:
    """After the parent merges and the child is re-evaluated, only residual diff remains."""
    residual = _classify(
        base_ref="main",
        merge_base_sha=MAIN_TIP,
        commits=CLEAN_COMMITS,
        declaration=None,
    )

    assert residual["state"] == "clean"


# ---------------------------------------------------------------------------
# Regression fixtures: #7308/#7309 contaminated vs #7389/#7390 clean
# ---------------------------------------------------------------------------


def test_regression_contaminated_7308_pattern_fails_before_review() -> None:
    """The exact #7308 contamination (foreign commits on the branch) must fail."""
    contaminated = [
        f"{'1' * 40} refactor: share PySF state slice constants (#7283)",
        f"{'2' * 40} merge remote-tracking branch 'origin/fix/issue-7283-shared-pysf-slices'",
        f"{'3' * 40} feat: intended work",
    ]
    result = _classify(commits=contaminated, declaration=None)

    assert result["state"] == "undeclared_stack"
    assert len(result["unexpected_commits"]) == 3


def test_regression_clean_7389_replacement_passes() -> None:
    """The clean #7389 replacement contains only its intended commit on main."""
    result = _classify(
        base_ref="main",
        merge_base_sha=MAIN_TIP,
        commits=[
            f"{'4' * 40} refactor: share PySF state slice constants (clean replacement) (#7389)"
        ],
    )

    assert result["state"] == "clean"


# ---------------------------------------------------------------------------
# Diagnostics and remediation
# ---------------------------------------------------------------------------


def test_diagnostics_include_required_fields() -> None:
    """Acceptance: the diagnostic block carries every issue #7515 item-5 field."""
    result = _classify(
        declaration=None,
        base_ref="main",
        merge_base_sha=PARENT_HEAD,
    )
    lines = render_diagnostics(result)
    joined = "\n".join(lines)

    assert "ancestry state: undeclared_stack" in joined
    assert "base ref: main" in joined
    assert f"merge base: {PARENT_HEAD}" in joined
    assert "unexpected commits:" in joined
    assert "unexpected paths:" in joined
    assert "declared parent: none" in joined
    assert "remediation:" in joined


def test_remediation_command_uses_parent_head() -> None:
    command = remediation_command(parent_head=PARENT_HEAD, branch="fix/child")

    assert command == f"git rebase --onto origin/main {PARENT_HEAD} fix/child"


def test_remediation_command_degrades_without_parent_head() -> None:
    command = remediation_command(parent_head="", branch="fix/child")

    assert command == "git rebase --onto origin/main fix/child"


# ---------------------------------------------------------------------------
# collect_ancestry_facts with a synthetic git repo (tmp_path)
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    """Initialize a synthetic repo with advanced main + a contaminated child.

    Layout (mirrors the #7308 failure): a child branch is built on top of
    another issue's unmerged foreign work, which itself branched from an older
    ``main``.  ``origin/main`` then advances, so the child's live merge base is
    strictly older than the live ``origin/main`` tip and the child carries
    foreign commits — the contaminated-ancestry pattern.
    """
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "test"], check=True)
    (path / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "base.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    base_sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Foreign branch commit (simulates another issue's unmerged work).
    subprocess.run(["git", "-C", str(path), "checkout", "-qb", "foreign"], check=True)
    (path / "foreign.txt").write_text("foreign\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "foreign.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "foreign work (#9999)"], check=True)
    foreign_sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Child branch built on the foreign work (contaminated pattern).
    subprocess.run(["git", "-C", str(path), "checkout", "-qb", "child"], check=True)
    (path / "child.txt").write_text("child\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "child.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "intended child work"], check=True)
    child_sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # main advances past the base; origin/main points at the advanced tip.
    subprocess.run(["git", "-C", str(path), "checkout", "-q", "main"], check=True)
    (path / "main-advance.txt").write_text("advanced\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "main-advance.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "advance main"], check=True)
    main_sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(path), "update-ref", "refs/remotes/origin/main", main_sha], check=True
    )
    path.joinpath("shas.json").write_text(
        json.dumps(
            {
                "base": base_sha,
                "main": main_sha,
                "foreign": foreign_sha,
                "child": child_sha,
            }
        ),
        encoding="utf-8",
    )


def test_collect_ancestry_facts_reports_contaminated_merge_base(tmp_path: Path) -> None:
    """Live git facts surface the foreign merge base and non-main commits.

    The synthetic child branch reproduces the #7308 pattern: its merge base is
    the old ``base`` commit (not the advanced ``origin/main`` tip), and the
    ``main..child`` enumeration contains the foreign issue's commit.
    """
    _init_repo(tmp_path)
    shas = json.loads(tmp_path.joinpath("shas.json").read_text(encoding="utf-8"))

    facts, error = collect_ancestry_facts(
        head_sha=shas["child"],
        base_ref="main",
        worktree=tmp_path,
        remote="origin",
    )

    assert error is None
    assert facts is not None
    assert facts["main_tip_sha"] == shas["main"]
    assert facts["merge_base_sha"] == shas["base"]
    assert facts["commits"] == [
        "intended child work",
        "foreign work (#9999)",
    ] or [commit.split(" ", 1)[1] for commit in facts["commits"]] == [
        "intended child work",
        "foreign work (#9999)",
    ]
    assert facts["changed_paths"] == ["child.txt", "foreign.txt"]

    state = ancestry_state(
        head_sha=shas["child"],
        base_ref="main",
        main_tip_sha=shas["main"],
        merge_base_sha=shas["base"],
        commits=facts["commits"],
    )
    # Contaminated pattern with no declaration fails closed (issue #7515).
    assert state["state"] == "undeclared_stack"
    assert state["unexpected_commits"] == facts["commits"]

    # With an exact parent declaration binding the foreign head, the same branch
    # is a valid declared stack (never independently mergeable).
    declared = ancestry_state(
        head_sha=shas["child"],
        base_ref="main",
        main_tip_sha=shas["main"],
        merge_base_sha=shas["base"],
        commits=facts["commits"],
        declaration=StackDeclaration(parent_pr=9999, parent_head=shas["foreign"]),
        parent_state="open",
    )
    assert declared["state"] == "stacked"


def test_collect_ancestry_facts_rejects_bad_inputs(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    shas = json.loads(tmp_path.joinpath("shas.json").read_text(encoding="utf-8"))

    facts, error = collect_ancestry_facts(
        head_sha="short-sha",
        base_ref="main",
        worktree=tmp_path,
    )
    assert facts is None
    assert "40-hex" in (error or "")

    facts, error = collect_ancestry_facts(
        head_sha=shas["child"],
        base_ref="",
        worktree=tmp_path,
    )
    assert facts is None
    assert "base ref" in (error or "")
