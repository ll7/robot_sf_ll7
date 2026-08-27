"""Tests for the canonical Dependabot risk-lane policy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jsonschema
import pytest

from scripts.dev.check_dependabot_update_policy import (
    PolicyError,
    _diff_vs_head,
    changed_files,
    changed_lock_package_names,
    check_workflow_action_pin_guard,
    classify_package_names,
    evaluate_update,
    filter_normalization_only_classifications,
    load_policy,
    validate_ci_workflow,
    validate_dependabot_config,
    validate_direct_dependency_coverage,
    validate_direct_update_lanes,
    validate_repository_structure,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "scripts/validation/dependabot_update_policy.v1.json"
SCHEMA_PATH = REPO_ROOT / "scripts/validation/dependabot_update_policy.v1.schema.json"
ACTION_REF = "example/demo-action"
OLD_ACTION_REF = f"{ACTION_REF}@{'a' * 40}"
NEW_ACTION_REF = f"{ACTION_REF}@{'b' * 40}"


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_workflow_pin_repo(
    tmp_path: Path,
    *,
    head_workflow: str,
    base_workflow: str | None = None,
    base_contract: str = OLD_ACTION_REF,
    head_contract: str | None = None,
) -> str:
    workflow_path = tmp_path / ".github/workflows/ci.yml"
    contract_path = tmp_path / "tests/dev/contract.py"
    workflow_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True)
    base_workflow = base_workflow or f"jobs:\n  test:\n    steps:\n      - uses: {OLD_ACTION_REF}\n"
    workflow_path.write_text(base_workflow, encoding="utf-8")
    contract_path.write_text(base_contract + "\n", encoding="utf-8")

    _run_git(tmp_path, "init", "--quiet")
    _run_git(tmp_path, "config", "user.name", "Dependabot policy test")
    _run_git(tmp_path, "config", "user.email", "dependabot-policy-test@example.invalid")
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "-c", "commit.gpgSign=false", "commit", "--quiet", "-m", "base")
    base_sha = _run_git(tmp_path, "rev-parse", "HEAD")

    workflow_path.write_text(head_workflow, encoding="utf-8")
    contract_path.write_text(
        (head_contract if head_contract is not None else base_contract) + "\n", encoding="utf-8"
    )
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "-c", "commit.gpgSign=false", "commit", "--quiet", "-m", "head")
    return base_sha


def test_policy_matches_its_schema() -> None:
    """The executable policy must satisfy its tracked schema."""
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(policy)


def test_live_policy_covers_direct_dependencies_and_ci_surfaces() -> None:
    """The current repository must be covered before new updates are admitted."""
    policy = validate_repository_structure(repo_root=REPO_ROOT)
    validate_ci_workflow(policy)
    validate_dependabot_config()


def test_issue_7480_package_set_is_split_into_distinct_risk_classes() -> None:
    """The historical mixed update cannot pass as one direct-risk lane."""
    policy = load_policy()
    names = {
        "numba",
        "orjson",
        "wandb",
        "pyarrow",
        "ruff",
        "pre-commit",
        "pylint",
        "mypy",
    }
    classified = classify_package_names(names, names, policy)
    assert {item["class"] for item in classified} == {
        "developer-tooling",
        "experiment-integrations",
        "high-impact-runtime",
        "serialization-data",
    }
    with pytest.raises(PolicyError, match="mixes direct risk classes"):
        validate_direct_update_lanes(classified)


def test_developer_tooling_lane_can_remain_grouped() -> None:
    """A bounded tooling-only group remains one review and rollback surface."""
    policy = load_policy()
    names = {"ruff", "pre-commit", "pylint", "mypy", "pytest"}
    classified = classify_package_names(names, names, policy)
    assert validate_direct_update_lanes(classified) == ["developer-tooling"]


def test_unknown_direct_package_fails_closed() -> None:
    """A new direct package must receive an explicit policy classification."""
    policy = load_policy()
    with pytest.raises(PolicyError, match="not classified"):
        classify_package_names({"new-runtime-package"}, {"new-runtime-package"}, policy)


def test_unlisted_transitive_package_uses_conservative_fallback() -> None:
    """An unlisted lock-only row remains visible and keeps compatibility evidence."""
    policy = load_policy()
    classified = classify_package_names({"transitive-helper"}, set(), policy)
    assert classified == [
        {
            "name": "transitive-helper",
            "direct": False,
            "class": "transitive-lock-package",
            "risk": "unknown",
            "update_lane": "individual",
            "required_jobs": ["fast-feedback", "compat-matrix"],
        }
    ]


def test_lock_row_changes_preserve_marker_variants() -> None:
    """A changed version or marker-specific row is treated as dependency evidence."""
    base = """
version = 1

[[package]]
name = "numba"
version = "0.66.0"
marker = "python_version >= '3.11'"
"""
    head = base.replace('version = "0.66.0"', 'version = "0.67.0"')
    assert changed_lock_package_names(base, head) == {"numba"}


def test_authoritative_changed_file_list_avoids_unneeded_base_lookup(tmp_path: Path) -> None:
    """Non-dependency PRs can use the collector output when base fetch is unavailable."""
    changed_file_list = tmp_path / "changed-files.txt"
    changed_file_list.write_text("README.md\n\n docs/dev/example.md \n", encoding="utf-8")
    assert changed_files(tmp_path, "missing-base", changed_file_list) == [
        "README.md",
        "docs/dev/example.md",
    ]


def test_exact_diff_replaces_stale_authoritative_base_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current base must exclude unrelated files from an older PR base."""
    changed_file_list = tmp_path / "changed-files.txt"
    changed_file_list.write_text("pyproject.toml\nuv.lock\nREADME.md\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.dev.check_dependabot_update_policy._diff_vs_head",
        lambda *args, **kwargs: "docs/research.md\n",
    )

    assert changed_files(tmp_path, "origin/main", changed_file_list) == ["docs/research.md"]


def test_diff_vs_head_falls_back_to_two_dot_when_three_dot_lacks_merge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shallow CI checkouts lack a merge base; the diff must fall back to two-dot."""

    def fake_git_text(repo_root, args):
        if args[:2] == ["diff", "--name-only"] and args[-1].endswith("...HEAD"):
            return None
        return "pyproject.toml\nuv.lock\n"

    monkeypatch.setattr("scripts.dev.check_dependabot_update_policy._git_text", fake_git_text)
    assert _diff_vs_head(tmp_path, "origin/main", ["--name-only"]) == "pyproject.toml\nuv.lock\n"


def test_diff_vs_head_three_dot_preferred_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full clone keeps the three-dot diff as the primary path."""
    calls: list[list[str]] = []

    def fake_git_text(repo_root, args):
        calls.append(args)
        if args[:2] == ["diff", "--name-only"]:
            return "README.md\n"
        return None

    monkeypatch.setattr("scripts.dev.check_dependabot_update_policy._git_text", fake_git_text)
    assert _diff_vs_head(tmp_path, "origin/main", ["--name-only"]) == "README.md\n"
    assert len(calls) == 1
    assert calls[0][-1] == "origin/main...HEAD"


def test_workflow_action_pin_guard_rejects_stale_head_reference(tmp_path: Path) -> None:
    """A replaced workflow pin cannot remain in a tracked contract surface."""
    head_workflow = f"jobs:\n  test:\n    steps:\n      - uses: {NEW_ACTION_REF}\n"
    base_sha = _make_workflow_pin_repo(tmp_path, head_workflow=head_workflow)

    with pytest.raises(
        PolicyError, match=f"{OLD_ACTION_REF}.*{NEW_ACTION_REF}.*tests/dev/contract.py"
    ):
        check_workflow_action_pin_guard(
            repo_root=tmp_path,
            base_ref=base_sha,
            changed_files=[".github/workflows/ci.yml"],
        )


def test_workflow_action_pin_guard_allows_clean_replacement(tmp_path: Path) -> None:
    """A replaced workflow pin passes after coupled references move to the new pin."""
    head_workflow = f"jobs:\n  test:\n    steps:\n      - uses: {NEW_ACTION_REF}\n"
    base_sha = _make_workflow_pin_repo(
        tmp_path,
        head_workflow=head_workflow,
        head_contract=NEW_ACTION_REF,
    )

    report = check_workflow_action_pin_guard(
        repo_root=tmp_path,
        base_ref=base_sha,
        changed_files=[".github/workflows/ci.yml"],
    )

    assert report["status"] == "pass"
    assert report["replacements"] == [
        {
            "workflow_file": ".github/workflows/ci.yml",
            "old_ref": OLD_ACTION_REF,
            "new_ref": NEW_ACTION_REF,
        }
    ]
    assert report["stale_references"] == []


def test_workflow_action_pin_guard_ignores_non_pin_workflow_changes(tmp_path: Path) -> None:
    """Unchanged pins do not force unrelated tracked references to move."""
    base_workflow = f"jobs:\n  test:\n    steps:\n      - uses: {OLD_ACTION_REF}\n"
    base_sha = _make_workflow_pin_repo(
        tmp_path,
        head_workflow=base_workflow + "# unrelated workflow comment\n",
    )

    report = check_workflow_action_pin_guard(
        repo_root=tmp_path,
        base_ref=base_sha,
        changed_files=[".github/workflows/ci.yml"],
    )

    assert report["status"] == "not_applicable"
    assert report["replacements"] == []


def test_workflow_action_pin_guard_ignores_block_scalar_text(tmp_path: Path) -> None:
    """Text that resembles ``uses`` inside a shell block is not an action declaration."""
    base_workflow = (
        f"jobs:\n  test:\n    steps:\n      - run: |\n          uses: {OLD_ACTION_REF}\n"
    )
    head_workflow = base_workflow.replace(OLD_ACTION_REF, NEW_ACTION_REF)
    base_sha = _make_workflow_pin_repo(
        tmp_path,
        base_workflow=base_workflow,
        head_workflow=head_workflow,
    )

    report = check_workflow_action_pin_guard(
        repo_root=tmp_path,
        base_ref=base_sha,
        changed_files=[".github/workflows/ci.yml"],
    )

    assert report["status"] == "not_applicable"
    assert report["replacements"] == []


def test_workflow_action_pin_guard_fails_closed_without_base_ref(tmp_path: Path) -> None:
    """A changed workflow cannot bypass the guard when the exact base is unavailable."""
    head_workflow = f"jobs:\n  test:\n    steps:\n      - uses: {NEW_ACTION_REF}\n"
    _make_workflow_pin_repo(tmp_path, head_workflow=head_workflow)

    with pytest.raises(PolicyError, match="unable to resolve workflow action-pin base ref"):
        check_workflow_action_pin_guard(
            repo_root=tmp_path,
            base_ref="missing-base",
            changed_files=[".github/workflows/ci.yml"],
        )


def test_workflow_step_uses_the_policy_checker() -> None:
    """The existing PR contract workflow must execute the policy checker."""
    workflow = (REPO_ROOT / ".github/workflows/pr-contract-check.yml").read_text(encoding="utf-8")
    assert "Validate dependency update policy" in workflow
    assert "scripts/dev/check_dependabot_update_policy.py" in workflow


def test_direct_dependency_coverage_rejects_unreviewed_name() -> None:
    """Coverage cannot be silently weakened by adding an unreviewed direct name."""
    policy = load_policy()
    with pytest.raises(PolicyError, match="missing from the canonical policy"):
        validate_direct_dependency_coverage({"unreviewed-package"}, policy)


def test_marker_only_direct_row_does_not_add_a_second_risk_class() -> None:
    policy = load_policy()
    classifications = classify_package_names({"numpy", "pylint"}, {"numpy", "pylint"}, policy)

    effective, normalized = filter_normalization_only_classifications(
        classifications,
        {"normalization_only_packages": ["numpy"]},
        changed_direct_names=set(),
    )

    assert normalized == ["numpy"]
    assert [item["name"] for item in effective] == ["pylint"]
    assert validate_direct_update_lanes(effective) == ["developer-tooling"]


def test_evaluate_update_exposes_resolution_evidence_and_filters_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_lock = """
version = 1

[[package]]
name = "numpy"
version = "2.0.0"

[[package]]
name = "pylint"
version = "3.3.7"
"""
    head_lock = base_lock.replace(
        'name = "numpy"\nversion = "2.0.0"',
        'name = "numpy"\nversion = "2.0.0"\nresolution-markers = ["python_version >= \'3.11\'"]',
    ).replace('name = "pylint"\nversion = "3.3.7"', 'name = "pylint"\nversion = "4.0.7"')
    (tmp_path / "uv.lock").write_text(head_lock, encoding="utf-8")
    policy = load_policy()

    monkeypatch.setattr(
        "scripts.dev.check_dependabot_update_policy.direct_dependency_names",
        lambda _repo_root: {"numpy", "pylint"},
    )
    monkeypatch.setattr(
        "scripts.dev.check_dependabot_update_policy.changed_files",
        lambda *_args, **_kwargs: ["uv.lock"],
    )
    monkeypatch.setattr(
        "scripts.dev.check_dependabot_update_policy.git_file_at_ref",
        lambda _repo_root, _base_ref, relative_path: (
            base_lock if relative_path == "uv.lock" else ""
        ),
    )

    report = evaluate_update(repo_root=tmp_path, base_ref="origin/main", policy=policy)

    assert report["status"] == "pass"
    assert report["workflow_action_pin_guard"]["status"] == "not_applicable"
    assert report["normalization_only_packages"] == ["numpy"]
    assert [item["name"] for item in report["effective_changed_packages"]] == ["pylint"]
    assert report["direct_risk_classes"] == ["developer-tooling"]
    evidence = report["resolution_evidence"]
    assert evidence["schema_version"] == "robot-sf.dependency-resolution-evidence.v1"
    assert evidence["material_packages"] == ["pylint"]
    assert evidence["locks"]["uv.lock"]["profiles"]
    assert evidence["locks"]["uv.lock"]["profile_ids"]
    assert evidence["locks"]["uv.lock"]["owners"][0]["id"] == "root"
    assert evidence["changed_package_classifications"][0]["name"] == "numpy"
