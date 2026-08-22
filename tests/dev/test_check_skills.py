"""Regression tests for the repo-local skill checker."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


def _load_check_skills_module():
    """Load scripts/dev/check_skills.py without requiring scripts to be a package."""
    module_path = Path(__file__).parents[2] / "scripts/dev/check_skills.py"
    spec = importlib.util.spec_from_file_location("check_skills", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _documented_handoff(readme: str) -> tuple[dict[str, object], str]:
    """Extract the flat handoff and resolver command blocks."""
    start_marker = "<!-- handoff.v2-example:start -->"
    end_marker = "<!-- handoff.v2-example:end -->"
    start = readme.index(start_marker)
    end = readme.index(end_marker, start)
    yaml_start = readme.index("```yaml", start) + len("```yaml")
    yaml_end = readme.index("```", yaml_start)
    assert yaml_end < end
    handoff = yaml.safe_load(readme[yaml_start:yaml_end])
    assert isinstance(handoff, dict)
    bash_start = readme.index("```bash", end) + len("```bash")
    bash_end = readme.index("```", bash_start)
    return handoff, readme[bash_start:bash_end]


def _handoff_doc_text() -> str:
    """Return the canonical shared-routing handoff document text."""
    return (Path(__file__).parents[2] / "docs/ai/agent_workflow_entrypoints.md").read_text(
        encoding="utf-8"
    )


def _assert_flat_handoff_contract(handoff: dict[str, object]) -> None:
    """Validate the documented identity and control fields independently of a resolver checkout."""
    allowed_fields = {
        "schema_version",
        "handoff_type",
        "task_id",
        "provider",
        "mode",
        "goal",
        "owned_paths",
        "forbidden_actions",
        "required_context",
        "required_output",
        "acceptance_gate",
        "validation_commands",
        "execution_mode",
        "dependencies",
        "budget",
        "stop_conditions",
        "side_effect_policy",
        "max_depth",
        "sync_barrier",
    }
    assert set(handoff) == allowed_fields
    assert handoff["schema_version"] == "handoff.v2"
    assert handoff["handoff_type"] == "request"
    assert handoff["task_id"] == "ROBOTSF-EXAMPLE"
    assert handoff["provider"] == "opencode_go"
    assert handoff["mode"] == "issue_implementation"
    assert handoff["execution_mode"] == "external_runtime"
    for field in ("task_id", "provider", "mode", "goal"):
        assert isinstance(handoff[field], str) and handoff[field].strip(), field
    for field in (
        "owned_paths",
        "forbidden_actions",
        "required_context",
        "required_output",
        "acceptance_gate",
        "validation_commands",
        "stop_conditions",
    ):
        value = handoff[field]
        assert isinstance(value, list) and value, field
        assert all(isinstance(item, str) and item.strip() for item in value), field
    dependencies = handoff["dependencies"]
    assert isinstance(dependencies, list)
    assert all(isinstance(item, str) and item.strip() for item in dependencies)
    required_output = handoff["required_output"]
    assert isinstance(required_output, list)
    assert required_output.count("final_status") == 1
    budget = handoff["budget"]
    assert isinstance(budget, dict)
    assert set(budget) == {"runtime_minutes"}
    assert budget["runtime_minutes"] == 30
    assert isinstance(budget["runtime_minutes"], int)
    assert not isinstance(budget["runtime_minutes"], bool)
    side_effect_policy = handoff["side_effect_policy"]
    assert isinstance(side_effect_policy, dict)
    assert set(side_effect_policy) == {"remote_mutation", "local_edits"}
    assert side_effect_policy == {"remote_mutation": False, "local_edits": True}
    assert isinstance(side_effect_policy["remote_mutation"], bool)
    assert isinstance(side_effect_policy["local_edits"], bool)
    assert handoff["max_depth"] == 0
    assert isinstance(handoff["max_depth"], int)
    assert not isinstance(handoff["max_depth"], bool)
    assert handoff["sync_barrier"] is None


def _assert_cli_contract(command_block: str) -> None:
    """Require only the production identity/head/output flags in the README command."""
    assert "```" not in command_block
    expected_fragments = {
        "--task-id ROBOTSF-EXAMPLE": 1,
        "--task-class issue_implementation": 1,
        "--risk R1": 1,
        '--handoff-file "$HANDOFF_FILE"': 1,
        '--frozen-head "$TARGET_HEAD"': 1,
        '--target-repo "$TARGET_REPO"': 1,
        '--out "${TMPDIR:-/tmp}/robotsf-route-plan.json"': 1,
    }
    for fragment, expected_count in expected_fragments.items():
        assert command_block.count(fragment) == expected_count, fragment
    for flag in (
        "--task-id",
        "--task-class",
        "--risk",
        "--handoff-file",
        "--frozen-head",
        "--target-repo",
        "--out",
    ):
        assert command_block.count(flag) == 1, flag
    for redundant_flag in ("--prompt", "--owned-paths", "--validation"):
        assert redundant_flag not in command_block, redundant_flag


def _run_shared_resolver_if_available(
    handoff: dict[str, object],
    command_block: str,
    target_repo: Path,
    tmp_path: Path,
) -> None:
    """Run the documented argv contract only when an explicit resolver checkout is available."""
    _assert_cli_contract(command_block)

    if "CODEX_ROUTING_REPO" not in os.environ:
        pytest.skip(
            "flat handoff contract validated; set CODEX_ROUTING_REPO to run the shared resolver"
        )
    routing_repo_value = os.environ["CODEX_ROUTING_REPO"]
    routing_repo = Path(routing_repo_value).expanduser().resolve()
    resolver = routing_repo / "scripts/resolve-route.py"
    assert resolver.is_file(), f"CODEX_ROUTING_REPO resolver missing: {resolver}"

    frozen_head = subprocess.run(
        ["git", "-C", str(target_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    handoff_path = tmp_path / "handoff.v2.yaml"
    handoff_path.write_text(yaml.safe_dump(handoff, sort_keys=False), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CODEX_ROUTING_REPO": str(routing_repo),
            "HANDOFF_FILE": str(handoff_path),
            "TMPDIR": str(tmp_path),
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", command_block],
        cwd=target_repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    route_plan_path = tmp_path / "robotsf-route-plan.json"
    artifact = json.loads(route_plan_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "route-plan.v1"
    assert artifact["task_id"] == handoff["task_id"]
    assert artifact["task_class"] == "issue_implementation"
    assert artifact["risk"] == "R1"
    assert artifact["repo_head"] == frozen_head
    assert artifact["handoff_schema_version"] == "handoff.v2"
    assert artifact["owned_paths"] == handoff["owned_paths"]
    assert artifact["validation_commands"] == handoff["validation_commands"]


def test_shared_route_planner_example_binds_reviewed_execution_contract() -> None:
    """Parse the flat handoff and bind its identity/control fields to the CLI example."""
    readme = _handoff_doc_text()

    assert '"$ROUTING_REPO/scripts/resolve-route.py"' in readme
    assert "--task-id ROBOTSF-EXAMPLE" in readme
    assert "--task-class issue_implementation" in readme
    assert "--risk R1" in readme
    assert '--frozen-head "$TARGET_HEAD"' in readme
    assert '--target-repo "$TARGET_REPO"' in readme
    assert "python3 ./scripts/resolve-route.py" not in readme
    handoff, command_block = _documented_handoff(readme)
    _assert_flat_handoff_contract(handoff)
    _assert_cli_contract(command_block)


def test_shared_route_planner_example_executes_shared_resolver_when_available(
    tmp_path: Path,
) -> None:
    """Run the exact argv contract only with an explicit, deterministic resolver checkout."""
    readme = _handoff_doc_text()
    target_repo = Path(__file__).parents[2]
    handoff, command_block = _documented_handoff(readme)
    _assert_flat_handoff_contract(handoff)
    _run_shared_resolver_if_available(handoff, command_block, target_repo, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("__unknown__", True),
        ("schema_version", "handoff.v1"),
        ("handoff_type", "result"),
        ("task_id", ""),
        ("mode", "read_only_review"),
        ("execution_mode", "in_turn"),
        ("owned_paths", [123]),
        ("dependencies", [""]),
        ("required_output", ["final_status", "final_status"]),
        ("stop_conditions", []),
        ("budget", []),
        ("budget", {"runtime_minutes": "30"}),
        ("budget", {"runtime_minutes": 30, "repair_loops": 2}),
        ("side_effect_policy", []),
        ("side_effect_policy", {"remote_mutation": False, "local_edits": "true"}),
        (
            "side_effect_policy",
            {"remote_mutation": False, "local_edits": True, "unexpected": False},
        ),
        ("max_depth", True),
        ("max_depth", -1),
        ("sync_barrier", {}),
    ],
    ids=[
        "unknown-top-level-control",
        "schema-version",
        "handoff-type",
        "empty-task-id",
        "mode",
        "execution-mode",
        "non-string-owned-path",
        "nonempty-dependencies-must-be-strings",
        "duplicate-final-status",
        "empty-stop-conditions",
        "budget-type",
        "budget-value-type",
        "budget-unknown-key",
        "side-effect-policy-type",
        "side-effect-policy-value-type",
        "side-effect-policy-unknown-key",
        "boolean-max-depth",
        "negative-max-depth",
        "malformed-sync-barrier",
    ],
)
def test_documented_handoff_rejects_malformed_or_unknown_controls(
    field: str,
    value: object,
) -> None:
    """Malformed or unknown v2 controls must fail the portable contract validator."""
    readme = _handoff_doc_text()
    handoff, _ = _documented_handoff(readme)
    if field == "__unknown__":
        handoff["unexpected_control"] = value
    else:
        handoff[field] = value

    with pytest.raises(AssertionError):
        _assert_flat_handoff_contract(handoff)


def test_read_yaml_fails_closed_for_empty_yaml(tmp_path: Path) -> None:
    """Empty YAML files should fail with a path-qualified checker error."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("", encoding="utf-8")

    with pytest.raises(AssertionError, match="YAML file is empty"):
        check_skills._read_yaml(yaml_path)


def test_read_yaml_fails_closed_for_non_mapping_yaml(tmp_path: Path) -> None:
    """Registry and schema YAML files should be top-level mappings."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    yaml_path = tmp_path / "scalar.yaml"
    yaml_path.write_text("not-a-map\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="YAML top level must be a mapping"):
        check_skills._read_yaml(yaml_path)


def test_registry_shape_reports_non_mapping_skill_metadata() -> None:
    """Malformed skill entries should produce checker errors instead of AttributeError."""
    check_skills = _load_check_skills_module()
    schema = {
        "allowed_categories": ["general"],
        "allowed_kinds": ["atomic"],
        "allowed_phases": ["context"],
        "allowed_write_scopes": ["filesystem"],
    }

    errors = check_skills._validate_registry_shape(
        {"version": 1, "skills": {"bad-skill": "oops"}},
        schema,
    )

    assert errors == ["bad-skill: metadata must be a dictionary"]


def test_frontmatter_fails_closed_for_non_mapping_yaml(tmp_path: Path) -> None:
    """Skill frontmatter must be a YAML mapping before downstream metadata checks run."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("---\n- nope\n---\n\nBody\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="frontmatter must be a YAML mapping"):
        check_skills._frontmatter(skill_path)


def _stage_content_contracts(tmp_path: Path, *skills: str) -> None:
    """Copy real declarative content contracts into the fixture repo root."""
    import shutil

    contracts = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "tests" / "contracts"
    for skill in skills:
        target_dir = tmp_path / ".agents" / "skills" / "tests" / "contracts"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            contracts / f"{skill}.content-contract.v1.yaml",
            target_dir / f"{skill}.content-contract.v1.yaml",
        )


def test_artifact_first_contract_passes_for_goal_autopilot(tmp_path: Path) -> None:
    """Artifact-first phrase and file requirements should pass for goal-autopilot style skills."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-autopilot")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-autopilot" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text("---\nname: goal-autopilot\n---\n", encoding="utf-8")
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing
or inconsistent.
The parent must inspect route evidence and run targeted local checks.
The routed manifest records terminal_state, scope_check, compact_artifacts, missing_artifact,
route_not_started, scope_violation, status.txt, and validation.txt. Parent acceptance remains
separate from route evidence.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
The active ledger records loaded context with skill/doc summaries, snapshot paths,
freshness keys, expected PR head SHA, worker artifact paths, and stale-state triggers.
Use compact_worktree_snapshot.py and compact_ci_snapshot.py before broad worktree or CI polling.
Pass ledger snapshot paths to workers, avoid repeating broad state polling, and run
fresh live checks before issue claim, push, PR publication, label/project mutation,
or merge-ready decisions.
Resolve the provider/model through the shared model-routing pointer using the phase task class
and current evidence state; do not select a model locally.
Use the shared model-routing pointer before dispatching any delegated phase. It owns the current
native tiers, evidenced escalation rule, and external provider budget alternatives. This skill
must not duplicate a model inventory or maintain a local sidecar route table. Route selection is
route evidence only; the controller still reviews artifacts, the diff, and validation locally
before accepting work.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-autopilot"},
        body,
    )
    assert errors == []


def test_artifact_first_contract_fails_when_missing_required_artifacts(tmp_path: Path) -> None:
    """Contracts should fail when required artifact filenames or evidence phrases are missing."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-autopilot")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-autopilot" / "SKILL.md"
    skill_path.parent.mkdir()
    body = "Delegated workers should run and report summary."
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-autopilot"},
        body,
    )
    assert any("result.json" in e for e in errors)
    assert any("contract 'artifact-first-review-order'" in e for e in errors)
    assert any("contract 'worker-output-limits'" in e for e in errors)
    assert any("contract 'active-ledger-reuse'" in e for e in errors)
    assert any("contract 'shared-model-routing'" in e for e in errors)


def test_artifact_first_contract_requires_canonical_result_markdown_case(
    tmp_path: Path,
) -> None:
    """The compact artifact contract should preserve RESULT.md casing exactly."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-autopilot")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-autopilot" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, result.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing
or inconsistent.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-autopilot"},
        body,
    )

    assert any("RESULT.md" in e for e in errors)


def test_goal_autopilot_contract_requires_active_ledger_reuse_terms(tmp_path: Path) -> None:
    """Goal autopilot should keep explicit ledger reuse and freshness-key guidance."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-autopilot")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-autopilot" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing
or inconsistent.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
The active ledger records only issue number, next action, and cleanup.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-autopilot"},
        body,
    )

    assert any("contract 'active-ledger-reuse'" in e for e in errors)


def test_active_routing_template_rejects_retired_spark_route(tmp_path: Path) -> None:
    """Active templates must use the shared resolver instead of retired Spark routing."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    template = tmp_path / "thread-profile.md"
    template.write_text(
        "Track Spark usage-limit resets and retry that model after reset.\n",
        encoding="utf-8",
    )

    errors = check_skills._validate_active_routing_template(template)

    assert errors == ["thread-profile.md: active routing template contains retired Spark routing"]


def test_goal_pr_review_contract_requires_compact_ci_snapshot_terms(tmp_path: Path) -> None:
    """Goal PR review should preserve compact PR/CI entry-point guidance."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-pr-review")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-pr-review" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
Start with snapshot_pr_queue.py, poll with watch_pr_ci_status.py, inspect status,conclusion,jobs,
return bounded excerpts, and keep full logs in private artifacts.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-pr-review"},
        body,
    )

    assert errors == []


def test_goal_pr_review_compact_ci_terms_are_case_insensitive(tmp_path: Path) -> None:
    """PR-review prose checks should tolerate sentence capitalization."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-pr-review" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
Start with scripts.dev.snapshot_pr_queue --active, poll with watch_pr_ci_status.py,
inspect status,conclusion,jobs.
Return bounded excerpts, and keep full logs in private artifacts.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-pr-review"},
        body,
    )

    assert errors == []


def test_goal_pr_review_contract_accepts_module_qualified_snapshot_command(
    tmp_path: Path,
) -> None:
    """Goal PR review accepts canonical module-qualified snapshot command guidance."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-pr-review" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
Start with `uv run python -m scripts.dev.snapshot_pr_queue --active`, poll with
watch_pr_ci_status.py, inspect status,conclusion,jobs, return bounded excerpts, and keep
full logs in private artifacts.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-pr-review"},
        body,
    )

    assert errors == []


def test_goal_pr_review_contract_requires_snapshot_queue_reference(tmp_path: Path) -> None:
    """Goal PR review should still require explicit snapshot queue command guidance."""
    check_skills = _load_check_skills_module()
    _stage_content_contracts(tmp_path, "goal-pr-review")
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-pr-review" / "SKILL.md"
    skill_path.parent.mkdir()
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
Poll with watch_pr_ci_status.py, inspect status,conclusion,jobs, return bounded excerpts,
and keep full logs in private artifacts.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-pr-review"},
        body,
    )

    assert any("one of" in error and "snapshot_pr_queue" in error for error in errors)


def test_goal_pr_review_snapshot_discovery_uses_the_active_cli_mode() -> None:
    """Queue-discovery guidance must use the CLI mode that selects active PRs."""
    skill_path = Path(__file__).parents[2] / ".agents/skills/goal-pr-review/SKILL.md"
    skill_text = skill_path.read_text(encoding="utf-8")

    assert "uv run python -m scripts.dev.snapshot_pr_queue --active" in skill_text


# -- broken-path detection tests ------------------------------------------------


def test_find_broken_paths_skips_non_path_placeholder(tmp_path: Path) -> None:
    """A backticked prose placeholder like ``SLURM/data-gated`` is not a path error.

    Regression for issue #3623: ``SLURM`` is a real top-level dir, so the path
    pattern matches the placeholder, but ``data-gated`` has no extension and only
    one path segment, so it must be treated as prose rather than a broken file.
    """
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "SKILL.md"
    body = "Route local-implementable vs `SLURM/data-gated` to `Success Probability`.\n"

    assert check_skills._find_broken_paths(skill_path, body) == []


def test_find_broken_paths_still_flags_broken_real_paths(tmp_path: Path) -> None:
    """Genuinely broken path references (extension or extra depth) are still caught."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "SKILL.md"
    body = (
        "See `docs/does_not_exist.md` and `SLURM/missing/template.sl` and "
        "`scripts/dev/no/such/file`.\n"
    )

    broken = check_skills._find_broken_paths(skill_path, body)

    assert any("docs/does_not_exist.md" in entry for entry in broken)
    assert any("SLURM/missing/template.sl" in entry for entry in broken)
    assert any("scripts/dev/no/such/file" in entry for entry in broken)


def test_find_broken_paths_allows_existing_paths(tmp_path: Path) -> None:
    """References that resolve on disk produce no errors."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("ok\n", encoding="utf-8")
    skill_path = tmp_path / "SKILL.md"
    body = "See `docs/real.md` for details.\n"

    assert check_skills._find_broken_paths(skill_path, body) == []


def test_looks_like_path_distinguishes_placeholders_from_paths() -> None:
    """Path-shaped tokens (extension or depth>=2) are paths; bare tokens are not."""
    check_skills = _load_check_skills_module()
    assert check_skills._looks_like_path("docs/guide.md") is True
    assert check_skills._looks_like_path("SLURM/templates/gpu.sl") is True
    assert check_skills._looks_like_path("scripts/dev/tool") is True
    assert check_skills._looks_like_path("SLURM/data-gated") is False
    assert check_skills._looks_like_path("docs/placeholder") is False


# -- preflight tests ------------------------------------------------------------


def test_check_command_git_is_available() -> None:
    """git should be available in a git repository checkout."""
    check_skills = _load_check_skills_module()
    ok, detail = check_skills._check_command("git", "--version")
    assert ok, f"git should be available but got: {detail}"
    assert "git version" in detail


def test_check_command_missing_command() -> None:
    """A nonexistent command should report missing."""
    check_skills = _load_check_skills_module()
    ok, detail = check_skills._check_command("this-command-does-not-exist-999", "--version")
    assert not ok
    assert "not found on PATH" in detail


def test_check_command_empty_output_is_safe() -> None:
    """A command with no version output should not crash preflight."""
    check_skills = _load_check_skills_module()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(check_skills.shutil, "which", lambda _cmd: "/usr/bin/tool")
        monkeypatch.setattr(
            check_skills.subprocess,
            "run",
            lambda *_args, **_kwargs: type(
                "Result",
                (),
                {"stdout": "", "stderr": "", "returncode": 0},
            )(),
        )
        ok, detail = check_skills._check_command("tool", "--version")
        assert ok
        assert detail == "unknown version"


def test_preflight_requires_no_reqs(tmp_path: Path) -> None:
    """Preflight should pass when a skill declares no requirements."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  minimal-skill:\n    requires: []\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("minimal-skill")
    assert rc == 0


def test_preflight_unknown_skill(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight should fail with a clear error for an unknown skill."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  real-skill:\n    requires: []\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("unknown-skill")
    assert rc == 1
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_preflight_json_output_unknown_skill(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight with --json should emit an error JSON object for an unknown skill."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  real-skill:\n    requires: []\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("unknown-skill", json_output=True)
    assert rc == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["skill"] == "unknown-skill"


def test_preflight_git_requires_in_skill(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight should check declared 'requires' against available tools."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  skill-with-git:\n    requires:\n      - git\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("skill-with-git")
    assert rc == 0
    captured = capsys.readouterr()
    assert "ok" in captured.out
    assert "git" in captured.out
    assert "PASS" in captured.out


def test_preflight_resolves_alias(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight should resolve an alias to its canonical skill name."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  my-skill:\n    aliases:\n      - my-alias\n    requires:\n      - git\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("my-alias")
    assert rc == 0
    captured = capsys.readouterr()
    assert "Preflight check for skill: my-skill" in captured.out


def test_preflight_json_output_passing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight with --json should emit structured JSON on success."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  json-skill:\n    requires:\n      - git\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("json-skill", json_output=True)
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["skill"] == "json-skill"
    assert payload["requires"] == ["git"]
    assert "checks" in payload
    assert payload["checks"]["git"]["status"] == "ok"
    assert payload["summary"]["available"] >= 1
    assert payload["summary"]["missing"] == 0
    assert payload["summary"]["unrecognized"] == 0


def test_preflight_multiple_requires_passing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight should check multiple requirements and pass when all are found."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  full-skill:\n    requires:\n      - git\n      - uv\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("full-skill")
    assert rc == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out


def test_preflight_unrecognized_requirement(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Preflight should fail closed on unrecognized requirements."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  exotic-skill:\n    requires:\n      - git\n      - some-exotic-tool\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("exotic-skill")
    assert rc == 1
    captured = capsys.readouterr()
    assert "UNRECOGNIZED" in captured.out
    assert "some-exotic-tool" in captured.out
    assert "Result: FAIL" in captured.out


def test_preflight_json_failure_sets_status(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """JSON preflight should mark the top-level status as fail for unsupported requirements."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  exotic-skill:\n    requires:\n      - some-exotic-tool\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("exotic-skill", json_output=True)
    assert rc == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "fail"
    assert payload["summary"]["unrecognized"] == 1


def test_preflight_publication_scout_linter_requirement_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A declared publication linter requirement should fail if its script is missing."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  scout-skill:\n    requires:\n      - publication-scout-linter\n",
        encoding="utf-8",
    )
    check_skills.REPO_ROOT = tmp_path
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("scout-skill")
    assert rc == 1
    captured = capsys.readouterr()
    assert "publication_scout_linter.py" in captured.out


def test_preflight_publication_scout_linter_requirement_present(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A declared publication-linter requirement should pass when the script exists."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  scout-skill:\n    requires:\n      - publication-scout-linter\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "dev" / "publication_scout_linter.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python3\nprint('ok')\n", encoding="utf-8")
    check_skills.REPO_ROOT = tmp_path
    check_skills.REGISTRY = registry_yaml
    rc = check_skills._preflight("scout-skill")
    assert rc == 0
    captured = capsys.readouterr()
    assert "publication-scout-linter" in captured.out
    assert "PASS" in captured.out


def test_parse_args_preflight() -> None:
    """--preflight flag should be parsed correctly."""
    check_skills = _load_check_skills_module()
    args = check_skills._parse_args(["--preflight", "my-skill"])
    assert args.preflight == "my-skill"
    assert args.json is False


def test_parse_args_preflight_with_json() -> None:
    """--preflight with --json should set both flags."""
    check_skills = _load_check_skills_module()
    args = check_skills._parse_args(["--preflight", "my-skill", "--json"])
    assert args.preflight == "my-skill"
    assert args.json is True


def test_main_dispatches_to_preflight(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """main() should dispatch to _preflight when --preflight is given."""
    check_skills = _load_check_skills_module()
    registry_yaml = tmp_path / "skills.yaml"
    registry_yaml.write_text(
        "version: 1\nskills:\n  preflight-skill:\n    requires:\n      - git\n",
        encoding="utf-8",
    )
    check_skills.REGISTRY = registry_yaml
    rc = check_skills.main(["--preflight", "preflight-skill"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Preflight check" in captured.out
    assert "PASS" in captured.out
