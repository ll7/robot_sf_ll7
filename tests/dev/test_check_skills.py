"""Regression tests for the repo-local skill checker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_check_skills_module():
    """Load scripts/dev/check_skills.py without requiring scripts to be a package."""
    module_path = Path(__file__).parents[2] / "scripts/dev/check_skills.py"
    spec = importlib.util.spec_from_file_location("check_skills", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_artifact_first_contract_passes_for_goal_autopilot(tmp_path: Path) -> None:
    """Artifact-first phrase and file requirements should pass for goal-autopilot style skills."""
    check_skills = _load_check_skills_module()
    check_skills.REPO_ROOT = tmp_path
    skill_path = tmp_path / "goal-autopilot" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text("---\nname: goal-autopilot\n---\n", encoding="utf-8")
    body = """
Artifact-first delegated review requires result.json, RESULT.md, diffstat.txt, and validation.json.
Treat worker exit success as route evidence only. Read raw logs only if artifacts are missing
or inconsistent.
The parent must inspect route evidence and run targeted local checks.
Worker output uses rg -l, rg --files, bounded sed -n, a 200 lines cap, private artifacts,
no broad rg -n ., and no full file reads.
The active ledger records loaded context with skill/doc summaries, snapshot paths,
freshness keys, expected PR head SHA, worker artifact paths, and stale-state triggers.
Use compact_worktree_snapshot.py and compact_ci_snapshot.py before broad worktree or CI polling.
Pass ledger snapshot paths to workers, avoid repeating broad state polling, and run
fresh live checks before issue claim, push, PR publication, label/project mutation,
or merge-ready decisions.
Spark sidecar routing supports gpt-5.3-codex-spark for tiny lookup, read-only review,
docs cross-check, and issue/file surface mapping. Spark prompts must report files inspected,
exact evidence, uncertainty, and recommended next prompt. Do not use Spark for GitHub mutation
or shell-executable fallback.
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
    assert any("artifact-first phrase requirement" in e for e in errors)
    assert any("worker-output limit requirement" in e for e in errors)
    assert any("active-ledger requirement" in e for e in errors)
    assert any("Spark sidecar routing requirement" in e for e in errors)


def test_artifact_first_contract_requires_canonical_result_markdown_case(
    tmp_path: Path,
) -> None:
    """The compact artifact contract should preserve RESULT.md casing exactly."""
    check_skills = _load_check_skills_module()
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

    assert any("active-ledger requirement" in e for e in errors)


def test_goal_pr_review_contract_requires_compact_ci_snapshot_terms(tmp_path: Path) -> None:
    """Goal PR review should preserve compact PR/CI entry-point guidance."""
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
Start with snapshot_pr_queue.py, poll with watch_pr_ci_status.py, inspect status,conclusion,jobs.
Return bounded excerpts, and keep full logs in private artifacts.
"""
    errors = check_skills._validate_artifact_first_contract(
        skill_path,
        {"name": "goal-pr-review"},
        body,
    )

    assert errors == []


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
