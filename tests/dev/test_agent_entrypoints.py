"""Tests for task-scoped context entrypoints and canonical task route contracts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO_ROOT / "AGENTS.md"
AGENTS_README = REPO_ROOT / ".agents" / "README.md"
ENTRYPOINTS_DOC = REPO_ROOT / "docs" / "ai" / "agent_workflow_entrypoints.md"
RELOCATED_GUIDANCE = REPO_ROOT / "docs" / "dev" / "agents" / "relocated-agents-guidance.md"
CAMERA_READY_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_camera_ready_benchmark.py"
CREATE_WORKTREE = REPO_ROOT / "scripts" / "dev" / "create_worktree.sh"
REVIEW_GUARD = REPO_ROOT / "scripts" / "dev" / "review_worktree_guard.py"
SHARED_VENV_WRAPPER = "scripts/dev/run_worktree_shared_venv.sh --"

CANONICAL_ROUTES = (
    "Read-only observation",
    "Documentation-only edit",
    "Implementation / runtime change",
    "Scientific / benchmark interpretation",
    "Environment / worktree repair",
)


def test_agent_workflow_entrypoints_defines_five_canonical_routes() -> None:
    """The entrypoints document must define the 5 canonical routes with required contract columns."""
    assert ENTRYPOINTS_DOC.is_file(), f"Missing {ENTRYPOINTS_DOC}"
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "## Task Routes And Preflight Discipline" in text

    # Verify each canonical route is present in the route table
    for route in CANONICAL_ROUTES:
        assert route in text, f"Missing canonical route in table: {route}"

    # Extract the route table rows
    table_match = re.search(
        r"\| Route \| Purpose \| Required context / evidence \| First deterministic command \| "
        r"Permitted mutations \| Authoritative acceptance command \|\n"
        r"\| (?:--- \| ){5}---\ \|\n"
        r"((?:\| \*\*.*?\*\* \| .*? \|\n)+)",
        text,
    )
    assert table_match is not None, "Failed to locate standard task route table format"
    table_body = table_match.group(1)

    for route in CANONICAL_ROUTES:
        assert f"**{route}**" in table_body, f"Route row not found: {route}"


def test_agent_workflow_entrypoints_commands_exist_or_valid() -> None:
    """Preflight and acceptance commands in the route table must point to real repository scripts."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    referenced_scripts = (
        "scripts/dev/watch_pr_ci_status.py",
        "scripts/tools/sync_ai_config.py",
        "scripts/dev/pr_ready_check.sh",
        "scripts/dev/check_worktree_capacity.py",
        "scripts/dev/check_worktree_optional_deps.py",
    )
    for script_rel in referenced_scripts:
        assert script_rel in text, f"Script not mentioned in entrypoints doc: {script_rel}"
        script_path = REPO_ROOT / script_rel
        assert script_path.is_file(), f"Referenced route script not found: {script_rel}"


def test_scientific_acceptance_route_uses_parser_backed_preflight() -> None:
    """The documented scientific acceptance route must use the real preflight CLI."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert (
        f"{SHARED_VENV_WRAPPER} uv run python scripts/tools/run_camera_ready_benchmark.py "
        "--config configs/benchmarks/camera_ready_baseline_safe.yaml --mode preflight"
    ) in text
    assert "robot_sf.benchmark.camera_ready_campaign --verify-only" not in text

    result = subprocess.run(
        [
            sys.executable,
            str(CAMERA_READY_RUNNER),
            "--config",
            str(REPO_ROOT / "configs/benchmarks/camera_ready_baseline_safe.yaml"),
            "--verify-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --verify-only" in result.stderr


def test_scientific_acceptance_route_dispatches_preflight(monkeypatch, tmp_path, capsys) -> None:
    """The documented ``--mode preflight`` path calls the canonical verifier."""
    from scripts.tools import run_camera_ready_benchmark

    config_path = tmp_path / "entrypoint-config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")
    sentinel_config = object()
    called: dict[str, object] = {}

    def fake_load_campaign_config(path: Path) -> object:
        assert path == config_path
        return sentinel_config

    def fake_prepare_campaign_preflight(config: object, **kwargs: object) -> dict[str, object]:
        assert config is sentinel_config
        called.update(kwargs)
        return {
            "campaign_id": "entrypoint-preflight",
            "campaign_root": tmp_path / "output",
            "validate_config_path": tmp_path / "output" / "validate_config.json",
            "preview_scenarios_path": tmp_path / "output" / "preview_scenarios.json",
            "matrix_summary_json_path": tmp_path / "output" / "matrix_summary.json",
            "matrix_summary_csv_path": tmp_path / "output" / "matrix_summary.csv",
            "amv_coverage_json_path": tmp_path / "output" / "amv_coverage.json",
            "amv_coverage_md_path": tmp_path / "output" / "amv_coverage.md",
            "comparability_json_path": None,
            "comparability_md_path": None,
        }

    monkeypatch.setattr(
        run_camera_ready_benchmark, "load_campaign_config", fake_load_campaign_config
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "prepare_campaign_preflight",
        fake_prepare_campaign_preflight,
    )
    monkeypatch.setattr(
        run_camera_ready_benchmark,
        "run_campaign",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight route must not run the benchmark campaign")
        ),
    )

    exit_code = run_camera_ready_benchmark.main(
        ["--config", str(config_path), "--mode", "preflight"]
    )

    assert exit_code == 0
    assert called["campaign_id"] is None
    assert json.loads(capsys.readouterr().out)["campaign_id"] == "entrypoint-preflight"


def test_documented_python_commands_use_project_environment() -> None:
    """Entrypoint guidance must not reintroduce bare Python interpreters."""
    for path in (AGENTS_MD, ENTRYPOINTS_DOC):
        text = path.read_text(encoding="utf-8")
        assert "python3" not in text
        assert re.search(r"(?m)^\s*python(?:3)?(?:\s|$)", text) is None


def test_fresh_worktree_commands_use_shared_environment_wrapper() -> None:
    """Fresh linked-worktree commands must show the complete shared-venv wrapper."""
    agents_text = AGENTS_MD.read_text(encoding="utf-8")
    agents_fresh_section = agents_text.split("## Fresh Worktree Bootstrap", maxsplit=1)[1].split(
        "If the current branch is not `main`", maxsplit=1
    )[0]
    for line in agents_fresh_section.splitlines():
        if "uv run" in line:
            assert SHARED_VENV_WRAPPER in line

    entrypoints_text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")
    route_table = entrypoints_text.split("## Task Routes And Preflight Discipline", maxsplit=1)[
        1
    ].split("### Protected read-only worktrees", maxsplit=1)[0]
    for line in route_table.splitlines():
        if "uv run" in line:
            assert SHARED_VENV_WRAPPER in line

    # All bash command examples on this page use the wrapper, except the explicit
    # external-project resolver, whose separate environment is intentional.
    for block in re.findall(r"```bash\n(.*?)\n```", entrypoints_text, flags=re.DOTALL):
        for line in block.splitlines():
            if "uv run" in line and "--project" not in line:
                assert SHARED_VENV_WRAPPER in line

    assert (
        f"{SHARED_VENV_WRAPPER} uv run pytest -q tests/dev/test_check_skills.py" in entrypoints_text
    )


def test_read_only_route_requires_explicit_protected_worktree_mode(tmp_path: Path) -> None:
    """A read-only linked checkout must be created with review mode and its guard enabled."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")
    protected_section = text.split("### Protected read-only worktrees", maxsplit=1)[1].split(
        "### Route Boundaries and Negative Rules", maxsplit=1
    )[0]
    assert 'MAIN_REPO_ROOT="$(git rev-parse --show-toplevel)"' in protected_section
    assert 'WORKTREE_PARENT="${WORKTREE_PARENT:-' in protected_section
    assert 'mkdir -p "$WORKTREE_PARENT"' in protected_section
    assert '! -d "$WORKTREE_PARENT"' in protected_section
    assert '! -w "$WORKTREE_PARENT"' in protected_section
    assert "create_worktree.sh" in protected_section
    assert "--mode review" in protected_section
    assert "robot-sf.worktree-mode=review" in protected_section

    repo = tmp_path / "repo"
    worktree = tmp_path / "review"
    branch = "review/entrypoint-contract"
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(repo)],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "agent-entrypoint-test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "agent-entrypoint@example.invalid"],
        check=True,
    )
    (repo / "fixture.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "fixture.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        capture_output=True,
        text=True,
        check=True,
    )

    code_block_match = re.search(r"```bash\n(?P<commands>.*?)\n```", protected_section, re.DOTALL)
    assert code_block_match is not None
    setup_commands = code_block_match.group("commands").split(
        "scripts/dev/create_worktree.sh", maxsplit=1
    )[0]
    fresh_env = os.environ.copy()
    fresh_env.pop("WORKTREE_PARENT", None)
    setup = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + setup_commands
            + '\nprintf "%s\\n" "$MAIN_REPO_ROOT" "$WORKTREE_PARENT"\n',
        ],
        cwd=repo,
        env=fresh_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert setup.returncode == 0, setup.stdout + setup.stderr
    expected_parent = repo.parent / f"{repo.name}.worktrees"
    assert setup.stdout.splitlines() == [str(repo), str(expected_parent)]
    assert expected_parent.is_dir()
    assert os.access(expected_parent, os.W_OK)

    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("fixture", encoding="utf-8")
    rejected_env = fresh_env | {"WORKTREE_PARENT": str(invalid_parent)}
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n" + setup_commands,
        ],
        cwd=repo,
        env=rejected_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode == 2
    assert "WORKTREE_PARENT must be an existing writable directory" in rejected.stderr

    try:
        created = subprocess.run(
            [
                str(CREATE_WORKTREE),
                "--path",
                str(worktree),
                "--branch",
                branch,
                "--base",
                "HEAD",
                "--minimum-free-bytes",
                "0",
                "--mode",
                "review",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert created.returncode == 0, created.stdout + created.stderr

        mode = subprocess.run(
            ["git", "-C", str(worktree), "config", "--get", "robot-sf.worktree-mode"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert mode.stdout.strip() == "review"

        guard = subprocess.run(
            [sys.executable, str(REVIEW_GUARD), "pre-push", "--worktree", str(worktree)],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert guard.returncode == 1, guard.stdout + guard.stderr
        payload = json.loads(guard.stdout)
        assert payload["mode"] == "review"
        assert payload["blocked"] is True
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(repo), "branch", "-D", branch],
            capture_output=True,
            text=True,
            check=False,
        )


def test_agent_workflow_entrypoints_documents_route_boundaries_and_negative_rules() -> None:
    """The entrypoints document must explicitly define the negative boundaries and linked owners."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "### Route Boundaries and Negative Rules" in text

    # Negative rule 1: Read-only review never mutates branches (ref #8321)
    assert "review_worktree_guard.py" in text
    assert "#8321" in text
    assert "never merge `origin/main` into the implementation branch" in text

    # Negative rule 2: Validation proportional to change risk
    assert "Validation proportional to change risk" in text

    # Negative rule 3: Environment blockers fail closed
    assert "Environment blockers are not relaxation licenses" in text
    assert "never authorizes lowering scientific gates" in text

    # Negative rule 4: Freshness before expensive proof (#7649)
    assert "Freshness before expensive proof" in text
    assert "#7649" in text

    # Negative rule 5: Observer / audit separation (#8304, #8307)
    assert "Separation of observer/audit collection from mutations" in text
    assert "#8304" in text
    assert "#8307" in text

    # Negative rule 6: Scientific indicator integrity
    assert "Integrity of scientific indicators" in text

    # Negative rule 7: Privacy and provenance boundaries
    assert "Privacy and provenance boundaries" in text


def test_compact_final_handoff_contract_fields() -> None:
    """The handoff contract must specify all standard acceptance elements."""
    text = ENTRYPOINTS_DOC.read_text(encoding="utf-8")

    assert "## Compact Final Handoff Contract" in text
    required_handoff_fields = (
        "Result",
        "Revisions",
        "Changed paths",
        "Validation evidence",
        "Unrun or unavailable checks",
        "Scientific scope & limitations",
        "Next disposition",
    )
    for field in required_handoff_fields:
        assert f"**{field}**" in text, f"Missing handoff contract field: {field}"


def test_agents_md_task_scoped_context_and_mode_specific_sync() -> None:
    """AGENTS.md must define task-scoped context entrypoints and mode-specific branch sync."""
    text = AGENTS_MD.read_text(encoding="utf-8")

    assert "## Task-Scoped Context Entrypoints" in text
    assert "Always-required core context:" in text
    assert "docs/maintainer_values.md" in text
    assert "AGENTS.md" in text
    assert "docs/ai/agent_workflow_entrypoints.md" in text

    # Verify mode-specific branch sync distinction
    assert "branch synchronization is mode-specific:" in text
    assert "For implementation worktrees, fetch latest `origin/main` and merge it early" in text
    assert "never merge `origin/main` into the implementation branch" in text
    assert "review_worktree_guard.py" in text
    assert "#8321" in text

    passive_review_line = next(
        line
        for line in text.splitlines()
        if line.startswith("- **Read-only observation / review**:")
    )
    assert ".agents/skills/implementation-verification/SKILL.md" in passive_review_line
    assert ".agents/skills/goal-pr-review/SKILL.md" not in passive_review_line


def test_relocated_guidance_mode_specific_sync() -> None:
    """relocated-agents-guidance.md must also reflect mode-specific branch sync."""
    text = RELOCATED_GUIDANCE.read_text(encoding="utf-8")

    assert "branch synchronization is mode-specific:" in text
    assert "never merge `origin/main` into the implementation branch" in text


def test_agents_readme_references_task_routes_and_mode_specific_sync() -> None:
    """.agents/README.md must reference task routes and mode-specific sync."""
    text = AGENTS_README.read_text(encoding="utf-8")

    assert "task-scoped context" in text
    assert "task route selection" in text
    assert "Branch synchronization is mode-specific" in text
