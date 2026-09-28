"""Exercise readiness impact routing against real Git change records (#9851)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
READY_SCRIPT = ROOT / "scripts/dev/pr_ready_check.sh"
IMPACT_ROOTS = (
    "robot_sf",
    "configs",
    "maps",
    "scripts/validation",
    "tests/validation",
    "tests/maps",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture_repo(tmp_path: Path, root: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Readiness Test")
    _git(repo, "config", "user.email", "readiness@example.invalid")
    for directory in (*IMPACT_ROOTS, "docs"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
        (repo / directory / "keep.txt").write_text(f"keep {directory}\n")
    (repo / root / "old.txt").write_text(f"old {root}\n")
    (repo / "docs/from.txt").write_text("rename source from docs\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture base")
    _git(repo, "branch", "base")
    return repo


def _classify(repo: Path) -> tuple[list[str], list[str]]:
    """Run the production classification block without launching test lanes."""
    source = READY_SCRIPT.read_text()
    block = (
        "changed_files=()\n"
        + source.split("changed_files=()\n", 1)[1].split(
            "# Validate that every changed test file", 1
        )[0]
    )
    shell = "\n".join(
        (
            "set -euo pipefail",
            "BASE_REF=base",
            "is_optional_readiness_path() { return 1; }",
            block,
            'for path in "${changed_files[@]}"; do printf "CHANGED:%s\\n" "$path"; done',
            'for path in "${pr_ready_extended_trigger_files[@]}"; do printf "IMPACT:%s\\n" "$path"; done',
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], cwd=repo, check=True, capture_output=True, text=True
    )
    changed = [
        line.removeprefix("CHANGED:")
        for line in result.stdout.splitlines()
        if line.startswith("CHANGED:")
    ]
    impact = [
        line.removeprefix("IMPACT:")
        for line in result.stdout.splitlines()
        if line.startswith("IMPACT:")
    ]
    return changed, impact


@pytest.mark.parametrize("root", IMPACT_ROOTS)
@pytest.mark.parametrize(
    "operation", ("add", "modify", "delete", "rename_out", "rename_in", "rename_within")
)
def test_impact_root_routes_every_git_change(tmp_path: Path, root: str, operation: str) -> None:
    repo = _fixture_repo(tmp_path, root)
    old = f"{root}/old.txt"
    if operation == "add":
        (repo / root / "new.txt").write_text("new\n")
        expected = {f"{root}/new.txt"}
    elif operation == "modify":
        (repo / old).write_text("modified\n")
        expected = {old}
    elif operation == "delete":
        (repo / old).unlink()
        expected = {old}
    elif operation == "rename_out":
        (repo / old).rename(repo / "docs/renamed.txt")
        expected = {old, "docs/renamed.txt"}
    elif operation == "rename_in":
        (repo / "docs/from.txt").rename(repo / root / "incoming.txt")
        expected = {"docs/from.txt", f"{root}/incoming.txt"}
    else:
        (repo / old).rename(repo / root / "renamed.txt")
        expected = {old, f"{root}/renamed.txt"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", operation)

    _, impact = _classify(repo)
    assert set(impact) == {path for path in expected if path.startswith(f"{root}/")}


def test_deleted_test_file_triggers_lane_without_becoming_pytest_target(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, "tests/validation")
    removed = repo / "tests/validation/test_removed.py"
    removed.write_text("def test_old(): pass\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add test to delete")
    _git(repo, "branch", "-f", "base")
    removed.unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "delete test")

    changed, impact = _classify(repo)
    assert "tests/validation/test_removed.py" in impact
    assert "tests/validation/test_removed.py" not in changed


def test_docs_only_change_does_not_route_extended_lane(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, "robot_sf")
    (repo / "docs/keep.txt").write_text("docs update\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "docs only")
    _, impact = _classify(repo)
    assert impact == []


def test_fast_pysf_only_change_does_not_route_extended_lane(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, "robot_sf")
    fast_pysf = repo / "fast-pysf"
    fast_pysf.mkdir()
    (fast_pysf / "module.py").write_text("value = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fast-pysf only")
    _, impact = _classify(repo)
    assert impact == []
