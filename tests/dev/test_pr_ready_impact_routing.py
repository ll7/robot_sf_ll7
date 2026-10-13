"""Exercise readiness impact routing against real Git change records (#9851)."""

from __future__ import annotations

import shlex
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
FAST_PYSF_RUNTIME_PATHS = (
    "fast-pysf/pysocialforce/runtime.py",
    "fast-pysf/pyproject.toml",
    "fast-pysf/uv.lock",
    "pyproject.toml",
    "uv.lock",
)
FAST_PYSF_EXISTING_LANE_PATHS = (
    "fast-pysf/docs",
    "fast-pysf/examples",
    "fast-pysf/benchmarks",
    "fast-pysf/maps",
    "fast-pysf/tests",
)
FIXTURE_DIRECTORIES = (
    *IMPACT_ROOTS,
    "docs",
    "fast-pysf/pysocialforce",
    *FAST_PYSF_EXISTING_LANE_PATHS,
)
FIXTURE_MANIFESTS = ("pyproject.toml", "uv.lock", "fast-pysf/pyproject.toml", "fast-pysf/uv.lock")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Readiness Test")
    _git(repo, "config", "user.email", "readiness@example.invalid")
    for directory in FIXTURE_DIRECTORIES:
        path = repo / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "keep.txt").write_text(f"keep {directory}\n", encoding="utf-8")
    for manifest in FIXTURE_MANIFESTS:
        path = repo / manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"manifest {manifest}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture base")
    _git(repo, "branch", "base")
    return repo


def _classification_block() -> str:
    source = READY_SCRIPT.read_text(encoding="utf-8")
    return source.split("changed_files=()\n", 1)[1].split(
        "# Validate that every changed test file", 1
    )[0]


def _classify(repo: Path) -> tuple[list[str], list[str]]:
    """Run the production classification block without launching test lanes."""
    shell = "\n".join(
        (
            "set -euo pipefail",
            "BASE_REF=base",
            "is_optional_readiness_path() { return 1; }",
            _classification_block(),
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
    """Deletion and both rename endpoints must route the affected impact root."""
    repo = _fixture_repo(tmp_path)
    old = f"{root}/old.txt"
    (repo / old).write_text(f"old {root}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "add impact fixture")
    _git(repo, "branch", "-f", "base")

    if operation == "add":
        (repo / root / "new.txt").write_text("new\n", encoding="utf-8")
        expected = {f"{root}/new.txt"}
    elif operation == "modify":
        (repo / old).write_text("modified\n", encoding="utf-8")
        expected = {old}
    elif operation == "delete":
        (repo / old).unlink()
        expected = {old}
    elif operation == "rename_out":
        (repo / old).rename(repo / "docs/renamed.txt")
        expected = {old, "docs/renamed.txt"}
    elif operation == "rename_in":
        (repo / "docs/keep.txt").rename(repo / root / "incoming.txt")
        expected = {"docs/keep.txt", f"{root}/incoming.txt"}
    else:
        (repo / old).rename(repo / root / "renamed.txt")
        expected = {old, f"{root}/renamed.txt"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", operation)

    changed, impact = _classify(repo)
    assert set(impact) == {path for path in expected if path.startswith(f"{root}/")}
    if operation in {"delete", "rename_out", "rename_within"}:
        assert old not in changed


def test_deleted_test_file_triggers_lane_without_becoming_pytest_target(tmp_path: Path) -> None:
    """A deleted validation test affects routing but cannot be collected as a target."""
    repo = _fixture_repo(tmp_path)
    removed = repo / "tests/validation/test_removed.py"
    removed.write_text("def test_old(): pass\n", encoding="utf-8")
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
    """A docs-only change is a negative control for impact routing."""
    repo = _fixture_repo(tmp_path)
    (repo / "docs/keep.txt").write_text("docs update\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "docs only")
    _, impact = _classify(repo)
    assert impact == []


@pytest.mark.parametrize("path", FAST_PYSF_RUNTIME_PATHS)
def test_fast_pysf_runtime_and_manifests_route_extended_lane(tmp_path: Path, path: str) -> None:
    """Runtime package and dependency manifests can affect uncovered adapter tests."""
    repo = _fixture_repo(tmp_path)
    changed_path = repo / path
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_text("changed\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fast-pysf runtime impact")

    _, impact = _classify(repo)
    assert path in impact


@pytest.mark.parametrize("directory", FAST_PYSF_EXISTING_LANE_PATHS)
def test_fast_pysf_non_runtime_subtrees_stay_on_existing_lanes(
    tmp_path: Path, directory: str
) -> None:
    """Docs, examples, benchmarks, maps, and fast-pysf tests do not over-trigger."""
    repo = _fixture_repo(tmp_path)
    changed_path = repo / directory / "keep.txt"
    changed_path.write_text("existing lane change\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fast-pysf existing lane change")

    _, impact = _classify(repo)
    assert impact == []


def _run_receipt_block(tmp_path: Path, *, required: bool, trigger_files: tuple[str, ...]) -> str:
    """Execute the shipped lane receipt block with a controlled lane stub."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = READY_SCRIPT.read_text(encoding="utf-8")
    receipt_block = source.split("pr_ready_extended_ran=0\n", 1)[1].split(
        'mark_pr_ready_progress "post_lane_checks"', 1
    )[0]
    trigger_array = " ".join(shlex.quote(path) for path in trigger_files)
    shell = "\n".join(
        (
            "set -euo pipefail",
            f"REPO_ROOT={shlex.quote(str(tmp_path))}",
            "SCRIPT_DIR=/unused",
            "pr_ready_uncovered_test_roots=(tests/validation tests/maps)",
            "optional_changed_files=()",
            f"pr_ready_extended_required={'1' if required else '0'}",
            f"pr_ready_extended_trigger_files=({trigger_array})",
            f"pr_ready_extended_reason={shlex.quote('changed impact paths: ' + ' '.join(trigger_files) if trigger_files else 'no changed paths matched the extended-lane impact roots')}",
            "pr_ready_extended_ran=0",
            "run_pr_ready_lane() { printf 'STUB_RUN:%s\\n' \"$1\"; }",
            "mark_pr_ready_progress() { :; }",
            receipt_block,
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], cwd=tmp_path, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return (tmp_path / "output/validation/pr_ready/lane_coverage.txt").read_text(encoding="utf-8")


def test_lane_coverage_receipt_records_extended_reason_for_run_and_skip(tmp_path: Path) -> None:
    """The production receipt states the extended lane decision and its reason."""
    run_receipt = _run_receipt_block(
        tmp_path / "run", required=True, trigger_files=("robot_sf/removed.py",)
    )
    assert "extended lane:  ran (tests/validation tests/maps)" in run_receipt
    assert "extended lane reason: changed impact paths: robot_sf/removed.py" in run_receipt

    skip_receipt = _run_receipt_block(tmp_path / "skip", required=False, trigger_files=())
    assert "extended lane:  NOT RUN" in skip_receipt
    assert "extended lane reason: no changed paths matched the extended-lane impact roots" in (
        skip_receipt
    )
