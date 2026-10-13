"""Exercise readiness impact routing against real Git change records (#9851)."""

from __future__ import annotations

import shlex
import shutil
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
            'for path in "${pr_ready_extended_trigger_files[@]}"; do printf "IMPACT:%q\\n" "$path"; done',
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


def _run_classification_with_stubbed_impact_producer(
    repo: Path, *, mode: str
) -> subprocess.CompletedProcess[str]:
    """Run production classification while stubbing only its Git impact producer."""
    shell = "\n".join(
        (
            "set -euo pipefail",
            "BASE_REF=base",
            "is_optional_readiness_path() { return 1; }",
            f"IMPACT_PRODUCER_MODE={shlex.quote(mode)}",
            "git() {",
            '  if [[ "$1" == diff && "$2" == --name-status && "$3" == -z &&',
            '    "$4" == --find-renames && "$5" == --diff-filter=ACDMRT ]]; then',
            '    case "$IMPACT_PRODUCER_MODE" in',
            "      producer_failure) printf 'D\\0robot_sf/removed.py\\0'; return 7 ;;",
            "      truncated_record) printf 'R100\\0tests/validation/old.txt\\0tests/validation/new.txt'; return 0 ;;",
            "      truncated_status) printf 'D'; return 0 ;;",
            "    esac",
            "  fi",
            '  command git "$@"',
            "}",
            _classification_block(),
        )
    )
    return subprocess.run(
        ["bash", "-c", shell], cwd=repo, check=False, capture_output=True, text=True
    )


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


@pytest.mark.parametrize("root", ("tests/validation", "tests/maps"))
def test_disappeared_extended_root_still_routes_deleted_path(tmp_path: Path, root: str) -> None:
    """A deleted last file still affects routing after its root vanishes."""
    repo = _fixture_repo(tmp_path)
    shutil.rmtree(repo / root)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "delete final extended root")

    changed, impact = _classify(repo)
    assert changed == []
    assert impact == [f"{root}/keep.txt"]


@pytest.mark.parametrize("root", ("tests/validation", "tests/maps"))
def test_rename_out_last_extended_root_still_routes_source_path(tmp_path: Path, root: str) -> None:
    """A rename out that removes the root still routes the old endpoint."""
    repo = _fixture_repo(tmp_path)
    source = repo / root / "keep.txt"
    source.rename(repo / "docs/renamed-from-extended-root.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "rename final extended root out")

    changed, impact = _classify(repo)
    assert f"{root}/keep.txt" not in changed
    assert impact == [f"{root}/keep.txt"]


@pytest.mark.parametrize(
    "mode, expected_exit, expected_message",
    (
        (
            "producer_failure",
            7,
            "Cannot resolve readiness impact scope from Git diff; refusing to classify the extended lane.",
        ),
        (
            "truncated_record",
            2,
            "Malformed Git impact record: missing rename destination for status R100.",
        ),
        ("truncated_status", 2, "Malformed Git impact record: truncated NUL framing."),
    ),
)
def test_impact_producer_failures_do_not_become_no_impact(
    tmp_path: Path, mode: str, expected_exit: int, expected_message: str
) -> None:
    """Git producer and NUL framing failures fail before lane classification."""
    repo = _fixture_repo(tmp_path)
    result = _run_classification_with_stubbed_impact_producer(repo, mode=mode)

    assert result.returncode == expected_exit
    assert expected_message in result.stderr
    assert "REASON:" not in result.stdout
    assert "no changed paths matched" not in result.stdout
    assert not (repo / "output/validation/pr_ready/lane_coverage.txt").exists()


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


def _run_classification_and_receipt(tmp_path: Path, *, change: str) -> tuple[str, str]:
    """Run production classification into the receipt block with only lane stubs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = _fixture_repo(tmp_path)
    if change == "delete":
        removed = repo / "robot_sf/removed.py"
        removed.write_text("removed\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "add deleted impact fixture")
        _git(repo, "branch", "-f", "base")
        removed.unlink()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "delete impact fixture")
    elif change == "docs":
        (repo / "docs/keep.txt").write_text("docs update\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "docs-only fixture")
    elif change == "delete_root":
        shutil.rmtree(repo / "tests/validation")
        shutil.rmtree(repo / "tests/maps")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "delete final extended roots")
    elif change == "rename_root_out":
        validation_source = repo / "tests/validation/keep.txt"
        maps_source = repo / "tests/maps/keep.txt"
        validation_source.rename(repo / "docs/renamed-from-root.txt")
        maps_source.rename(repo / "docs/renamed-from-maps-root.txt")
        validation_source.parent.rmdir()
        maps_source.parent.rmdir()
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "rename final extended roots out")
    elif change == "newline_path":
        changed_path = repo / "robot_sf/line\nname.py"
        changed_path.write_text("original newline path\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "add newline impact fixture")
        _git(repo, "branch", "-f", "base")
        changed_path.write_text("changed newline path\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "modify newline impact fixture")
    else:
        raise ValueError(f"unsupported fixture change: {change}")

    source = READY_SCRIPT.read_text(encoding="utf-8")
    classification_block = _classification_block()
    receipt_block = (
        "pr_ready_extended_ran=0\n"
        + source.split("pr_ready_extended_ran=0\n", 1)[1].split(
            'mark_pr_ready_progress "post_lane_checks"', 1
        )[0]
    )
    shell = "\n".join(
        (
            "set -euo pipefail",
            "BASE_REF=base",
            "is_optional_readiness_path() { return 1; }",
            classification_block,
            "printf 'REQUIRED:%s\\n' \"$pr_ready_extended_required\"",
            "printf 'REASON:%s\\n' \"$pr_ready_extended_reason\"",
            'for path in "${pr_ready_extended_trigger_files[@]}"; do printf \'IMPACT:%q\\n\' "$path"; done',
            f"REPO_ROOT={shlex.quote(str(repo))}",
            "SCRIPT_DIR=/unused",
            "run_pr_ready_lane() { printf 'STUB_RUN:%s\\n' \"$1\"; }",
            "mark_pr_ready_progress() { :; }",
            receipt_block,
        )
    )
    result = subprocess.run(
        ["bash", "-c", shell], cwd=repo, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    receipt = (repo / "output/validation/pr_ready/lane_coverage.txt").read_text(encoding="utf-8")
    return result.stdout + result.stderr, receipt


def test_lane_coverage_receipt_records_extended_reason_for_run_and_skip(tmp_path: Path) -> None:
    """Production classification feeds the lane decision and reason into its receipt."""
    run_output, run_receipt = _run_classification_and_receipt(tmp_path / "run", change="delete")
    assert "REQUIRED:1" in run_output
    assert "REASON:changed impact paths: robot_sf/removed.py" in run_output
    assert "IMPACT:robot_sf/removed.py" in run_output
    assert "STUB_RUN:extended" in run_output
    assert "extended lane:  ran (tests/validation tests/maps)" in run_receipt
    assert "extended lane reason: changed impact paths: robot_sf/removed.py" in run_receipt

    skip_output, skip_receipt = _run_classification_and_receipt(tmp_path / "skip", change="docs")
    assert "REQUIRED:0" in skip_output
    assert "REASON:no changed paths matched the extended-lane impact roots" in skip_output
    assert "STUB_RUN:extended" not in skip_output
    assert "extended lane:  NOT RUN" in skip_receipt
    assert "extended lane reason: no changed paths matched the extended-lane impact roots" in (
        skip_receipt
    )


def test_lane_coverage_receipt_quotes_newline_impact_path(tmp_path: Path) -> None:
    """Production receipt rendering keeps a newline-bearing impact path on one line."""
    run_output, run_receipt = _run_classification_and_receipt(tmp_path, change="newline_path")
    quoted_path = "$'robot_sf/line\\nname.py'"
    expected_reason = f"changed impact paths: {quoted_path}"

    assert f"REASON:{expected_reason}\n" in run_output
    assert f"IMPACT:{quoted_path}\n" in run_output
    assert "STUB_RUN:extended" in run_output
    reason_lines = [
        line for line in run_receipt.splitlines() if line.startswith("  extended lane reason:")
    ]
    assert reason_lines == [f"  extended lane reason: {expected_reason}"]
    assert "robot_sf/line\nname.py" not in run_receipt


@pytest.mark.parametrize(
    "change, expected_paths",
    (
        (
            "delete_root",
            ("tests/maps/keep.txt", "tests/validation/keep.txt"),
        ),
        (
            "rename_root_out",
            ("tests/maps/keep.txt", "tests/validation/keep.txt"),
        ),
    ),
)
def test_lane_coverage_receipt_records_required_impact_without_surviving_roots(
    tmp_path: Path, change: str, expected_paths: tuple[str, ...]
) -> None:
    """A required impact with no executable roots is recorded without a false run."""
    run_output, run_receipt = _run_classification_and_receipt(tmp_path, change=change)

    assert "REQUIRED:1" in run_output
    for expected_path in expected_paths:
        assert f"IMPACT:{expected_path}" in run_output
    assert "STUB_RUN:extended" not in run_output
    assert "extended lane:  NOT RUN (no surviving test roots)" in run_receipt
    assert "extended lane reason: changed impact paths:" in run_receipt
    for expected_path in expected_paths:
        assert expected_path in run_receipt
    assert "extended lane disposition: required impact; no surviving tests" in run_receipt
