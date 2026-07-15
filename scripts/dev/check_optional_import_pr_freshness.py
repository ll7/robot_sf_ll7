#!/usr/bin/env python3
"""PR readiness gate: ensure optional-import guard snapshot is refreshed when inventory changes.

Verifies that any PR adding, removing, or changing optional-import exception occurrences
under robot_sf/ also contains an update to tests/fixtures/optional_import_guards.json.
This prevents leaving origin/main red for unrelated branches when a PR changes optional guards.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_MODULE = SCRIPT_DIR.parents[1] / "tests" / "test_optional_import_guard_inventory.py"


def get_git_root() -> Path:
    """Get the root directory of the current Git repository."""
    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(res.stdout.strip()).resolve()


def _load_collector_helpers() -> tuple[object, object, object, set[str]]:
    """Load AST collector helpers from the inventory test module."""
    if not TEST_MODULE.exists():
        raise FileNotFoundError(f"Inventory test module not found at {TEST_MODULE}")
    spec = importlib.util.spec_from_file_location("_optional_import_ratchet_collector", TEST_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load collector from {TEST_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Clean up sys.modules to prevent cache pollution
    sys.modules.pop(spec.name, None)

    return (
        module._caught_type_names,
        module._spelling_key,
        module._has_pragma,
        module._TRACKED,
    )


_caught_type_names, _spelling_key, _has_pragma, _TRACKED = _load_collector_helpers()


def count_guards_in_source(src: str, filepath: str) -> dict[str, int]:
    """Parse python source code and count occurrences of each optional-import spelling."""
    counts: dict[str, int] = {}
    if not src:
        return counts
    try:
        tree = ast.parse(src, filename=filepath)
    except SyntaxError:
        return counts

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        names = _caught_type_names(node)
        if not (_TRACKED & set(names)):
            continue
        key = _spelling_key(names)
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_merge_base(base_ref: str) -> str:
    """Resolve the common ancestor used for a base-versus-HEAD comparison."""
    res = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    merge_base = res.stdout.strip()
    if not merge_base:
        raise RuntimeError(f"Could not resolve a merge base between {base_ref!r} and HEAD")
    return merge_base


def get_git_diff_files(base_commit: str) -> list[str]:
    """Get Python files in ``robot_sf/`` changed from the merge base to the worktree."""
    # Modified/added/deleted tracked files
    res = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "--", "robot_sf/"],
        capture_output=True,
        text=True,
        check=True,
    )
    changed = [
        line.strip()
        for line in res.stdout.splitlines()
        if line.strip() and line.strip().endswith(".py")
    ]

    # Untracked files
    res_untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "robot_sf/"],
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = [
        line.strip()
        for line in res_untracked.stdout.splitlines()
        if line.strip() and line.strip().endswith(".py")
    ]

    return sorted(set(changed + untracked))


def get_base_content(base_commit: str, filepath: str) -> str:
    """Read the content of a file at the merge base from git."""
    res = subprocess.run(
        ["git", "show", f"{base_commit}:{filepath}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
        return res.stdout
    return ""


def is_snapshot_modified(base_commit: str) -> bool:
    """True if the snapshot has changes relative to the merge base."""
    res = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            base_commit,
            "--",
            "tests/fixtures/optional_import_guards.json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(res.stdout.strip())


def check_freshness(base_commit: str, git_root: Path) -> list[tuple[str, int, int]]:
    """Compare guard occurrences between the merge base and current worktree."""
    changed_files = get_git_diff_files(base_commit)
    if not changed_files:
        return []

    head_totals: dict[str, int] = {}
    base_totals: dict[str, int] = {}

    for filepath in changed_files:
        # Read HEAD content from disk (if exists)
        disk_path = git_root / filepath
        if disk_path.is_file():
            try:
                head_src = disk_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                head_src = ""
        else:
            head_src = ""

        # Read BASE content from git
        base_src = get_base_content(base_commit, filepath)

        head_counts = count_guards_in_source(head_src, filepath)
        base_counts = count_guards_in_source(base_src, filepath)

        for k, count in head_counts.items():
            head_totals[k] = head_totals.get(k, 0) + count
        for k, count in base_counts.items():
            base_totals[k] = base_totals.get(k, 0) + count

    # Determine if any spelling count has changed (increased or decreased or added/removed)
    mismatched_spellings = []
    all_keys = set(head_totals.keys()) | set(base_totals.keys())
    for key in sorted(all_keys):
        head_c = head_totals.get(key, 0)
        base_c = base_totals.get(key, 0)
        if head_c != base_c:
            mismatched_spellings.append((key, base_c, head_c))

    return mismatched_spellings


def main() -> int:
    """CLI entry point to verify optional-import guard snapshot freshness in PRs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base ref to compare against (default: origin/main)",
    )
    args = parser.parse_args()

    base_ref = args.base_ref
    git_root = get_git_root()

    # Resolve base ref to verify it exists
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        print(
            f"check_optional_import_pr_freshness: BASE_REF '{base_ref}' "
            "does not resolve to a valid commit. Skipping check."
        )
        return 0

    comparison_base = get_merge_base(base_ref)
    mismatched_spellings = check_freshness(comparison_base, git_root)
    if not mismatched_spellings:
        return 0

    # Mismatch detected: requires snapshot update
    if is_snapshot_modified(comparison_base):
        return 0

    print(
        "ERROR: Optional-import guard inventory count changed, but "
        "tests/fixtures/optional_import_guards.json was not updated in this diff.",
        file=sys.stderr,
    )
    print(f"BASE_REF: {base_ref}", file=sys.stderr)
    print("Mismatched counts in changed files:", file=sys.stderr)
    for key, base_c, head_c in mismatched_spellings:
        print(
            f"  - '{key}': count in changed files went from {base_c} to {head_c}",
            file=sys.stderr,
        )
    print("\nTo fix, please run:", file=sys.stderr)
    print("  uv run python scripts/dev/generate_optional_import_snapshot.py", file=sys.stderr)
    print("And add tests/fixtures/optional_import_guards.json to your commit.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
