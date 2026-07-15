"""Tests for check_optional_import_pr_freshness.py validation logic.

Verifies that:
- The check script succeeds (exit 0) when no optional-import guard changed.
- The check script fails (exit 1) when a guard count changes but the snapshot is not updated in the same diff.
- The check script succeeds (exit 0) when a guard count changes and the snapshot is updated in the same diff.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_optional_import_pr_freshness.py"


@pytest.mark.base_sensitive
def test_script_exits_zero_on_no_changes() -> None:
    """The check script must exit 0 when compared against HEAD with no changes."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"Expected 0 exit, got {result.returncode}. Output:\n{result.stderr}"
    )


@pytest.mark.base_sensitive
def test_script_enforces_snapshot_update_on_change(tmp_path: Path) -> None:
    """The check script must fail when guard count changes but the snapshot is stale, and succeed once updated."""
    # Set up dummy Git repository
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=repo, check=True)

    # 1. Create initial state
    robot_sf = repo / "robot_sf"
    robot_sf.mkdir()
    seed_file = robot_sf / "seed.py"
    seed_file.write_text(
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        encoding="utf-8",
    )

    fixtures = repo / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    snapshot = fixtures / "optional_import_guards.json"
    snapshot.write_text("{}", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo, check=True, capture_output=True
    )

    # 2. Modify robot_sf/seed.py to add another ImportError guard
    seed_file.write_text(
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
        "\n"
        "try:\n"
        "    import another_mod\n"
        "except ImportError:\n"
        "    another_mod = None\n",
        encoding="utf-8",
    )

    # Run check script against initial commit (HEAD).
    # Since tests/fixtures/optional_import_guards.json was NOT updated, this must fail.
    res_fail = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    assert res_fail.returncode == 1, (
        f"Expected exit 1 since snapshot is not updated. Output:\n{res_fail.stdout}\n{res_fail.stderr}"
    )
    assert "Optional-import guard inventory count changed, but" in res_fail.stderr

    # 3. Simulate updating tests/fixtures/optional_import_guards.json in the working tree
    snapshot.write_text('{"dummy_updated": true}', encoding="utf-8")

    # Now the check script should pass since tests/fixtures/optional_import_guards.json is modified in the diff vs HEAD.
    res_pass = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    assert res_pass.returncode == 0, (
        f"Expected exit 0 since snapshot has been modified. Output:\n{res_pass.stdout}\n{res_pass.stderr}"
    )
