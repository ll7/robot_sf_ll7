"""Tests for the broad-exception inventory ratchet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "check_broad_exceptions.py"


def _git(repo: Path, *args: str) -> None:
    """Run a git command in a temporary repository."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    """Create a tiny tracked repository with one ratcheted Python file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    target = repo / "scripts" / "demo" / "tool.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except RuntimeError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_broad_exception_ratchet_passes_generated_baseline(tmp_path: Path) -> None:
    """A freshly generated baseline passes without source changes."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)

    write_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert write_result.returncode == 0, write_result.stderr

    check_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check_result.returncode == 0, check_result.stderr
    assert "Broad exception ratchet passed with 0 entries." in check_result.stdout


def test_broad_exception_ratchet_fails_on_added_broad_catch(tmp_path: Path) -> None:
    """Adding an unapproved broad catch fails against the existing baseline."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    (repo / "scripts" / "demo" / "tool.py").write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Broad exception count increased from 0 to 1." in result.stderr
    assert "Unapproved broad exception handlers were added:" in result.stderr
    assert "scripts/demo/tool.py:4: except Exception:" in result.stderr


def test_broad_exception_ratchet_fails_on_unapproved_replacement(tmp_path: Path) -> None:
    """Moving a broad catch to a new fingerprint fails even when count is unchanged."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)
    script = repo / "scripts" / "demo" / "tool.py"
    script.write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    script.write_text(
        "def helper() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Broad exception count increased" not in result.stderr
    assert "Unapproved broad exception handlers were added:" in result.stderr
    assert "scripts/demo/tool.py:4: except Exception:" in result.stderr
