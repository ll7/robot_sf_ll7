"""Shared fixtures for development-script tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_FOCUSED_TESTS = REPO_ROOT / "scripts" / "dev" / "run_focused_tests.sh"


@pytest.fixture()
def helper_repo(tmp_path: Path) -> Path:
    """Return a tiny git repo with the focused-test helper and a fake uv binary."""
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(RUN_FOCUSED_TESTS, scripts_dir / "run_focused_tests.sh")
    shutil.copy2(REPO_ROOT / "scripts" / "dev" / "common_setup.sh", scripts_dir / "common_setup.sh")
    shutil.copy2(
        REPO_ROOT / "scripts" / "dev" / "clean_generated_output.py",
        scripts_dir / "clean_generated_output.py",
    )
    (repo / "output" / "coverage").mkdir(parents=True)
    (repo / "output" / "coverage" / "tracked.txt").write_text("tracked", encoding="utf-8")
    (repo / "bin").mkdir()
    (repo / "bin" / "uv").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "mkdir -p output/coverage",
                "printf fake > output/coverage/generated.txt",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "bin" / "uv").chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.com",
            "add",
            "scripts/dev/run_focused_tests.sh",
            "scripts/dev/common_setup.sh",
            "scripts/dev/clean_generated_output.py",
            "bin/uv",
            "output/coverage/tracked.txt",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.com",
            "commit",
            "-m",
            "seed tracked focused test fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo
