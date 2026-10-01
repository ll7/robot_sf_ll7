"""Coverage shards recorded under different runner workspace roots must combine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

coverage = pytest.importorskip("coverage")

REPO_ROOT = Path(__file__).resolve().parents[2]
SELF_HOSTED = "/home/runner/_work/robot_sf_ll7/robot_sf_ll7"
GITHUB_HOSTED = "/home/runner/work/robot_sf_ll7/robot_sf_ll7"


def _write_shard(path: Path, root: str, rel: str, lines: list[int]) -> None:
    data = coverage.CoverageData(basename=str(path))
    data.add_lines({f"{root}/{rel}": lines})
    data.write()


def test_paths_config_combines_self_hosted_and_github_hosted_roots(tmp_path: Path) -> None:
    """Both workspace roots map onto the repository source and merge into one file."""
    # Coverage resolves the first, relative alias against the combine cwd.
    # Give it its own source tree: the real checkout may itself be SELF_HOSTED.
    canonical = tmp_path / "canonical"
    for relative in ("robot_sf/__init__.py", "fast-pysf/pysocialforce/__init__.py"):
        source = canonical / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic coverage source\n", encoding="utf-8")
    shards = tmp_path / "shards"
    shards.mkdir()
    _write_shard(shards / ".coverage.a", SELF_HOSTED, "robot_sf/__init__.py", [1, 2])
    _write_shard(shards / ".coverage.b", GITHUB_HOSTED, "robot_sf/__init__.py", [3])
    _write_shard(
        shards / ".coverage.c",
        SELF_HOSTED,
        "fast-pysf/pysocialforce/__init__.py",
        [1],
    )
    out = tmp_path / ".coverage"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "combine",
            f"--rcfile={REPO_ROOT / 'pyproject.toml'}",
            f"--data-file={out}",
            str(shards),
        ],
        cwd=canonical,
        check=True,
        capture_output=True,
        text=True,
    )
    data = coverage.CoverageData(basename=str(out))
    data.read()
    files = {Path(f).resolve() for f in data.measured_files()}
    init = (canonical / "robot_sf" / "__init__.py").resolve()
    assert init in files
    assert files == {init, (canonical / "fast-pysf/pysocialforce/__init__.py").resolve()}
    assert not any(
        f.startswith((SELF_HOSTED + "/", GITHUB_HOSTED + "/")) for f in data.measured_files()
    )
    assert sorted(data.lines(str(init)) or []) == [1, 2, 3]
