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


@pytest.mark.parametrize("hosted_checkout", [False, True])
def test_paths_config_combines_self_hosted_and_github_hosted_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hosted_checkout: bool
) -> None:
    """Both workspace roots map onto the repository source and merge into one file."""
    if hosted_checkout:
        checkout = tmp_path / "home/runner/_work/robot_sf_ll7/robot_sf_ll7"
        for rel in ("robot_sf/__init__.py", "fast-pysf/pysocialforce/__init__.py"):
            source = checkout / rel
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes((REPO_ROOT / rel).read_bytes())
        (checkout / "pyproject.toml").write_bytes((REPO_ROOT / "pyproject.toml").read_bytes())
        monkeypatch.setitem(globals(), "REPO_ROOT", checkout)

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
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = coverage.CoverageData(basename=str(out))
    data.read()
    files = {Path(f).resolve() for f in data.measured_files()}
    init = (REPO_ROOT / "robot_sf" / "__init__.py").resolve()
    # A canonical checkout can itself live under /home/runner in hosted CI.
    # Require exactly the local source paths, regardless of the checkout's root.
    assert files == {
        init,
        (REPO_ROOT / "fast-pysf/pysocialforce/__init__.py").resolve(),
    }
    assert sorted(data.lines(str(init)) or []) == [1, 2, 3]
