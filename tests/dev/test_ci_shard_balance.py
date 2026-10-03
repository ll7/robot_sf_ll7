"""Exercise shard scheduling through the shell driver and real pytest-split."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cache_mode", ["cold", "partial", "warm"])
def test_run_tests_parallel_spreads_unknown_prefix_and_preserves_partition(
    tmp_path: Path, cache_mode: str
) -> None:
    """A missing prefix must spread, with each real collected node selected once."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts" / "dev"
    scripts.mkdir(parents=True)
    for name in ("run_tests_parallel.sh", "common_setup.sh"):
        target = scripts / name
        target.write_bytes((ROOT / "scripts" / "dev" / name).read_bytes())
        target.chmod(0o755)
    support = repo / "tests" / "support"
    support.mkdir(parents=True)
    (support / "optional_test_allowlist.txt").write_text("", encoding="utf-8")
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    items_source = (
        "import pytest\n"
        "@pytest.mark.parametrize('index', range(5))\n"
        "def test_00_unknown(index): assert index >= 0\n"
        "@pytest.mark.parametrize('index', range(5))\n"
        "def test_10_known(index): assert index >= 0\n"
    )
    for filename in ("test_first.py", "test_second.py"):
        (repo / filename).write_text(items_source, encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def pytest_collection_finish(session):\n"
        " Path(os.environ['SELECTED']).write_text(json.dumps([i.nodeid for i in session.items]))\n",
        encoding="utf-8",
    )
    fake_bin = repo / "fake-bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\nset -eu\n"
        '[ "$1" = run ] && [ "$2" = pytest ] || exit 99\n'
        'shift 2\nexec "$REAL_PYTHON" -m pytest -p pytest_split.plugin "$@"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    expected = {
        f"{filename}::test_{kind}[{index}]"
        for filename in ("test_first.py", "test_second.py")
        for kind in ("00_unknown", "10_known")
        for index in range(5)
    }
    cache = (
        {}
        if cache_mode == "cold"
        else {node: 1.0 for node in expected if cache_mode == "warm" or "10_known" in node}
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_NUM_WORKERS": "1",
        "PYTEST_SHARD_COUNT": "5",
        "ROBOT_SF_SHARD_INCLUDE_SLOW": "1",
        "ROBOT_SF_PYTEST_COVERAGE": "0",
        "PYTEST_DEBUG_TEMPROOT": "fixture",
        "CI": "true",
        "PYTEST_ORDER_MODE": "failed-first",
    }
    groups = []
    for index in range(1, 6):
        (repo / ".test_durations").write_text(json.dumps(cache), encoding="utf-8")
        # Each job can restore a different failure snapshot. Equal test names
        # in separate modules must still form the same partition.
        lastfailed = repo / ".pytest_cache/v/cache/lastfailed"
        lastfailed.parent.mkdir(parents=True, exist_ok=True)
        lastfailed.write_text(
            json.dumps({f"test_second.py::test_00_unknown[{index - 1}]": True}),
            encoding="utf-8",
        )
        selected = tmp_path / f"selected-{index}.json"
        result = subprocess.run(
            [
                str(scripts / "run_tests_parallel.sh"),
                "--collect-only",
                "-q",
                "test_first.py",
                "test_second.py",
            ],
            cwd=repo,
            env={**env, "PYTEST_SHARD_INDEX": str(index), "SELECTED": str(selected)},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        groups.append(set(json.loads(selected.read_text(encoding="utf-8"))))
    assert set.union(*groups) == expected
    assert sum(map(len, groups)) == len(expected)
    assert [sum("00_unknown" in node for node in group) for group in groups] == [2] * 5


def test_preregistration_source_path_node_ids_are_checkout_independent() -> None:
    """Absolute source-path test values must not become checkout-dependent shard IDs."""
    prefix = "tests/validation/test_issue_6969_stage_b_preregistration.py::test_non_relative_source_path_fails_closed"
    env = {**os.environ, "PYTEST_ADDOPTS": ""}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", prefix],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert collected == [f"{prefix}[absolute]", f"{prefix}[parent]"]
