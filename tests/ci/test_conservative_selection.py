"""Uncertain input mappings must never remove an unrelated slow witness."""

import subprocess

import pytest

from scripts.dev.affected_test_selection import affected_tests
from tests.support.environment_guards import configure_git_identity


@pytest.mark.parametrize(
    "changed",
    [
        "notebooks/01_run_first_episode.ipynb",
        "examples/advanced/app.py",
        "maps/svg_maps/corridor.svg",
        "configs/scenarios/corridor.yaml",
        "robot_sf/schema.json",
        "tests/conftest.py",
        "tests/nested/conftest.py",
        "pyproject.toml",
        "uv.lock",
        "setup.cfg",
        ".github/actions/setup/action.yml",
        ".github/workflows/ci.yml",
        "scripts/dev/tool.py",
        "tools/app.py",
        "utilities/app.py",
        "robot_sf_carla_bridge/app.py",
        "SLURM/runner.py",
        "benchmarks/app.py",
        "fast-pysf/pysocialforce/forces.py",
        "robot_sf/pinned.py",
        "outside/removed.py",
    ],
)
def test_uncertain_changes_include_every_test(tmp_path, changed):
    """Data hops, implicit fixtures, vendoring and literal pins select the full set."""
    files = {
        "tests/test_slow.py": "import pytest\npytestmark = pytest.mark.slow\ndef test_witness(): assert True\n",
        "tests/test_pin.py": 'PIN = "pinned.py"\nDIRECTORY = "notebooks"\n',
        "tests/test_unrelated.py": "def test_other(): assert True\n",
        "configs/scenarios/corridor.yaml": "map_file: maps/svg_maps/corridor.svg\n",
        "fast-pysf/tests/test_force.py": "from pysocialforce.forces import Force\n",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    configure_git_identity(tmp_path)
    subprocess.run(["git", "add", *files], cwd=tmp_path, check=True)
    expected = {
        "tests/test_slow.py",
        "tests/test_pin.py",
        "tests/test_unrelated.py",
        "fast-pysf/tests/test_force.py",
    }
    assert expected <= set(affected_tests(tmp_path, {changed}))
