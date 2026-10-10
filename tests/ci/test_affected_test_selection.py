"""Exercise PR admission through real pytest collection and temporary Git changes."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_changed_slow_pin_is_admitted(tmp_path):
    """A slow pin must execute when its production path changes."""
    pin = tmp_path / "test_temporary_affected_pin.py"
    pin.write_text(
        'import pytest\npytestmark = pytest.mark.slow\nPIN = "robot_sf/benchmark/metrics.py"\ndef test_pin(): assert False, "pin executed"\n'
    )
    env = {**os.environ, "ROBOT_SF_AFFECTED_TEST_PATHS": str(pin), "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            str(pin),
            "-m",
            "not slow or affected",
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "pin executed" in result.stdout


def test_imports_pins_deletes_and_renames(tmp_path):
    """Use Git's real diff and exclude deleted test paths from execution."""
    from scripts.dev.affected_test_selection import affected_tests, changed_paths

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    files = {
        "robot_sf/old.py": "VALUE = 1\n",
        "robot_sf/bridge.py": "from .old import VALUE\n",
        "robot_sf/schema.json": "{}\n",
        "tests/test_import.py": "from robot_sf.bridge import VALUE\n",
        "tests/test_pin.py": 'PIN = "robot_sf" / "schema.json"\n',
        "tests/test_directory.py": 'PIN = "robot_sf/schemas"\n',
        "tests/test_unrelated.py": 'PIN = "robot_sf/other.json"\n',
        "tests/test_deleted.py": "pass\n",
        "robot_sf/schemas/gone.json": "{}\n",
    }
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git("add", *files)
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "robot_sf/old.py").rename(tmp_path / "robot_sf/new.py")
    (tmp_path / "robot_sf/schema.json").write_text('{"changed": true}\n')
    (tmp_path / "robot_sf/schemas/gone.json").unlink()
    (tmp_path / "tests/test_deleted.py").unlink()
    git("add", *files, "robot_sf/new.py")
    git("commit", "-m", "change")
    paths = changed_paths(tmp_path, base)
    assert {"robot_sf/old.py", "robot_sf/new.py", "robot_sf/schemas/gone.json"} <= paths
    assert affected_tests(tmp_path, paths) == [
        "tests/test_directory.py",
        "tests/test_import.py",
        "tests/test_pin.py",
        "tests/test_unrelated.py",
    ]


def test_mapped_change_excludes_unrelated_slow_and_always_admits_pins(tmp_path):
    """Mapped paths select real dependents without paying for unrelated slow files."""
    from scripts.dev.affected_test_selection import affected_tests

    files = {
        "robot_sf/model.py": "VALUE = 1\n",
        "tests/test_model.py": "from robot_sf.model import VALUE\n",
        "tests/test_reference.py": 'PIN = "robot_sf" / "model.py"\n',
        "tests/test_pin.py": "def test_pin(): pass\n",
        "tests/test_inventory.py": "def test_inventory(): pass\n",
        "tests/test_unrelated.py": "import pytest\npytestmark = pytest.mark.slow\n",
        "tests/scoped/conftest.py": "from robot_sf.model import VALUE\n",
        "tests/scoped/test_fixture.py": "def test_fixture(): pass\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *files], cwd=tmp_path, check=True)
    assert affected_tests(tmp_path, {"robot_sf/model.py"}) == [
        "tests/scoped/test_fixture.py",
        "tests/test_inventory.py",
        "tests/test_model.py",
        "tests/test_pin.py",
        "tests/test_reference.py",
    ]
    assert affected_tests(tmp_path, set()) == ["tests/test_inventory.py", "tests/test_pin.py"]


def test_mapped_direct_consumer_does_not_hide_dynamic_slow_consumer(tmp_path):
    """A static consumer cannot prove an unselected dynamic slow consumer is safe."""
    from scripts.dev.affected_test_selection import selection_reasons

    files = {
        "configs/a.yaml": "ok\n",
        "tests/test_direct.py": (
            "PIN = 'configs/a.yaml'\n"
            "def test_direct():\n"
            "    assert PIN\n"
        ),
        "tests/test_dynamic_slow.py": (
            "from pathlib import Path\n"
            "import pytest\n"
            "pytestmark = pytest.mark.slow\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "def _config_path():\n"
            "    return ''.join(chr(value) for value in "
            "[99, 111, 110, 102, 105, 103, 115, 47, 97, 46, 121, 97, 109, 108])\n"
            "def test_dynamic_consumer():\n"
            "    assert (ROOT / _config_path()).read_text() == 'ok\\n'\n"
        ),
        "tests/test_unrelated_slow.py": (
            "import pytest\n"
            "pytestmark = pytest.mark.slow\n"
            "def test_unrelated():\n"
            "    assert True\n"
        ),
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", *files], cwd=tmp_path, check=True)

    mode, reasons = selection_reasons(tmp_path, {"configs/a.yaml"})

    assert mode == "full"
    assert set(reasons) == {
        "tests/test_direct.py",
        "tests/test_dynamic_slow.py",
        "tests/test_unrelated_slow.py",
    }
    assert reasons["tests/test_dynamic_slow.py"].startswith(
        "fallback: unproven slow-test dependency"
    )


def test_unchanged_pr_still_admits_inventory(tmp_path):
    """Every PR executes integrity witnesses even when its diff is empty."""
    from scripts.dev.affected_test_selection import affected_tests

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_inventory.py").write_text("def test_inventory(): pass\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "tests/test_inventory.py"], cwd=tmp_path, check=True)
    assert affected_tests(tmp_path, set()) == ["tests/test_inventory.py"]
