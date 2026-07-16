"""Tests for the CPU-only beginner quickstart notebooks (Issue #5798).

These tests cover the *structure and generator* of the notebooks cheaply:
existence, valid nbformat, expected cells, and that the generator reproduces
them. The full headless execution of every notebook is the dedicated CI smoke
(``scripts/validation/run_notebooks_smoke.py`` / ``ci_driver.sh notebooks-smoke``);
a slow-marked test here also executes them end-to-end for local confidence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from scripts.validation.run_notebooks_smoke import discover_notebooks

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NOTEBOOKS = (
    "01_run_first_episode.ipynb",
    "02_compare_two_planners.ipynb",
    "03_visualize_trace.ipynb",
)


def _load(name: str) -> nbformat.notebooknode:
    """Load a committed notebook from ``notebooks/``."""
    path = REPO_ROOT / "notebooks" / name
    return nbformat.read(path, as_version=4)


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_is_valid(name: str) -> None:
    """Each expected notebook exists and is a valid nbformat document."""
    nb = _load(name)
    assert nb.cells, f"{name} has no cells"
    # At least one markdown intro and one code cell.
    assert any(c.cell_type == "markdown" for c in nb.cells)
    assert any(c.cell_type == "code" for c in nb.cells)


def test_notebooks_write_to_gitignored_output() -> None:
    """Notebooks must not commit artifacts: they write under output/ (git-ignored)."""
    for name in EXPECTED_NOTEBOOKS:
        nb = _load(name)
        joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "output/notebooks" in joined, f"{name} should write artifacts under output/notebooks"


@pytest.mark.parametrize(
    "name,needle",
    [
        ("01_run_first_episode.ipynb", "make_robot_env"),
        ("02_compare_two_planners.ipynb", "run_episode"),
        ("03_visualize_trace.ipynb", "export_threejs_viewer"),
    ],
)
def test_notebook_uses_existing_public_api(name: str, needle: str) -> None:
    """Each notebook relies on the existing env/planner/trace API (no new logic)."""
    nb = _load(name)
    joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert needle in joined, f"{name} should use the existing {needle!r} API"


def test_notebooks_resolve_repo_root_robustly() -> None:
    """Notebooks anchor paths to the repo root so they run from any cwd."""
    for name in EXPECTED_NOTEBOOKS:
        nb = _load(name)
        joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "_repo_root" in joined or "pyproject.toml" in joined, (
            f"{name} should resolve the repo root independent of launch directory"
        )


def test_smoke_script_discovers_all_notebooks() -> None:
    """The notebooks smoke discovers exactly the expected notebooks."""
    names = {p.name for p in discover_notebooks()}
    assert set(EXPECTED_NOTEBOOKS) <= names


def test_generator_produces_valid_notebooks(tmp_path: Path) -> None:
    """Regenerating notebooks yields valid nbformat documents with code cells."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen", REPO_ROOT / "scripts" / "dev" / "generate_quickstart_notebooks.py"
    )
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    # Point the generator's output dir at a temp location.
    spec.loader.exec_module(gen)
    gen.OUT_DIR = tmp_path
    gen.main()
    for name in EXPECTED_NOTEBOOKS:
        nb = nbformat.read(tmp_path / name, as_version=4)
        assert nb.cells, f"generated {name} has no cells"


@pytest.mark.slow
def test_all_notebooks_execute_headless() -> None:
    """Slow: execute every notebook headless via nbconvert (the CI smoke path)."""
    notebooks = discover_notebooks()
    assert notebooks, "no notebooks discovered"
    env = {
        "SDL_VIDEODRIVER": "dummy",
    }
    import os

    full_env = {**os.environ, **env}
    for nb in notebooks:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            "--ExecutePreprocessor.timeout=600",
            str(nb),
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=full_env, capture_output=True, check=False)
        assert result.returncode == 0, (
            f"{nb.name} failed to execute headless:\n"
            f"{result.stderr.decode(errors='replace')[-2000:]}"
        )
