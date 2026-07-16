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


def test_notebook_01_seeds_action_space_explicitly() -> None:
    """Notebook 01 must seed the factory, reset, and action space so the trace is reproducible."""
    nb = _load("01_run_first_episode.ipynb")
    joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "make_robot_env(debug=False, seed=SEED)" in joined, (
        "notebook 01 should seed the factory via make_robot_env(seed=SEED)"
    )
    assert "env.reset(seed=SEED)" in joined, "notebook 01 should seed reset explicitly"
    assert "env.action_space.seed(SEED)" in joined, (
        "notebook 01 should seed the Gymnasium action space so action sampling is reproducible"
    )


def test_notebook_01_does_not_call_set_global_seed() -> None:
    """The redundant set_global_seed call crashes the kernel on the installed Torch/TF stack."""
    nb = _load("01_run_first_episode.ipynb")
    joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from robot_sf.common.seed import set_global_seed" not in joined, (
        "notebook 01 must not import set_global_seed (kernel-crash path)"
    )
    assert "set_global_seed(" not in joined, (
        "notebook 01 must not call set_global_seed (kernel-crash path)"
    )


def test_notebook_01_action_reward_trace_is_reproducible() -> None:
    """Two seeded runs of notebook 01's policy must produce identical action/reward traces."""
    from robot_sf.gym_env.environment_factory import make_robot_env

    def run_trace() -> list[tuple[float, float]]:
        SEED = 87234
        env = make_robot_env(debug=False, seed=SEED)
        env.reset(seed=SEED)
        env.action_space.seed(SEED)
        rewards: list[tuple[float, float]] = []
        for _ in range(1, 61):
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            rewards.append((float(action[0]), float(reward)))
            if terminated or truncated:
                env.reset(seed=SEED)
                env.action_space.seed(SEED)
        env.exit()
        return rewards

    assert run_trace() == run_trace(), (
        "notebook 01 seeded action/reward trace must be identical across runs"
    )


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


def test_notebook_03_does_not_glob_and_pick_last_recording() -> None:
    """Notebook 03 must not rely on glob-last JSONL discovery, which silently selects a stale artifact.

    The previous `candidates[-1]` lookup picked the lexicographically-last file in the
    recordings directory; a higher episode id left by a prior run was selected instead of
    the freshly recorded episode, producing a cryptic `IndexError`/`ValueError`.
    """
    nb = _load("03_visualize_trace.ipynb")
    joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "candidates[-1]" not in joined, (
        "notebook 03 must not use glob-and-pick-last (candidates[-1]) recording discovery"
    )
    assert "last_recorded_jsonl" in joined, (
        "notebook 03 should locate its recording via env.last_recorded_jsonl (deterministic)"
    )
    assert "FileNotFoundError" in joined, (
        "notebook 03 should fail clearly when no recording was produced"
    )


def test_notebook_03_locates_fresh_recording_with_stale_present(tmp_path: Path) -> None:
    """Reproduces Issue #5827: a stale higher-episode-id recording must NOT be selected.

    We record one deterministic episode, then drop a stale `ep0001` file next to the fresh
    `ep0000`, and assert the discovery selects the *fresh* file rather than the stale one.
    """
    import numpy as np

    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.training.scenario_loader import (
        build_robot_config_from_scenario,
        load_scenarios,
    )

    scenario_name = "quickstart_demo_crossing_basic"
    scenario_path = REPO_ROOT / "configs/scenarios/single/quickstart_demo.yaml"
    scenario = next(s for s in load_scenarios(scenario_path) if s["name"] == scenario_name)
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)

    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    # Arm a stale higher episode id, simulating a prior run's leftover artifact.
    stale = recording_dir / "notebook_quickstart_demo_crossing_basic_random_270_ep0001.jsonl"
    stale.write_text('{"event": "step", "state": {}}\n')

    env = make_robot_env(
        config=config,
        seed=270,
        debug=False,
        recording_enabled=True,
        use_jsonl_recording=True,
        recording_dir=str(recording_dir),
        suite_name="notebook",
        scenario_name=scenario_name,
        algorithm_name="random",
        recording_seed=270,
    )
    try:
        env.reset(seed=270)
        for _ in range(5):
            _obs, _rew, terminated, truncated, _info = env.step(np.zeros(2, dtype=np.float32))
            if terminated or truncated:
                break
        env.end_episode_recording()
        assert env.last_recorded_jsonl is not None, (
            "env.last_recorded_jsonl must point at the recorded file after a recording"
        )
        recorded = env.last_recorded_jsonl
        assert Path(recorded).exists(), f"recorded file missing: {recorded}"
        assert "ep0000" in Path(recorded).name, (
            f"fresh recording should be ep0000, got {Path(recorded).name}"
        )
        assert Path(recorded).name != stale.name, (
            "discovery must not select the stale higher episode id"
        )
    finally:
        env.close_recorder()
        env.exit()
