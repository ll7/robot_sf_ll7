"""Tests for the CPU-only beginner quickstart notebooks (Issue #5798).

These tests cover the *structure and generator* of the notebooks cheaply:
existence, valid nbformat, expected cells, and that the generator reproduces
them. The full headless execution of every notebook is the dedicated CI smoke
(``scripts/validation/run_notebooks_smoke.py`` / ``ci_driver.sh notebooks-smoke``);
a slow-marked test here also executes them end-to-end for local confidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
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
        ("01_run_first_episode.ipynb", "from robot_sf import make_env"),
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
    assert "make_env(debug=False, seed=SEED)" in joined, (
        "notebook 01 should seed the factory via the public facade make_env(seed=SEED)"
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
        env.close()
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


def test_generator_preserves_close_lifecycle_in_canonical_notebooks(tmp_path: Path) -> None:
    """Generated notebooks must keep the public close() lifecycle contract."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_lifecycle", REPO_ROOT / "scripts" / "dev" / "generate_quickstart_notebooks.py"
    )
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.OUT_DIR = tmp_path
    gen.main()

    for name in ("01_run_first_episode.ipynb", "03_visualize_trace.ipynb"):
        nb = nbformat.read(tmp_path / name, as_version=4)
        joined = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "env.close()" in joined, f"generated {name} must close its environment"
        assert "env.exit()" not in joined, f"generated {name} must not reintroduce env.exit()"


def test_generator_is_byte_reproducible_in_fresh_directories(tmp_path: Path) -> None:
    """Fresh generator runs must not randomize notebook cell IDs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_reproducible", REPO_ROOT / "scripts" / "dev" / "generate_quickstart_notebooks.py"
    )
    assert spec and spec.loader
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    outputs: list[dict[str, bytes]] = []
    for directory in (tmp_path / "first", tmp_path / "second"):
        gen.OUT_DIR = directory
        gen.main()
        outputs.append({path.name: path.read_bytes() for path in directory.glob("*.ipynb")})

    assert outputs[0] == outputs[1]


_FACADE_OWNED_INTERNAL_MODULES = (
    "robot_sf.gym_env.environment_factory",
    "robot_sf.gym_env.robot_env",
    "robot_sf.training.scenario_loader",
)
_INTERNAL_IMPORT_MARKER = "# internal-import-exception:"


def _code_source(name: str) -> str:
    """Return the joined code-cell source of one committed notebook."""
    nb = _load(name)
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebooks_use_public_facade_for_owned_operations(name: str) -> None:
    """Internal modules owned by the public facade must not appear in notebooks."""
    joined = _code_source(name)
    for module in _FACADE_OWNED_INTERNAL_MODULES:
        assert f"from {module} import" not in joined, (
            f"{name} must use the public facade instead of {module}"
        )
    assert "from robot_sf import" in joined, f"{name} should import the supported facade"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_internal_imports_are_documented_exceptions(name: str) -> None:
    """Every remaining internal import must name why no facade equivalent exists."""
    for lineno, line in enumerate(_code_source(name).splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("from robot_sf."):
            continue
        assert _INTERNAL_IMPORT_MARKER in stripped, (
            f"{name}:{lineno} internal import lacks {_INTERNAL_IMPORT_MARKER!r}: {stripped}"
        )


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
        env.close()


def _load_notebook_generator():
    """Load the notebook generator module from the repository scripts tree."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_setup_cleanup", REPO_ROOT / "scripts" / "dev" / "generate_quickstart_notebooks.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_cell_source(nb, needle: str) -> str:
    """Return the first generated code cell containing ``needle``."""

    for cell in nb.cells:
        if cell.cell_type == "code" and needle in cell.source:
            return cell.source
    raise AssertionError(f"no generated code cell contains {needle!r}")


class _FakeRaisingEnv:
    """Environment whose setup reset always fails, recording close calls."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.action_space = types.SimpleNamespace(seed=lambda *args, **kwargs: None)

    def reset(self, **kwargs):
        """Fail during setup to exercise the cleanup path."""
        raise RuntimeError("setup boom")

    def close(self) -> None:
        """Record the cleanup call."""
        self.close_calls += 1


def test_notebook_01_setup_failure_closes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #8745 review: a failed reset in notebook 01 must close the environment."""

    gen = _load_notebook_generator()
    nb = gen.build_notebook_01()
    setup_source = _generated_cell_source(nb, "os.environ.setdefault")
    env_source = _generated_cell_source(nb, "env.reset(seed=SEED)")
    fake_env = _FakeRaisingEnv()

    import robot_sf

    monkeypatch.setattr(robot_sf, "make_env", lambda **kwargs: fake_env)
    namespace: dict = {}
    exec(compile(setup_source, "<notebook-01-setup>", "exec"), namespace)  # noqa: S102

    with pytest.raises(RuntimeError, match="setup boom"):
        exec(compile(env_source, "<notebook-01-env>", "exec"), namespace)  # noqa: S102

    assert fake_env.close_calls == 1, "failed reset must close the created environment"


def test_notebook_03_setup_failure_closes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #8745 review: failed setup in notebook 03 must close the environment."""

    gen = _load_notebook_generator()
    nb = gen.build_notebook_03()
    setup_source = _generated_cell_source(nb, "os.environ.setdefault")
    env_source = _generated_cell_source(nb, "env = make_env(")
    fake_env = _FakeRaisingEnv()

    import robot_sf
    from robot_sf.baselines import random_policy

    class _FakePlanner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def reset(self, **kwargs) -> None:
            pass

    monkeypatch.setattr(robot_sf, "make_env", lambda **kwargs: fake_env)
    monkeypatch.setattr(robot_sf, "load_scenario", lambda name: {"name": name})
    monkeypatch.setattr(random_policy, "RandomPlanner", _FakePlanner)
    namespace: dict = {}
    exec(compile(setup_source, "<notebook-03-setup>", "exec"), namespace)  # noqa: S102

    with pytest.raises(RuntimeError, match="setup boom"):
        exec(compile(env_source, "<notebook-03-env>", "exec"), namespace)  # noqa: S102

    assert fake_env.close_calls == 1, "failed setup must close the created environment"


def test_canonical_notebook_strips_transient_state() -> None:
    """Canonicalization removes ids, outputs, execution counts, and env metadata."""
    gen = _load_notebook_generator()
    nb = gen.build_notebook_01()
    nb.cells[0].id = "abc123"
    nb.cells[0].metadata["kernelspec"] = {"display_name": "Local Python"}
    nb.metadata["kernelspec"] = {
        "display_name": "Local Python",
        "language": "python",
        "name": "python3",
    }
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    code_cells[0]["execution_count"] = 7
    code_cells[0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "leak"}]

    code_index = next(index for index, cell in enumerate(nb.cells) if cell.cell_type == "code")
    canonical = gen._canonical_notebook(nb)
    payload = json.dumps(canonical)

    assert "abc123" not in payload
    assert "execution_count" not in json.dumps(canonical["cells"][code_index])
    assert "output_type" not in payload
    assert canonical["metadata"]["kernelspec"] == {"language": "python", "name": "python3"}
    assert gen._transient_state_issues(nb) == [
        f"cells[{code_index}].outputs",
        f"cells[{code_index}].execution_count",
    ]


def test_parity_report_matches_committed_notebooks() -> None:
    """The committed notebooks are canonical generator output."""
    gen = _load_notebook_generator()

    report = gen.notebook_parity_report()

    assert report["schema"] == "quickstart_notebook_parity.v1"
    assert report["status"] == "match"
    assert report["mismatch_count"] == 0
    assert len(report["notebooks"]) == 3


def test_parity_detects_cell_source_drift(tmp_path: Path) -> None:
    """A manual source edit is reported with the exact mismatch path."""
    gen = _load_notebook_generator()
    gen.OUT_DIR = tmp_path
    gen.main()
    target = tmp_path / "01_run_first_episode.ipynb"
    nb = nbformat.read(target, as_version=4)
    code_cell = next(cell for cell in nb.cells if cell.cell_type == "code")
    code_cell.source = code_cell.source + "\n# manual edit\n"
    nbformat.write(nb, target)

    report = gen.notebook_parity_report(out_dir=tmp_path)

    entry = next(item for item in report["notebooks"] if item["notebook"] == target.name)
    assert entry["status"] == "drift"
    assert "cell_source_changed" in entry["reason_codes"]
    assert any(path.endswith(".source") for path in entry["mismatch_paths"])


def test_parity_detects_committed_execution_state(tmp_path: Path) -> None:
    """Committed outputs or execution counts fail the parity check."""
    gen = _load_notebook_generator()
    gen.OUT_DIR = tmp_path
    gen.main()
    target = tmp_path / "01_run_first_episode.ipynb"
    nb = nbformat.read(target, as_version=4)
    code_cell = next(cell for cell in nb.cells if cell.cell_type == "code")
    code_cell.execution_count = 1
    code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="executed")]
    nbformat.write(nb, target)

    report = gen.notebook_parity_report(out_dir=tmp_path)

    entry = next(item for item in report["notebooks"] if item["notebook"] == target.name)
    assert entry["status"] == "drift"
    assert "transient_state_present" in entry["reason_codes"]
    assert entry["transient_state_issues"]


def test_parity_detects_missing_notebook(tmp_path: Path) -> None:
    """A missing committed notebook fails closed."""
    gen = _load_notebook_generator()

    report = gen.notebook_parity_report(("01_run_first_episode.ipynb",), out_dir=tmp_path)

    assert report["status"] == "drift"
    entry = report["notebooks"][0]
    assert entry["status"] == "missing"
    assert entry["reason_codes"] == ["missing_committed_notebook"]


def test_parity_ignores_transient_ids_and_display_names(tmp_path: Path) -> None:
    """Transient ids and environment-specific kernelspec names do not drift."""
    gen = _load_notebook_generator()
    gen.OUT_DIR = tmp_path
    gen.main()
    target = tmp_path / "02_compare_two_planners.ipynb"
    nb = nbformat.read(target, as_version=4)
    for index, cell in enumerate(nb.cells):
        cell.id = f"transient-{index}"
    nb.metadata["kernelspec"] = {
        "display_name": "Different local kernel",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(nb, target)

    report = gen.notebook_parity_report(out_dir=tmp_path)

    assert report["status"] == "match"
    assert all(entry["status"] == "match" for entry in report["notebooks"])


def test_parity_report_is_deterministic(tmp_path: Path) -> None:
    """Repeated parity reports are equal."""
    gen = _load_notebook_generator()
    gen.OUT_DIR = tmp_path
    gen.main()

    assert gen.notebook_parity_report(out_dir=tmp_path) == gen.notebook_parity_report(
        out_dir=tmp_path
    )
