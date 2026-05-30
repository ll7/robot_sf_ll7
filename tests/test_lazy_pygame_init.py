"""Regression tests for lazy pygame initialization in robot environments."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace


def _run_probe(code: str) -> str:
    """Run an isolated Python probe and return stdout."""
    env = os.environ.copy()
    env.update(
        {
            "DISPLAY": "",
            "MPLBACKEND": "Agg",
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_headless_robot_env_creation_does_not_import_pygame() -> None:
    """Pure-simulation robot env creation should avoid importing pygame."""
    stdout = _run_probe(
        """
import sys
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=False)
try:
    print("pygame_imported", "pygame" in sys.modules)
    print("sim_ui", getattr(env, "sim_ui", None))
finally:
    env.close()
"""
    )

    assert "pygame_imported False" in stdout
    assert "sim_ui None" in stdout


def test_debug_robot_env_materializes_pygame_on_first_render() -> None:
    """Debug robot envs should keep pygame lazy until the first render call."""
    stdout = _run_probe(
        """
import sys
from robot_sf.gym_env.environment_factory import make_robot_env

env = make_robot_env(debug=True)
try:
    print("pygame_after_create", "pygame" in sys.modules)
    print("sim_ui_present", getattr(env, "sim_ui", None) is not None)
    env.reset(seed=123)
    env.render()
    print("pygame_after_render", "pygame" in sys.modules)
finally:
    env.close()
"""
    )

    assert "pygame_after_create False" in stdout
    assert "sim_ui_present True" in stdout
    assert "pygame_after_render True" in stdout


def test_lazy_simulation_view_replays_pending_attributes(monkeypatch) -> None:
    """LazySimulationView should replay setup mutations on materialization."""
    from robot_sf.render.lazy_sim_view import LazySimulationView

    created = {}

    class FakeSimulationView:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.render_calls = []
            self.manual_view_modes = []
            created["view"] = self

        def set_manual_view_mode(self, view_mode):
            self.manual_view_modes.append(view_mode)

        def render(self, *args, **kwargs):
            self.render_calls.append((args, kwargs))
            return "rendered"

    def fake_import_module(name: str):
        assert name == "robot_sf.render.sim_view"
        return SimpleNamespace(SimulationView=FakeSimulationView)

    monkeypatch.setattr("robot_sf.render.lazy_sim_view.importlib.import_module", fake_import_module)

    view = LazySimulationView(record_video=True, width=320, height=200)
    view.observation_space_mode = "grid"
    view.set_manual_view_mode("ego_up")

    assert bool(view)
    assert view.materialized is False
    assert view.record_video is True
    assert view.render("state", target_fps=12) == "rendered"
    assert view.materialized is True

    materialized = created["view"]
    assert materialized.kwargs["width"] == 320
    assert materialized.observation_space_mode == "grid"
    assert materialized.manual_view_modes == ["ego_up"]
    assert materialized.render_calls == [(("state",), {"target_fps": 12})]


def test_lazy_simulation_view_propagates_tracked_attribute_mutations(monkeypatch) -> None:
    """Tracked view attributes should stay synchronized before and after materialization."""
    from robot_sf.render.lazy_sim_view import LazySimulationView

    created = {}

    class FakeSimulationView:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.width = kwargs["width"]
            self.height = kwargs["height"]
            self.record_video = kwargs["record_video"]
            created["view"] = self

        def render(self):
            return "rendered"

    def fake_import_module(name: str):
        assert name == "robot_sf.render.sim_view"
        return SimpleNamespace(SimulationView=FakeSimulationView)

    monkeypatch.setattr("robot_sf.render.lazy_sim_view.importlib.import_module", fake_import_module)

    view = LazySimulationView(record_video=False, width=320, height=200)
    view.width = 640
    view.record_video = True

    assert view.render() == "rendered"

    materialized = created["view"]
    assert materialized.kwargs["width"] == 640
    assert materialized.kwargs["record_video"] is True

    view.height = 480

    assert materialized.height == 480


def test_lazy_simulation_view_forwards_manual_view_mode_after_materialization(monkeypatch) -> None:
    """Manual view mode changes should forward to an already materialized SimulationView."""
    from robot_sf.render.lazy_sim_view import LazySimulationView

    created = {}

    class FakeSimulationView:
        def __init__(self, **_kwargs):
            self.manual_view_modes = []
            created["view"] = self

        def set_manual_view_mode(self, view_mode):
            self.manual_view_modes.append(view_mode)

        def render(self):
            return "rendered"

    def fake_import_module(name: str):
        assert name == "robot_sf.render.sim_view"
        return SimpleNamespace(SimulationView=FakeSimulationView)

    monkeypatch.setattr("robot_sf.render.lazy_sim_view.importlib.import_module", fake_import_module)

    view = LazySimulationView(record_video=False)
    assert view.render() == "rendered"
    view.set_manual_view_mode("ego_up")

    assert created["view"].manual_view_modes == ["ego_up"]


def test_pysocialforce_visualization_export_remains_available() -> None:
    """The bundled pysocialforce package should still expose SimulationView lazily."""
    import pysocialforce

    assert pysocialforce.SimulationView.__name__ == "SimulationView"
