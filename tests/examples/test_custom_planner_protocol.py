"""Focused tests for the custom planner-protocol tutorial (issue #8737)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.protocol import LocalPlannerProtocol
from robot_sf.robot.differential_drive import DifferentialDriveSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "examples/advanced/37_custom_planner_protocol.py"


def _load_tutorial_module():
    """Load the tutorial example by file path (numeric filenames are not importable)."""
    spec = importlib.util.spec_from_file_location("tutorial_planner", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tutorial_planner"] = module
    spec.loader.exec_module(module)
    return module


_tutorial = _load_tutorial_module()


def test_planner_satisfies_protocol() -> None:
    """The tutorial class structurally satisfies LocalPlannerProtocol."""
    assert isinstance(_tutorial.TutorialPlanner(), LocalPlannerProtocol)


def test_same_inputs_produce_same_actions_and_diagnostics() -> None:
    """Same seed and inputs deterministically reproduce actions and diagnostics."""
    first = _tutorial.TutorialPlanner()
    first.reset(seed=11)
    second = _tutorial.TutorialPlanner()
    second.reset(seed=11)
    observation: dict[str, object] = {}
    assert [first.plan(observation) for _ in range(3)] == [
        second.plan(observation) for _ in range(3)
    ]
    assert first.diagnostics() == second.diagnostics()


def test_invalid_commands_fail_with_validation_error() -> None:
    """Wrong shape, non-finite, and out-of-bounds commands raise explicitly."""
    with pytest.raises(ValueError, match="must be a"):
        _tutorial.validate_command((0.3,))
    with pytest.raises(ValueError, match="finite"):
        _tutorial.validate_command((float("nan"), 0.0))
    with pytest.raises(ValueError, match="outside bounds"):
        _tutorial.validate_command((5.0, 0.0))
    with pytest.raises(ValueError, match="outside bounds"):
        _tutorial.validate_command((0.3, 9.0))


def test_close_is_idempotent() -> None:
    """Closing twice must not raise."""
    planner = _tutorial.TutorialPlanner()
    planner.close()
    planner.close()


def test_run_episode_projects_velocity_command_to_default_drive_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tutorial passes velocity/rate commands through the canonical projection."""
    robot = SimpleNamespace(pose=((0.0, 0.0), 0.0), current_speed=(0.0, 0.0))
    config = SimpleNamespace(
        robot_config=DifferentialDriveSettings(),
        sim_config=SimpleNamespace(time_per_step_in_secs=0.2),
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.config = config
            self.simulator = SimpleNamespace(robots=[robot])
            self.actions: list[np.ndarray] = []

        def reset(self):
            return {}, {}

        def step(self, action):
            self.actions.append(action)
            return {}, 0.0, False, False, {}

        def close(self):
            pass

    env = FakeEnv()
    monkeypatch.setattr(_tutorial, "make_robot_env", lambda debug=False: env)
    summary = _tutorial.run_episode(1)

    assert np.allclose(env.actions[0], np.array([1.5, 0.0]))
    assert summary["steps"] == 1


def test_run_episode_closes_planner_when_environment_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner resources are released even when the environment cannot be built."""
    closed = False

    class TrackingPlanner:
        def __init__(self) -> None:
            pass

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(_tutorial, "TutorialPlanner", TrackingPlanner)
    monkeypatch.setattr(
        _tutorial,
        "make_robot_env",
        lambda debug=False: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        _tutorial.run_episode(1)
    assert closed


def test_run_episode_resets_planner_after_internal_environment_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An environment episode reset is followed by the matching planner reset."""
    reset_seeds: list[int | None] = []

    class TrackingPlanner(_tutorial.TutorialPlanner):
        def reset(self, *, seed: int | None = None) -> None:
            reset_seeds.append(seed)
            super().reset(seed=seed)

    class FakeEnv:
        config = SimpleNamespace(
            robot_config=DifferentialDriveSettings(),
            sim_config=SimpleNamespace(time_per_step_in_secs=0.2),
        )
        simulator = SimpleNamespace(
            robots=[SimpleNamespace(pose=((0.0, 0.0), 0.0), current_speed=(0.0, 0.0))]
        )

        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            return {}, {}

        def step(self, action):
            return {}, 0.0, self.reset_count == 1, False, {}

        def close(self):
            pass

    env = FakeEnv()
    monkeypatch.setattr(_tutorial, "TutorialPlanner", TrackingPlanner)
    monkeypatch.setattr(_tutorial, "make_robot_env", lambda debug=False: env)

    summary = _tutorial.run_episode(2)

    assert reset_seeds == [_tutorial.SEED, _tutorial.SEED]
    assert env.reset_count == 2
    assert summary["steps"] == 2


def test_step_budgets_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment and CLI step budgets share a hard upper bound."""
    assert _tutorial._step_budget(10**9) == _tutorial.MAX_STEPS
    monkeypatch.setenv("ROBOT_SF_EXAMPLES_MAX_STEPS", str(10**9))
    assert _tutorial._step_budget(1) == _tutorial.MAX_STEPS
    assert _tutorial.parse_args(["--horizon", str(10**9)]).horizon == _tutorial.MAX_STEPS


def test_manifest_entry_is_registered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The manifest registration check is independent of the current working directory."""
    import yaml

    monkeypatch.chdir(tmp_path)
    manifest_path = REPO_ROOT / "examples/examples_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest["examples"]
        if entry["path"] == "advanced/37_custom_planner_protocol.py"
    ]
    assert len(entries) == 1
    assert entries[0]["ci_enabled"] is True
