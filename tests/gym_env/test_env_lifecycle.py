"""Lifecycle tests for the environment ``close()``/``exit()`` contract (issue #8245)."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pytest

import robot_sf.gym_env.base_env as base_env_module
from robot_sf.gym_env.base_env import BaseEnv, _warn_exit_deprecated


class _FakeSimUi:
    """Record teardown calls so lifecycle tests can assert on them."""

    def __init__(self) -> None:
        self.exit_calls = 0

    def exit_simulation(self) -> None:
        self.exit_calls += 1


class _MinimalEnv(BaseEnv):
    """Minimal concrete BaseEnv exercising only the lifecycle contract."""

    def __init__(self) -> None:
        self.sim_ui: Any = None
        self.recorded_states: list[Any] = []
        super().__init__()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return {}, {}

    def step(self, action: Any):
        return {}, 0.0, False, False, {}

    @property
    def action_space(self):  # pragma: no cover - not exercised here
        raise NotImplementedError

    @property
    def observation_space(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


def test_close_releases_sim_ui_and_is_idempotent() -> None:
    env = _MinimalEnv()
    sim_ui = _FakeSimUi()
    env.sim_ui = sim_ui

    env.close()
    assert sim_ui.exit_calls == 1
    assert env.sim_ui is None

    env.close()
    assert sim_ui.exit_calls == 1


def test_close_without_sim_ui_is_safe() -> None:
    env = _MinimalEnv()
    env.close()
    assert env.sim_ui is None


def test_exit_alias_still_tears_down_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.gym_env.base_env as base_env_module

    monkeypatch.setattr(base_env_module, "_EXIT_DEPRECATION_WARNED", False)
    env = _MinimalEnv()
    sim_ui = _FakeSimUi()
    env.sim_ui = sim_ui

    with pytest.warns(DeprecationWarning, match="use env.close()"):
        env.exit()
    assert sim_ui.exit_calls == 1
    assert env.sim_ui is None

    # The process-level guard fires the warning at most once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env.exit()
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert sim_ui.exit_calls == 1


def test_warn_exit_deprecated_helper_warns_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_env_module, "_EXIT_DEPRECATION_WARNED", False)
    with pytest.warns(DeprecationWarning):
        _warn_exit_deprecated()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_exit_deprecated()
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _simulation_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A BaseSimulationEnv stub created without the config machinery.

    The module itself uses the same ``__new__`` pattern for its recording
    helper. The render import chain needs pygame (an optional extra), so the
    core dependency lane stubs it: the lifecycle paths under test never touch
    pygame functionality.
    """
    if "pygame" not in sys.modules:
        try:
            import pygame  # noqa: F401
        except ModuleNotFoundError:
            from unittest.mock import MagicMock

            monkeypatch.setitem(sys.modules, "pygame", MagicMock())
    from robot_sf.gym_env.abstract_envs import BaseSimulationEnv

    class _StubSimulationEnv(BaseSimulationEnv):
        """Concrete enough to instantiate without the config machinery."""

        def _create_spaces(self) -> None:
            pass

        def _setup_environment(self) -> None:
            pass

        def render(self, **kwargs: Any) -> None:
            pass

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return {}, {}

        def step(self, action: Any):
            return {}, 0.0, False, False, {}

    env = _StubSimulationEnv.__new__(_StubSimulationEnv)
    env.sim_ui = None
    env.recorded_states = []
    return env


def test_simulation_env_close_releases_sim_ui_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _simulation_env(monkeypatch)
    sim_ui = _FakeSimUi()
    env.sim_ui = sim_ui

    env.close()
    assert sim_ui.exit_calls == 1
    assert env.sim_ui is None

    env.close()
    assert sim_ui.exit_calls == 1


def test_simulation_env_exit_alias_warns_and_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_env_module, "_EXIT_DEPRECATION_WARNED", False)
    env = _simulation_env(monkeypatch)
    sim_ui = _FakeSimUi()
    env.sim_ui = sim_ui

    with pytest.warns(DeprecationWarning, match="use env.close()"):
        env.exit()
    assert sim_ui.exit_calls == 1
    assert env.sim_ui is None
