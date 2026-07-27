"""Unit tests locking the Fast-PySF backend factory contract.

``fast_pysf_factory`` is a thin wrapper over
:func:`robot_sf.sim.simulator.init_simulators`. These tests mock the
``init_simulators`` boundary so no real simulator is constructed and no maps are
loaded, while pinning argument forwarding (``env_config``/``map_def`` identity,
``random_start_pos=True``, and both ``peds`` obstacle-force values),
first-simulator selection, and unchanged exception propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robot_sf.sim.backends import fast_pysf_backend
from robot_sf.sim.backends.fast_pysf_backend import fast_pysf_factory


@pytest.mark.parametrize("peds", [True, False])
def test_factory_forwards_peds_obstacle_force_flag(
    peds: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both ``peds`` modes forward to the matching ``peds_have_obstacle_forces`` value."""
    env_config = object()
    map_def = object()
    selected_sim = object()
    mock_init = MagicMock(return_value=[selected_sim])
    monkeypatch.setattr(fast_pysf_backend, "init_simulators", mock_init)

    result = fast_pysf_factory(env_config, map_def, peds=peds)

    assert result is selected_sim
    mock_init.assert_called_once_with(
        env_config, map_def, random_start_pos=True, peds_have_obstacle_forces=peds
    )


def test_factory_forwards_env_config_and_map_def_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``env_config`` and ``map_def`` are forwarded by identity with ``random_start_pos=True``."""
    env_config = object()
    map_def = object()
    mock_init = MagicMock(return_value=[object()])
    monkeypatch.setattr(fast_pysf_backend, "init_simulators", mock_init)

    fast_pysf_factory(env_config, map_def, peds=True)

    positional_args, keyword_args = mock_init.call_args
    assert positional_args[0] is env_config
    assert positional_args[1] is map_def
    assert keyword_args["random_start_pos"] is True


def test_factory_selects_first_returned_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory returns index 0 of the simulator list, never a later element."""
    first_sim = object()
    second_sim = object()
    mock_init = MagicMock(return_value=[first_sim, second_sim])
    monkeypatch.setattr(fast_pysf_backend, "init_simulators", mock_init)

    result = fast_pysf_factory(object(), object(), peds=True)

    assert result is first_sim
    assert result is not second_sim


def test_factory_propagates_initialization_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions from ``init_simulators`` propagate unchanged with no fallback."""
    error = RuntimeError("simulator initialization failed")
    mock_init = MagicMock(side_effect=error)
    monkeypatch.setattr(fast_pysf_backend, "init_simulators", mock_init)

    with pytest.raises(RuntimeError, match="simulator initialization failed") as exc_info:
        fast_pysf_factory(object(), object(), peds=True)

    assert exc_info.value is error
