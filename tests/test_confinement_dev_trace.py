"""Regression fixtures for the predeclared descriptive trace estimands."""

import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "confinement_dev_trace",
    Path(os.environ.get("CONFINEMENT_CODE_ROOT", Path(__file__).parents[1]))
    / "scripts/dev/confinement_dev_trace.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
P = {"name": "door", "axis": 0, "entry": 0, "exit": 1, "lateral_min": -1, "lateral_max": 1}


def frames(positions, dt=0.1):
    result = []
    for i, p in enumerate(positions):
        prev = positions[max(0, i - 1)]
        result.append(
            {
                "time": i * dt,
                "position": p,
                "velocity": [(p[j] - prev[j]) / dt for j in (0, 1)],
                "boundary": False,
            }
        )
    return result


def test_complete_crossing_and_reverse():
    for ps in [[[-0.1, 0], [0.4, 0], [1.1, 0]], [[1.1, 0], [0.4, 0], [-0.1, 0]]]:
        assert MODULE.actor_indicators(frames(ps), [P])["passages"] == ["door"]


def test_bypass_and_left_censor_are_not_traversals():
    assert not MODULE.actor_indicators(frames([[-0.1, 2], [1.1, 2]]), [P])["passages"]
    x = MODULE.actor_indicators(frames([[0.5, 0], [1.1, 0]]), [P])
    assert not x["passages"] and x["initial_inside"]


def test_respawn_cannot_complete_or_bridge_a_crossing():
    fs = frames([[-0.1, 0], [0.3, 0], [1.1, 0]])
    fs[2]["boundary"] = True
    assert not MODULE.actor_indicators(fs, [P])["passages"]


def test_stationarity_has_full_three_second_duration():
    assert not MODULE.actor_indicators(frames([[0, 0]] * 30), [P])["stationary"]
    assert MODULE.actor_indicators(frames([[0, 0]] * 31), [P])["stationary"]
    assert not MODULE.actor_indicators(frames([[0, 4]] * 40), [P])["stationary"]
    fs = frames([[0, 0]] * 40)
    fs[20]["boundary"] = True
    assert not MODULE.actor_indicators(fs, [P])["stationary"]


def test_velocity_mismatch_fails_closed():
    fs = frames([[-0.1, 0], [0.2, 0]])
    fs[1]["velocity"] = [0, 0]
    with pytest.raises(ValueError, match="velocity"):
        MODULE.actor_indicators(fs, [P])


def test_seed_block_draws_are_shared_and_repeatable():
    draws = MODULE.seed_draws(100)
    assert draws == MODULE.seed_draws(100)
    assert len(draws) == 100 and all(len(d) == 30 for d in draws)
    assert MODULE.interval([0.0] * 50 + [1.0] * 50) == [0.0, 1.0]


def test_campaign_seeds_are_exactly_development_schedule():
    spec = importlib.util.spec_from_file_location(
        "campaign", Path(__file__).parents[1] / "scripts/dev/confinement_dev_trace_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.allowed_seed(1091) and module.allowed_seed(1120)
    assert not module.allowed_seed(111) and not module.allowed_seed(1121)
