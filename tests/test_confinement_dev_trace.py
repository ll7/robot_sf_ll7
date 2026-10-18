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


def test_campaign_serializes_nonfinite_unused_metric_as_null():
    spec = importlib.util.spec_from_file_location(
        "campaign", Path(__file__).parents[1] / "scripts/dev/confinement_dev_trace_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.safe({"unobserved_metric": float("inf")}) == {"unobserved_metric": None}


def test_conditional_rates_recompute_denominators_and_empty_strata():
    rows = []
    for seed in range(1091, 1121):
        for policy in ("orca", "hybrid"):
            rows.append(
                {
                    "seed": seed,
                    "group": "portal",
                    "policy": policy,
                    "success": policy == "hybrid",
                    "traversal": True,
                }
            )
    summary = MODULE.group_success(rows, replicates=100)
    rates = {(r["policy"], r["stratum"]): r for r in summary["rates"]}
    assert rates["hybrid", "traversal"]["interval"] == [1.0, 1.0]
    assert rates["orca", "traversal"]["interval"] == [0.0, 0.0]
    assert rates["orca", "no_traversal"]["rate"] is None
    assert rates["orca", "no_traversal"]["empty_replicates"] == 100
    assert summary["contrasts"][0]["interval"] == [1.0, 1.0]


def test_first_step_zero_velocity_sentinel_uses_position_speed():
    fs = frames([[-0.1, 0], [0.2, 0]])
    fs[1]["velocity"] = [0, 0]
    fs[1]["first_step_sentinel"] = True
    result = MODULE.actor_indicators(fs, [P])
    assert result["first_step_sentinels"] == 1
    assert not result["stationary"]


def test_oblique_entry_uses_finite_plane_intersection():
    fs = frames([[-0.2, 1.2], [0.2, 0.6], [0.8, 0.2], [1.2, 0]])
    assert MODULE.actor_indicators(fs, [P])["passages"] == ["door"]


def test_two_necks_must_be_completed_without_respawn_between_them():
    second = {**P, "name": "second", "entry": 2, "exit": 3}
    fs = frames([[-0.1, 0], [0.6, 0], [1.1, 0], [1.9, 0], [2.5, 0], [3.1, 0]])
    fs[3]["boundary"] = True
    result = MODULE.actor_indicators(fs, [P, second])
    assert result["passages"] == ["door", "second"]
    assert not result["all_passages_uninterrupted"]


def test_resume_adopts_only_complete_verified_identities():
    spec = importlib.util.spec_from_file_location(
        "campaign", Path(__file__).parents[1] / "scripts/dev/confinement_dev_trace_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {("cell", 1091, "orca"), ("cell", 1092, "orca")}
    assert module.remaining_identities(expected, {("cell", 1091, "orca")}) == {
        ("cell", 1092, "orca")
    }
    with pytest.raises(ValueError, match="unexpected"):
        module.remaining_identities(expected, {("different", 1091, "orca")})


def test_missing_actor_observation_breaks_crossing_segment():
    ped = {"id": "slot-0", "position": [-0.1, 0], "velocity": [0, 0]}
    trace = {
        "dt": 0.1,
        "reset": {"pedestrians": [ped]},
        "steps": [
            {"time_s": 0.1, "pedestrians": [{**ped, "position": [0.3, 0]}]},
            {"time_s": 0.2, "pedestrians": []},
            {"time_s": 0.3, "pedestrians": [{**ped, "position": [1.1, 0], "velocity": [4, 0]}]},
        ],
    }
    record = {"algorithm_metadata": {"simulation_step_trace": trace}}
    assert not MODULE.episode_indicators(record, [P])["traversal"]


def test_unfinished_attempt_is_reported_as_censored():
    result = MODULE.actor_indicators(frames([[-0.1, 0], [0.3, 0], [0.6, 0]]), [P])
    assert result["unfinished_passages"] == ["door"]
    assert result["terminal_inside"] and not result["passages"]
