"""Characterization baseline tests for ``robot_sf/planner/socnav.py`` public adapters.

This file is the pre-decomposition behavior lock for the SocNav planner-family
adapters (issue #4963, part of #4770). It pins the observable output of the
public factory + ``plan()`` surface so the follow-up god-class decomposition
refactor (#4986) can prove behavior preservation by diffing against these
golden values.

Pinned surfaces (additive golden-value style, deterministic, seeded fixtures):
1. The public factories (``make_*_policy``) construction and the resulting
   ``SocNavPlannerPolicy`` + wrapped adapter type, for every family.
2. ``plan()`` exact ``(v, omega)`` output for a small set of fixed synthetic
   observations, pinned for at least the ORCA, HRVO, and social-force families.
3. The ``diagnostics()`` dict keys the ``TrivialReferencePlannerAdapter``
   exposes publicly, plus the shared ``config`` metadata surface.
4. Optional-dependency gating that mirrors the ``try/except ImportError`` guards
   at the top of ``socnav.py``: rvo2-backed ORCA is ``skipif``-guarded on
   ``rvo2``; SACADRL/TF and prediction/torch families are ``skipif``-guarded on
   their respective backends.

Determinism is asserted explicitly in
``test_characterization_is_deterministic_across_instances`` (two consecutive
fresh instances must reproduce every golden command). This test does not modify
production code; if a value reveals a genuine bug, do NOT fix it here — file a
separate fix issue (same contract as ``test_metrics_characterization.py`` and
the sibling ``test_hybrid_rule_local_planner_characterization.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner import socnav
from robot_sf.planner.socnav import (
    HRVOPlannerAdapter,
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    TrivialReferencePlannerAdapter,
    make_hrvo_policy,
    make_orca_policy,
    make_prediction_policy,
    make_sacadrl_policy,
    make_social_force_policy,
)

# ---------------------------------------------------------------------------
# Shared observation builders.
#
# The payload shape mirrors the compact SocNav structured observation used by
# the sibling ``tests/test_socnav_planner_adapter.py`` so this baseline covers
# the same contract the rest of the suite exercises. Every fixture is a
# module-level constant so the golden values below are tied to an unchanging
# input; changing a fixture requires recomputing the golden values in the same
# commit.
# ---------------------------------------------------------------------------


def _obs(
    *,
    goal: tuple[float, float] = (5.0, 0.0),
    heading: float = 0.0,
    peds: list[tuple[float, float]] | None = None,
) -> dict:
    """Build a minimal SocNav observation with optional pedestrians.

    The pedestrian block always carries a single-slot capacity; when ``peds`` is
    provided the positions/velocities/count are filled to match the SocNav
    adapter's ``_extract_pedestrians`` contract.
    """
    ped_positions = [] if peds is None else peds
    max_peds = max(1, len(ped_positions))
    positions = np.zeros((max_peds, 2), dtype=np.float32)
    velocities = np.zeros((max_peds, 2), dtype=np.float32)
    if ped_positions:
        positions[: len(ped_positions)] = np.array(ped_positions, dtype=np.float32)
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([float(len(ped_positions))], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


# Fixed scenarios. The same constants feed the per-family plan() pins and the
# determinism lock, so a single source of truth drives every golden value.

# Open corridor: robot aligned with a distant goal, no pedestrians.
_OBS_OPEN_AHEAD = _obs(goal=(5.0, 0.0), heading=0.0)

# Near goal: within the reference tolerance distance, but the ORCA/HRVO path
# keys on preferred-velocity magnitude rather than the goal-tolerance stop, so
# this scenario distinguishes the social-force stop from the ORCA/HRVO forward
# command.
_OBS_NEAR_GOAL = _obs(goal=(0.2, 0.0), heading=0.0)

# Head-on pedestrian: a single pedestrian directly between robot and goal,
# driving the ORCA slowdown, the HRVO lateral avoidance, and the social-force
# repulsion path.
_OBS_HEAD_ON = _obs(goal=(5.0, 0.0), heading=0.0, peds=[(2.0, 0.0)])

# Goal to the left: goal axis rotated 90 deg from heading, driving a pure turn.
_OBS_GOAL_LEFT = _obs(goal=(0.0, 5.0), heading=0.0)


def _orca_fallback_adapter(config: SocNavPlannerConfig | None = None) -> ORCAPlannerAdapter:
    """Create an ORCA adapter forced onto the deterministic heuristic fallback.

    The heuristic fallback is pure Python and therefore fully portable across
    platforms; it is the always-green primary lock for the ORCA family. The
    rvo2-backed path is pinned separately under a skipif guard.
    """
    adapter = ORCAPlannerAdapter(config or SocNavPlannerConfig(), allow_fallback=True)
    adapter._fallback_warned = True  # silence the one-shot fallback warning
    return adapter


# ===========================================================================
# 1. Factory construction + resulting policy/adapter type
# ===========================================================================


@pytest.mark.parametrize(
    ("factory", "kwargs", "adapter_type"),
    [
        (make_social_force_policy, {}, SocialForcePlannerAdapter),
        (make_orca_policy, {"allow_fallback": True}, ORCAPlannerAdapter),
        (make_hrvo_policy, {}, HRVOPlannerAdapter),
        (make_sacadrl_policy, {"allow_fallback": True}, SACADRLPlannerAdapter),
        (make_prediction_policy, {"allow_fallback": True}, PredictionPlannerAdapter),
    ],
)
def test_factory_builds_policy_wrapping_correct_adapter(factory, kwargs, adapter_type) -> None:
    """Each ``make_*_policy`` must return a SocNavPlannerPolicy over the right adapter."""
    policy = factory(SocNavPlannerConfig(), **kwargs)

    assert isinstance(policy, SocNavPlannerPolicy)
    assert isinstance(policy.adapter, adapter_type)


def test_factory_wraps_a_single_shared_adapter() -> None:
    """``SocNavPlannerPolicy`` delegates to exactly one adapter attribute."""
    policy = make_hrvo_policy(SocNavPlannerConfig(max_linear_speed=0.9))

    assert set(vars(policy).keys()) == {"adapter"}
    assert policy.adapter.config.max_linear_speed == pytest.approx(0.9)


def test_factory_default_config_matches_socnav_planner_config() -> None:
    """Factories must default to a ``SocNavPlannerConfig`` when none is passed."""
    policy = make_social_force_policy()

    assert isinstance(policy.adapter.config, SocNavPlannerConfig)


# ===========================================================================
# 2. plan() golden values: ORCA heuristic, HRVO (pure Python), social-force
# ===========================================================================


# --- ORCA heuristic fallback (always available, pure Python) ---------------


@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        ("open_ahead", _OBS_OPEN_AHEAD, (3.0, 0.0)),
        ("near_goal", _OBS_NEAR_GOAL, (3.0, 0.0)),
        ("head_on", _OBS_HEAD_ON, (pytest.approx(0.09166666616996129), 0.0)),
        ("goal_left", _OBS_GOAL_LEFT, (2.4000000000000004, 1.0)),
    ],
)
def test_orca_heuristic_plan_golden(label, observation, golden) -> None:
    """ORCA heuristic fallback must return the pinned ``(v, omega)`` per scenario."""
    command = _orca_fallback_adapter().plan(observation)

    assert command == golden, f"ORCA heuristic {label}: {command} != {golden}"
    # Bounded command contract (independent of the golden value).
    assert 0.0 <= command[0] <= 3.0 + 1e-9
    assert abs(command[1]) <= 1.0 + 1e-9


# --- HRVO (pure-Python local solver, no optional backend) ------------------


@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        ("open_ahead", _OBS_OPEN_AHEAD, (3.0, 0.0)),
        ("near_goal", _OBS_NEAR_GOAL, (3.0, 0.0)),
        ("head_on", _OBS_HEAD_ON, (2.6795521968903904, -0.7281393895749115)),
        ("goal_left", _OBS_GOAL_LEFT, (2.4000000000000004, 1.0)),
    ],
)
def test_hrvo_plan_golden(label, observation, golden) -> None:
    """HRVO local solver must return the pinned ``(v, omega)`` per scenario."""
    adapter = HRVOPlannerAdapter(SocNavPlannerConfig())

    command = adapter.plan(observation)

    assert command == golden, f"HRVO {label}: {command} != {golden}"
    assert 0.0 <= command[0] <= 3.0 + 1e-9
    assert abs(command[1]) <= 1.0 + 1e-9


# --- Social-force (pysocialforce backend, skipif-guarded) ------------------


_SOCIAL_FORCE_AVAILABLE = socnav.sf_forces is not None
skipif_no_sf = pytest.mark.skipif(
    not _SOCIAL_FORCE_AVAILABLE,
    reason="pysocialforce (fast-pysf) is required for the SocialForcePlannerAdapter path",
)


@skipif_no_sf
@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        ("open_ahead", _OBS_OPEN_AHEAD, (pytest.approx(0.19999996298023964), 0.0)),
        ("near_goal", _OBS_NEAR_GOAL, (0.0, 0.0)),
        (
            "head_on",
            _OBS_HEAD_ON,
            (pytest.approx(0.19823034917009186), pytest.approx(0.006774436119217686)),
        ),
        ("goal_left", _OBS_GOAL_LEFT, (pytest.approx(0.09999998149011982), 1.0)),
    ],
)
def test_social_force_plan_golden(label, observation, golden) -> None:
    """Social-force adapter must return the pinned ``(v, omega)`` per scenario."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())

    command = adapter.plan(observation)

    assert command == golden, f"Social-force {label}: {command} != {golden}"
    assert 0.0 <= command[0] <= 3.0 + 1e-9
    assert abs(command[1]) <= 1.0 + 1e-9


# ===========================================================================
# 3. Backend-available ORCA rvo2 path (skipif-guarded)
# ===========================================================================


_RVO2_AVAILABLE = socnav.rvo2 is not None
skipif_no_rvo2 = pytest.mark.skipif(
    not _RVO2_AVAILABLE,
    reason="rvo2 ('orca' extra) is required for the benchmark-ready ORCA path",
)


@skipif_no_rvo2
@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        # rvo2 shares the ORCA preferred-velocity projection for the open-space
        # and goal-left cases, so those values match the heuristic fallback.
        ("open_ahead", _OBS_OPEN_AHEAD, (3.0, 0.0)),
        ("near_goal", _OBS_NEAR_GOAL, (3.0, 0.0)),
        # The head-on case resolves through the rvo2 C++ solver; allow a small
        # tolerance for platform float-ordering differences.
        ("head_on", _OBS_HEAD_ON, (pytest.approx(0.09166666865348816), 0.0)),
        ("goal_left", _OBS_GOAL_LEFT, (2.4000000000000004, 1.0)),
    ],
)
def test_orca_rvo2_plan_golden(label, observation, golden) -> None:
    """The rvo2-backed ORCA path must return the pinned ``(v, omega)`` when installed."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)

    command = adapter.plan(observation)

    assert command == golden, f"ORCA rvo2 {label}: {command} != {golden}"
    assert 0.0 <= command[0] <= 3.0 + 1e-9
    assert abs(command[1]) <= 1.0 + 1e-9


# ===========================================================================
# 4. Backend-gated families: SACADRL (TF), prediction (torch)
# ===========================================================================


_TF_AVAILABLE = socnav.tf is not None
_TORCH_AVAILABLE = socnav.torch is not None
skipif_no_tf = pytest.mark.skipif(
    not _TF_AVAILABLE,
    reason="tensorflow is required for the SACADRL family",
)
skipif_no_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch is required for the prediction family",
)


def _force_fallback(adapter) -> None:
    """Force a learned adapter onto its deterministic heuristic fallback path."""
    adapter._load_error = RuntimeError("forced missing model for characterization")
    adapter._model = None
    adapter._fallback_warned = True


@skipif_no_tf
@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        # SA-CADRL fallback delegates to the sampling heuristic, which moves at
        # max speed toward an aligned goal and ignores the head-on pedestrian.
        ("open_ahead", _OBS_OPEN_AHEAD, (3.0, 0.0)),
        ("head_on", _OBS_HEAD_ON, (3.0, 0.0)),
    ],
)
def test_sacadrl_fallback_plan_golden(label, observation, golden) -> None:
    """SA-CADRL adapter fallback must return the pinned ``(v, omega)`` per scenario."""
    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    _force_fallback(adapter)

    command = adapter.plan(observation)

    assert command == golden, f"SACADRL fallback {label}: {command} != {golden}"


@skipif_no_torch
@pytest.mark.parametrize(
    ("label", "observation", "golden"),
    [
        # Predictive fallback uses constant-velocity prediction; the head-on
        # pedestrian triggers a TTC-driven slowdown.
        ("open_ahead", _OBS_OPEN_AHEAD, (3.0, 0.0)),
        ("head_on", _OBS_HEAD_ON, (0.30000000000000004, 0.0)),
    ],
)
def test_prediction_fallback_plan_golden(label, observation, golden) -> None:
    """Predictive adapter fallback must return the pinned ``(v, omega)`` per scenario."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    _force_fallback(adapter)

    command = adapter.plan(observation)

    assert command == golden, f"Prediction fallback {label}: {command} != {golden}"


# ===========================================================================
# 5. diagnostics() / metadata characterization
# ===========================================================================


# TrivialReferencePlannerAdapter is the only SocNav-family adapter that exposes
# an explicit ``diagnostics()`` dict; pin its public key set and values.
_TRIVIAL_DIAGNOSTICS_KEYS = frozenset({"adapter", "steps", "contract"})


def test_trivial_reference_diagnostics_shape_and_values() -> None:
    """``TrivialReferencePlannerAdapter.diagnostics()`` pins the diagnostic contract."""
    adapter = TrivialReferencePlannerAdapter(SocNavPlannerConfig())

    diagnostics = adapter.diagnostics()

    assert frozenset(diagnostics.keys()) == _TRIVIAL_DIAGNOSTICS_KEYS
    assert diagnostics == {
        "adapter": "TrivialReferencePlannerAdapter",
        "steps": 0,
        "contract": "diagnostic_reference_only",
    }


def test_trivial_reference_diagnostics_counts_steps() -> None:
    """``diagnostics()['steps']`` must reflect the number of ``plan()`` calls."""
    adapter = TrivialReferencePlannerAdapter(SocNavPlannerConfig())

    adapter.plan(_OBS_OPEN_AHEAD)
    adapter.plan(_OBS_GOAL_LEFT)

    assert adapter.diagnostics()["steps"] == 2


def test_trivial_reference_plan_golden() -> None:
    """Pin the reference adapter command so its public contract is locked too."""
    adapter = TrivialReferencePlannerAdapter(SocNavPlannerConfig())

    # Aligned goal: forward motion with no turn.
    assert adapter.plan(_OBS_OPEN_AHEAD) == (3.0, 0.0)
    # Goal to the left: heading error of pi/2 -> alignment 0.5, distance 5 m
    # gives linear = clip(5 * 0.5, 0, 3.0) = 2.5, and a saturated left turn.
    command = adapter.plan(_OBS_GOAL_LEFT)
    assert command == (2.5, 1.0)


@pytest.mark.parametrize(
    "factory",
    [make_social_force_policy, make_hrvo_policy],
)
def test_factory_adapter_exposes_socnav_planner_config_metadata(factory) -> None:
    """Every family adapter exposes its config as the shared metadata surface."""
    policy = factory(SocNavPlannerConfig(max_linear_speed=2.5, max_angular_speed=0.8))

    assert isinstance(policy.adapter.config, SocNavPlannerConfig)
    assert policy.adapter.config.max_linear_speed == pytest.approx(2.5)
    assert policy.adapter.config.max_angular_speed == pytest.approx(0.8)


def test_sampling_adapter_plan_golden() -> None:
    """Pin the base SamplingPlannerAdapter heuristic command (shared fallback core)."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig())

    assert adapter.plan(_OBS_OPEN_AHEAD) == (3.0, 0.0)
    assert adapter.plan(_OBS_NEAR_GOAL) == (0.0, 0.0)


# ===========================================================================
# 6. Determinism lock: the golden commands must reproduce across instances.
# ===========================================================================


def test_characterization_is_deterministic_across_instances() -> None:
    """Two fresh adapter instances must produce identical commands per scenario."""
    # (label, factory, observation, golden). The factory returns a fresh adapter
    # so each scenario is evaluated against an untouched instance, exactly as the
    # per-family golden tests above construct theirs.
    expected = [
        ("orca_heuristic_open", _orca_fallback_adapter, _OBS_OPEN_AHEAD, (3.0, 0.0)),
        (
            "orca_heuristic_head_on",
            _orca_fallback_adapter,
            _OBS_HEAD_ON,
            (pytest.approx(0.09166666616996129), 0.0),
        ),
        (
            "hrvo_open",
            lambda: HRVOPlannerAdapter(SocNavPlannerConfig()),
            _OBS_OPEN_AHEAD,
            (3.0, 0.0),
        ),
        (
            "hrvo_head_on",
            lambda: HRVOPlannerAdapter(SocNavPlannerConfig()),
            _OBS_HEAD_ON,
            (2.6795521968903904, -0.7281393895749115),
        ),
        (
            "hrvo_goal_left",
            lambda: HRVOPlannerAdapter(SocNavPlannerConfig()),
            _OBS_GOAL_LEFT,
            (2.4000000000000004, 1.0),
        ),
        (
            "trivial_open",
            lambda: TrivialReferencePlannerAdapter(SocNavPlannerConfig()),
            _OBS_OPEN_AHEAD,
            (3.0, 0.0),
        ),
        (
            "sampling_open",
            lambda: SamplingPlannerAdapter(SocNavPlannerConfig()),
            _OBS_OPEN_AHEAD,
            (3.0, 0.0),
        ),
    ]
    if _SOCIAL_FORCE_AVAILABLE:
        expected.append(
            (
                "social_force_open",
                lambda: SocialForcePlannerAdapter(SocNavPlannerConfig()),
                _OBS_OPEN_AHEAD,
                (pytest.approx(0.19999996298023964), 0.0),
            ),
        )

    for label, factory, observation, golden in expected:
        first = factory().plan(observation)
        second = factory().plan(observation)
        assert first == golden, f"{label}: first run {first} != golden {golden}"
        assert second == golden, f"{label}: second run {second} != golden {golden}"
        assert first == second, f"{label}: not deterministic across instances"
