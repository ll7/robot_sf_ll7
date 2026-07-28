"""Focused coverage for the extracted ORCA + HRVO planner-family module."""

import numpy as np

from robot_sf.planner import socnav
from robot_sf.planner import socnav_orca as orca


def _observation(*, goal=(5.0, 0.0), heading=0.0, pedestrians=None) -> dict:
    """Build a compact nested SocNav observation for adapter unit tests.

    Mirrors the helper shape used by tests/test_socnav_planner_adapter.py so the
    ORCA heuristic path receives a well-formed robot/goal/pedestrian payload.
    """
    if pedestrians is None:
        positions = np.zeros((1, 2), dtype=np.float32)
        count = 0.0
    else:
        positions = np.asarray(pedestrians, dtype=np.float32).reshape(-1, 2)
        count = float(positions.shape[0])
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([count], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def test_facade_wildcard_import_includes_lazy_public_exports() -> None:
    """Lazy ORCA/HRVO symbols stay visible through facade introspection and wildcard import."""
    expected = {"ORCAPlannerAdapter", "HRVOPlannerAdapter", "make_orca_policy", "make_hrvo_policy"}
    assert expected <= set(dir(socnav))
    assert expected <= set(socnav.__all__)
    assert expected <= socnav._ORCA_LAZY_EXPORTS
    assert socnav.ORCAPlannerAdapter is orca.ORCAPlannerAdapter
    assert socnav.HRVOPlannerAdapter is orca.HRVOPlannerAdapter
    assert socnav.make_orca_policy is orca.make_orca_policy
    assert socnav.make_hrvo_policy is orca.make_hrvo_policy


def test_lazy_resolution_caches_into_facade_globals() -> None:
    """Accessing a lazy name binds it into the facade module globals after first resolution."""
    # Touch the lazy attributes through the facade to trigger __getattr__ resolution.
    _ = socnav.ORCAPlannerAdapter
    _ = socnav.HRVOPlannerAdapter
    _ = socnav.make_orca_policy
    _ = socnav.make_hrvo_policy
    for name in (
        "ORCAPlannerAdapter",
        "HRVOPlannerAdapter",
        "make_orca_policy",
        "make_hrvo_policy",
    ):
        assert name in socnav.__dict__


def test_orca_adapter_importable_and_instantiable() -> None:
    """The ORCA adapter can be imported and instantiated from the extracted module."""
    adapter = orca.ORCAPlannerAdapter()
    assert isinstance(adapter, orca.SamplingPlannerAdapter)
    assert adapter.config is not None
    assert adapter.config.max_linear_speed == 3.0


def test_hrvo_adapter_is_orca_subclass() -> None:
    """HRVO is a pure-Python velocity-obstacle adapter subclassing ORCAPlannerAdapter."""
    adapter = orca.HRVOPlannerAdapter()
    assert isinstance(adapter, orca.ORCAPlannerAdapter)
    assert isinstance(adapter, orca.SamplingPlannerAdapter)


def test_factories_wrap_correct_adapter_types() -> None:
    """Factory functions wrap the correct adapter inside the policy."""
    assert isinstance(orca.make_orca_policy().adapter, orca.ORCAPlannerAdapter)
    assert isinstance(orca.make_hrvo_policy().adapter, orca.HRVOPlannerAdapter)


def test_orca_heuristic_fallback_produces_finite_action(monkeypatch) -> None:
    """Facade re-exported ORCA adapter falls back to the heuristic plan when rvo2 is unavailable.

    Patching the facade's ``rvo2`` handle propagates to the extracted module because ORCA reads
    the live ``_socnav.rvo2`` attribute at call time (mirroring the SA-CADRL ``_socnav.tf`` path).
    """
    monkeypatch.setattr(socnav, "rvo2", None)
    from robot_sf.planner.socnav import ORCAPlannerAdapter

    adapter = ORCAPlannerAdapter(allow_fallback=True)
    linear, angular = adapter.plan(_observation(goal=(2.0, 0.0)))
    assert np.isfinite(linear)
    assert np.isfinite(angular)
    assert linear >= 0.0


def test_orca_heuristic_fallback_slows_for_head_on_pedestrian(monkeypatch) -> None:
    """The ORCA heuristic path reduces speed for a blocking head-on pedestrian."""
    monkeypatch.setattr(socnav, "rvo2", None)
    adapter = orca.ORCAPlannerAdapter(allow_fallback=True)

    linear_free, _angular_free = adapter.plan(_observation(goal=(5.0, 0.0)))
    linear_blocked, angular_blocked = adapter.plan(
        _observation(goal=(5.0, 0.0), pedestrians=[(2.0, 0.0)])
    )
    assert linear_blocked < linear_free
    assert np.isfinite(angular_blocked)
