"""Focused coverage for the extracted SocialForce planner-family module."""

import numpy as np

from robot_sf.planner import socnav
from robot_sf.planner import socnav_social_force as sf


def test_facade_wildcard_import_includes_lazy_public_exports() -> None:
    """Lazy public symbols remain visible through facade introspection and wildcard import."""
    assert "SocialForcePlannerAdapter" in dir(socnav)
    assert "make_social_force_policy" in dir(socnav)
    assert "SocialForcePlannerAdapter" in socnav.__all__
    assert "make_social_force_policy" in socnav.__all__
    assert socnav.SocialForcePlannerAdapter is sf.SocialForcePlannerAdapter
    assert socnav.make_social_force_policy is sf.make_social_force_policy


def test_social_force_adapter_importable_and_instantiable() -> None:
    """The adapter can be imported and instantiated from the extracted module."""
    adapter = sf.SocialForcePlannerAdapter()
    assert isinstance(adapter, sf.SamplingPlannerAdapter)
    assert adapter.config is not None
    assert adapter.config.social_force_repulsion_weight == 0.8


def test_factory_produces_policy_with_correct_adapter_type() -> None:
    """Factory function wraps the correct adapter inside the policy."""
    policy = sf.make_social_force_policy()
    assert isinstance(policy.adapter, sf.SocialForcePlannerAdapter)


def test_adapter_constructs_finite_action_via_facade() -> None:
    """Facade re-exported adapter produces finite actions."""
    from robot_sf.planner.socnav import SocialForcePlannerAdapter, SocNavPlannerConfig

    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    obs = {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "speed": np.array([0.0, 0.0]),
            "radius": np.array([0.5]),
        },
        "goal": {"current": np.array([2.0, 0.0])},
        "pedestrians": {
            "positions": np.zeros((0, 2)),
            "velocities": np.zeros((0, 2)),
            "count": np.array([0]),
            "radius": np.array([0.3]),
        },
        "sim": {"timestep": np.array([0.1])},
    }
    v, w = adapter.plan(obs)
    assert np.isfinite(v)
    assert np.isfinite(w)
