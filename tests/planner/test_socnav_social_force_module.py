"""Focused coverage for the extracted SocialForce planner-family module."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
from pysocialforce.config import SOCIAL_FORCE_KERNEL_WRAPPED_V2

from robot_sf.planner import socnav
from robot_sf.planner import socnav_social_force as sf
from robot_sf.prediction._contract_utils import stable_config_hash


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


def test_predictive_surface_mode_requires_positive_body_radii() -> None:
    """Predictive surface distances cannot use zero-valued placeholder bodies."""
    import pytest

    from robot_sf.planner.socnav import SocNavPlannerConfig

    with pytest.raises(ValueError, match="robot_radius must be finite and positive"):
        SocNavPlannerConfig(
            predictive_clearance_model="surface_v2",
            predictive_robot_radius=0.0,
            predictive_pedestrian_radius=0.4,
        )


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


def test_kernel_provenance_is_absent_for_legacy_default_and_present_when_selected() -> None:
    """Missing selectors keep legacy diagnostics unchanged; opt-in runs identify the kernel."""
    legacy = sf.SocialForcePlannerAdapter(sf.SocNavPlannerConfig()).diagnostics()
    assert "kernel_version" not in legacy
    assert "social_force_kernel" not in legacy

    wrapped = sf.SocialForcePlannerAdapter(
        sf.SocNavPlannerConfig(social_force_kernel_version=SOCIAL_FORCE_KERNEL_WRAPPED_V2)
    ).diagnostics()
    assert wrapped["kernel_version"] == SOCIAL_FORCE_KERNEL_WRAPPED_V2
    assert wrapped["kernel_resolution_mode"] == "explicit"
    assert wrapped["social_force_kernel"]["angle_wrap"] is True


def test_socnav_kernel_selector_preserves_default_serialization_and_explicit_identity() -> None:
    """Missing and explicitly selected legacy kernels keep distinct identities."""
    from pysocialforce.config import SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1

    from robot_sf.planner.socnav_base import SocNavPlannerConfig

    legacy_default = SocNavPlannerConfig()
    explicit_legacy = SocNavPlannerConfig(
        social_force_kernel_version=SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1
    )
    wrapped = SocNavPlannerConfig(social_force_kernel_version=SOCIAL_FORCE_KERNEL_WRAPPED_V2)

    assert legacy_default.social_force_kernel_resolution_mode == "defaulted_missing"
    legacy_default.__post_init__()
    assert legacy_default.social_force_kernel_version == (SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1)
    assert "social_force_kernel_version" not in asdict(legacy_default)
    assert "social_force_kernel_version" not in legacy_default.to_dict()
    assert legacy_default != explicit_legacy
    assert explicit_legacy.to_dict()["social_force_kernel_version"] == (
        SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1
    )
    assert wrapped.to_dict()["social_force_kernel_version"] == SOCIAL_FORCE_KERNEL_WRAPPED_V2
    assert SocNavPlannerConfig(**wrapped.to_dict()) == wrapped
    assert stable_config_hash(legacy_default.to_dict()) != stable_config_hash(wrapped.to_dict())


def test_issue_9750_clearance_opt_ins_preserve_legacy_config_bytes() -> None:
    """New v0.8 geometry and sampling selectors stay out of v1 defaults."""
    default = sf.SocNavPlannerConfig()
    payload = default.to_dict()
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()

    assert "predictive_clearance_model" not in asdict(default)
    assert "predictive_clearance_model" not in payload
    assert "sampling_repulsion_weight" not in asdict(default)
    assert "sampling_repulsion_weight" not in payload
    assert hashlib.sha256(encoded).hexdigest() == (
        "51c9ab100f7165ef1438acc8a545bc0fe31a0e079db3b1796fca874c5cb0e1d2"
    )

    selected = sf.SocNavPlannerConfig(
        predictive_clearance_model="surface_v2",
        sampling_repulsion_weight=0.0,
    )
    selected_payload = selected.to_dict()
    assert selected_payload["predictive_clearance_model"] == "surface_v2"
    assert selected_payload["sampling_repulsion_weight"] == 0.0
    assert sf.SocNavPlannerConfig(**selected_payload) == selected
