"""Tests for the optional CBF safety-filter planner wrapper."""

from __future__ import annotations

import pytest

from robot_sf.planner.cbf_safety_filter import (
    CBF_VARIANT_DYNAMIC_PARABOLIC,
    CBFFilterContext,
    CBFObstacleState,
    CBFSafetyFilterConfig,
    CbfSafetyFilterConfig,
    CbfSafetyFilterPlannerWrapper,
    CollisionConeCbfSafetyFilter,
    apply_cbf_safety_filter,
    build_cbf_safety_filter_config,
)


def _head_on_observation() -> dict[str, object]:
    return {
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.8, 0.0],
            "heading": [0.0],
            "radius": [0.3],
        },
        "agents": [
            {
                "position": [0.8, 0.0],
                "velocity": [-0.6, 0.0],
                "radius": 0.3,
            }
        ],
    }


def _public_context() -> CBFFilterContext:
    return CBFFilterContext(
        robot_position_m=(0.0, 0.0),
        robot_heading_rad=0.0,
        robot_radius_m=0.3,
        obstacles=(
            CBFObstacleState(
                position_m=(1.0, 0.0),
                velocity_mps=(-0.5, 0.0),
                radius_m=0.35,
            ),
        ),
    )


def test_public_apply_cbf_filter_disabled_returns_original_command() -> None:
    """Public pure-function API preserves disabled commands."""

    result = apply_cbf_safety_filter(0.8, 0.1, _public_context(), CBFSafetyFilterConfig())

    assert result["qp_status"] == "disabled"
    assert result["filtered_linear_velocity"] == 0.8
    assert result["filtered_angular_velocity"] == 0.1


def test_public_apply_cbf_filter_no_obstacles_passes_through() -> None:
    """No obstacle constraints keep enabled CBF pass-through."""

    context = CBFFilterContext(
        robot_position_m=(0.0, 0.0),
        robot_heading_rad=0.0,
        robot_radius_m=0.3,
        obstacles=(),
    )
    result = apply_cbf_safety_filter(
        0.8,
        0.1,
        context,
        CBFSafetyFilterConfig(enabled=True),
    )

    assert result["qp_status"] == "pass_through"
    assert result["active_constraint_count"] == 0


def test_public_apply_cbf_filter_projects_head_on_command() -> None:
    """Head-on obstacle clips the nominal forward command."""

    result = apply_cbf_safety_filter(
        0.8,
        0.1,
        _public_context(),
        CBFSafetyFilterConfig(enabled=True),
    )

    assert result["qp_status"] in {"filtered", "fallback_infeasible"}
    assert result["filtered_linear_velocity"] < 0.8
    assert result["intervened"] is True


def test_public_apply_cbf_filter_dynamic_parabolic_scaffold_fails_closed() -> None:
    """DPCBF is scaffolded but intentionally unavailable in first slice."""

    with pytest.raises(NotImplementedError, match="dynamic_parabolic_cbf_v1"):
        apply_cbf_safety_filter(
            0.8,
            0.1,
            _public_context(),
            CBFSafetyFilterConfig(enabled=True, variant=CBF_VARIANT_DYNAMIC_PARABOLIC),
        )


def test_public_apply_cbf_filter_rejects_malformed_state() -> None:
    """Non-finite public context state fails closed."""

    context = CBFFilterContext(
        robot_position_m=(float("nan"), 0.0),
        robot_heading_rad=0.0,
        robot_radius_m=0.3,
        obstacles=(),
    )

    with pytest.raises(ValueError, match="robot_position_m"):
        apply_cbf_safety_filter(0.8, 0.1, context, CBFSafetyFilterConfig(enabled=True))


def test_collision_cone_cbf_projects_head_on_command() -> None:
    """Approaching obstacle should force a lower-speed best-effort command."""

    filter_ = CollisionConeCbfSafetyFilter(
        CbfSafetyFilterConfig(
            enabled=True,
            alpha=1.0,
            safety_margin=0.1,
            max_linear_speed=1.0,
            max_angular_speed=2.0,
        )
    )

    decision = filter_.filter_command(_head_on_observation(), (0.8, 0.0))

    assert decision.decision_label in {"cbf_projected", "cbf_best_effort"}
    assert decision.filtered_action[0] < 0.8
    assert decision.prediction_source == "current_state"
    assert decision.violated_constraints == ("collision_cone_cbf_agent_0",)
    assert filter_.diagnostics()["decision_count"] == 1


def test_collision_cone_cbf_leaves_clear_command_feasible() -> None:
    """Obstacle moving away should not alter the nominal command."""

    observation = _head_on_observation()
    observation["agents"][0]["velocity"] = [0.6, 0.0]  # type: ignore[index]
    filter_ = CollisionConeCbfSafetyFilter(
        CbfSafetyFilterConfig(enabled=True, alpha=1.0, max_linear_speed=1.0)
    )

    decision = filter_.filter_command(observation, (0.2, 0.0))

    assert decision.decision_label == "cbf_feasible"
    assert decision.filtered_action == pytest.approx((0.2, 0.0))


def test_wrapper_disabled_returns_nominal_command_unchanged() -> None:
    """Disabled wrapper preserves nominal planner output exactly."""

    class _Planner:
        def plan(self, _observation: dict[str, object]) -> tuple[float, float]:
            return 0.8, 0.1

    wrapper = CbfSafetyFilterPlannerWrapper(_Planner(), CbfSafetyFilterConfig(enabled=False))

    assert wrapper.plan(_head_on_observation()) == (0.8, 0.1)
    assert wrapper.last_decision is None


def test_build_cbf_config_rejects_unimplemented_dynamic_parabolic_variant() -> None:
    """Dynamic-parabolic CBF is deliberately out of this first bounded slice."""

    with pytest.raises(ValueError, match="collision_cone"):
        build_cbf_safety_filter_config({"enabled": True, "variant": "dynamic_parabolic"})


def test_build_cbf_config_validates_nested_mapping_and_bounds() -> None:
    """Config builder handles nested map-runner payloads and rejects bad bounds."""

    cfg = build_cbf_safety_filter_config({"cbf_safety_filter": {"enabled": True, "alpha": 0.5}})
    assert cfg.enabled is True
    assert cfg.alpha == pytest.approx(0.5)

    with pytest.raises(ValueError, match="Unknown"):
        build_cbf_safety_filter_config({"unexpected": True})
    with pytest.raises(ValueError, match="alpha"):
        build_cbf_safety_filter_config({"alpha": -0.1})
    with pytest.raises(ValueError, match="safety_margin"):
        build_cbf_safety_filter_config({"safety_margin": -0.1})
    with pytest.raises(ValueError, match="max_projection_passes"):
        build_cbf_safety_filter_config({"max_projection_passes": 0})


def test_cbf_filter_supports_pedestrian_dict_payload() -> None:
    """Pedestrian matrix observations exercise benchmark observation extraction."""

    filter_ = CollisionConeCbfSafetyFilter(
        CbfSafetyFilterConfig(enabled=True, alpha=1.0, max_linear_speed=1.0)
    )

    decision = filter_.filter_command(
        {
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.8, 0.0],
                "heading": [0.0],
                "radius": [0.3],
            },
            "pedestrians": {
                "positions": [[0.8, 0.0]],
                "velocities": [[-0.6, 0.0]],
                "radius": [0.3],
            },
        },
        (0.8, 0.0),
    )

    assert decision.decision_label in {"cbf_projected", "cbf_best_effort"}
    assert decision.filtered_action[0] < 0.8


def test_cbf_filter_supports_drive_state_and_empty_obstacles() -> None:
    """Drive-state observations without obstacles pass through as feasible."""

    filter_ = CollisionConeCbfSafetyFilter(
        CbfSafetyFilterConfig(
            enabled=True,
            max_linear_speed=0.5,
            max_angular_speed=1.0,
        )
    )

    decision = filter_.filter_command(
        {"drive_state": [1.0, 2.0, 0.25, 0.1, 0.0]},
        (0.0, 0.4),
    )

    assert decision.decision_label == "cbf_feasible"
    assert decision.filtered_action == pytest.approx((0.0, 0.0))


def test_wrapper_enabled_forwards_diagnostics_reset_and_close() -> None:
    """Enabled wrapper filters commands while forwarding planner lifecycle calls."""

    class _Planner:
        def __init__(self) -> None:
            self.reset_seed = None
            self.closed = False

        def plan(self, _observation: dict[str, object]) -> tuple[float, float]:
            return 0.8, 0.0

        def diagnostics(self) -> dict[str, object]:
            return {"planner": "dummy"}

        def reset(self, *, seed: int | None = None) -> str:
            self.reset_seed = seed
            return "reset"

        def close(self) -> None:
            self.closed = True

    planner = _Planner()
    wrapper = CbfSafetyFilterPlannerWrapper(
        planner,
        CbfSafetyFilterConfig(enabled=True, alpha=1.0, max_linear_speed=1.0),
    )

    assert wrapper.plan(_head_on_observation())[0] < 0.8
    assert wrapper.last_decision is not None
    assert wrapper.diagnostics()["wrapped_planner"] == {"planner": "dummy"}
    assert wrapper.reset(seed=7) == "reset"
    assert planner.reset_seed == 7
    wrapper.close()
    assert planner.closed is True
