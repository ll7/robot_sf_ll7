"""Contract and import-boundary tests for ``robot_sf.core``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robot_sf.adversarial.config import Pose2D as ExistingPose2D
from robot_sf.benchmark.types import EpisodeRecord as ExistingEpisodeRecord
from robot_sf.core import (
    CORE_CONTRACT_VERSION,
    DT_DECOMPOSITION_STAGE_ORDER,
    ActorState,
    EpisodeRecord,
    ForceBreakdown,
    ForceComponent,
    ObservationSnapshot,
    Pose2D,
    SimTime,
    TransitionRecord,
    Twist2D,
    WorldFrame,
)
from robot_sf.prediction.oracle_transition_trace import (
    ForceComponentRecord,
    ForceComponents,
    OracleTransitionTraceV1,
)
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianCoordinateFrame,
    PedestrianObservationSnapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_importing_core_keeps_optional_runtime_backends_unloaded() -> None:
    """The additive package remains safe for lightweight tools and docs probes."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import robot_sf.core; "
                "forbidden = {'pygame', 'torch', 'stable_baselines3'} & set(sys.modules); "
                "assert not forbidden, forbidden"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_existing_contracts_are_re_exported_by_identity() -> None:
    """Core names must not create parallel pose, observation, or evidence types."""

    assert Pose2D is ExistingPose2D
    assert WorldFrame is PedestrianCoordinateFrame
    assert ObservationSnapshot is PedestrianObservationSnapshot
    assert ForceComponent is ForceComponentRecord
    assert ForceBreakdown is ForceComponents
    assert TransitionRecord is OracleTransitionTraceV1
    assert EpisodeRecord is ExistingEpisodeRecord
    assert CORE_CONTRACT_VERSION == "core_contract.v1"


def test_dt_decomposition_stage_order_is_frozen_and_type_only() -> None:
    """Expose the documented fixed-step order without owning simulator wiring."""

    assert isinstance(DT_DECOMPOSITION_STAGE_ORDER, tuple)
    assert DT_DECOMPOSITION_STAGE_ORDER == (
        "start_of_step_state",
        "post_behaviour_pedestrian_state",
        "force_evaluation_state",
        "component_forces",
        "final_pre_cap_force",
        "uncapped_velocity",
        "applied_capped_velocity",
        "integrated_state",
        "observation",
        "recorded_transition",
    )


def test_sim_time_is_dt_derived_and_round_trips() -> None:
    """Discrete and continuous time remain finite and serializable."""

    value = SimTime.from_step(4, 0.1)

    assert value == SimTime.from_dict(value.to_dict())
    assert value.step_index == 4
    assert value.seconds == pytest.approx(0.4)
    assert value.advance(0.1) == SimTime(5, 0.5)

    with pytest.raises(ValueError, match="positive"):
        SimTime.from_step(1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        SimTime(0, float("nan"))


def test_twist_is_finite_and_round_trips() -> None:
    """Linear and angular velocity use explicit SI units and signed values."""

    value = Twist2D(vx=-0.5, vy=1.25, omega=0.3)

    assert value == Twist2D.from_dict(value.to_dict())
    assert value.velocity_xy == (-0.5, 1.25)
    assert value.angular_velocity_rad_s == pytest.approx(0.3)

    with pytest.raises(ValueError, match="finite"):
        Twist2D(vx=0.0, vy=float("inf"))


def test_actor_state_round_trip_separates_source_and_track_identity() -> None:
    """Actor identity, optional track identity, frame, validity, and time persist."""

    value = ActorState(
        actor_id="pedestrian-7",
        track_id="track-2",
        pose=Pose2D(1.0, -2.0, 0.25),
        twist=Twist2D(0.4, -0.1, 0.05),
        time=SimTime.from_step(3, 0.2),
        coordinate_frame=WorldFrame.GLOBAL_XY,
        valid=True,
        source_identity="observation_tracker",
    )

    restored = ActorState.from_dict(value.to_dict())

    assert restored == value
    assert restored.frame is WorldFrame.GLOBAL_XY
    assert restored.position_xy == (1.0, -2.0)
    assert restored.velocity_xy == (0.4, -0.1)
    assert restored.step_index == 3
    assert restored.timestamp_s == pytest.approx(0.6)


def test_actor_state_rejects_nonfinite_pose_and_unknown_serialized_keys() -> None:
    """State values and schema evolution fail closed rather than being coerced."""

    with pytest.raises(ValueError, match="finite"):
        ActorState(
            actor_id="pedestrian-7",
            pose=Pose2D(float("nan"), 0.0),
            twist=Twist2D(0.0, 0.0),
            time=SimTime(0, 0.0),
        )

    payload = ActorState(
        actor_id="pedestrian-7",
        pose=Pose2D(0.0, 0.0),
        twist=Twist2D(0.0, 0.0),
        time=SimTime(0, 0.0),
    ).to_dict()
    payload["future_field"] = "must-not-be-dropped"
    with pytest.raises(ValueError, match="keys mismatch"):
        ActorState.from_dict(payload)


def test_new_value_objects_are_frozen() -> None:
    """Core state cannot mutate after it crosses a consumer boundary."""

    with pytest.raises(FrozenInstanceError):
        SimTime(0, 0.0).seconds = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        Twist2D(0.0, 0.0).vx = 1.0  # type: ignore[misc]
