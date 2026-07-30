"""Protocol and conformance tests for ``LocalPlannerProtocol`` (#6492).

Exercises the canonical local-planner protocol on exactly two representative
families before any 32-planner migration:

1. the native ``LidarOccupancyPlannerAdapter`` (``plan() -> tuple`` family), and
2. the baseline ``SocialForcePlanner`` (``step() -> dict`` family) wrapped by the
   new ``BaselineStepToLocalAdapter``.

These tests cover lifecycle order, reset seed forwarding/ignoring, plan result
shape, diagnostics ``planner_type`` plus explicit unavailable fields, and close
idempotence. They make no benchmark, metric, or performance claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_sf.baselines.social_force import SFPlannerConfig, SocialForcePlanner
from robot_sf.planner.lidar_occupancy import (
    LidarOccupancyGridConfig,
    LidarOccupancyPlannerAdapter,
)
from robot_sf.planner.protocol import (
    DIAGNOSTICS_UNAVAILABLE_KEY,
    DIAGNOSTICS_UNAVAILABLE_REASON_KEY,
    PLANNER_TYPE_KEY,
    BaselineStepToLocalAdapter,
    LocalPlannerProtocol,
    normalize_planner_diagnostics,
)

# ``SocialForcePlanner`` is constructed (never edited) in these tests, then
# wrapped by the canonical baseline->local adapter; the import above sits first
# only to satisfy first-party import ordering.

# ---------------------------------------------------------------------------
# Fixtures and minimal helpers
# ---------------------------------------------------------------------------


def _lidar_config() -> LidarOccupancyGridConfig:
    """Return a compact deterministic LiDAR adapter config."""
    return LidarOccupancyGridConfig(
        resolution=0.5,
        width=4.0,
        height=4.0,
        max_range=4.0,
        angle_min=0.0,
        angle_max=0.0,
        obstacle_inflation_cells=0,
        normalized_rays=False,
        normalized_drive_state=False,
    )


def _lidar_observation() -> dict[str, Any]:
    """Return a minimal LiDAR observation that converts successfully."""
    return {
        "rays": np.asarray([4.0], dtype=np.float32),
        "drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
    }


def _baseline_observation() -> dict[str, Any]:
    """Return a minimal baseline observation accepted by ``SocialForcePlanner.step``."""
    return {
        "dt": 0.1,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "radius": 0.3,
        },
        "agents": [],
        "obstacles": [],
    }


class _RecordingInner:
    """Stub native inner planner recording reset calls for seed-forwarding tests."""

    def __init__(self, *, accepts_seed: bool) -> None:
        self.accepts_seed = accepts_seed
        self.reset_calls: list[Any] = []
        self.plan_calls = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.reset_calls.append((args, kwargs))

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        del observation
        self.plan_calls += 1
        return 0.5, 0.25


class _SeedlessInner:
    """Stub native inner planner with a seedless ``reset`` signature."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        del observation
        return 0.0, 0.0


class _RecordingBaseline:
    """Stub baseline planner recording reset/close for forwarding tests."""

    def __init__(self, *, accepts_seed: bool, action: dict[str, float]) -> None:
        self.accepts_seed = accepts_seed
        self.action = action
        self.reset_calls: list[Any] = []
        self.close_calls = 0
        self.step_calls = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.reset_calls.append((args, kwargs))

    def step(self, obs: Any) -> dict[str, float]:
        del obs
        self.step_calls += 1
        return self.action

    def close(self) -> None:
        self.close_calls += 1


class _SeedlessBaseline:
    """Stub baseline planner with seedless reset and no diagnostics."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, obs: Any) -> dict[str, float]:
        del obs
        return {"v": 1.0, "omega": 0.0}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Protocol runtime conformance
# ---------------------------------------------------------------------------


def test_native_lidar_adapter_satisfies_protocol() -> None:
    """The native LiDAR adapter must be recognized as a LocalPlannerProtocol."""
    adapter = LidarOccupancyPlannerAdapter(_SeedlessInner(), _lidar_config())
    assert isinstance(adapter, LocalPlannerProtocol)


def test_baseline_step_adapter_satisfies_protocol() -> None:
    """The canonical baseline->local adapter must satisfy the protocol."""
    baseline = SocialForcePlanner(SFPlannerConfig(action_space="unicycle"), seed=1)
    adapter = BaselineStepToLocalAdapter(baseline)
    assert isinstance(adapter, LocalPlannerProtocol)


# ---------------------------------------------------------------------------
# Plan result shape
# ---------------------------------------------------------------------------


def test_native_lidar_adapter_plan_returns_float_pair() -> None:
    """Native ``plan`` must return a 2-tuple of finite floats."""
    adapter = LidarOccupancyPlannerAdapter(_RecordingInner(accepts_seed=True), _lidar_config())
    command = adapter.plan(_lidar_observation())
    assert isinstance(command, tuple)
    assert len(command) == 2
    linear, angular = command
    assert isinstance(linear, float)
    assert isinstance(angular, float)
    assert np.isfinite(linear) and np.isfinite(angular)


@pytest.mark.parametrize("action_space", ["unicycle", "velocity"])
def test_baseline_adapter_plan_returns_float_pair(action_space: str) -> None:
    """The baseline adapter must convert both baseline action spaces to a float pair."""
    baseline = SocialForcePlanner(SFPlannerConfig(action_space=action_space), seed=1)
    adapter = BaselineStepToLocalAdapter(baseline)
    command = adapter.plan(_baseline_observation())
    assert isinstance(command, tuple)
    assert len(command) == 2
    linear, angular = command
    assert isinstance(linear, float)
    assert isinstance(angular, float)
    assert linear >= 0.0  # both conversions yield a non-negative linear speed


def test_baseline_adapter_unicycle_maps_v_omega_directly() -> None:
    """Unicycle ``{"v", "omega"}`` actions must map 1:1 onto the command tuple."""
    adapter = BaselineStepToLocalAdapter(
        _RecordingBaseline(accepts_seed=True, action={"v": 1.5, "omega": -0.7})
    )
    assert adapter.plan({}) == (1.5, -0.7)


def test_baseline_adapter_velocity_converts_to_speed_with_zero_angular() -> None:
    """Holonomic ``{"vx", "vy"}`` actions convert to ``(speed, 0.0)`` explicitly."""
    adapter = BaselineStepToLocalAdapter(
        _RecordingBaseline(accepts_seed=True, action={"vx": 3.0, "vy": 4.0})
    )
    assert adapter.plan({}) == (5.0, 0.0)


def test_baseline_adapter_rejects_invalid_action_dict() -> None:
    """An action dict without recognized keys must fail closed, not silently coerce."""
    adapter = BaselineStepToLocalAdapter(_RecordingBaseline(accepts_seed=True, action={"x": 1.0}))
    with pytest.raises(ValueError, match="v/omega or vx/vy"):
        adapter.plan({})


def test_baseline_adapter_rejects_non_dict_action() -> None:
    """A non-dict baseline action must raise TypeError, not be coerced."""

    class _BadBaseline:
        def step(self, obs: Any) -> Any:
            del obs
            return [1.0, 2.0]

    adapter = BaselineStepToLocalAdapter(_BadBaseline())
    with pytest.raises(TypeError, match="must return a dict"):
        adapter.plan({})


# ---------------------------------------------------------------------------
# Reset seed forwarding (when used) and ignoring (when not)
# ---------------------------------------------------------------------------


def test_native_lidar_adapter_reset_forwards_seed_when_used() -> None:
    """Native adapter must forward the seed keyword to a seed-aware inner planner."""
    inner = _RecordingInner(accepts_seed=True)
    adapter = LidarOccupancyPlannerAdapter(inner, _lidar_config())
    adapter.reset(seed=123)
    assert inner.reset_calls == [((), {"seed": 123})]


def test_native_lidar_adapter_reset_ignores_seed_when_not_used() -> None:
    """Native adapter must still reset a seedless inner planner without crashing."""
    inner = _SeedlessInner()
    adapter = LidarOccupancyPlannerAdapter(inner, _lidar_config())
    adapter.reset(seed=123)
    assert inner.reset_calls == 1


def test_baseline_adapter_reset_forwards_seed_when_used() -> None:
    """The baseline adapter must forward the seed keyword to a seed-aware planner."""
    baseline = _RecordingBaseline(accepts_seed=True, action={"v": 1.0, "omega": 0.0})
    adapter = BaselineStepToLocalAdapter(baseline)
    adapter.reset(seed=7)
    assert baseline.reset_calls == [((), {"seed": 7})]


def test_baseline_adapter_reset_ignores_seed_when_not_used() -> None:
    """The baseline adapter must tolerate a seedless baseline reset signature."""
    baseline = _SeedlessBaseline()
    adapter = BaselineStepToLocalAdapter(baseline)
    adapter.reset(seed=99)  # must not raise
    assert baseline.reset_calls == 1


def test_baseline_adapter_reset_no_seed_calls_seedless_baseline() -> None:
    """Resetting without a seed must still call a seedless baseline reset once."""
    baseline = _SeedlessBaseline()
    adapter = BaselineStepToLocalAdapter(baseline)
    adapter.reset()
    assert baseline.reset_calls == 1


# ---------------------------------------------------------------------------
# Diagnostics: planner_type plus explicit unavailable fields (fail-closed)
# ---------------------------------------------------------------------------


def test_normalize_passthrough_when_planner_type_present() -> None:
    """A valid planner_type must pass through unchanged with no unavailable keys."""
    raw = {"planner_type": "DWAPlannerAdapter", "last_decision": {"v": 1.0}}
    normalized = normalize_planner_diagnostics(raw, fallback_planner_type="Fallback")
    assert normalized["planner_type"] == "DWAPlannerAdapter"
    assert normalized["last_decision"] == {"v": 1.0}
    assert DIAGNOSTICS_UNAVAILABLE_KEY not in normalized


def test_normalize_synthesizes_when_planner_type_missing() -> None:
    """Missing planner_type must be synthesized with explicit unavailable fields."""
    raw = {"lidar_occupancy_adapter": {"converted_observations": 3}}
    normalized = normalize_planner_diagnostics(
        raw, fallback_planner_type="LidarOccupancyPlannerAdapter"
    )
    assert normalized["planner_type"] == "LidarOccupancyPlannerAdapter"
    assert normalized["lidar_occupancy_adapter"] == {"converted_observations": 3}
    assert normalized[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]
    assert PLANNER_TYPE_KEY in normalized[DIAGNOSTICS_UNAVAILABLE_REASON_KEY]


def test_normalize_fail_closed_on_non_mapping() -> None:
    """A non-mapping raw payload must be normalized fail-closed with a reason."""
    normalized = normalize_planner_diagnostics(None, fallback_planner_type="SocialForcePlanner")
    assert normalized["planner_type"] == "SocialForcePlanner"
    assert normalized[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]
    assert "NoneType" in normalized[DIAGNOSTICS_UNAVAILABLE_REASON_KEY]


def test_normalize_fail_closed_on_invalid_planner_type() -> None:
    """A non-string planner_type must be replaced and recorded as unavailable."""
    normalized = normalize_planner_diagnostics(
        {"planner_type": 42}, fallback_planner_type="Fallback"
    )
    assert normalized["planner_type"] == "Fallback"
    assert normalized[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]
    assert "42" in normalized[DIAGNOSTICS_UNAVAILABLE_REASON_KEY]


def test_native_lidar_adapter_diagnostics_behavior_unchanged() -> None:
    """The native adapter's raw diagnostics must remain unchanged (no planner_type)."""
    adapter = LidarOccupancyPlannerAdapter(_SeedlessInner(), _lidar_config())
    raw = adapter.diagnostics()
    # Existing behavior preserved: nested adapter payload, no top-level planner_type.
    assert "lidar_occupancy_adapter" in raw
    assert PLANNER_TYPE_KEY not in raw


def test_native_lidar_diagnostics_normalize_fail_closed() -> None:
    """The native raw diagnostics normalize fail-closed to a synthesized planner_type."""
    adapter = LidarOccupancyPlannerAdapter(_SeedlessInner(), _lidar_config())
    normalized = normalize_planner_diagnostics(
        adapter.diagnostics(), fallback_planner_type="LidarOccupancyPlannerAdapter"
    )
    assert normalized["planner_type"] == "LidarOccupancyPlannerAdapter"
    assert normalized[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]


def test_baseline_adapter_diagnostics_fail_closed_without_diagnostics_method() -> None:
    """The real SocialForcePlanner has no diagnostics(); the adapter must fail closed."""
    baseline = SocialForcePlanner(SFPlannerConfig(action_space="unicycle"), seed=1)
    adapter = BaselineStepToLocalAdapter(baseline)
    diagnostics = adapter.diagnostics()
    assert diagnostics[PLANNER_TYPE_KEY] == "SocialForcePlanner"
    assert diagnostics[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]
    assert "did not return a mapping" in diagnostics[DIAGNOSTICS_UNAVAILABLE_REASON_KEY]


def test_baseline_adapter_explicit_planner_type_overrides_default() -> None:
    """An explicit planner_type must be honored and still fail closed on the fields."""
    baseline = _SeedlessBaseline()
    adapter = BaselineStepToLocalAdapter(baseline, planner_type="social_force")
    diagnostics = adapter.diagnostics()
    assert diagnostics[PLANNER_TYPE_KEY] == "social_force"
    assert diagnostics[DIAGNOSTICS_UNAVAILABLE_KEY] == [PLANNER_TYPE_KEY]


# ---------------------------------------------------------------------------
# Close idempotence
# ---------------------------------------------------------------------------


def test_native_lidar_adapter_close_is_idempotent_noop() -> None:
    """The native adapter's new close() is a no-op and idempotent."""
    adapter = LidarOccupancyPlannerAdapter(_SeedlessInner(), _lidar_config())
    adapter.close()
    adapter.close()  # second call must not raise
    # plan/reset/diagnostics still work after close (close is terminal but harmless).
    adapter.reset(seed=0)
    assert adapter.plan(_lidar_observation()) == (0.0, 0.0)


def test_baseline_adapter_close_is_idempotent_and_forwards_once() -> None:
    """The baseline adapter must forward close exactly once and stay idempotent."""
    baseline = _RecordingBaseline(accepts_seed=True, action={"v": 1.0, "omega": 0.0})
    adapter = BaselineStepToLocalAdapter(baseline)
    adapter.close()
    adapter.close()
    assert baseline.close_calls == 1


# ---------------------------------------------------------------------------
# Lifecycle order
# ---------------------------------------------------------------------------


def test_native_lidar_adapter_full_lifecycle_order() -> None:
    """reset -> plan -> diagnostics -> close must run cleanly in order on the native adapter."""
    adapter = LidarOccupancyPlannerAdapter(_RecordingInner(accepts_seed=True), _lidar_config())
    adapter.reset(seed=42)
    command = adapter.plan(_lidar_observation())
    assert command == (0.5, 0.25)
    raw = adapter.diagnostics()
    assert raw["lidar_occupancy_adapter"]["converted_observations"] == 1
    adapter.close()


def test_baseline_adapter_full_lifecycle_order() -> None:
    """reset -> plan -> diagnostics -> close must run cleanly on the baseline adapter."""
    baseline = SocialForcePlanner(SFPlannerConfig(action_space="unicycle"), seed=1)
    adapter = BaselineStepToLocalAdapter(baseline)
    adapter.reset(seed=2)
    command = adapter.plan(_baseline_observation())
    assert isinstance(command, tuple) and len(command) == 2
    diagnostics = adapter.diagnostics()
    assert diagnostics[PLANNER_TYPE_KEY] == "SocialForcePlanner"
    adapter.close()


def test_baseline_adapter_wraps_real_social_force_end_to_end() -> None:
    """End-to-end proof: the real SocialForcePlanner is exercisable through the protocol."""
    baseline = SocialForcePlanner(SFPlannerConfig(action_space="unicycle"), seed=0)
    adapter = BaselineStepToLocalAdapter(baseline)
    assert isinstance(adapter, LocalPlannerProtocol)
    adapter.reset(seed=0)
    linear, angular = adapter.plan(_baseline_observation())
    assert isinstance(linear, float) and isinstance(angular, float)
    diagnostics = adapter.diagnostics()
    assert isinstance(diagnostics[PLANNER_TYPE_KEY], str) and diagnostics[PLANNER_TYPE_KEY]
    adapter.close()


@pytest.mark.parametrize(
    "rel_path,cls_name,method_name",
    [
        ("robot_sf/planner/risk_dwa.py", "RiskDWAPlannerAdapter", "plan"),
        ("robot_sf/planner/mppi_social.py", "MPPISocialPlannerAdapter", "plan"),
        ("robot_sf/planner/predictive_mppi.py", "PredictiveMPPIAdapter", "plan"),
        ("robot_sf/planner/safety_barrier.py", "SafetyBarrierPlannerAdapter", "plan"),
        ("robot_sf/planner/stream_gap.py", "StreamGapPlannerAdapter", "plan"),
        ("robot_sf/planner/guarded_ppo.py", "GuardedPPOAdapter", "_violated_constraints"),
        ("robot_sf/planner/visibility_planner.py", "VisibilityPlanner", "_nearest_neighbor_order"),
    ],
)
def test_diagnostics_insertion_did_not_drop_preceding_return(
    rel_path: str, cls_name: str, method_name: str
) -> None:
    """Regression guard (#6505): ``diagnostics()`` insertion must not erase a return.

    The mechanical insertion of ``diagnostics()`` replaced the final ``return`` of
    several ``plan()``/helper methods, silently making them fall through to
    ``None``. Existing behavioral coverage did not reach those final returns, so
    the regression passed CI. Assert each affected concrete method still contains
    a ``return`` statement so the regression fails fast if it recurs.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    tree = ast.parse((repo_root / rel_path).read_text())
    cls = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == cls_name
    )
    method = next(
        member
        for member in cls.body
        if isinstance(member, ast.FunctionDef) and member.name == method_name
    )
    assert any(isinstance(node, ast.Return) for node in ast.walk(method)), (
        f"{cls_name}.{method_name} in {rel_path} lost its return statement"
    )
