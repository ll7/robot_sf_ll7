"""Issue #6506 parity guard for canonical planner construction and diagnostics.

Issue #6506 replaces the duplicated planner bridges at the benchmark harness
boundary with the canonical adapter introduced by issue #6492
(``LocalPlannerProtocol``). Its scoped wiring covers the baseline
``step(obs) -> dict`` path in ``robot_sf/benchmark/runner.py`` and the external-
MPC ``plan(observation) -> tuple`` bridge in ``robot_sf/benchmark/map_runner/__init__.py``.

The #6771 injection points (merged as #6813) now let the runner construct
``BaselineStepToLocalAdapter`` with its existing process-isolated step executor
and world-velocity projector. The map-runner external-MPC bridge reuses the same
canonical adapter with callbacks for its external observation format and
heading-aware action projection. Both paths retain their existing execution-mode
metadata, cleanup, and fail-closed behavior. Their diagnostics continue through
the canonical ``normalize_planner_diagnostics`` path: the runner native-command
arm (``run_episode``) and the map-runner adapter arm
(``_build_common_adapter_policy._planner_stats``) share one diagnostics schema
that always carries a string ``planner_type``. This module pins the parity
contract required by #6506:

    "A parity assertion that native vs adapter execution-mode metadata and
    planner_diagnostics are unchanged for representative planners."

Concretely these tests pin, for representative planners across all three
execution modes:

* the ``planner_kinematics.execution_mode`` classification (``native`` /
  ``adapter`` / ``mixed``) plus the zeroed ``adapter_impact`` counter schema,
  which the diagnostics unification must not alter, and
* the canonical ``planner_diagnostics`` propagation path: both the runner
  native-command arm and the map-runner SocNav-family adapter arm must route
  through ``normalize_planner_diagnostics`` so the payload carries a string
  ``planner_type`` while every existing counter is preserved unchanged.
"""

from __future__ import annotations

import math
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.benchmark import map_runner, runner
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.map_runner import build_map_policy
from robot_sf.benchmark.runner import NATIVE_COMMAND_DIAGNOSTICS_KEY, run_episode
from robot_sf.planner.protocol import BaselineStepToLocalAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping

# Representative planners spanning every execution-mode family the #6506
# canonical wiring must preserve. The ``algo_key`` values are the canonical keys the
# benchmark registry resolves via ``canonical_algorithm_name``; they are the
# stable handles both ``runner.py`` and ``map_runner.py`` classify against.
EXECUTION_MODE_REPRESENTATIVES: list[tuple[str, str]] = [
    # Native family: baseline ``step(obs) -> dict`` planners routed through the
    # canonical adapter in ``runner.py`` (including the process-isolated arm).
    ("goal", "native"),
    ("sac", "native"),
    ("simple_policy", "native"),
    ("native_command", "native"),
    # Adapter family: planner ``plan(observation) -> tuple`` arm wired through
    # the SocNav-family adapter construction in ``map_runner.py``.
    ("social_force", "adapter"),
    ("orca", "adapter"),
    ("prediction_planner", "adapter"),
    ("drl_vo", "adapter"),
    # Mixed family: learned-policy ``step()`` arms in ``map_runner.py`` that fuse
    # native command spaces with an adapter projection.
    ("ppo", "mixed"),
    ("guarded_ppo", "mixed"),
]

# Minimal inline native-command stub: one JSON state frame in, one JSON velocity
# command out. Keeps the parity test free of any external planner binary or GPU.
_NATIVE_STUB_SRC = (
    "import sys, json\n"
    "req = json.loads(sys.stdin.readline())\n"
    "pos = req['robot_pos']; goal = req['robot_goal']\n"
    "dx = goal[0] - pos[0]; dy = goal[1] - pos[1]\n"
    "d = (dx * dx + dy * dy) ** 0.5\n"
    "print(json.dumps({'v': min(1.0, d) if d > 1e-6 else 0.0, 'omega': 0.0}))\n"
    "sys.stdout.flush()\n"
)


def _native_command_scenario() -> dict[str, object]:
    """Return a minimal native-command scenario invoking the inline stub."""
    return {
        "id": "nc_parity",
        "num_pedestrians": 0,
        "algo": "native_command",
        "native_command": {
            "argv": [sys.executable, "-c", _NATIVE_STUB_SRC],
            "env": {},
            "timeout_s": 2.0,
            "persistent": False,
        },
    }


@pytest.mark.parametrize("algo_key, expected_mode", EXECUTION_MODE_REPRESENTATIVES)
def test_execution_mode_classification_baseline(algo_key: str, expected_mode: str) -> None:
    """Execution-mode classification is unchanged for representative planners.

    The #6506 canonical wiring must preserve the native/adapter/mixed label each
    representative earns today via ``enrich_algorithm_metadata``.
    """
    metadata = enrich_algorithm_metadata(algo=algo_key, adapter_impact_requested=True)
    planner_kinematics = metadata["planner_kinematics"]
    assert planner_kinematics["execution_mode"] == expected_mode
    # ``adapter_active`` is the derived parity flag the benchmark reports: it is
    # True exactly for adapter/mixed modes and False for native. A flip here
    # would change the execution-mode classification surfaced to consumers.
    assert planner_kinematics["adapter_active"] is (expected_mode in {"adapter", "mixed"})

    # The canonical construction path must retain the additive
    # adapter-impact counters for every execution-mode family. Actual values
    # are accumulated at runtime, so metadata initialization is deliberately
    # zeroed here rather than fabricated by this contract test.
    adapter_impact = metadata["adapter_impact"]
    assert adapter_impact == {
        "requested": True,
        "native_steps": 0,
        "adapted_steps": 0,
        "status": "pending",
    }


def test_map_runner_socnav_adapter_preserves_adapter_metadata() -> None:
    """The map-runner entry point retains its adapter classification baseline."""
    policy, metadata = build_map_policy("social_force", {})
    try:
        planner_kinematics = metadata["planner_kinematics"]
        assert planner_kinematics["execution_mode"] == "adapter"
        assert planner_kinematics["adapter_active"] is True
        assert planner_kinematics["adapter_name"] == "SocialForcePlannerAdapter"
        assert getattr(policy, "_planner_adapter", None) is not None
    finally:
        close = getattr(policy, "_planner_close", None)
        if callable(close):
            close()


def test_native_command_arm_propagates_native_mode_and_diagnostics() -> None:
    """The runner's canonical ``step()->dict`` arm keeps native mode + diagnostics.

    The native-command arm is the representative of the baselines ``step()``
    family in ``runner.py``. It must classify as ``native`` and surface its
    counters under ``algorithm_metadata.planner_diagnostics`` (the
    ``NATIVE_COMMAND_DIAGNOSTICS_KEY`` propagation path). The #6506 wiring
    routes the payload through the canonical
    ``normalize_planner_diagnostics`` path, so it must carry a string
    ``planner_type`` while every existing counter is preserved unchanged.
    """
    record = run_episode(
        _native_command_scenario(),
        seed=7,
        algo="native_command",
        horizon=8,
        dt=0.1,
        record_forces=False,
    )
    algorithm_metadata = record["algorithm_metadata"]
    assert algorithm_metadata["planner_kinematics"]["execution_mode"] == "native"
    assert algorithm_metadata["planner_kinematics"]["adapter_active"] is False

    # The diagnostics key the native arm writes must round-trip into the
    # episode algorithm_metadata block under its canonical name.
    assert NATIVE_COMMAND_DIAGNOSTICS_KEY in algorithm_metadata
    diagnostics = algorithm_metadata[NATIVE_COMMAND_DIAGNOSTICS_KEY]
    for counter_key in (
        "expansion_limit_hits",
        "runtime_bound_exits",
        "fallback_count",
        "commitment_invalidations",
        "process_spawns",
    ):
        assert isinstance(diagnostics[counter_key], int)
        assert diagnostics[counter_key] >= 0
    # A healthy per-episode native run records one spawn per step and no fallback.
    assert diagnostics["process_spawns"] == 8
    assert diagnostics["fallback_count"] == 0
    assert len(diagnostics["planner_step_runtime_seconds"]) == 8

    # Canonical #6492 diagnostics schema: the unified propagation path must
    # guarantee a string ``planner_type``. The native-command arm does not
    # produce one natively, so the fail-closed normalizer synthesizes it from
    # the algorithm key and records the synthesis explicitly.
    assert isinstance(diagnostics.get("planner_type"), str)
    assert diagnostics["planner_type"] == "native_command"
    assert diagnostics.get("diagnostics_unavailable") == ["planner_type"]


def test_diagnostics_propagation_routes_through_canonical_adapter() -> None:
    """Both harness arms route ``planner_diagnostics`` through the #6492 adapter.

    The #6506 unification imports the single canonical adapter from #6492
    (``normalize_planner_diagnostics``) into both ``runner.py`` and
    ``map_runner.py`` so the native-command arm and the SocNav-family adapter
    arm share one fail-closed diagnostics propagation path. Each payload must
    carry a string ``planner_type`` while preserving the counters/values the
    arm already produced.
    """
    from robot_sf.planner.protocol import PLANNER_TYPE_KEY

    # Native-command arm (runner.py): ``planner_type`` synthesized from the algo key.
    record = run_episode(
        _native_command_scenario(),
        seed=3,
        algo="native_command",
        horizon=4,
        dt=0.1,
        record_forces=False,
    )
    native_diag = record["algorithm_metadata"][NATIVE_COMMAND_DIAGNOSTICS_KEY]
    assert native_diag[PLANNER_TYPE_KEY] == "native_command"
    assert native_diag["process_spawns"] == 4

    # SocNav-family adapter arm (map_runner.py): ``planner_type`` synthesized
    # from the adapter class name by the canonical normalizer.
    policy, _meta = build_map_policy("social_force", {})
    try:
        stats_fn = getattr(policy, "_planner_stats", None)
        assert callable(stats_fn), "SocNav adapter policy must expose _planner_stats"
        adapter_diag = stats_fn()
        assert adapter_diag[PLANNER_TYPE_KEY] == "SocialForcePlannerAdapter"
    finally:
        close = getattr(policy, "_planner_close", None)
        if callable(close):
            close()


def test_adapter_diagnostics_preserve_mapping_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical diagnostics path preserves non-dict Mapping payloads."""

    class MappingDiagnosticsAdapter:
        """Fixture adapter that exposes a valid immutable Mapping diagnostics payload."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, _obs: dict[str, object]) -> tuple[float, float]:
            """Return a no-op unicycle command for policy construction."""
            return 0.0, 0.0

        def diagnostics(self) -> Mapping[str, object]:
            """Return counters through a Mapping implementation rather than dict."""
            return MappingProxyType({"planner_type": "mapping_fixture", "preserved_counter": 7})

    monkeypatch.setattr(map_runner, "SocialForcePlannerAdapter", MappingDiagnosticsAdapter)
    policy, _meta = map_runner.build_map_policy("social_force", {})
    stats_fn = getattr(policy, "_planner_stats", None)
    assert callable(stats_fn), "SocNav adapter policy must expose _planner_stats"
    assert stats_fn() == {"planner_type": "mapping_fixture", "preserved_counter": 7}


def test_adapter_diagnostics_keep_primary_schema_over_foresight_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supplementary foresight data cannot replace canonical primary schema fields."""

    class CollidingDiagnosticsAdapter:
        """Fixture whose supplementary diagnostics deliberately reuse reserved keys."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, _obs: dict[str, object]) -> tuple[float, float]:
            """Return a no-op unicycle command for policy construction."""
            return 0.0, 0.0

        def diagnostics(self) -> dict[str, object]:
            """Return the primary planner diagnostics schema and a counter."""
            return {"planner_type": "primary_fixture", "preserved_counter": 7}

        def foresight_diagnostics(self) -> dict[str, object]:
            """Return supplemental data that attempts to collide with reserved keys."""
            return {
                "planner_type": "foresight_fixture",
                "diagnostics_unavailable": ["foresight_fixture"],
                "diagnostics_unavailable_reason": "must not replace primary schema",
                "foresight_counter": 11,
            }

    monkeypatch.setattr(map_runner, "SocialForcePlannerAdapter", CollidingDiagnosticsAdapter)
    policy, _meta = map_runner.build_map_policy("social_force", {})
    stats_fn = getattr(policy, "_planner_stats", None)
    assert callable(stats_fn), "SocNav adapter policy must expose _planner_stats"
    assert stats_fn() == {
        "planner_type": "primary_fixture",
        "preserved_counter": 7,
        "foresight_counter": 11,
    }


def test_adapter_diagnostics_fail_closed_for_non_mapping_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed adapter diagnostics retain their source-type reason and foresight counters."""

    class InvalidDiagnosticsAdapter:
        """Fixture adapter with malformed primary diagnostics and valid supplementary counters."""

        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, _obs: dict[str, object]) -> tuple[float, float]:
            """Return a no-op unicycle command for policy construction."""
            return 0.0, 0.0

        def diagnostics(self) -> str:
            """Return an invalid payload that the canonical normalizer must report."""
            return "not-a-mapping"

        def foresight_diagnostics(self) -> dict[str, int]:
            """Return valid supplemental counters that must survive diagnostics normalization."""
            return {"foresight_counter": 7}

    monkeypatch.setattr(map_runner, "SocialForcePlannerAdapter", InvalidDiagnosticsAdapter)
    policy, _meta = map_runner.build_map_policy("social_force", {})
    stats_fn = getattr(policy, "_planner_stats", None)
    assert callable(stats_fn), "SocNav adapter policy must expose _planner_stats"

    assert stats_fn() == {
        "planner_type": "InvalidDiagnosticsAdapter",
        "diagnostics_unavailable": ["planner_type"],
        "diagnostics_unavailable_reason": "diagnostics() did not return a mapping (got str)",
        "foresight_counter": 7,
    }


def test_native_command_diagnostics_fail_closed_for_non_mapping_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner normalizes malformed live diagnostics instead of retaining stale data."""
    original_create_policy = runner._create_robot_policy

    def _create_with_invalid_diagnostics(*args: object, **kwargs: object) -> tuple[object, object]:
        policy, metadata = original_create_policy(*args, **kwargs)
        policy.diagnostics = lambda: "not-a-mapping"
        return policy, metadata

    monkeypatch.setattr(runner, "_create_robot_policy", _create_with_invalid_diagnostics)
    record = runner.run_episode(
        _native_command_scenario(),
        seed=11,
        algo="native_command",
        horizon=2,
        dt=0.1,
        record_forces=False,
    )

    diagnostics = record["algorithm_metadata"][NATIVE_COMMAND_DIAGNOSTICS_KEY]
    assert diagnostics["planner_type"] == "native_command"
    assert diagnostics["diagnostics_unavailable"] == ["planner_type"]
    assert diagnostics["diagnostics_unavailable_reason"] == (
        "diagnostics() did not return a mapping (got str)"
    )


class _RecordingMpcPlanner:
    """Stub external-MPC ``step(obs) -> dict`` planner that records its input obs."""

    def __init__(self, action: dict[str, object]) -> None:
        """Store the action the stub returns from every ``step`` call."""
        self.action = action
        self.received_obs: dict[str, object] | None = None

    def step(self, obs: dict[str, object]) -> dict[str, object]:
        """Record the observation and return the configured action dict."""
        self.received_obs = obs
        return self.action


def _external_mpc_adapter(stub: _RecordingMpcPlanner) -> object:
    """Build an external-MPC adapter over a stub planner."""
    return map_runner._ExternalMPCAdapter(
        stub,
        algo_config={},
        robot_kinematics=None,
        planner_name="StubExternalMPC",
    )


def test_external_mpc_adapter_uses_canonical_step_adapter() -> None:
    """The external-MPC bridge is the canonical adapter with injected callbacks."""
    adapter = _external_mpc_adapter(_RecordingMpcPlanner({"v": 0.5, "omega": 0.1}))
    assert isinstance(adapter, BaselineStepToLocalAdapter)


def test_external_mpc_adapter_forwards_mpc_observation_format() -> None:
    """The MPC bridge preserves its observation transform and unicycle output."""
    stub = _RecordingMpcPlanner({"v": 0.5, "omega": 0.1})
    adapter = _external_mpc_adapter(stub)
    obs = {
        "robot": {"heading": [0.0], "position": [0.0, 0.0]},
        "goal": {"current": [1.0, 0.0]},
        "obstacles": [{"x": 1.0, "y": 2.0}],
    }

    linear, angular = adapter.plan(obs)

    assert stub.received_obs == map_runner._obs_to_external_mpc_format(obs)
    assert stub.received_obs is not None and "obstacles" in stub.received_obs
    assert (linear, angular) == (0.5, 0.1)


def test_external_mpc_adapter_converts_holonomic_action() -> None:
    """The MPC bridge preserves heading-error angular derivation."""
    stub = _RecordingMpcPlanner({"vx": 1.0, "vy": 0.0})
    adapter = _external_mpc_adapter(stub)

    assert adapter.plan({"robot": {"heading": [0.0]}}) == (1.0, 0.0)
    assert adapter.plan({"robot": {"heading": [math.pi / 2]}}) == (1.0, -1.0)


def test_external_mpc_adapter_rejects_non_dict_action() -> None:
    """The MPC bridge fails closed on a non-dict ``step`` action payload."""

    class InvalidPlanner(_RecordingMpcPlanner):
        def __init__(self) -> None:
            super().__init__({})

        def step(self, obs: dict[str, object]) -> object:
            self.received_obs = obs
            return "not-a-dict"

    adapter = _external_mpc_adapter(InvalidPlanner())
    with pytest.raises(TypeError, match="StubExternalMPC"):
        adapter.plan({"robot": {"heading": [0.0]}})


class _StubStepRunner:
    """Minimal process-runner stand-in for the baseline canonical-adapter test."""

    def __init__(self, action: dict[str, float]) -> None:
        """Store the action returned by the isolated-step stand-in."""
        self.action = action
        self.calls = 0
        self.closed = False

    def step(self, _obs: object) -> dict[str, float]:
        """Return the configured action and record the isolated call."""
        self.calls += 1
        return self.action

    def close(self) -> None:
        """Record worker cleanup."""
        self.closed = True


def test_runner_baseline_policy_uses_canonical_adapter_without_output_change() -> None:
    """The runner adapter preserves the existing world-velocity projection."""
    action = {"v": 1.5, "omega": 0.2}
    step_runner = _StubStepRunner(action)
    planner = object()
    metadata: dict[str, object] = {}
    timeout_metadata: dict[str, object] = {}
    policy = runner._build_baseline_policy_fn(
        algo="stub",
        planner=planner,
        observation_cls=lambda **kwargs: kwargs,
        step_runner=step_runner,
        timeout_metadata=timeout_metadata,
        metadata=metadata,
        retry_budget=0,
        robot_radius=0.3,
        ped_radius=0.35,
    )
    robot_pos = np.array([0.0, 0.0])
    robot_vel = np.array([0.0, 1.0])
    robot_goal = np.array([3.0, 0.0])

    actual = policy(robot_pos, robot_vel, robot_goal, np.empty((0, 2)), 0.1)
    expected = runner._action_to_velocity(
        action,
        robot_pos,
        robot_vel,
        robot_goal,
        "stub",
    )

    assert isinstance(policy._planner_adapter, BaselineStepToLocalAdapter)
    assert step_runner.calls == 1
    np.testing.assert_allclose(actual, expected)
    policy.close()
    assert step_runner.closed is True
