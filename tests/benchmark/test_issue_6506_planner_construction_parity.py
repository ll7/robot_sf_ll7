"""Issue #6506 parity guard for unified planner-construction diagnostics.

Issue #6506 asks the benchmark harness to replace its two duplicated planner
bridging layers -- the baselines ``step(obs) -> dict`` arm in
``robot_sf/benchmark/runner.py`` (native-command / simple policy) and the
planner ``plan(observation) -> tuple`` arm in ``robot_sf/benchmark/map_runner.py``
(SocNav-family adapter construction) -- with the single canonical adapter
introduced by issue #6492 (``LocalPlannerProtocol``).

#6492 has now merged its canonical adapter (``robot_sf/planner/protocol.py``).
#6506 wires both harness entry points' ``planner_diagnostics`` propagation
through that single canonical adapter's fail-closed normalizer
(``normalize_planner_diagnostics``): the runner native-command arm
(``run_episode``) and the map-runner SocNav-family adapter arm
(``_build_common_adapter_policy._planner_stats``) now share one diagnostics
schema that always carries a string ``planner_type``. The full step()->dict /
plan()->tuple *bridging* collapse remains blocked behind #6506 stop-condition 3
(the canonical adapter is a proof-of-concept and lacks the process-isolated,
heading-based, holonomic-passthrough, and learned-action conversions the two
arms actually use); this module pins the contract the diagnostics unification
must preserve -- validation requirement #4 of the #6506 contract:

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

import sys
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark import map_runner, runner
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.map_runner import build_map_policy
from robot_sf.benchmark.runner import NATIVE_COMMAND_DIAGNOSTICS_KEY, run_episode

if TYPE_CHECKING:
    from collections.abc import Mapping

# Representative planners spanning every execution-mode family the #6506
# unification must preserve. The ``algo_key`` values are the canonical keys the
# benchmark registry resolves via ``canonical_algorithm_name``; they are the
# stable handles both ``runner.py`` and ``map_runner.py`` classify against.
EXECUTION_MODE_REPRESENTATIVES: list[tuple[str, str]] = [
    # Native family: baselines ``step(obs) -> dict`` arm wired in ``runner.py``
    # (simple goal-seeking policy, SAC learned policy, native-command process arm).
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

    The unified canonical adapter from #6492/#6506 must route every planner
    through one construction path while preserving the native/adapter/mixed
    label each representative earns today via ``enrich_algorithm_metadata``.
    """
    metadata = enrich_algorithm_metadata(algo=algo_key, adapter_impact_requested=True)
    planner_kinematics = metadata["planner_kinematics"]
    assert planner_kinematics["execution_mode"] == expected_mode
    # ``adapter_active`` is the derived parity flag the benchmark reports: it is
    # True exactly for adapter/mixed modes and False for native. A flip here
    # would change the execution-mode classification surfaced to consumers.
    assert planner_kinematics["adapter_active"] is (expected_mode in {"adapter", "mixed"})

    # The eventual single construction path must retain the additive
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
    """The runner.py ``step()->dict`` arm keeps native mode + planner_diagnostics.

    The native-command arm is the representative of the baselines ``step()``
    family in ``runner.py``. It must classify as ``native`` and surface its
    counters under ``algorithm_metadata.planner_diagnostics`` (the
    ``NATIVE_COMMAND_DIAGNOSTICS_KEY`` propagation path). After the #6506 rewire
    the payload is routed through the canonical ``normalize_planner_diagnostics``
    adapter from #6492, so it must additionally carry a string ``planner_type``
    while every existing counter is preserved unchanged.
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
