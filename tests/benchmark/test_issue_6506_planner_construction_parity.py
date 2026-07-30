"""Issue #6506 pre-refactor parity baseline for planner construction.

Issue #6506 asks the benchmark harness to replace its two duplicated planner
bridging layers -- the baselines ``step(obs) -> dict`` arm in
``robot_sf/benchmark/runner.py`` (native-command / simple policy) and the
planner ``plan(observation) -> tuple`` arm in ``robot_sf/benchmark/map_runner.py``
(SocNav-family adapter construction) -- with the single canonical adapter
introduced by issue #6492 (``LocalPlannerProtocol``).

That canonical adapter does not exist yet: #6492 is open with no merged PR and
``robot_sf/planner/protocol.py`` is absent. The unification is therefore blocked
on #6492, so this module captures the *current* contract the eventual refactor
must preserve -- validation requirement #4 of the #6506 contract:

    "A parity assertion that native vs adapter execution-mode metadata and
    planner_diagnostics are unchanged for representative planners."

Concretely these tests pin, for representative planners across all three
execution modes:

* the ``planner_kinematics.execution_mode`` classification (``native`` /
  ``adapter`` / ``mixed``), and
* the ``planner_diagnostics`` propagation path for the native-command arm
  (``runner.py`` ``step()`` bridging).

When the #6492 canonical adapter lands and #6506 rewires both harness entry
points through it, these assertions must still pass; a change here is the
signal that the unification altered execution-mode classification or
diagnostics propagation and must be investigated rather than silently merged.
"""

from __future__ import annotations

import sys

import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.runner import NATIVE_COMMAND_DIAGNOSTICS_KEY, run_episode

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
    metadata = enrich_algorithm_metadata(algo=algo_key)
    planner_kinematics = metadata["planner_kinematics"]
    assert planner_kinematics["execution_mode"] == expected_mode
    # ``adapter_active`` is the derived parity flag the benchmark reports: it is
    # True exactly for adapter/mixed modes and False for native. A flip here
    # would change the execution-mode classification surfaced to consumers.
    assert planner_kinematics["adapter_active"] is (expected_mode in {"adapter", "mixed"})


def test_native_command_arm_propagates_native_mode_and_diagnostics() -> None:
    """The runner.py ``step()->dict`` arm keeps native mode + planner_diagnostics.

    The native-command arm is the representative of the baselines ``step()``
    family in ``runner.py``. It must classify as ``native`` and surface its
    counters under ``algorithm_metadata.planner_diagnostics`` (the
    ``NATIVE_COMMAND_DIAGNOSTICS_KEY`` propagation path). The #6506 rewire must
    not alter either signal.
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
