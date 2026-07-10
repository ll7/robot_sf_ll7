"""Hybrid arbitration activation + command-source telemetry check (issue #5001).

Context
-------
The retained-trace analysis from issue #4914 reported **0 ``command_source_changes``
across all 1,165 episodes** for the hybrid planners. Zero handoffs is consistent with
three very different explanations:

1. the hybrid arbitration mechanism never activating (genuine non-activation);
2. the ``command_source`` telemetry not capturing the switch (a wiring gap);
3. the tested scenario distribution never crossing an activation threshold.

Until mechanism activation is verified independently, the 0/1,165 result must **not**
be read as evidence about hybrid effectiveness (issue #5001 acceptance criterion 3).

What this module proves
-----------------------
* ``test_hybrid_portfolio_arbitration_produces_source_handoff`` drives the *real*
  :class:`~robot_sf.planner.hybrid_portfolio.HybridPortfolioAdapter` arbitration through
  regime changes (open cruise -> emergency clearance -> caution density) and shows the
  per-step ``selected_head`` label genuinely changes. Only the child planner heads are
  stubbed; the head-selection logic (``_desired_head`` / ``_switch_head`` /
  ``_record_decision``) under test is the production code path.
* ``test_command_source_changes_counted_when_wired`` feeds that captured per-step
  source sequence into
  :func:`~robot_sf.benchmark.safety_predicates.oscillatory_control_predicate` and shows a
  **nonzero** ``command_source_changes`` — the mechanism *does* produce a countable
  handoff when the telemetry is actually populated.
* ``test_episode_runner_callsite_always_reports_zero`` pins the root cause of the
  0/1,165 field result: the episode runner
  (``robot_sf/benchmark/map_runner_episode.py:597``) calls
  ``oscillatory_control_predicate(positions, headings, speeds, dt=dt)`` and never passes
  ``command_sources``. With that optional signal absent, ``command_source_changes`` is
  hard-wired to 0 (``safety_predicates.py:98-100``) for *every* episode, regardless of
  how many real head handoffs occurred.

Classification of the 0/1,165 result
-------------------------------------
**Telemetry failure (wiring gap), not genuine non-activation.** The arbitration mechanism
is reachable and, when its per-step source labels are threaded through, yields a nonzero
count (first two tests). The retained-trace surface reports 0 only because the runner
never supplies ``command_sources`` to the predicate (third test). Wiring the per-step
planner source into the episode runner changes benchmark telemetry output and is therefore
deferred as post-freeze work (freeze 2026-07-18); see
``docs/context/issue_5001_state.yaml``.

This module changes no benchmark metric semantics: it only reads existing production code
and the existing optional predicate signal.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from robot_sf.benchmark.safety_predicates import oscillatory_control_predicate
from robot_sf.planner.hybrid_portfolio import (
    HybridPortfolioAdapter,
    HybridPortfolioConfig,
)


class _StubHead:
    """Minimal planner head that returns a fixed command and supports reset().

    Only the hybrid *arbitration* logic is under test here; the child heads merely
    need to return a distinct, deterministic command so we can confirm the selected
    command tracks the selected head.
    """

    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command
        self.reset_calls = 0

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the fixed command for this head."""
        return self.command

    def reset(self) -> None:
        """Count resets so the adapter's reset fan-out stays observable."""
        self.reset_calls += 1


def _observation(ped_xy: list[list[float]] | None) -> dict[str, Any]:
    """Build a hybrid-portfolio observation with the robot at the origin.

    Args:
        ped_xy: Pedestrian ``(x, y)`` positions, or ``None`` for an empty crowd.

    Returns:
        Observation dict consumed by ``HybridPortfolioAdapter._extract_ped_clearance``.
    """
    return {
        "robot": {"position": [0.0, 0.0]},
        "pedestrians": {"positions": ped_xy if ped_xy is not None else []},
    }


# Head-distinct commands so a source handoff is also a command handoff.
_RISK_DWA_CMD = (1.0, 0.0)
_ORCA_CMD = (0.0, 0.0)
_PREDICTION_CMD = (0.5, 0.2)


def _build_adapter() -> HybridPortfolioAdapter:
    """Construct a hybrid adapter with deterministic stub heads and no hysteresis.

    ``hysteresis_steps=0`` makes every step re-evaluate the desired head, so the
    regime sequence deterministically maps to a head sequence.

    Returns:
        A ready-to-drive :class:`HybridPortfolioAdapter`.
    """
    config = HybridPortfolioConfig(hysteresis_steps=0)
    return HybridPortfolioAdapter(
        hybrid_config=config,
        risk_dwa=_StubHead(_RISK_DWA_CMD),  # type: ignore[arg-type]
        orca=_StubHead(_ORCA_CMD),  # type: ignore[arg-type]
        prediction=_StubHead(_PREDICTION_CMD),  # type: ignore[arg-type]
        mppi=None,
    )


# Regime observations chosen to cross the documented arbitration thresholds
# (emergency_clearance=0.55, caution_clearance=1.0, near_field_distance=2.5):
#   - open cruise       -> risk_dwa   (no pedestrian in the near field)
#   - emergency clearance-> orca       (pedestrian at 0.4 m <= 0.55 m)
#   - caution density    -> prediction (pedestrian at 0.9 m in (0.55, 1.0])
_REGIME_SEQUENCE = [
    _observation([[100.0, 100.0]]),  # open cruise -> risk_dwa
    _observation([[0.4, 0.0]]),  # emergency -> orca
    _observation([[0.9, 0.0]]),  # caution -> prediction
    _observation([[100.0, 100.0]]),  # open cruise -> risk_dwa
]
_EXPECTED_HEADS = ["risk_dwa", "orca", "prediction", "risk_dwa"]
_EXPECTED_COMMANDS = [_RISK_DWA_CMD, _ORCA_CMD, _PREDICTION_CMD, _RISK_DWA_CMD]


def _drive(adapter: HybridPortfolioAdapter) -> tuple[list[str], list[tuple[float, float]]]:
    """Run the regime sequence and capture per-step selected head and command.

    Returns:
        ``(selected_heads, selected_commands)`` aligned to ``_REGIME_SEQUENCE``.
    """
    heads: list[str] = []
    commands: list[tuple[float, float]] = []
    for observation in _REGIME_SEQUENCE:
        command = adapter.plan(observation)
        assert adapter._last_decision is not None
        heads.append(str(adapter._last_decision["selected_head"]))
        commands.append(command)
    return heads, commands


def test_hybrid_portfolio_arbitration_produces_source_handoff() -> None:
    """The real arbitration path must change its selected source across regimes."""
    adapter = _build_adapter()
    heads, commands = _drive(adapter)

    assert heads == _EXPECTED_HEADS
    assert commands == _EXPECTED_COMMANDS
    # At least one handoff is the whole point: the mechanism activates.
    handoffs = sum(1 for a, b in pairwise(heads) if a != b)
    assert handoffs >= 1
    assert handoffs == 3  # risk_dwa->orca->prediction->risk_dwa


def test_selected_command_tracks_selected_head() -> None:
    """The selected command must change consistently with the selected head."""
    adapter = _build_adapter()
    heads, commands = _drive(adapter)

    # Every distinct head maps to a distinct command, so a source change implies
    # a command (trajectory) change — the two telemetry views stay consistent.
    for step in range(1, len(heads)):
        head_changed = heads[step] != heads[step - 1]
        command_changed = commands[step] != commands[step - 1]
        assert head_changed == command_changed


def test_command_source_changes_counted_when_wired() -> None:
    """Threading the captured source sequence yields a nonzero change count."""
    adapter = _build_adapter()
    heads, _commands = _drive(adapter)

    # A benign straight trajectory of matching length; only command_source_changes
    # is under test here.
    n = len(heads)
    positions = np.array([[float(i), 0.0] for i in range(n)])
    headings = np.zeros(n)
    velocities = np.ones(n)

    result = oscillatory_control_predicate(
        positions,
        headings,
        velocities,
        dt=0.1,
        command_sources=heads,
    )
    assert result["fields"]["command_source_changes"] == 3
    assert result["fields"]["command_source_changes"] > 0


def test_episode_runner_callsite_always_reports_zero() -> None:
    """The runner's argument signature structurally forces a zero change count.

    This replicates ``map_runner_episode.py:597`` — ``oscillatory_control_predicate``
    is called with positions/headings/speeds and ``dt`` but *without*
    ``command_sources``. This is the root cause of the 0/1,165 field result: with the
    optional signal absent, ``command_source_changes`` is 0 for every episode no matter
    how many real head handoffs occurred, so the retained-trace 0/1,165 is a telemetry
    gap, not evidence of genuine non-activation.
    """
    adapter = _build_adapter()
    heads, _commands = _drive(adapter)
    # Sanity: the underlying mechanism genuinely handed off in this scenario.
    assert any(a != b for a, b in pairwise(heads))

    n = len(heads)
    positions = np.array([[float(i), 0.0] for i in range(n)])
    headings = np.zeros(n)
    speeds = np.ones(n)

    # Exact episode-runner call signature (no command_sources).
    result = oscillatory_control_predicate(positions, headings, speeds, dt=0.1)
    assert result["fields"]["command_source_changes"] == 0
