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
* ``test_absent_command_sources_remain_backward_compatible`` preserves the predicate's
  original zero-valued behavior for non-hybrid callers that do not provide the optional
  command-source signal.

Classification of the 0/1,165 result
-------------------------------------
**Telemetry failure (wiring gap), not genuine non-activation.** The arbitration mechanism
is reachable and, when its per-step source labels are threaded through, yields a nonzero
count (first two tests). The retained-trace surface reported 0 because the runner did not
supply ``command_sources`` to the predicate. Issue #5081 repairs that runner wiring while
this predecessor module keeps the original activation and compatibility evidence.

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


def _integrate_commands(commands: list[tuple[float, float]], *, dt: float = 0.1) -> np.ndarray:
    """Integrate the selected unicycle commands into a deterministic trajectory.

    The fixture deliberately uses a small, transparent integrator rather than a
    second planner: this makes the causal chain under test explicit — selected
    source changes the selected command, which changes the next trajectory step.
    """
    position = np.zeros(2)
    heading = 0.0
    positions: list[np.ndarray] = []
    for linear_speed, angular_speed in commands:
        position = position + dt * linear_speed * np.array([np.cos(heading), np.sin(heading)])
        heading += dt * angular_speed
        positions.append(position.copy())
    return np.asarray(positions)


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
    """Source transitions must consistently change selected commands and trajectory."""
    adapter = _build_adapter()
    heads, commands = _drive(adapter)
    trajectory = _integrate_commands(commands)
    trajectory_steps = np.diff(np.vstack([np.zeros(2), trajectory]), axis=0)

    # Every distinct head maps to a distinct command, so a source change must
    # change both the command and the resulting unicycle trajectory step.
    for step in range(1, len(heads)):
        head_changed = heads[step] != heads[step - 1]
        command_changed = commands[step] != commands[step - 1]
        trajectory_changed = not np.allclose(trajectory_steps[step], trajectory_steps[step - 1])
        assert head_changed == command_changed == trajectory_changed


def test_command_source_changes_counted_when_wired() -> None:
    """Threading the captured source sequence yields a nonzero change count."""
    adapter = _build_adapter()
    heads, commands = _drive(adapter)

    # The production-selected commands generate this deterministic trajectory;
    # source handoffs therefore feed both source telemetry and motion evidence.
    positions = _integrate_commands(commands)
    n = len(heads)
    headings = np.zeros(n)
    velocities = np.asarray([command[0] for command in commands])

    result = oscillatory_control_predicate(
        positions,
        headings,
        velocities,
        dt=0.1,
        command_sources=heads,
    )
    assert result["fields"]["command_source_changes"] == 3
    assert result["fields"]["command_source_changes"] > 0


def test_absent_command_sources_remain_backward_compatible() -> None:
    """Non-hybrid callers without source telemetry should retain a zero count."""
    n = len(_EXPECTED_HEADS)
    positions = np.array([[float(i), 0.0] for i in range(n)])
    headings = np.zeros(n)
    speeds = np.ones(n)

    result = oscillatory_control_predicate(positions, headings, speeds, dt=0.1)
    assert result["fields"]["command_source_changes"] == 0
