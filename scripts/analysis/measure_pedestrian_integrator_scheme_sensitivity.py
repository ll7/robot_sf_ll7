#!/usr/bin/env python3
"""Measure explicit versus semi-implicit pedestrian trajectory divergence.

This CPU-only diagnostic runs the native PySocialForce pedestrian interaction
law at the standard 0.1-second timestep for the repository's cautious,
standard, and hurried speed archetypes.  It reports numerical trajectory
divergence only: it is not a benchmark campaign, a realism calibration, or a
planner-ranking claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from pysocialforce import Simulator
from pysocialforce.config import SceneConfig, SimulatorConfig
from pysocialforce.scene import EXPLICIT_EULER, SEMI_IMPLICIT_EULER

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "pedestrian-integrator-scheme-sensitivity.v1"
CLAIM_BOUNDARY = (
    "diagnostic_only: native two-pedestrian Social Force trajectories at a fixed timestep; "
    "not a full benchmark campaign, pedestrian-realism calibration, or planner-ranking claim."
)
ARCHETYPE_SPEED_FACTORS = {"cautious": 0.7, "standard": 1.0, "hurried": 1.4}


@dataclass(frozen=True)
class DivergenceSummary:
    """One archetype's explicit-versus-semi-implicit trajectory divergence."""

    archetype: str
    desired_speed_factor: float
    final_mean_position_divergence_m: float
    max_position_divergence_m: float
    rms_position_divergence_m: float


def _initial_state(speed: float) -> np.ndarray:
    """Return a deterministic opposing-pedestrian interaction state."""
    return np.array(
        [
            [-1.0, 0.0, speed, 0.0, 4.0, 0.0],
            [1.0, 0.0, -speed, 0.0, -4.0, 0.0],
        ],
        dtype=float,
    )


def _trajectory(*, speed: float, dt: float, steps: int, integration_scheme: str) -> np.ndarray:
    """Run one native Social Force trajectory and return positions for every step."""
    config = SimulatorConfig(
        scene_config=SceneConfig(dt_secs=dt, integration_scheme=integration_scheme)
    )
    simulator = Simulator(_initial_state(speed), config=config)
    positions = [simulator.peds.pos().copy()]
    for _ in range(steps):
        simulator.step_once()
        positions.append(simulator.peds.pos().copy())
    return np.asarray(positions, dtype=float)


def measure_scheme_sensitivity(*, dt: float = 0.1, steps: int = 60) -> dict[str, object]:
    """Measure named-archetype trajectory divergence at one fixed timestep.

    Args:
        dt: Positive timestep in seconds. The standard comparison uses ``0.1``.
        steps: Positive number of native simulator steps per archetype.

    Returns:
        A JSON-serializable diagnostic report with per-archetype divergence metrics.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    summaries: list[DivergenceSummary] = []
    for archetype, speed_factor in ARCHETYPE_SPEED_FACTORS.items():
        explicit = _trajectory(
            speed=speed_factor,
            dt=dt,
            steps=steps,
            integration_scheme=EXPLICIT_EULER,
        )
        semi_implicit = _trajectory(
            speed=speed_factor,
            dt=dt,
            steps=steps,
            integration_scheme=SEMI_IMPLICIT_EULER,
        )
        divergence = np.linalg.norm(explicit - semi_implicit, axis=-1)
        summaries.append(
            DivergenceSummary(
                archetype=archetype,
                desired_speed_factor=speed_factor,
                final_mean_position_divergence_m=float(np.mean(divergence[-1])),
                max_position_divergence_m=float(np.max(divergence)),
                rms_position_divergence_m=float(np.sqrt(np.mean(divergence**2))),
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "diagnostic-only",
        "timestep_s": dt,
        "steps": steps,
        "schemes": {"baseline": SEMI_IMPLICIT_EULER, "comparison": EXPLICIT_EULER},
        "archetype_source": "configs/research/pedestrian_archetypes_v1.yaml",
        "trajectory_divergence": [asdict(summary) for summary in summaries],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.1, help="Fixed timestep in seconds.")
    parser.add_argument("--steps", type=int, default=60, help="Steps per archetype.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scheme-sensitivity diagnostic."""
    args = build_parser().parse_args(argv)
    report = measure_scheme_sensitivity(dt=args.dt, steps=args.steps)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Pedestrian integrator scheme sensitivity (diagnostic-only)")
        for summary in report["trajectory_divergence"]:
            print(
                f"{summary['archetype']}: max={summary['max_position_divergence_m']:.6f} m, "
                f"final_mean={summary['final_mean_position_divergence_m']:.6f} m"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
