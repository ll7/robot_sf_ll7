"""Deterministic scenario generator for the Social Navigation Benchmark.

This module converts high-level scenario parameter dictionaries into
`pysocialforce.Simulator` instances (or raw state + obstacles) in a
deterministic fashion (seeded), ensuring reproducibility.

Scenario parameter keys (expected):
    id: str (scenario id)
    density: one of {"low","med","high"}
    flow: one of {"uni","bi","cross","merge"}
    obstacle: one of {"open","bottleneck","maze"}
    groups: float fraction in {0.0,0.2,0.4}
    speed_var: {"low","high"}
    goal_topology: {"point","swap","circulate"}
    robot_context: {"ahead","behind","embedded"}
    repeats: int (ignored here, used by runner)

Returned structure:
    {
        "simulator": pysocialforce.Simulator,
        "state": np.ndarray shape (N,7),
        "obstacles": list[tuple[float,float,float,float]],
        "groups": list[int] (group id per agent, -1 if none),
        "metadata": { original params + derived fields }
    }

Notes:
 - We keep geometry simple initially (rectangular area 10m x 6m).
 - Densities map to approximate counts: low=10, med=25, high=40.
 - Obstacles layouts are coarse placeholders (can be evolved later).
 - Group assignment is random among eligible fraction, groups of size 2-4.
 - All randomness uses a local numpy Generator seeded with `seed`.
 - Initial velocities set to zero; policies/env will update.
 - Desired relaxation time (tau) set to 1.0 placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Optional heavy import delayed until needed
    import pysocialforce as pysf
except ImportError:  # pragma: no cover - allow import failure during docs builds
    pysf = None  # type: ignore


AREA_WIDTH = 10.0
AREA_HEIGHT = 6.0

_DENSITY_COUNTS = {"low": 10, "med": 25, "high": 40}


@dataclass
class GeneratedScenario:
    """GeneratedScenario class."""

    simulator: Any
    state: np.ndarray
    obstacles: list[tuple[float, float, float, float]]
    groups: list[int]
    metadata: dict[str, Any]


def _select_counts(params: dict[str, Any]) -> int:
    """Select counts.

    Args:
        params: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    density = params.get("density", "med")
    return int(_DENSITY_COUNTS.get(density, _DENSITY_COUNTS["med"]))


def _sample_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample positions.

    Args:
        rng: Auto-generated placeholder description.
        n: Auto-generated placeholder description.

    Returns:
        np.ndarray: Auto-generated placeholder description.
    """
    # Uniform in central bounding box with small margin
    margin = 0.5
    xs = rng.uniform(margin, AREA_WIDTH - margin, size=n)
    ys = rng.uniform(margin, AREA_HEIGHT - margin, size=n)
    return np.stack([xs, ys], axis=1)


def _build_obstacles(kind: str) -> list[tuple[float, float, float, float]]:
    """Build obstacles.

    Args:
        kind: Auto-generated placeholder description.

    Returns:
        list[tuple[float, float, float, float]]: Auto-generated placeholder description.
    """
    if kind == "open":
        return []
    if kind == "bottleneck":
        # Two vertical walls with a small gap centered
        gap_y1, gap_y2 = 2.5, 3.5
        x = AREA_WIDTH / 2
        return [
            (x, 0.0, x, gap_y1),  # lower segment
            (x, gap_y2, x, AREA_HEIGHT),  # upper segment
        ]
    if kind == "maze":
        # Simple grid-like three segments
        w2 = AREA_WIDTH / 3
        return [
            (w2, 0.0, w2, AREA_HEIGHT * 0.6),
            (2 * w2, AREA_HEIGHT * 0.4, 2 * w2, AREA_HEIGHT),
            (w2, AREA_HEIGHT * 0.6, 2 * w2, AREA_HEIGHT * 0.6),
        ]
    return []


def _assign_goals(flow: str, goal_topology: str, pos: np.ndarray) -> np.ndarray:
    """Assign goals.

    Args:
        flow: Auto-generated placeholder description.
        goal_topology: Auto-generated placeholder description.
        pos: Auto-generated placeholder description.

    Returns:
        np.ndarray: Auto-generated placeholder description.
    """
    n = pos.shape[0]
    goals = np.zeros_like(pos)
    if flow == "uni":
        # Move left->right: goals at right boundary
        goals[:, 0] = AREA_WIDTH - 0.2
        goals[:, 1] = pos[:, 1]
    elif flow == "bi":
        half = n // 2
        goals[:half, 0] = AREA_WIDTH - 0.2
        goals[:half, 1] = pos[:half, 1]
        goals[half:, 0] = 0.2
        goals[half:, 1] = pos[half:, 1]
    elif flow == "cross":
        # Half move horizontally, half vertically
        half = n // 2
        goals[:half, 0] = AREA_WIDTH - pos[:half, 0]
        goals[:half, 1] = pos[:half, 1]
        goals[half:, 0] = pos[half:, 0]
        goals[half:, 1] = AREA_HEIGHT - pos[half:, 1]
    elif flow == "merge":
        # All toward a central x then exit right
        cx = AREA_WIDTH * 0.6
        goals[:, 0] = cx
        goals[:, 1] = AREA_HEIGHT / 2
    # Adjust for goal topology variants
    if goal_topology == "swap":
        goals = pos[::-1].copy()
    elif goal_topology == "circulate":
        # shift positions circularly
        goals = np.roll(pos, shift=1, axis=0)
    return goals


def _assign_groups(rng: np.random.Generator, n: int, fraction: float) -> list[int]:
    """Assign groups.

    Args:
        rng: Auto-generated placeholder description.
        n: Auto-generated placeholder description.
        fraction: Auto-generated placeholder description.

    Returns:
        list[int]: Auto-generated placeholder description.
    """
    if fraction <= 0:
        return [-1] * n
    num_grouped = int(round(n * fraction))
    indices = rng.permutation(n)[:num_grouped]
    group_ids = [-1] * n
    current_gid = 0
    i = 0
    while i < len(indices):
        group_size = int(rng.integers(2, 5))  # 2-4
        members = indices[i : i + group_size]
        for m in members:
            group_ids[m] = current_gid
        current_gid += 1
        i += group_size
    return group_ids


def _speed_variation(speed_var: str) -> float:
    """Speed variation.

    Args:
        speed_var: Auto-generated placeholder description.

    Returns:
        float: Auto-generated placeholder description.
    """
    return 0.2 if speed_var == "low" else 0.5


def generate_scenario(params: dict[str, Any], seed: int) -> GeneratedScenario:
    """Generate a deterministic scenario.

    Parameters
    ----------
    params : dict
        Scenario parameter dictionary (see module docstring).
    seed : int
        RNG seed for reproducibility.
    """
    # Special preset for testing/validation: guaranteed contact at t=0
    # Places one pedestrian exactly at the default robot start (0.3, 3.0)
    # with goal equal to its position (no desired motion). This ensures
    # min distance < D_COLL at the first timestep, exercising the collision
    # counting pipeline end-to-end.
    preset = str(params.get("preset", "")).strip().lower()
    if preset == "collision_sanity":
        n = 1
        pos = np.array([[0.3, 3.0]], dtype=float)
        goals = pos.copy()  # no movement desired
        state = np.zeros((n, 7), dtype=float)
        state[:, 0:2] = pos
        state[:, 4:6] = goals
        state[:, 6] = 1.0
        obstacles: list[tuple[float, float, float, float]] = []
        groups: list[int] = [-1]
        metadata = {**params, "n_agents": n, "area": AREA_WIDTH * AREA_HEIGHT, "seed": seed}
        simulator = None if pysf is None else pysf.Simulator(state=state, obstacles=None)  # type: ignore[arg-type]
        return GeneratedScenario(
            simulator=simulator,
            state=state,
            obstacles=obstacles,
            groups=groups,
            metadata=metadata,
        )

    rng = np.random.default_rng(seed)
    n = _select_counts(params)
    pos = _sample_positions(rng, n)
    goals = _assign_goals(params.get("flow", "uni"), params.get("goal_topology", "point"), pos)
    speed_std = _speed_variation(params.get("speed_var", "low"))
    # Desired speeds around 1.3 m/s with variation
    desired_speeds = rng.normal(loc=1.3, scale=speed_std, size=n)
    desired_speeds = np.clip(desired_speeds, 0.2, 2.0)

    # State: [x,y,vx,vy,goalx,goaly,tau]
    state = np.zeros((n, 7), dtype=float)
    state[:, 0:2] = pos
    state[:, 4:6] = goals
    state[:, 6] = 1.0  # tau placeholder

    obstacles = _build_obstacles(params.get("obstacle", "open"))
    groups = _assign_groups(rng, n, float(params.get("groups", 0.0)))

    # Attach group influence via metadata; downstream can use groups list
    metadata = {
        **params,
        "n_agents": n,
        "area": AREA_WIDTH * AREA_HEIGHT,
        "seed": seed,
        "speed_std": speed_std,
        "group_count": len({g for g in groups if g >= 0}),
    }

    if pysf is None:
        simulator = None  # pragma: no cover
    else:
        # pysocialforce expects None (not empty list) for no obstacles; empty list triggers
        # a broadcasting issue inside EnvState._update_obstacles_raw.
        sim_obstacles = obstacles if len(obstacles) > 0 else None
        simulator = pysf.Simulator(state=state, obstacles=sim_obstacles)  # type: ignore[arg-type]

    return GeneratedScenario(
        simulator=simulator,
        state=state,
        obstacles=obstacles,
        groups=groups,
        metadata=metadata,
    )


__all__ = ["GeneratedScenario", "generate_scenario"]
