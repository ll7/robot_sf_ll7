"""Thin adapter around the bundled `fast-pysf` (pysocialforce) simulator to
provide point-wise force queries and convenient force-field sampling.

This adapter uses the real APIs present in the vendored `fast-pysf` copy
(`pysocialforce.simulator`, `pysocialforce.forces`, `pysocialforce.scene`).

Design goals:
- Reliable: call the canonical functions where available.
- Defensive: fall back to simple heuristics when an API is missing.
- Documented: clear docstrings and type hints.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pysocialforce as pysf
from pysocialforce import forces as pf_forces


class FastPysfWrapper:
    """Adapter that exposes get_forces_at() and get_force_field().

    The implementation composes individual force contributions using the
    canonical functions defined in `pysocialforce.forces`:
    - `social_force_ped_ped` for pairwise pedestrian-social interactions
    - `obstacle_force` for obstacle contributions
    - (optional) desired force is computed inline when a goal is provided

    Notes:
    - All returned forces are 2D numpy arrays (x, y).
    - For performance sample many points use `get_force_field` which batches
      calls in Python; for very large grids consider implementing a native
      batched kernel inside `fast-pysf`.
    """

    def __init__(self, simulator: pysf.Simulator):
        self.sim = simulator
        # Named caches for precomputed force grids. Each cache is a dict with
        # keys: 'xs' (1D array), 'ys' (1D array), 'field' (H,W,2 array).
        self._force_grid_caches: dict[str, dict] = {}

    def _ped_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays (N,2) positions and velocities for all pedestrians."""
        pos = self.sim.peds.pos()
        vel = self.sim.peds.vel()
        return pos, vel

    def _social_params(self):
        cfg = self.sim.config.social_force_config
        return cfg.n, cfg.n_prime, cfg.lambda_importance, cfg.gamma, cfg.factor

    def _get_obstacles_raw(self) -> np.ndarray:
        # Returns array shape (M,6) where cols 0:4 are line endpoints and 4:6 orthovec
        return self.sim.get_raw_obstacles()

    # --- small helper functions to reduce complexity ---
    def _compute_desired_force(self, p: np.ndarray, desired_goal: Sequence[float]) -> np.ndarray:
        goal = np.asarray(desired_goal, dtype=float).reshape(2)
        dir_vec = goal - p
        dist = np.linalg.norm(dir_vec)
        if dist != 0:
            dir_unit = dir_vec / dist
        else:
            dir_unit = np.zeros(2)

        tau = float(self.sim.config.desired_force_config.relaxation_time)
        # Compute a robust max_speed without triggering numpy warnings
        max_speed = 1.0
        try:
            speeds = np.asarray(self.sim.peds.max_speeds, dtype=float)
            if speeds.size > 0:
                finite = np.isfinite(speeds) & (speeds > 0)
                if np.any(finite):
                    max_speed = float(speeds[finite].mean())
        except Exception:
            # Fall back to default 1.0 if anything goes wrong
            max_speed = 1.0
        v_des = dir_unit * max_speed
        f_des = (v_des - np.zeros(2)) / tau
        f_des = f_des * float(self.sim.config.desired_force_config.factor)
        return f_des

    def _compute_social_force_at_point(self, p: np.ndarray) -> np.ndarray:
        total = np.zeros(2, dtype=float)
        ped_pos, ped_vel = self._ped_positions_and_velocities()
        if ped_pos.shape[0] == 0:
            return total

        n, n_prime, lambda_importance, gamma, factor = self._social_params()
        for i in range(ped_pos.shape[0]):
            other_pos = ped_pos[i]
            other_vel = ped_vel[i]
            pos_diff = (p - other_pos).astype(float)
            vel_diff = (np.zeros(2) - other_vel).astype(float)
            try:
                f_x, f_y = pf_forces.social_force_ped_ped(
                    pos_diff,
                    vel_diff,
                    int(n),
                    int(n_prime),
                    float(lambda_importance),
                    float(gamma),
                )
                total += np.array([f_x, f_y], dtype=float) * float(factor)
            except Exception:
                d = p - other_pos
                r = np.linalg.norm(d)
                if r > 1e-6:
                    total += (d / (r * r)) * 1.0
        return total

    def _compute_obstacle_force_at_point(self, p: np.ndarray) -> np.ndarray:
        total = np.zeros(2, dtype=float)
        raw_obs = self._get_obstacles_raw()
        if raw_obs is None or len(raw_obs) == 0:
            return total

        ped_radius = float(self.sim.peds.agent_radius)
        for row in raw_obs:
            line = tuple(map(float, row[:4]))
            ortho = tuple(map(float, row[4:6]))
            try:
                fx, fy = pf_forces.obstacle_force(line, ortho, p.astype(float), ped_radius)
                total += np.array([fx, fy], dtype=float) * float(
                    self.sim.config.obstacle_force_config.factor
                )
            except Exception:
                # ignore obstacle errors
                pass
        return total

    def _compute_robot_force_at_point(self, p: np.ndarray, robot_state: dict) -> np.ndarray:
        # Defensive: try common names for robot interaction functions
        for name in ("robot_force", "robot_interaction_force_on_point", "force_robot"):
            if hasattr(pf_forces, name):
                fn = getattr(pf_forces, name)
                try:
                    return np.asarray(
                        fn(p, robot_state, getattr(self.sim, "scene", None)), dtype=float
                    )
                except Exception:
                    return np.zeros(2, dtype=float)
        return np.zeros(2, dtype=float)

    # --- public API ---
    def get_forces_at(
        self,
        point: Sequence[float],
        include_desired: bool = False,
        desired_goal: Optional[Sequence[float]] = None,
        include_robot: bool = False,
        robot_state: Optional[dict] = None,
    ) -> np.ndarray:
        """Compute the total force vector at an arbitrary 2D `point`.

        This function composes smaller helpers; keep it short to satisfy complexity
        limits from linters.
        """
        p = np.asarray(point, dtype=float).reshape(2)
        total = np.zeros(2, dtype=float)

        if include_desired and desired_goal is not None:
            total += self._compute_desired_force(p, desired_goal)

        total += self._compute_social_force_at_point(p)
        total += self._compute_obstacle_force_at_point(p)

        if include_robot and robot_state is not None:
            total += self._compute_robot_force_at_point(p, robot_state)

        return total

    def get_force_field(self, xs: Sequence[float], ys: Sequence[float], **kwargs) -> np.ndarray:
        """Sample forces on the grid defined by 1D arrays `xs`, `ys`.

        Returns an array shaped (len(ys), len(xs), 2) suitable for plotting with
        `matplotlib.quiver`.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        forces = np.vstack([self.get_forces_at(p, **kwargs) for p in pts])
        return forces.reshape(len(ys), len(xs), 2)

    def build_force_grid_cache(
        self, xs: Sequence[float], ys: Sequence[float], name: str = "default", **kwargs
    ) -> None:
        """Precompute and store a sampled force grid under the given `name`.

        The cached grid can later be interpolated with `interpolate_force`.
        If a cache with the same name exists it will be overwritten.
        """
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        field = self.get_force_field(xs_arr, ys_arr, **kwargs)
        self._force_grid_caches[name] = {"xs": xs_arr, "ys": ys_arr, "field": field}

    def interpolate_force(self, x: float, y: float, name: str = "default") -> np.ndarray:
        """Return bilinearly interpolated force at (x, y) using cached grid `name`.

        Raises ValueError if `name` is not present.
        """
        if name not in self._force_grid_caches:
            raise ValueError(
                f"No cached grid named {name!r}. Call build_force_grid_cache(..., name='{name}') first."
            )

        cache = self._force_grid_caches[name]
        xs = cache["xs"]
        ys = cache["ys"]
        field = cache["field"]

        # outside grid -> nearest sample
        if not (xs[0] <= x <= xs[-1]) or not (ys[0] <= y <= ys[-1]):
            ix = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 1))
            iy = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 1))
            return field[iy, ix]

        ix = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
        iy = int(np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2))

        x1, x2 = xs[ix], xs[ix + 1]
        y1, y2 = ys[iy], ys[iy + 1]
        Q11 = field[iy, ix]
        Q21 = field[iy, ix + 1]
        Q12 = field[iy + 1, ix]
        Q22 = field[iy + 1, ix + 1]

        if x2 == x1 or y2 == y1:
            return Q11

        tx = (x - x1) / (x2 - x1)
        ty = (y - y1) / (y2 - y1)

        top = Q11 * (1 - tx) + Q21 * tx
        bottom = Q12 * (1 - tx) + Q22 * tx
        return top * (1 - ty) + bottom * ty

    def clear_force_grid_cache(self, name: Optional[str] = None) -> None:
        """Clear a specific cache by `name` or all caches if `name` is None."""
        if name is None:
            self._force_grid_caches.clear()
        else:
            self._force_grid_caches.pop(name, None)
