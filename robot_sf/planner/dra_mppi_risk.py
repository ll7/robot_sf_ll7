"""Shared Monte Carlo integration of independent pedestrian collision events."""

from math import ceil, exp, floor, pi, sqrt

import numpy as np
from numba import njit

_GAUSSIAN_TAIL_SIGMAS = 6.0


@njit(cache=True)
def _disk_probability(center, radius, disk_area, grid, x_prefix, y_density, applicable, sums):
    """Sum density values for each disk member and combine the marginal masses.

    Returns:
        Probability of at least one pedestrian in the disk.
    """
    px, py = center
    x0, y0, dx, dy = grid
    rows = y_density.shape[1]
    columns = x_prefix.shape[1] - 1
    bottom = max(0, ceil((py - radius - y0) / dy))
    top = min(rows - 1, floor((py + radius - y0) / dy))
    points_in_disk = 0
    for row in range(bottom, top + 1):
        offset = y0 + row * dy - py
        half_width = sqrt(max(0.0, radius**2 - offset**2))
        left = max(0, ceil((px - half_width - x0) / dx))
        right = min(columns - 1, floor((px + half_width - x0) / dx))
        if right < left:
            continue
        points_in_disk += right - left + 1
        for i, j in enumerate(applicable):
            sums[i] += y_density[j, row] * (x_prefix[j, right + 1] - x_prefix[j, left])
    if points_in_disk == 0:
        # A zero denominator is possible for a deliberately undersampled grid.
        return 1.0
    complement = 1.0
    for density_sum in sums:
        marginal = disk_area * density_sum / points_in_disk
        complement *= 1.0 - min(1.0, max(0.0, marginal))
    return 1.0 - complement


@njit(cache=True)
def _prepare_densities(means, sigma, bounds, grid, outside2, near, x_prefix, y_density):
    """Populate shared Gaussian axes for the pedestrians who could overlap a disk.

    Returns:
        Number of prepared pedestrian densities.
    """
    x_min, x_max, y_min, y_max = bounds
    x0, y0, dx, dy = grid
    rows = y_density.shape[1]
    columns = x_prefix.shape[1] - 1
    near_count = 0
    density_scale = 1.0 / (sqrt(2.0 * pi) * sigma)
    for j in range(len(means)):
        mx, my = means[j]
        gap_x = max(x_min - mx, mx - x_max, 0.0)
        gap_y = max(y_min - my, my - y_max, 0.0)
        if gap_x**2 + gap_y**2 > outside2:
            continue
        near[near_count] = j
        near_count += 1
        x_prefix[j, 0] = 0.0
        for col in range(columns):
            z = (x0 + col * dx - mx) / sigma
            x_prefix[j, col + 1] = x_prefix[j, col] + density_scale * exp(-0.5 * z**2)
        for row in range(rows):
            z = (y0 + row * dy - my) / sigma
            y_density[j, row] = density_scale * exp(-0.5 * z**2)

    return near_count


@njit(cache=True)
def joint_collision_probabilities(
    rollouts: np.ndarray,
    means: np.ndarray,
    std: np.ndarray,
    radius: float,
    shifts: np.ndarray,
    columns: int,
    rows: int,
) -> np.ndarray:
    """Integrate each marginal over the collision disk and combine their complements.

    At each stage one uniformly shifted rectangular grid is shared by all
    rollouts. For isotropic Gaussian marginals, x-prefix sums and y densities
    sum the same points as a full disk-membership calculation.

    Returns:
        Stage collision probabilities with shape (trajectory, stage).
    """
    count, horizon, _ = rollouts.shape
    joint = np.zeros((count, horizon))
    if means.shape[0] == 0:
        return joint
    disk_area = pi * radius**2
    peds = means.shape[0]
    x_prefix = np.empty((peds, columns + 1))
    y_density = np.empty((peds, rows))
    near = np.empty(peds, dtype=np.int64)
    applicable = np.empty(peds, dtype=np.int64)
    sums = np.empty(peds)
    for t in range(horizon):
        sigma = std[t]
        tail = _GAUSSIAN_TAIL_SIGMAS * sigma
        outside2 = (radius + tail) ** 2
        inside2 = max(0.0, radius - tail) ** 2
        x = rollouts[:, t, 0]
        y = rollouts[:, t, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        dx = (x_max - x_min + 2.0 * radius) / columns
        dy = (y_max - y_min + 2.0 * radius) / rows
        x0 = x_min - radius + shifts[t, 0] * dx
        y0 = y_min - radius + shifts[t, 1] * dy

        bounds = (x_min, x_max, y_min, y_max)
        grid = (x0, y0, dx, dy)
        near_count = _prepare_densities(
            means[:, t], sigma, bounds, grid, outside2, near, x_prefix, y_density
        )
        for k in range(count):
            px, py = x[k], y[k]
            used = 0
            for i in range(near_count):
                j = near[i]
                mx, my = means[j, t]
                distance2 = (mx - px) ** 2 + (my - py) ** 2
                if distance2 < inside2:
                    # The collision disk contains a six-sigma disk of the prediction.
                    joint[k, t] = 1.0
                    used = 0
                    break
                if distance2 <= outside2:
                    applicable[used] = j
                    sums[used] = 0.0
                    used += 1
            if used == 0:
                continue

            joint[k, t] = _disk_probability(
                (px, py),
                radius,
                disk_area,
                grid,
                x_prefix,
                y_density,
                applicable[:used],
                sums[:used],
            )
    return joint
