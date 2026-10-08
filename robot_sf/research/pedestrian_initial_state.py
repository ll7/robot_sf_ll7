"""Rigid-disc input admissibility and source-equivalent holding layouts.

Seyfried section 2/Figure 2: 60 people, 3.3/m², 4 m corridor, first holding
section centre 3 m upstream (https://arxiv.org/abs/physics/0702004).
Liao section 2: up to 350 people, 3/m², holding semicircle radius 8.618 m
(https://doi.org/10.1016/j.trpro.2014.09.005). Enlargements are engineering
protocol equivalents; people and empirical flow targets are unchanged.
"""

from __future__ import annotations

import numpy as np

JITTER_M = 0.00025
START_GAP_M = 0.02


def initial_admissibility(state, segments, radius_m) -> dict[str, object]:
    """Measure initial disc separation and wall clearance before any model step.

    Returns:
        Input verdict and independently readable distances and violation counts.
    """
    values = np.asarray(state, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] not in (6, 7)
        or not np.isfinite(values).all()
        or isinstance(radius_m, (bool, np.bool_))
        or not np.isscalar(radius_m)
        or not np.isfinite(radius_m)
        or radius_m <= 0
    ):
        raise ValueError("invalid initial state/radius")
    xy = values[:, :2]
    walls = np.asarray(segments, dtype=float)
    if walls.size and (
        walls.shape not in ((len(walls), 4), (len(walls), 2, 2)) or not np.isfinite(walls).all()
    ):
        raise ValueError("expected finite wall segments with four coordinates")
    pairs = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)[np.triu_indices(len(xy), 1)]
    minimum = float(pairs.min()) if pairs.size else None
    wall_minimum = None
    wall_overlaps = 0
    if len(segments):
        walls = walls.reshape(-1, 2, 2)
        vector = walls[:, 1] - walls[:, 0]
        length2 = np.sum(vector * vector, axis=1)
        delta = xy[:, None] - walls[None, :, 0]
        alpha = np.sum(delta * vector, axis=-1) / np.where(length2 > 0, length2, 1)
        nearest = walls[None, :, 0] + np.clip(alpha, 0, 1)[:, :, None] * vector
        distances = np.linalg.norm(xy[:, None] - nearest, axis=-1)
        wall_minimum = float(distances.min())
        wall_overlaps = int(np.sum(distances < radius_m))
    required = 2 * radius_m + START_GAP_M
    return {
        "admissible": (minimum is None or minimum >= required - 1e-12) and wall_overlaps == 0,
        "persons": len(xy),
        "radius_m": float(radius_m),
        "minimum_centre_spacing_m": minimum,
        "required_centre_spacing_m": required,
        "overlapping_pairs": int(np.sum(pairs < 2 * radius_m)),
        "pairs_below_start_gap": int(np.sum(pairs < required - 1e-12)),
        "minimum_wall_centre_clearance_m": wall_minimum,
        "initial_wall_overlap_count": wall_overlaps,
    }


def require_initial_admissibility(state, segments, radius_m) -> dict[str, object]:
    """Refuse invalid protocol inputs before constructing or stepping the model.

    Returns:
        The admissible initial-state receipt.
    """
    receipt = initial_admissibility(state, segments, radius_m)
    if not receipt["admissible"]:
        raise ValueError("inadmissible initial state: " + str(receipt))
    return receipt


def attach_initial_receipts(row, audits, holdings) -> None:
    """Attach actual simulated input geometry rather than the inherited constructor's labels."""
    if audits:
        row["initial_admissibility"] = audits[0]
    if holdings:
        row["holding_layout"] = holdings[0]
        row["initial_density_m2"] = holdings[0]["new_density_persons_m2"]
        row["initial_min_pair_distance_m"] = audits[0]["minimum_centre_spacing_m"]
        row["initial_footprint_pair_overlaps"] = audits[0]["overlapping_pairs"]


def holding_state(
    state, segments, *, wide, radius_m, seed, aperture_width_m
) -> tuple[np.ndarray, list[tuple[float, ...]], dict[str, object]]:
    """Retain all people; expand the holding area only when required by disc spacing.

    Returns:
        Repaired state, wall segments and old/new source-equivalence receipt.
    """
    if not 1001 <= seed <= 1030:
        raise ValueError("holding layouts require dev seeds1001-1030")
    n = 350 if wide else 60
    if len(state) != n:
        raise ValueError("holding population differs from source protocol")
    rng = np.random.default_rng(seed)
    gap = 2 * radius_m + START_GAP_M
    pitch = gap + 2 * np.sqrt(2) * JITTER_M + 1e-12
    source_density = 3.0 if wide else 3.3
    old_width, old_length = 4.0, 60 / (4 * 3.3)
    old_radius = 8.618
    if wide:
        pitch = max(pitch, np.sqrt(2 / (np.sqrt(3) * source_density)))
        holding_radius = old_radius
        while True:
            bound = int(np.ceil(holding_radius / pitch)) + 2
            sites = [
                ((column + (row % 2) * 0.5) * pitch, row * pitch * np.sqrt(3) / 2)
                for row in range(-2 * bound, 2 * bound + 1)
                for column in range(-bound, 1)
                if (column + (row % 2) * 0.5) * pitch <= -radius_m - START_GAP_M - JITTER_M
                and ((column + (row % 2) * 0.5) * pitch) ** 2 + (row * pitch * np.sqrt(3) / 2) ** 2
                <= (holding_radius - JITTER_M) ** 2
            ]
            if len(sites) >= n:
                break
            holding_radius += 0.01
        xy = np.asarray(sites)[rng.choice(len(sites), n, replace=False)]
        area = np.pi * holding_radius**2 / 2
        old_area = np.pi * old_radius**2 / 2
        new_half = max(10.0, holding_radius + radius_m + START_GAP_M)
        segments = [
            tuple(
                np.copysign(new_half, value) if i in (1, 3) and abs(value) == 10 else value
                for i, value in enumerate(segment)
            )
            for segment in segments
        ]
        size = {"semicircle_radius_m": holding_radius, "area_m2": area}
        old_size = {"semicircle_radius_m": old_radius, "area_m2": old_area}
    else:
        dy = pitch * np.sqrt(3) / 2
        rows = int(np.floor((old_width - 2 * (radius_m + JITTER_M)) / dy)) + 1
        if rows < 1:
            raise ValueError("disc does not fit source upstream corridor")
        rows = min(rows, n)
        counts = np.full(rows, n // rows)
        order = list(range(0, rows, 2)) + list(range(1, rows, 2))
        counts[order[: n % rows]] += 1
        columns = int(counts.max())
        xy = np.asarray(
            [
                (
                    (column - (columns - 1) / 2 + (row % 2) * 0.5) * pitch,
                    (row - (rows - 1) / 2) * dy,
                )
                for row, count in enumerate(counts)
                for column in range(int(count))
            ]
        )
        xy[:, 0] -= (xy[:, 0].max() + xy[:, 0].min()) / 2
        holding_length = max(old_length, np.ptp(xy[:, 0]) + 2 * (radius_m + JITTER_M))
        xy[:, 0] += -3.0 - holding_length / 3
        area = old_width * holding_length
        size = {"width_m": old_width, "length_m": float(holding_length), "area_m2": float(area)}
        old_size = {"width_m": old_width, "length_m": old_length, "area_m2": old_width * old_length}
    xy += rng.uniform(-JITTER_M, JITTER_M, xy.shape)
    result = np.asarray(state, dtype=float).copy()
    result[:, :2] = xy
    result[:, 5] = np.clip(
        xy[:, 1],
        -max(0, aperture_width_m / 2 - radius_m - 0.05),
        max(0, aperture_width_m / 2 - radius_m - 0.05),
    )
    receipt = {
        "source": "Liao et al. 2014 section 2; https://doi.org/10.1016/j.trpro.2014.09.005"
        if wide
        else "Seyfried et al. section 2/Figure 2; https://arxiv.org/abs/physics/0702004",
        "equivalence": "author-approved nonoverlapping holding area, 2026-10-02",
        "persons": n,
        "old_density_persons_m2": source_density,
        "new_density_persons_m2": float(n / area),
        "old_holding_area": old_size,
        "new_holding_area": size,
        "lattice_pitch_m": float(pitch),
        "jitter_bound_m": JITTER_M,
        "admissibility": initial_admissibility(result, segments, radius_m),
    }
    return result, segments, receipt
