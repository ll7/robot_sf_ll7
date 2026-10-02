"""Opt-in frictionless body exclusion and finite-range body-edge wall response.

The legacy simulator never calls these kernels. Constraints act after the
historical velocity integration; swept tests prevent a large step tunnelling
through a wall or exchanging two bodies. Equal-mass pair corrections preserve
the centre of mass. A capped solve records a deterministic repair fallback rather than aborting an episode.
"""

from __future__ import annotations

import numpy as np
from numba import njit

PROJECTION_RULE = "projection_v1"
WALL_RULE = "bounded_edge_v1"
MAX_PASSES = 256
SEPARATION_MARGIN_M = 1e-6
WALL_AMPLITUDE_M_S2 = 3.0
WALL_DECAY_M = 0.04
WALL_RANGE_M = 0.20
WALL_NORMAL_BLEND_M = 0.04


def validate_contact_rules(config) -> None:
    """Reject unknown explicit laws before constructing or stepping a simulator."""
    pairs = getattr(config, "pedestrian_contact_rule", None)
    walls = getattr(config.obstacle_force_config, "wall_contact_rule", None)
    if pairs not in (None, PROJECTION_RULE) or walls not in (None, WALL_RULE):
        raise ValueError("unsupported explicit pedestrian contact or wall rule")


@njit(cache=True)
def closest_point(p, segment):
    """Return the closest finite-segment surface point.

    Returns:
        Surface point in world coordinates.
    """
    a = segment[:2]
    delta = segment[2:4] - a
    denominator = np.dot(delta, delta)
    t = max(0.0, min(1.0, np.dot(p - a, delta) / denominator)) if denominator else 0.0
    return a + t * delta


@njit(cache=True)
def bounded_wall_force(
    positions, obstacles, radius, amplitude, decay, influence, blend=WALL_NORMAL_BLEND_M
):
    """Blend nearby surface normals across a finite clearance tolerance.

    Returns:
        Bounded wall acceleration, continuous at medial axes and range cutoff.
    """
    result = np.zeros_like(positions)
    for i in range(len(positions)):
        nearest = np.inf
        for segment in obstacles:
            delta = positions[i] - closest_point(positions[i], segment)
            nearest = min(nearest, np.sqrt(np.dot(delta, delta)))
        clearance = nearest - radius
        if clearance >= influence:
            continue
        normal = np.zeros(2)
        weight_sum = 0.0
        for segment in obstacles:
            # AABB rejection also avoids irrelevant endpoint calculations.
            if not near_segment(positions[i], positions[i], segment, nearest + blend):
                continue
            delta = positions[i] - closest_point(positions[i], segment)
            distance = np.sqrt(np.dot(delta, delta))
            weight = max(0.0, 1.0 - (distance - nearest) / blend)
            if weight:
                normal += weight * delta / max(distance, 1e-12)
                weight_sum += weight
        strength = amplitude * (np.exp(-max(0.0, clearance) / decay) - np.exp(-influence / decay))
        if weight_sum:
            result[i] = strength * normal / weight_sum
    return result


@njit(cache=True)
def near_segment(start, end, segment, radius):
    """Cull capsules whose expanded AABB cannot intersect the swept centre.

    Returns:
        Whether the swept AABBs can intersect.
    """
    for axis in range(2):
        if max(start[axis], end[axis]) < min(segment[axis], segment[axis + 2]) - radius:
            return False
        if min(start[axis], end[axis]) > max(segment[axis], segment[axis + 2]) + radius:
            return False
    return True


@njit(cache=True)
def circle_hit(start, end, centre, radius):
    """Find first segment/circle contact.

    Returns:
        First contact time and outward normal, or time two for no contact.
    """
    offset = start - centre
    motion = end - start
    squared = np.dot(offset, offset)
    if squared < radius * radius:
        distance = np.sqrt(squared)
        normal = offset / distance if distance > 1e-12 else np.array([1.0, 0.0])
        return 0.0, normal
    aa = np.dot(motion, motion)
    bb = np.dot(offset, motion)
    cc = squared - radius * radius
    disc = bb * bb - aa * cc
    if aa > 0 and bb < 0 and disc >= 0:
        t = (-bb - np.sqrt(disc)) / aa
        if 0 <= t <= 1:
            return t, (offset + t * motion) / radius
    return 2.0, np.zeros(2)


@njit(cache=True)
def push_out(point, obstacles, radius):
    """Repair a penetrating start deterministically, using every segment.

    Returns:
        Deterministically pushed-out position.
    """
    result = point.copy()
    for _ in range(16):
        changed = False
        for segment in obstacles:
            if not near_segment(result, result, segment, radius):
                continue
            delta = result - closest_point(result, segment)
            distance = np.sqrt(np.dot(delta, delta))
            if distance < radius:
                if distance > 1e-12:
                    normal = delta / distance
                else:
                    normal = segment[4:6].copy()
                    norm = np.sqrt(np.dot(normal, normal))
                    normal = normal / norm if norm else np.array([1.0, 0.0])
                result += (radius - distance) * normal
                changed = True
        if not changed:
            break
    return result


@njit(cache=True)
def wall_project(start, end, obstacles, radius):
    """Apply all swept capsule constraints, including shared-vertex ties.

    Returns:
        Corrected endpoint; an endpoint never shadows the segment interior.
    """
    result = end.copy()
    for segment in obstacles:
        if not near_segment(start, result, segment, radius):
            continue
        a, b = segment[:2], segment[2:4]
        along = b - a
        length = np.sqrt(np.dot(along, along))
        # Line and BOTH endpoint planes are separate constraints.
        if length > 0:
            tangent = along / length
            n = np.array([-tangent[1], tangent[0]])
            signed = np.dot(start - a, n)
            if signed < 0:
                n = -n
                signed = -signed
            finish = np.dot(result - a, n)
            if finish < radius:
                t = max(0.0, (signed - radius) / (signed - finish)) if signed > finish else 0.0
                crossing = start + t * (result - start)
                coordinate = np.dot(crossing - a, tangent)
                if 0 <= coordinate <= length:
                    result += (radius - finish) * n
        for centre in (a, b):
            t, normal = circle_hit(start, result, centre, radius)
            if t <= 1:
                correction = radius - np.dot(result - centre, normal)
                if correction > 0:
                    result += correction * normal
    return result


@njit(cache=True)
def grid_pairs(p, previous, cell):
    """Conservative uniform-grid candidates, expanded for relative swept motion.

    Returns:
        Ordered candidate pairs. Rebuilt each constraint pass, no stale admission.
    """
    n = len(p)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    coords = np.floor(p / cell).astype(np.int64)
    lower = coords[0].copy()
    for i in range(n):
        for axis in range(2):
            lower[axis] = min(lower[axis], coords[i, axis])
    # Numba does not implement axis minima for int arrays on all supported versions.
    stride = 1
    for i in range(n):
        stride = max(stride, coords[i, 1] - lower[1] + 1)
    reach = 1
    for i in range(n):
        motion = np.sqrt(np.sum((p[i] - previous[i]) ** 2))
        reach = max(reach, int(np.ceil(1 + 2 * motion / cell)))
    # Huge steps: the all-pair path remains conservative without huge cell scans.
    if (2 * reach + 1) ** 2 > n:
        result = np.empty((n * (n - 1) // 2, 2), dtype=np.int64)
        k = 0
        for i in range(n):
            for j in range(i):
                result[k, 0], result[k, 1] = i, j
                k += 1
        return result
    keys = (coords[:, 0] - lower[0]) * stride + coords[:, 1] - lower[1]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    result = np.empty((n * (n - 1) // 2, 2), dtype=np.int64)
    k = 0
    for i in range(n):
        for cx in range(coords[i, 0] - reach, coords[i, 0] + reach + 1):
            for cy in range(coords[i, 1] - reach, coords[i, 1] + reach + 1):
                if cy < lower[1] or cy >= lower[1] + stride:
                    continue
                key = (cx - lower[0]) * stride + cy - lower[1]
                first = np.searchsorted(sorted_keys, key)
                last = np.searchsorted(sorted_keys, key, side="right")
                for index in range(first, last):
                    j = order[index]
                    if j < i:
                        result[k, 0], result[k, 1] = i, j
                        k += 1
    return result[:k]


@njit(cache=True)
def geometry_valid(p, obstacles, radius, pairs, walls):
    """Independent scalar geometry admission, without the swept constraint margin.

    Returns:
        Whether every physical body constraint is satisfied.
    """
    if pairs:
        for i in range(len(p)):
            for j in range(i):
                dx, dy = p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]
                if dx * dx + dy * dy < (2 * radius) ** 2:
                    return False
    if walls:
        for i in range(len(p)):
            for segment in obstacles:
                if near_segment(p[i], p[i], segment, radius):
                    delta = p[i] - closest_point(p[i], segment)
                    if np.dot(delta, delta) < radius**2:
                        return False
    return True


@njit(cache=True)
def project_step(previous, proposed, obstacles, radius, pairs, walls, max_passes, fixed_mask=None):
    """Grid Gauss-Seidel over-relaxation, swept constraints and start push-out.

    Returns:
        Corrected positions, pass count, and geometry admission status.
    """
    p = proposed.copy()
    old_positions = previous.copy()
    target = 2 * radius + SEPARATION_MARGIN_M
    wall_radius = radius + SEPARATION_MARGIN_M
    if walls:
        for i in range(len(p)):
            old_positions[i] = push_out(previous[i], obstacles, wall_radius)
            p[i] += old_positions[i] - previous[i]
            p[i] = push_out(p[i], obstacles, wall_radius)
    for iteration in range(max_passes):
        maximum = 0.0
        if pairs:
            candidates = grid_pairs(p, old_positions, target)
            # Alternate order to avoid systematic queue-direction bias.
            for index in range(len(candidates)):
                pair = candidates[index if iteration % 2 == 0 else len(candidates) - 1 - index]
                i, j = pair[0], pair[1]
                dx, dy = p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]
                squared = dx * dx + dy * dy
                if squared >= target * target:
                    ox = old_positions[i, 0] - old_positions[j, 0]
                    oy = old_positions[i, 1] - old_positions[j, 1]
                    mx, my = dx - ox, dy - oy
                    motion_squared = mx * mx + my * my
                    t = (
                        max(0.0, min(1.0, -(ox * mx + oy * my) / motion_squared))
                        if motion_squared
                        else 0.0
                    )
                    cx, cy = ox + t * mx, oy + t * my
                    if cx * cx + cy * cy >= target * target:
                        continue
                delta = p[i] - p[j]
                distance = np.sqrt(np.dot(delta, delta))
                correction = max(0.0, target - distance)
                normal = delta / distance if distance > 1e-12 else np.array([1.0, 0.0])
                old = old_positions[i] - old_positions[j]
                if np.dot(old, old) >= target * target:
                    t, swept_n = circle_hit(old, delta, np.zeros(2), target)
                    if t <= 1:
                        swept_correction = target - np.dot(delta, swept_n)
                        if swept_correction > correction:
                            correction, normal = swept_correction, swept_n
                if correction > 0:
                    fixed_i = fixed_mask is not None and fixed_mask[i]
                    fixed_j = fixed_mask is not None and fixed_mask[j]
                    if fixed_i and fixed_j:
                        return p, iteration + 1, False
                    weight_i = 0.0 if fixed_i else (1.0 if fixed_j else 0.5)
                    weight_j = 0.0 if fixed_j else (1.0 if fixed_i else 0.5)
                    # 1.6 over-relaxation accelerates contact chains without stiffness/dt coupling.
                    p[i] += 1.6 * weight_i * correction * normal
                    p[j] -= 1.6 * weight_j * correction * normal
                    maximum = max(maximum, correction)
        if walls:
            for i in range(len(p)):
                corrected = wall_project(old_positions[i], p[i], obstacles, wall_radius)
                maximum = max(maximum, np.sqrt(np.sum((corrected - p[i]) ** 2)))
                p[i] = corrected
        if maximum < 1e-6 and geometry_valid(p, obstacles, radius, pairs, walls):
            return p, iteration + 1, True
    return p, max_passes, False


@njit(cache=True)
def project_velocity(positions, velocity, obstacles, radius, pairs, walls, fixed):
    """Remove closing relative normal velocity; never add displacement divided by dt.

    Returns:
        Velocities with closing contact components removed.
    """
    v = velocity.copy()
    candidates = grid_pairs(positions, positions, 2 * radius + 1e-5)
    for _ in range(8):
        if pairs:
            for pair in candidates:
                i, j = pair[0], pair[1]
                delta = positions[i] - positions[j]
                distance = np.sqrt(np.dot(delta, delta))
                if distance > 2 * radius + 1e-4 or distance < 1e-12:
                    continue
                normal = delta / distance
                closing = np.dot(v[i] - v[j], normal)
                if closing < 0 and not (fixed[i] and fixed[j]):
                    wi = 0.0 if fixed[i] else (1.0 if fixed[j] else 0.5)
                    wj = 0.0 if fixed[j] else (1.0 if fixed[i] else 0.5)
                    v[i] -= wi * closing * normal
                    v[j] += wj * closing * normal
        if walls:
            for i in range(len(positions)):
                for segment in obstacles:
                    if not near_segment(positions[i], positions[i], segment, radius + 1e-5):
                        continue
                    delta = positions[i] - closest_point(positions[i], segment)
                    distance = np.sqrt(np.dot(delta, delta))
                    if distance < radius + 1e-5 and distance > 1e-12:
                        normal = delta / distance
                        closing = np.dot(v[i], normal)
                        if closing < 0 and not fixed[i]:
                            v[i] -= closing * normal
    return v


def apply_contact_step(sim, previous: np.ndarray) -> None:
    """Project geometry and closing velocities; cap, count and continue on fallback."""
    pairs = getattr(sim.config, "pedestrian_contact_rule", None) == PROJECTION_RULE
    walls = getattr(sim.config.obstacle_force_config, "wall_contact_rule", None) == WALL_RULE
    if not pairs and not walls:
        return
    obstacles = np.asarray(sim.get_raw_obstacles(), dtype=float).reshape(-1, 6)
    current = sim.peds.pos()
    previous_positions = previous[:, :2]
    fixed = np.zeros(sim.peds.size(), dtype=np.bool_)
    fixed_indices = getattr(sim.config, "contact_prescribed_indices", ())
    if fixed_indices:
        fixed[list(fixed_indices)] = True
        current[fixed] = previous_positions[fixed] + sim.peds.d_t * previous[fixed, 2:4]
    radius = float(sim.peds.agent_radius)
    corrected, passes, converged = project_step(
        previous_positions, current, obstacles, radius, pairs, walls, MAX_PASSES, fixed
    )
    sim.contact_projection_fallback_count = getattr(sim, "contact_projection_fallback_count", 0)
    sim.contact_projection_unresolved_count = getattr(sim, "contact_projection_unresolved_count", 0)
    if not converged:
        sim.contact_projection_fallback_count += 1
        # Endpoint-only deterministic push-out resolves incompatible swept planes.
        corrected, _, converged = project_step(
            corrected, corrected, obstacles, radius, pairs, walls, 1024, fixed
        )
        if (
            not converged
            and geometry_valid(previous_positions, obstacles, radius, pairs, walls)
            and not fixed.any()
        ):
            corrected = previous_positions.copy()
            converged = True
        if not converged:
            sim.contact_projection_unresolved_count += 1
        sim.peds.state[~fixed, 2:4] = 0.0
    sim.peds.state[:, :2] = corrected
    velocity = project_velocity(corrected, sim.peds.vel(), obstacles, radius, pairs, walls, fixed)
    speed = np.linalg.norm(velocity, axis=1)
    cap = getattr(sim.peds, "contact_step_speed_caps", sim.peds.max_speeds)
    over = speed > cap
    velocity[over] *= (cap[over] / speed[over])[:, None]
    sim.peds.state[:, 2:4] = velocity
    sim.contact_projection_passes = passes


def contact_law_metadata(sim) -> dict[str, object]:
    """Read effective opt-in contact, wall and integration values from live objects.

    Returns:
        Live law and integration manifest.
    """
    obstacle = sim.config.obstacle_force_config
    wall_force = next((f for f in sim.forces if hasattr(f, "contact_wall_parameters")), None)
    parameters = dict(wall_force.contact_wall_parameters) if wall_force is not None else {}
    return {
        "pedestrian_contact": {
            "law": getattr(sim.config, "pedestrian_contact_rule", None),
            "radius_m": float(sim.peds.agent_radius),
            "max_passes": MAX_PASSES,
            "separation_margin_m": SEPARATION_MARGIN_M,
            "sliding_friction": False,
            "solver": "uniform_grid_alternating_gauss_seidel_1.6",
            "velocity": "closing_normal_removal_then_speed_cap",
            "cap_fallback": "endpoint_push_out_then_admissible_previous",
            "swept_pair_guard": True,
            "prescribed_indices": list(getattr(sim.config, "contact_prescribed_indices", ())),
        },
        "wall": {
            "law": getattr(obstacle, "wall_contact_rule", None),
            "radius_m": float(sim.peds.agent_radius),
            **parameters,
            "surface_aggregation": "nearest_with_finite_normal_blend",
            "nonpenetration": "swept_capsule_projection",
        },
        "integration": {"dt_s": float(sim.peds.d_t), "integrator": sim.peds.integration_scheme},
    }


def validate_physics_manifest(sim, declared: dict[str, object]) -> None:
    """Reject a declaration that differs from the effective live simulator."""
    if declared != contact_law_metadata(sim):
        raise ValueError("declared pedestrian physics differs from live simulator")
