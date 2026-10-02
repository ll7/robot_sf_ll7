"""Opt-in frictionless body exclusion and finite-range body-edge wall response.

The legacy simulator never calls these kernels. Constraints act after the
historical velocity integration; swept tests prevent a large step tunnelling
through a wall or exchanging two bodies. Equal-mass pair corrections preserve
the centre of mass. A failed constraint solve is an error, never accepted overlap.
"""

from __future__ import annotations

import numpy as np
from numba import njit

PROJECTION_RULE = "projection_v1"
WALL_RULE = "bounded_edge_v1"
MAX_PASSES = 4096
SEPARATION_MARGIN_M = 1e-6
WALL_AMPLITUDE_M_S2 = 3.0
WALL_DECAY_M = 0.04
WALL_RANGE_M = 0.20


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
def bounded_wall_force(positions, obstacles, radius, amplitude, decay, influence):
    """Use the nearest surface, so tessellation does not multiply wall strength.

    Equal-distance surface normals are averaged, preserving symmetry in a gap.
    The shifted exponential vanishes continuously at the finite influence range.

    Returns:
        Bounded wall acceleration for each pedestrian.
    """
    result = np.zeros_like(positions)
    for i in range(len(positions)):
        nearest = np.inf
        normal = np.zeros(2)
        ties = 0
        for segment in obstacles:
            delta = positions[i] - closest_point(positions[i], segment)
            distance = np.sqrt(np.dot(delta, delta))
            if distance < nearest - 1e-10:
                nearest = distance
                normal = delta / max(distance, 1e-12)
                ties = 1
            elif abs(distance - nearest) <= 1e-10:
                normal += delta / max(distance, 1e-12)
                ties += 1
        clearance = nearest - radius
        if clearance < influence and ties:
            strength = amplitude * (
                np.exp(-max(0.0, clearance) / decay) - np.exp(-influence / decay)
            )
            result[i] = strength * normal / ties
    return result


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
def wall_project(start, end, obstacles, radius):
    """Project swept disc motion onto blocking capsule tangent planes.

    Returns:
        Corrected endpoint of the attempted movement.
    """
    result = end.copy()
    for segment in obstacles:
        a, b = segment[:2], segment[2:4]
        along = b - a
        length = np.sqrt(np.dot(along, along))
        hit_t = 2.0
        hit_n = np.zeros(2)
        hit_c = a
        for centre in (a, b):
            t, normal = circle_hit(start, result, centre, radius)
            if t < hit_t:
                hit_t, hit_n, hit_c = t, normal, centre
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
                if 0 <= coordinate <= length and t < hit_t:
                    hit_t, hit_n, hit_c = t, n, a
        if hit_t <= 1:
            correction = radius - np.dot(result - hit_c, hit_n)
            if correction > 0:
                result += correction * hit_n
    return result


@njit(cache=True)
def project_step(previous, proposed, obstacles, radius, pairs, walls, max_passes, fixed_mask=None):  # noqa: PLR0912
    """Alternate pair and swept-wall constraints until every disc is admissible.

    Returns:
        Corrected positions, pass count and explicit convergence status.
    """
    p = proposed.copy()
    target = 2 * radius + SEPARATION_MARGIN_M
    wall_radius = radius + SEPARATION_MARGIN_M
    for iteration in range(max_passes):
        maximum = 0.0
        if pairs:
            for i in range(len(p)):
                for j in range(i):
                    dx, dy = p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]
                    squared = dx * dx + dy * dy
                    if squared >= target * target:
                        ox = previous[i, 0] - previous[j, 0]
                        oy = previous[i, 1] - previous[j, 1]
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
                    # Use the old side when a separated pair would exchange sides.
                    old = previous[i] - previous[j]
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
                        p[i] += weight_i * correction * normal
                        p[j] -= weight_j * correction * normal
                        maximum = max(maximum, correction)
        if walls:
            for i in range(len(p)):
                fixed = wall_project(previous[i], p[i], obstacles, wall_radius)
                maximum = max(maximum, np.sqrt(np.sum((fixed - p[i]) ** 2)))
                p[i] = fixed
        if maximum < 1e-6:
            # Admit actual body geometry, rather than requiring convergence to
            # an artificial subnanometre margin in a long contact chain.
            valid = True
            if pairs:
                for i in range(len(p)):
                    for j in range(i):
                        delta = p[i] - p[j]
                        if np.dot(delta, delta) < (2 * radius) ** 2:
                            valid = False
            if walls:
                for i in range(len(p)):
                    for segment in obstacles:
                        delta = p[i] - closest_point(p[i], segment)
                        if np.dot(delta, delta) < radius**2:
                            valid = False
            if valid:
                return p, iteration + 1, True
    return p, max_passes, False


def apply_contact_step(sim, previous: np.ndarray) -> None:
    """Apply explicitly selected laws and update only corrected velocities."""
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
    corrected, passes, converged = project_step(
        previous_positions,
        current,
        obstacles,
        float(sim.peds.agent_radius),
        pairs,
        walls,
        MAX_PASSES,
        fixed,
    )
    if not converged:
        sim.peds.state = previous.copy()
        raise RuntimeError("PEDCONTACT constraint solve did not converge; step rejected")
    changed = np.any(corrected != current, axis=1)
    sim.peds.state[changed, 2:4] = (corrected[changed] - previous_positions[changed]) / sim.peds.d_t
    sim.peds.state[:, :2] = corrected
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
            "swept_pair_guard": True,
            "prescribed_indices": list(getattr(sim.config, "contact_prescribed_indices", ())),
        },
        "wall": {
            "law": getattr(obstacle, "wall_contact_rule", None),
            "radius_m": float(sim.peds.agent_radius),
            **parameters,
            "surface_aggregation": "nearest_with_tied_normal_average",
            "nonpenetration": "swept_capsule_projection",
        },
        "integration": {"dt_s": float(sim.peds.d_t), "integrator": sim.peds.integration_scheme},
    }


def validate_physics_manifest(sim, declared: dict[str, object]) -> None:
    """Reject a declaration that differs from the effective live simulator."""
    if declared != contact_law_metadata(sim):
        raise ValueError("declared pedestrian physics differs from live simulator")
