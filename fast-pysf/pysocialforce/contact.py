"""Opt-in bounded normal velocity projection; never repairs positions or substeps."""

from __future__ import annotations

import numpy as np
from numba import njit

from pysocialforce.forces import closest_point_on_segment

RULE = "projection_v2"
MARGIN_M = 1e-9


@njit(nogil=True)
def _constraints(xy, caps, walls, radius, dt, indices, normals, gaps):
    """Build reachable supporting halfspaces and refuse already overlapping starts.

    Returns:
        Number of reachable pair and wall constraints.
    """
    count = 0
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            dx, dy = xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]
            distance2 = dx * dx + dy * dy
            if distance2 < (2 * radius - 1e-10) ** 2:
                raise ValueError("contact requires nonoverlapping initial bodies")
            reach = 2 * radius + dt * (caps[i] + caps[j]) + MARGIN_M
            if distance2 > reach * reach:
                continue
            distance = np.sqrt(distance2)
            indices[count] = i, j
            normals[count, 0], normals[count, 1] = dx / distance, dy / distance
            gaps[count] = distance - 2 * radius
            count += 1
        for wall in walls:
            x, y = closest_point_on_segment(wall[:4], xy[i])
            dx, dy = xy[i, 0] - x, xy[i, 1] - y
            distance2 = dx * dx + dy * dy
            if distance2 < (radius - 1e-10) ** 2:
                raise ValueError("contact requires nonoverlapping initial walls")
            if distance2 > (radius + dt * caps[i] + MARGIN_M) ** 2:
                continue
            distance = np.sqrt(distance2)
            indices[count] = i, -1
            normals[count, 0], normals[count, 1] = dx / distance, dy / distance
            gaps[count] = distance - radius
            count += 1
    return count


@njit(nogil=True, inline="always")
def _cap(velocity, i, caps):
    """Keep each corrected velocity inside its existing execution speed ball."""
    norm2 = velocity[i, 0] ** 2 + velocity[i, 1] ** 2
    if norm2 > caps[i] ** 2:
        ratio = caps[i] / np.sqrt(norm2)
        velocity[i, 0] *= ratio
        velocity[i, 1] *= ratio


@njit(nogil=True, inline="always")
def _gap_rate(k, velocity, indices, normals, gaps, dt):
    """Return predicted normal clearance per second for one constraint."""
    i, j = indices[k]
    dx, dy = velocity[i, 0], velocity[i, 1]
    if j >= 0:
        dx -= velocity[j, 0]
        dy -= velocity[j, 1]
    return gaps[k] / dt + dx * normals[k, 0] + dy * normals[k, 1]


@njit(nogil=True, inline="always")
def _root(parent, i):
    """Find a contact candidate component without changing its membership.

    Returns:
        Component root index.
    """
    while parent[i] != i:
        i = parent[i]
    return i


@njit(nogil=True)
def _solve(velocity, caps, fixed, indices, normals, gaps, count, dt, passes):  # noqa: PLR0913
    """Run exactly the bounded passes; halt only unresolved reachable components.

    Returns:
        Fallback activation and the remaining unresolved constraint count.
    """
    target = MARGIN_M / dt
    for _ in range(passes):
        for k in range(count):
            deficit = target - _gap_rate(k, velocity, indices, normals, gaps, dt)
            if deficit <= 0:
                continue
            i, j = indices[k]
            movable_i = not fixed[i]
            movable_j = j >= 0 and not fixed[j]
            weight = int(movable_i) + int(movable_j)
            if weight == 0:
                continue
            impulse = deficit / weight
            nx, ny = normals[k, 0], normals[k, 1]
            if movable_i:
                velocity[i, 0] += impulse * nx
                velocity[i, 1] += impulse * ny
                _cap(velocity, i, caps)
            if movable_j:
                velocity[j, 0] -= impulse * nx
                velocity[j, 1] -= impulse * ny
                _cap(velocity, j, caps)
    failed = np.zeros(count, dtype=np.bool_)
    for k in range(count):
        failed[k] = _gap_rate(k, velocity, indices, normals, gaps, dt) < -1e-10
    fallback = np.any(failed)
    if fallback:
        # Stopping a body can create contact with a following neighbor. Stop the
        # whole reachable component in one graph pass; isolated walkers keep moving.
        parent = np.arange(len(velocity))
        for k in range(count):
            i, j = indices[k]
            if j >= 0:
                parent[_root(parent, i)] = _root(parent, j)
        halted = np.zeros(len(velocity), dtype=np.bool_)
        for k in range(count):
            if failed[k]:
                halted[_root(parent, indices[k, 0])] = True
        for i in range(len(velocity)):
            if halted[_root(parent, i)] and not fixed[i]:
                velocity[i, 0], velocity[i, 1] = 0.0, 0.0
    unresolved = 0
    for k in range(count):
        unresolved += _gap_rate(k, velocity, indices, normals, gaps, dt) < -1e-10
    return fallback, unresolved


@njit(nogil=True)
def _project(xy, velocity, caps, walls, radius, dt, fixed, buffers, passes):  # noqa: PLR0913
    """Bind broad phase and solve in one compiled call.

    Returns:
        Fallback activation and unresolved count.
    """
    for i in range(len(velocity)):
        if not np.isfinite(velocity[i, 0]) or not np.isfinite(velocity[i, 1]):
            raise ValueError("contact requires finite capped velocities")
    indices, normals, gaps = buffers
    count = _constraints(xy, caps, walls, radius, dt, indices, normals, gaps)
    return _solve(velocity, caps, fixed, indices, normals, gaps, count, dt, passes)


class VelocityProjection:
    """Bind the native integration hook to live geometry, holds and speed caps."""

    def __init__(self, sim, passes):
        """Bind a simulator and an already validated correction-pass budget."""
        self.sim = sim
        self.passes = passes
        self.capacity = -1
        self.population = -1

    def __call__(self, velocity):
        """Return capped contact velocities, refusing infeasible prescribed motion.

        Returns:
            Corrected velocities used for both displacement and stored motion.
        """
        sim = self.sim
        peds = sim.peds
        n = peds.size()
        walls = np.asarray(sim.get_raw_obstacles(), dtype=float).reshape(-1, 6)
        capacity = n * (n - 1) // 2 + n * len(walls)
        if capacity != self.capacity or n != self.population:
            self.indices = np.empty((capacity, 2), dtype=np.int64)
            self.normals = np.empty((capacity, 2))
            self.gaps = np.empty(capacity)
            self.capacity = capacity
            self.population = n
            self.fixed = np.zeros(n, dtype=np.bool_)
        projected = velocity
        fixed = self.fixed
        fixed.fill(False)
        held = getattr(peds, "contact_held_indices", ())
        prescribed = getattr(sim.config, "contact_prescribed_indices", ())
        caps = peds.max_speeds
        if held:
            indices = list(held)
            fixed[indices] = True
            projected[indices] = 0.0
        if prescribed:
            indices = list(prescribed)
            fixed[indices] = True
            projected[indices] = peds.vel()[indices]
            caps = caps.copy()
            caps[indices] = np.linalg.norm(projected[indices], axis=1)
        fallback, unresolved = _project(
            peds.pos(),
            projected,
            caps,
            walls,
            peds.agent_radius,
            peds.d_t,
            fixed,
            (self.indices, self.normals, self.gaps),
            self.passes,
        )
        sim.contact_projection_passes = self.passes
        sim.contact_projection_fallback_count += int(fallback)
        sim.contact_projection_unresolved_count += int(unresolved)
        if unresolved:
            raise ValueError("contact prescribed motion is infeasible; step refused")
        return projected


def install_projection(sim):
    """Install only an explicitly selected rule; default integration stays unchanged."""
    rule = getattr(sim.config, "pedestrian_contact_rule", None)
    if rule is None:
        return
    if rule != RULE:
        raise ValueError("unsupported pedestrian contact rule; select projection_v2")
    passes = getattr(sim.config, "pedestrian_contact_passes", 3)
    if type(passes) is not int or not 3 <= passes <= 10:
        raise ValueError("contact passes must be an integer in 3..10")
    if sim.peds.integration_scheme != "semi_implicit_euler":
        raise ValueError("projection_v2 requires semi_implicit_euler")
    if not np.isfinite(sim.peds.agent_radius) or sim.peds.agent_radius <= 0:
        raise ValueError("contact requires a positive finite body radius")
    sim.contact_projection_passes = 0
    sim.contact_projection_fallback_count = 0
    sim.contact_projection_unresolved_count = 0
    sim.peds.velocity_projector = VelocityProjection(sim, passes)
