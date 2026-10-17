"""Opt-in bounded normal velocity projection; never repairs positions or substeps."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numba import njit
from scipy import spatial

# The installed typing namespace omits this public compiled extension export.
cKDTree = cast("Any", spatial).cKDTree

RULE = "projection_v2"
MARGIN_M = 1e-8


@njit(nogil=True)
def _constraints(xy, caps, walls, radius, dt, buffers, broad_phase, wall_geometry):
    """Build reachable supporting halfspaces and refuse already overlapping starts.

    Returns:
        Number of reachable pair and wall constraints.
    """
    indices, normals, gap_rates = buffers
    pairs, order = broad_phase
    count = 0
    cursor = 0
    for i in order:
        while cursor < len(pairs) and pairs[cursor, 0] == i:
            j = pairs[cursor, 1]
            cursor += 1
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
            gap_rates[count] = (distance - 2 * radius) / dt
            count += 1
        for wall_index, wall in enumerate(walls):
            wx, wy, denominator = wall_geometry[wall_index]
            t = 0.0
            if denominator != 0.0:
                t = ((xy[i, 0] - wall[0]) * wx + (xy[i, 1] - wall[1]) * wy) / denominator
                t = max(0.0, min(1.0, t))
            x, y = wall[0] + t * wx, wall[1] + t * wy
            dx, dy = xy[i, 0] - x, xy[i, 1] - y
            distance2 = dx * dx + dy * dy
            if distance2 < (radius - 1e-10) ** 2:
                raise ValueError("contact requires nonoverlapping initial walls")
            if distance2 > (radius + dt * caps[i] + MARGIN_M) ** 2:
                continue
            distance = np.sqrt(distance2)
            indices[count] = i, -wall_index - 1
            normals[count, 0], normals[count, 1] = dx / distance, dy / distance
            gap_rates[count] = (distance - radius) / dt
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
def _gap_rate(k, velocity, indices, normals, gap_rates):
    """Return predicted normal clearance per second for one constraint."""
    i, j = indices[k]
    dx, dy = velocity[i, 0], velocity[i, 1]
    if j >= 0:
        dx -= velocity[j, 0]
        dy -= velocity[j, 1]
    return gap_rates[k] + dx * normals[k, 0] + dy * normals[k, 1]


@njit(nogil=True, inline="always")
def _halfspace(k, actor, velocity, fixed, indices, normals, gap_rates, reciprocal):
    """Orient one neighboring normal bound towards the actor being resolved.

    Returns:
        Oriented unit normal and its lower velocity bound.
    """
    i, j = indices[k]
    nx, ny = normals[k, 0], normals[k, 1]
    if actor != i:
        j = i
        nx, ny = -nx, -ny
    bound = -gap_rates[k]
    if j >= 0:
        if reciprocal and not fixed[j]:
            bound *= 0.5
        else:
            bound += nx * velocity[j, 0] + ny * velocity[j, 1]
    return nx, ny, bound


@njit(nogil=True)
def _actor_velocity(  # noqa: PLR0913
    actor,
    proposal,
    velocity,
    caps,
    fixed,
    indices,
    normals,
    gap_rates,
    offsets,
    neighbors,
    reciprocal,
):
    """Project onto a local 2-D normal polytope intersected with the speed ball.

    Each violated line is solved exactly by clipping its tangent interval against
    preceding lines and the speed circle. A single contact leaves tangent motion
    unchanged. No contact iteration, position repair or timestep subdivision is
    hidden in this finite per-neighbor line scan.

    Returns:
        Whether the local feasible projection was found.
    """
    vx, vy = proposal[actor, 0], proposal[actor, 1]
    cap2 = caps[actor] * caps[actor]
    for cursor in range(offsets[actor], offsets[actor + 1]):
        nx, ny, bound = _halfspace(
            neighbors[cursor], actor, velocity, fixed, indices, normals, gap_rates, reciprocal
        )
        if nx * vx + ny * vy >= bound - 1e-12:
            continue
        circle = cap2 - bound * bound
        if circle < -1e-12:
            return False
        px, py = nx * bound, ny * bound
        tx, ty = -ny, nx
        extent = np.sqrt(max(0.0, circle))
        low, high = -extent, extent
        for earlier in range(offsets[actor], cursor):
            ax, ay, lower = _halfspace(
                neighbors[earlier], actor, velocity, fixed, indices, normals, gap_rates, reciprocal
            )
            slope = ax * tx + ay * ty
            intercept = lower - ax * px - ay * py
            if abs(slope) < 1e-14:
                if intercept > 1e-12:
                    return False
            elif slope > 0:
                low = max(low, intercept / slope)
            else:
                high = min(high, intercept / slope)
        if low > high + 1e-12:
            return False
        tangent = tx * proposal[actor, 0] + ty * proposal[actor, 1]
        tangent = max(low, min(high, tangent))
        vx, vy = px + tx * tangent, py + ty * tangent
    velocity[actor, 0], velocity[actor, 1] = vx, vy
    return True


@njit(nogil=True)
def _unresolved(velocity, indices, normals, gap_rates, count, tolerance):
    """Count constraints outside the numerical normal-rate tolerance.

    Returns:
        Number of violated constraints.
    """
    unresolved = 0
    for k in range(count):
        unresolved += _gap_rate(k, velocity, indices, normals, gap_rates) < -tolerance
    return unresolved


@njit(nogil=True)
def _solve(xy, velocity, caps, fixed, indices, normals, gap_rates, count, passes, order):  # noqa: PLR0913
    """Resolve local normal budgets, then retain feasibility in bounded actor sweeps.

    The first pass shares a pair's available approach distance. Later passes let
    each actor reclaim unused distance against the already feasible neighbor
    velocities. Actors at the front of the mean flow go first, preserving convoy
    motion. Every update satisfies all incident contacts; no graph-wide halt is
    needed. This emergency path never halts a reachable component. A failed local
    line solve retains only that actor's prior feasible velocity; infeasible
    prescribed motion is refused before integration.

    Returns:
        Emergency activation and remaining infeasible constraint count.
    """
    if _unresolved(velocity, indices, normals, gap_rates, count, 1e-12) == 0:
        return False, 0
    n = len(velocity)
    proposal = velocity.copy()
    offsets = np.zeros(n + 1, dtype=np.int64)
    for k in range(count):
        i, j = indices[k]
        offsets[i + 1] += 1
        if j >= 0:
            offsets[j + 1] += 1
    for i in range(n):
        offsets[i + 1] += offsets[i]
    neighbors = np.empty(offsets[n], dtype=np.int64)
    cursor = offsets[:-1].copy()
    for k in range(count):
        i, j = indices[k]
        neighbors[cursor[i]] = k
        cursor[i] += 1
        if j >= 0:
            neighbors[cursor[j]] = k
            cursor[j] += 1
    flow_x, flow_y = 0.0, 0.0
    for i in order:
        if not fixed[i]:
            flow_x += proposal[i, 0]
            flow_y += proposal[i, 1]
            velocity[i, 0], velocity[i, 1] = 0.0, 0.0
    if abs(flow_x) + abs(flow_y) > 1e-12:
        forward = xy[order, 0] * flow_x + xy[order, 1] * flow_y
        # Stable ties start from physical coordinate order, never state-row IDs.
        order = order[np.argsort(forward, kind="mergesort")][::-1]
    fallback = False
    for iteration in range(passes):
        for i in order:
            if not fixed[i]:
                if not _actor_velocity(
                    i,
                    proposal,
                    velocity,
                    caps,
                    fixed,
                    indices,
                    normals,
                    gap_rates,
                    offsets,
                    neighbors,
                    iteration == 0,
                ):
                    fallback = True
    return fallback, _unresolved(velocity, indices, normals, gap_rates, count, 1e-10)


@njit(nogil=True, inline="always")
def _row_dot(k, other, fixed, indices, normals):
    """Return the free-actor Gram entry for two contact normals."""
    i, j = indices[k]
    u, v = indices[other]
    dot = normals[k, 0] * normals[other, 0] + normals[k, 1] * normals[other, 1]
    weight = 0
    if not fixed[i]:
        weight += int(i == u) - int(i == v)
    if j >= 0 and not fixed[j]:
        weight += int(j == v) - int(j == u)
    return weight * dot


@njit(nogil=True, inline="always")
def _impulse(k, amount, velocity, fixed, indices, normals):
    """Apply only the selected contact normal to its movable incident actors."""
    i, j = indices[k]
    if not fixed[i]:
        velocity[i, 0] += amount * normals[k, 0]
        velocity[i, 1] += amount * normals[k, 1]
    if j >= 0 and not fixed[j]:
        velocity[j, 0] -= amount * normals[k, 0]
        velocity[j, 1] -= amount * normals[k, 1]


@njit(nogil=True)
def _delete_factor(factor, selected, multipliers, m, deleted):
    """Delete one active normal with finite Givens updates of its Gram factor."""
    # The factor is stored as lower-triangular L. Row deletion and triangular
    # solves therefore traverse contiguous memory instead of strided columns.
    for row in range(deleted, m - 1):
        for col in range(m):
            factor[row, col] = factor[row + 1, col] if col <= row + 1 else 0.0
    for col in range(deleted, m - 1):
        upper, lower = factor[col, col], factor[col, col + 1]
        norm = np.sqrt(upper * upper + lower * lower)
        cosine, sine = upper / norm, lower / norm
        for row in range(col, m - 1):
            upper, lower = factor[row, col], factor[row, col + 1]
            factor[row, col] = cosine * upper + sine * lower
            factor[row, col + 1] = -sine * upper + cosine * lower
    for a in range(deleted, m - 1):
        selected[a], multipliers[a] = selected[a + 1], multipliers[a + 1]


@njit(nogil=True, inline="always")
def _cache_slot(k, indices, n):
    """Return a stable pair or wall key."""
    i, j = indices[k]
    # A pair's physical normal row is unchanged when both endpoints and the
    # normal orientation swap. Keep its cache key independent of that swap.
    if j >= 0:
        return min(i, j), max(i, j)
    return i, n - j - 1


@njit(nogil=True)
def _store_basis(warm, indices, count, selected, multipliers, m):
    """Retain only nonnegative active normal multipliers."""
    warm[:] = 0.0
    for a in range(m):
        i, j = _cache_slot(selected[a], indices, len(warm))
        warm[i, j] = max(0.0, multipliers[a])


@njit(nogil=True)
def _seed_basis(  # noqa: PLR0913
    velocity, fixed, indices, normals, gap_rates, count, warm, factor, selected, multipliers, active
):
    """Seed a dual-feasible basis, dropping negative cached rows before pivoting.

    This changes algebraic initialization only; the same exact active-normal
    solve verifies every reachable inequality before accepting the result.
    Returns:
        Number of retained, independent active rows.
    """
    m = 0
    for k in range(count):
        i, j = _cache_slot(k, indices, len(warm))
        if warm[i, j] > 1e-12 and _row_dot(k, k, fixed, indices, normals) > 0:
            selected[m] = k
            m += 1
    for _ in range(6):
        if m == 0:
            return 0
        rows = selected[:m]
        first, second = indices[rows, 0], indices[rows, 1]
        movable_first = ~fixed[first]
        movable_second = (second >= 0) & ~fixed[np.maximum(second, 0)]
        # Vectorized free-actor incidence Gram matrix, with the same scalar
        # products and row order as _row_dot (no change to the contact QP).
        weights = (
            (first[:, None] == first[None, :]).astype(np.float64)
            - (first[:, None] == second[None, :])
        ) * movable_first[:, None] + (
            (second[:, None] == second[None, :]).astype(np.float64)
            - (second[:, None] == first[None, :])
        ) * movable_second[:, None]
        nx, ny = normals[rows, 0], normals[rows, 1]
        matrix = weights * (nx[:, None] * nx[None, :] + ny[:, None] * ny[None, :])
        rhs = np.empty(m)
        for a in range(m):
            rhs[a] = -_gap_rate(selected[a], velocity, indices, normals, gap_rates)
        try:
            lower = np.linalg.cholesky(matrix)
            lam = np.linalg.solve(matrix, rhs)
        except Exception:  # noqa: BLE001 - compiled Numba permits only Exception matching.
            # Failed cached Gram algebra leaves the velocity untouched. The
            # independent exact active-normal solve starts without that cache.
            return 0
        retained = 0
        negative = False
        for a in range(m):
            if lam[a] < 0.0:
                negative = True
            else:
                selected[retained] = selected[a]
                retained += 1
        if negative:
            m = retained
            continue
        for a in range(m):
            multipliers[a] = lam[a]
            active[selected[a]] = True
            for b in range(a + 1):
                factor[a, b] = lower[a, b]
            _impulse(selected[a], lam[a], velocity, fixed, indices, normals)
        return m
    return 0


@njit(nogil=True)
def _normal_projection(velocity, fixed, indices, normals, gap_rates, count, warm):  # noqa: PLR0912
    """Solve coupled active normals with incremental dual Cholesky updates.

    This is one exact normal projection, not repeated pair sweeps. Only active
    rows sharing a movable actor couple; disjoint pairs have zero Gram entries.
    Every finite basis exchange removes a normal or accepts the incoming one.

    Returns:
        Whether the normal projection attained numerical feasibility.
    """
    if _unresolved(velocity, indices, normals, gap_rates, count, 1e-11) == 0:
        warm[:] = 0.0
        return True
    dimension = min(count, 2 * len(velocity))
    factor = np.empty((dimension, dimension))
    selected = np.empty(dimension, dtype=np.int64)
    multipliers = np.zeros(dimension)
    active = np.zeros(count, dtype=np.bool_)
    h = np.empty(dimension)
    y = np.empty(dimension)
    r = np.empty(dimension)
    m = _seed_basis(
        velocity,
        fixed,
        indices,
        normals,
        gap_rates,
        count,
        warm,
        factor,
        selected,
        multipliers,
        active,
    )
    for _ in range(2 * count + dimension + 1):
        incoming, deficit = -1, 1e-11
        for k in range(count):
            if not active[k]:
                violation = -_gap_rate(k, velocity, indices, normals, gap_rates)
                if violation > deficit:
                    incoming, deficit = k, violation
        if incoming < 0:
            if _unresolved(velocity, indices, normals, gap_rates, count, 1e-10) == 0:
                _store_basis(warm, indices, count, selected, multipliers, m)
                return True
            return False
        new_multiplier = 0.0
        for _ in range(dimension + 1):
            linked = False
            for a in range(m):
                h[a] = _row_dot(incoming, selected[a], fixed, indices, normals)
                linked = linked or h[a] != 0.0
                y[a], r[a] = 0.0, 0.0
            if linked:
                for a in range(m):
                    value = h[a]
                    for b in range(a):
                        value -= factor[a, b] * y[b]
                    y[a] = value / factor[a, a]
                for a in range(m):
                    r[a] = y[a]
                for a in range(m - 1, -1, -1):
                    r[a] /= factor[a, a]
                    for b in range(a):
                        r[b] -= factor[a, b] * r[a]
            denominator = _row_dot(incoming, incoming, fixed, indices, normals)
            for a in range(m):
                denominator -= y[a] * y[a]
            dual_step, leaving = np.inf, -1
            for a in range(m):
                if r[a] > 1e-12:
                    bound = max(0.0, multipliers[a]) / r[a]
                    if bound < dual_step:
                        dual_step, leaving = bound, a
            primal_step = np.inf
            if denominator > 1e-12:
                primal_step = (
                    max(0.0, -_gap_rate(incoming, velocity, indices, normals, gap_rates))
                    / denominator
                )
            step = min(dual_step, primal_step)
            if not np.isfinite(step):
                return False
            for a in range(m):
                amount = step * r[a]
                multipliers[a] -= amount
                _impulse(selected[a], -amount, velocity, fixed, indices, normals)
            _impulse(incoming, step, velocity, fixed, indices, normals)
            new_multiplier += step
            if primal_step <= dual_step:
                if m >= dimension:
                    return False
                for a in range(m):
                    factor[m, a] = y[a]
                factor[m, m] = np.sqrt(denominator)
                selected[m], multipliers[m] = incoming, new_multiplier
                active[incoming] = True
                m += 1
                break
            active[selected[leaving]] = False
            _delete_factor(factor, selected, multipliers, m, leaving)
            m -= 1
        else:
            return False
    return False


@njit(nogil=True)
def _pair_sweeps(velocity, caps, fixed, indices, normals, gap_rates, count, target, passes):  # noqa: PLR0913
    """Run bounded, friction-free normal impulses on a small wall-free cell."""
    for _ in range(passes):
        for k in range(count):
            deficit = target - _gap_rate(k, velocity, indices, normals, gap_rates)
            if deficit <= 0.0:
                continue
            i, j = indices[k]
            weight = int(not fixed[i]) + int(not fixed[j])
            if weight == 0:
                continue
            impulse = deficit / weight
            nx, ny = normals[k, 0], normals[k, 1]
            if not fixed[i]:
                velocity[i, 0] += impulse * nx
                velocity[i, 1] += impulse * ny
                _cap(velocity, i, caps)
            if not fixed[j]:
                velocity[j, 0] -= impulse * nx
                velocity[j, 1] -= impulse * ny
                _cap(velocity, j, caps)


@njit(nogil=True)
def _project(  # noqa: PLR0912, PLR0913
    xy, velocity, caps, walls, radius, dt, fixed, buffers, passes, broad_phase, wall_geometry
):
    """Bind broad phase and solve in one compiled call.

    Returns:
        Fallback activation and unresolved count.
    """
    for i in range(len(velocity)):
        if not np.isfinite(velocity[i, 0]) or not np.isfinite(velocity[i, 1]):
            raise ValueError("contact requires finite capped velocities")
    indices, normals, gap_rates, warm = buffers
    _, order = broad_phase
    count = _constraints(
        xy, caps, walls, radius, dt, (indices, normals, gap_rates), broad_phase, wall_geometry
    )
    for i in range(len(velocity)):
        if caps[i] == 0.0:
            fixed[i] = True
    if count == 0:
        # No supporting pair or wall can be reached in this capped displacement.
        # Match the old no-constraint cache/cap effects without entering a solver.
        if not 4 < len(velocity) <= 30:
            warm[:] = 0.0
            for i in range(len(velocity)):
                if not fixed[i]:
                    _cap(velocity, i, caps)
        return False, 0
    proposal = velocity.copy()
    wall_free = True
    for k in range(count):
        wall_free = wall_free and indices[k, 1] >= 0
    if 4 < len(velocity) <= 30 and wall_free:
        _pair_sweeps(
            velocity, caps, fixed, indices, normals, gap_rates, count, MARGIN_M / dt, passes
        )
        if _unresolved(velocity, indices, normals, gap_rates, count, 1e-10) == 0:
            return False, 0
    else:
        guarded = gap_rates[:count] - MARGIN_M / dt
        for _ in range(passes - 1):
            if not _normal_projection(velocity, fixed, indices, normals, guarded, count, warm):
                break
            for i in range(len(velocity)):
                if not fixed[i]:
                    _cap(velocity, i, caps)
            if _unresolved(velocity, indices, normals, gap_rates, count, 1e-10) == 0:
                return False, 0
    # A single local reciprocal safety pass, never component halting. This
    # emergency is reported separately from ordinary normal/cap resolution.
    velocity[:] = proposal
    _, unresolved = _solve(xy, velocity, caps, fixed, indices, normals, gap_rates, count, 1, order)
    return True, unresolved


class VelocityProjection:
    """Bind the native integration hook to live geometry, holds and speed caps."""

    def __init__(self, sim, passes):
        """Bind a simulator and an already validated correction-pass budget."""
        self.sim = sim
        self.passes = passes
        self.capacity = -1
        self.population = -1
        self.radius = getattr(sim.config, "pedestrian_contact_radius_m", sim.peds.agent_radius)
        self.wall_source = None

    def __call__(self, velocity):
        """Return capped contact velocities, refusing infeasible prescribed motion.

        Returns:
            Corrected velocities used for both displacement and stored motion.
        """
        sim = self.sim
        peds = sim.peds
        n = peds.size()
        walls = sim.get_raw_obstacles()
        source = walls
        if walls.ndim != 2:
            walls = walls.reshape(-1, 6)
        # EnvState already caches raw segments. Cache their static projection
        # vectors too, invalidating on the supported setter or in-place edits.
        if source is not self.wall_source or not np.array_equal(
            walls[:, :4], self.wall_coordinates
        ):
            self.wall_source = source
            self.wall_coordinates = walls[:, :4].copy()
            dx, dy = walls[:, 2] - walls[:, 0], walls[:, 3] - walls[:, 1]
            self.wall_geometry = np.column_stack((dx, dy, dx * dx + dy * dy))
        capacity = n * (n - 1) // 2 + n * len(walls)
        if capacity != self.capacity or n != self.population:
            self.indices = np.empty((capacity, 2), dtype=np.int64)
            self.normals = np.empty((capacity, 2))
            self.gap_rates = np.empty(capacity)
            self.capacity = capacity
            self.population = n
            self.fixed = np.zeros(n, dtype=np.bool_)
            self.warm = np.zeros((n, n + len(walls)))
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
        xy = peds.pos()
        # Valid nonoverlapping bodies have distinct positions: lexicographic
        # geometry order gives every physical actor/pair the same priority after
        # relabeling, including walls and the emergency solver. State rows stay
        # untouched so controller indices and recorded actor identities remain valid.
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        # Include pairs reachable during this one capped displacement, rather
        # than only already touching bodies.
        reach = 2 * self.radius + 2 * peds.d_t * np.max(caps, initial=0.0) + MARGIN_M
        pairs = cKDTree(xy).query_pairs(np.nextafter(reach, np.inf), output_type="ndarray")
        if len(pairs):
            reverse = rank[pairs[:, 0]] > rank[pairs[:, 1]]
            pairs[reverse] = pairs[reverse, ::-1]
            pairs = pairs[np.lexsort((rank[pairs[:, 1]], rank[pairs[:, 0]]))]
        fallback, unresolved = _project(
            xy,
            projected,
            caps,
            walls,
            self.radius,
            peds.d_t,
            fixed,
            (self.indices, self.normals, self.gap_rates, self.warm),
            self.passes,
            (pairs, order),
            self.wall_geometry,
        )
        sim.contact_projection_passes = self.passes
        if fallback:
            sim.contact_projection_fallback_count += 1
        if unresolved:
            sim.contact_projection_unresolved_count += int(unresolved)
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
    radius = getattr(sim.config, "pedestrian_contact_radius_m", sim.peds.agent_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("contact requires a positive finite body radius")
    sim.contact_projection_passes = 0
    sim.contact_projection_fallback_count = 0
    sim.contact_projection_unresolved_count = 0
    sim.peds.velocity_projector = VelocityProjection(sim, passes)
