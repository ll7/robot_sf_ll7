"""Descriptive, predeclared development trace analysis; no simulator changes.

The fixed campaign contract lives in docs/context/evidence/confinement_dev_trace_20261018.
"""

from __future__ import annotations

import itertools
import math
import random


def seed_draws(count=10000):
    """Resample complete 30-seed blocks once, jointly for every estimand."""
    rng = random.Random(42)
    return [[rng.randrange(30) for _ in range(30)] for _ in range(count)]


def interval(values):
    """Use the dissertation's rounded percentile indices on defined replicates."""
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    return [xs[max(0, round(0.025 * n) - 1)], xs[min(n - 1, round(0.975 * n))]]


def face_distance(position, passage):
    """Centre distance to the nearest finite entry/exit segment, in metres."""
    axis = passage["axis"]
    x, y = position[axis], position[1 - axis]
    dy = max(passage["lateral_min"] - y, 0, y - passage["lateral_max"])
    return math.hypot(min(abs(x - passage["entry"]), abs(x - passage["exit"])), dy)


def _passage_transition(a, b, passage, direction):
    """Intersect a linear sample segment with finite faces of the aperture."""
    axis = passage["axis"]
    x0, x1, y0, y1 = a[axis], b[axis], a[1 - axis], b[1 - axis]
    lo, hi = passage["entry"], passage["exit"]
    low, high = passage["lateral_min"], passage["lateral_max"]
    dx = x1 - x0
    if x0 < lo <= x1:
        direction = 1
    elif x0 > hi >= x1:
        direction = -1
    if direction is None:
        return None, False
    if dx:
        ends = sorted(((lo - x0) / dx, (hi - x0) / dx))
        left, right = max(0, ends[0]), min(1, ends[1])
    else:
        left, right = 0, 1
    if left > right or any(not low <= y0 + t * (y1 - y0) <= high for t in (left, right)):
        return None, False
    completed = (direction == 1 and x1 >= hi) or (direction == -1 and x1 <= lo)
    retreated = (direction == 1 and x1 < lo) or (direction == -1 and x1 > hi)
    return (None if completed or retreated else direction), completed


def actor_indicators(frames, passages):
    """Count complete aperture crossings without bridging reset/respawn boundaries."""
    found, active = set(), {}
    uninterrupted = set()
    all_passages_uninterrupted = False
    stationary = False
    slow_duration = 0.0
    max_error = 0.0
    excluded = 0
    first_step_sentinels = 0
    max_sentinel_discrepancy = 0.0
    first = frames[0]["position"]
    inside = any(
        p["entry"] < first[p["axis"]] < p["exit"]
        and p["lateral_min"] <= first[1 - p["axis"]] <= p["lateral_max"]
        for p in passages
    )
    for prev, cur in itertools.pairwise(frames):
        dt = cur["time"] - prev["time"]
        if dt <= 0:
            raise ValueError("nonpositive trace dt")
        a, b = prev["position"], cur["position"]
        delta = [b[j] - a[j] for j in (0, 1)]
        if not all(math.isfinite(v) for v in a + b + cur["velocity"]):
            raise ValueError("nonfinite trace")
        error = max(abs(cur["velocity"][j] - delta[j] / dt) for j in (0, 1))
        sentinel = cur.get("first_step_sentinel", False) and cur["velocity"] == [0, 0]
        if sentinel:
            first_step_sentinels += 1
            max_sentinel_discrepancy = max(max_sentinel_discrepancy, error)
        else:
            max_error = max(max_error, error)
        if error > 1e-6 and not sentinel:
            raise ValueError("recorded velocity differs from position differences")
        if cur.get("boundary") or math.hypot(*delta) > 10 * dt:
            active.clear()
            uninterrupted.clear()
            slow_duration = 0.0
            excluded += 1
            continue
        near = all(min(face_distance(pos, p) for p in passages) <= 2 for pos in (a, b))
        slow_duration = slow_duration + dt if math.hypot(*delta) / dt < 0.05 and near else 0.0
        stationary |= slow_duration >= 3 - 1e-9
        for p in passages:
            name = p["name"]
            direction, completed = _passage_transition(a, b, p, active.get(name))
            if direction is None:
                active.pop(name, None)
            else:
                active[name] = direction
            if completed:
                found.add(name)
                uninterrupted.add(name)
                all_passages_uninterrupted |= len(uninterrupted) == len(passages)
    return {
        "passages": sorted(found),
        "all_passages_uninterrupted": all_passages_uninterrupted,
        "stationary": stationary,
        "initial_inside": inside,
        "first_step_sentinels": first_step_sentinels,
        "max_sentinel_discrepancy": max_sentinel_discrepancy,
        "excluded_transitions": excluded,
        "max_velocity_error": max_error,
    }


def episode_indicators(record, passages):
    """Convert stable trace IDs into actor tracks, retaining reset positions."""
    trace = record["algorithm_metadata"]["simulation_step_trace"]
    reset = trace["reset"]
    tracks = {}
    # Reset payload uses the same actor records as frames.
    for ped in reset["pedestrians"]:
        tracks[str(ped["id"])] = [
            {
                "time": 0.0,
                "position": ped["position"],
                "velocity": ped["velocity"],
                "boundary": False,
            }
        ]
    for step_index, step in enumerate(trace["steps"]):
        excluded = {str(x) for x in step.get("respawn_actor_ids", [])}
        for ped in step["pedestrians"]:
            key = str(ped["id"])
            fs = tracks.setdefault(key, [])
            fs.append(
                {
                    "time": step["time_s"],
                    "position": ped["position"],
                    "velocity": ped["velocity"],
                    "boundary": key in excluded,
                    "first_step_sentinel": step_index == 0,
                }
            )
    indicators = [actor_indicators(fs, passages) for fs in tracks.values() if fs]
    return {
        "traversal": any(x["passages"] for x in indicators),
        "stationary": any(x["stationary"] for x in indicators),
        "actors": len(indicators),
        "first_step_sentinels": sum(x["first_step_sentinels"] for x in indicators),
        "max_sentinel_discrepancy": max(
            (x["max_sentinel_discrepancy"] for x in indicators), default=0
        ),
        "stationary_actors": sum(x["stationary"] for x in indicators),
        "initial_inside_actors": sum(x["initial_inside"] for x in indicators),
        "excluded_transitions": sum(x["excluded_transitions"] for x in indicators),
        "max_velocity_error": max((x["max_velocity_error"] for x in indicators), default=0),
        "passage_actors": {
            p["name"]: sum(p["name"] in x["passages"] for x in indicators) for p in passages
        },
        "all_passages_actors": sum(x["all_passages_uninterrupted"] for x in indicators),
        "duration_s": trace["steps"][-1]["time_s"],
    }


def group_success(rows, replicates=10000):
    """Estimate conditional rates and paired contrasts from shared seed-block draws."""
    seeds = list(range(1091, 1121))
    if any(row["seed"] not in seeds for row in rows):
        raise ValueError("unexpected development seed")
    draws = seed_draws(replicates)
    rates, distributions = [], {}
    for group in sorted({r["group"] for r in rows}):
        policies = sorted({r["policy"] for r in rows if r["group"] == group})
        for policy in policies:
            selected = [r for r in rows if r["group"] == group and r["policy"] == policy]
            for stratum in ("unconditional", "traversal", "no_traversal"):
                subset = [
                    r
                    for r in selected
                    if stratum == "unconditional" or r["traversal"] == (stratum == "traversal")
                ]
                counts = [
                    (
                        sum(r["success"] for r in subset if r["seed"] == s),
                        sum(r["seed"] == s for r in subset),
                    )
                    for s in seeds
                ]
                successes = sum(x[0] for x in counts)
                denominator = sum(x[1] for x in counts)
                samples = []
                for draw in draws:
                    n = sum(counts[i][0] for i in draw)
                    d = sum(counts[i][1] for i in draw)
                    samples.append(n / d if d else None)
                defined = [x for x in samples if x is not None]
                distributions[group, policy, stratum] = samples
                rates.append(
                    {
                        "group": group,
                        "policy": policy,
                        "stratum": stratum,
                        "successes": successes,
                        "episodes": denominator,
                        "rate": successes / denominator if denominator else None,
                        "interval": interval(defined),
                        "empty_replicates": replicates - len(defined),
                    }
                )
    contrasts = []
    lookup = {(r["group"], r["policy"], r["stratum"]): r for r in rates}
    for group in sorted({r["group"] for r in rows}):
        for policy in sorted(
            {r["policy"] for r in rows if r["group"] == group} - {"orca", "socnav_sampling"}
        ):
            hybrid = lookup[group, policy, "traversal"]["rate"]
            baseline = lookup[group, "orca", "traversal"]["rate"]
            hs = distributions[group, policy, "traversal"]
            bs = distributions[group, "orca", "traversal"]
            diffs = [h - b for h, b in zip(hs, bs, strict=True) if h is not None and b is not None]
            contrasts.append(
                {
                    "group": group,
                    "policy": policy,
                    "difference": hybrid - baseline
                    if hybrid is not None and baseline is not None
                    else None,
                    "interval": interval(diffs),
                    "empty_replicates": replicates - len(diffs),
                }
            )
    return {
        "rates": rates,
        "contrasts": contrasts,
        "bootstrap_replicates": replicates,
        "bootstrap_rng": "random.Random(42)",
        "resampling_unit": "complete development seed block",
    }
