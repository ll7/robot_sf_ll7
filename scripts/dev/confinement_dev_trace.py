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


def actor_indicators(frames, passages):  # noqa: C901 - explicit predeclared state machine.
    """Count complete aperture crossings without bridging reset/respawn boundaries."""
    found, active = set(), {}
    stationary = False
    slow_duration = 0.0
    max_error = 0.0
    excluded = 0
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
        max_error = max(max_error, error)
        if error > 1e-6:
            raise ValueError("recorded velocity differs from position differences")
        if cur.get("boundary") or math.hypot(*delta) > 10 * dt:
            active.clear()
            slow_duration = 0.0
            excluded += 1
            continue
        near = all(min(face_distance(pos, p) for p in passages) <= 2 for pos in (a, b))
        slow_duration = slow_duration + dt if math.hypot(*delta) / dt < 0.05 and near else 0.0
        stationary |= slow_duration >= 3 - 1e-9
        for p in passages:
            name, axis = p["name"], p["axis"]
            lo, hi = p["entry"], p["exit"]
            low, high = p["lateral_min"], p["lateral_max"]
            x0, x1 = a[axis], b[axis]
            # Convex aperture: both endpoints inside the lateral bounds suffice.
            lateral = low <= a[1 - axis] <= high and low <= b[1 - axis] <= high
            if not lateral:
                active.pop(name, None)
                continue
            if x0 < lo and x1 >= lo:
                active[name] = 1
            elif x0 > hi and x1 <= hi:
                active[name] = -1
            direction = active.get(name)
            if (direction == 1 and x1 >= hi) or (direction == -1 and x1 <= lo):
                found.add(name)
                active.pop(name, None)
            elif (direction == 1 and x1 < lo) or (direction == -1 and x1 > hi):
                active.pop(name, None)
    return {
        "passages": sorted(found),
        "stationary": stationary,
        "initial_inside": inside,
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
    for step in trace["steps"]:
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
                }
            )
    indicators = [actor_indicators(fs, passages) for fs in tracks.values() if fs]
    return {
        "traversal": any(x["passages"] for x in indicators),
        "stationary": any(x["stationary"] for x in indicators),
        "actors": len(indicators),
        "stationary_actors": sum(x["stationary"] for x in indicators),
        "initial_inside_actors": sum(x["initial_inside"] for x in indicators),
        "excluded_transitions": sum(x["excluded_transitions"] for x in indicators),
        "max_velocity_error": max((x["max_velocity_error"] for x in indicators), default=0),
        "passage_actors": {
            p["name"]: sum(p["name"] in x["passages"] for x in indicators) for p in passages
        },
        "all_passages_actors": sum(len(x["passages"]) == len(passages) for x in indicators),
        "duration_s": trace["steps"][-1]["time_s"],
    }
