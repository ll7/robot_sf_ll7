"""Implementation-independent bicycle diagnostic motion metrics."""

import math


def displacement_stuck_steps(trace, reset_pose):
    """Count full overlapping 2s windows with <.05m net motion and a requested command.

    At least one raw (pre-projection) linear or yaw component must exceed 1e-6
    during the window. Count its ending step; windows overlap at the .1s stride.
    Neither creep nor the commanded/achieved yaw law appears in the motion test.
    """
    count = 0
    for end in range(19, len(trace)):
        start = end - 19
        initial = trace[start - 1]["pose"][:2] if start else reset_pose[0]
        moved = math.dist(initial, trace[end]["pose"][:2])
        issued = any(max(map(abs, t["cmd"])) > 1e-6 for t in trace[start : end + 1])
        count += moved < 0.05 and issued
    return count
