"""Step-trace invariant checks for benchmark episode rows (#9979).

The checker reads episode rows that carry ``algorithm_metadata.simulation_step_trace``
(schema ``simulation-step-trace.v1``) and tests five invariant families:

* ``a_goal_heading``: the robot moves towards ``goal.current`` and not, in a
  sustained way, towards ``goal.next`` or towards the world origin (the #9883 class).
* ``b_drive_limits``: speed, acceleration and yaw rate stay inside the drive limits
  read from the environment config of the row (never hard-coded here).
* ``c_clearance_contact``: recorded surface clearance matches a recomputation with the
  real radii, and contact is consistent with the collision flags.
* ``d_position_jump``: no position jump above ``v_max * dt`` plus tolerance.
* ``e_termination``: the termination reason agrees with the final state.

Rows without a step trace still get the row-level subset of (b) and (e); they are
counted as ``no_trace`` so that missing coverage is explicit.

Minimal trace fields the runner must emit for the full check: per step ``robot.position``,
``robot.heading``, ``goal.current``, ``goal.next`` (pre-step values), ``collision``
(``pedestrian``/``obstacle``/``robot``), ``pedestrians[].position`` and
``pedestrians[].surface_clearance_m``; per trace ``dt`` and ``reset.robot``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

INVARIANTS = (
    "a_goal_heading",
    "b_drive_limits",
    "c_clearance_contact",
    "d_position_jump",
    "e_termination",
)

_TIMEOUT_REASONS = {"terminated", "truncated", "max_steps"}


@dataclass(frozen=True)
class DriveLimits:
    """Drive limits and radii resolved from the environment config of an episode."""

    max_linear_speed: float | None
    max_angular_speed: float | None
    max_linear_accel: float | None
    max_linear_decel: float | None
    max_angular_accel: float | None
    robot_radius: float
    ped_radius: float
    goal_radius: float
    source: str


@dataclass(frozen=True)
class Tolerances:
    """Numeric tolerances and detection thresholds; all overridable from the CLI."""

    speed_rel: float = 0.01
    speed_abs: float = 0.01
    accel_rel: float = 0.02
    accel_abs: float = 0.05
    yaw_rel: float = 0.01
    yaw_abs: float = 0.01
    yaw_accel_rel: float = 0.02
    yaw_accel_abs: float = 0.10
    jump_abs: float = 0.05
    clearance_abs: float = 1e-4
    flagless_contact_clearance: float = 0.0
    phantom_contact_clearance: float = 0.5
    goal_slack: float = 1.0
    # invariant (a)
    window_steps: int = 10
    min_window_displacement: float = 0.5
    max_cos_to_current: float = -0.2
    min_cos_to_alt: float = 0.85
    min_flagged_steps: int = 10
    # steps immediately after a waypoint switch are skipped (heading still swinging)
    leg_settle_steps: int = 5


@dataclass
class Violation:
    """One invariant violation with its location."""

    invariant: str
    kind: str
    episode_id: str
    step_start: int | None = None
    step_end: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def resolve_limits(row: dict[str, Any], overrides: dict[str, float] | None = None) -> DriveLimits:
    """Resolve drive limits and radii from the environment config of a row.

    The row's ``scenario_params.robot_config`` / ``simulation_config`` are applied on top of
    the dataclass defaults of the named robot type, so a config change is picked up here.

    Returns:
        DriveLimits for the episode.
    """
    from robot_sf.robot.differential_drive import DifferentialDriveSettings  # noqa: PLC0415
    from robot_sf.sim.sim_config import SimulationSettings  # noqa: PLC0415

    params = row.get("scenario_params") or {}
    robot_cfg = dict(params.get("robot_config") or {})
    sim_cfg = dict(params.get("simulation_config") or {})
    robot_type = str(robot_cfg.pop("type", "differential_drive"))
    fields = DifferentialDriveSettings.__dataclass_fields__
    ddrive = DifferentialDriveSettings(**{k: v for k, v in robot_cfg.items() if k in fields})
    sim_fields = SimulationSettings.__dataclass_fields__
    ped_radius = float(sim_cfg.get("ped_radius", sim_fields["ped_radius"].default))
    goal_radius = float(sim_cfg.get("goal_radius", sim_fields["goal_radius"].default))
    limits = {
        "max_linear_speed": ddrive.max_linear_speed,
        "max_angular_speed": ddrive.max_angular_speed,
        "max_linear_accel": ddrive.max_linear_accel,
        "max_linear_decel": ddrive.max_linear_decel,
        "max_angular_accel": ddrive.max_angular_accel,
        "robot_radius": ddrive.radius,
        "ped_radius": ped_radius,
        "goal_radius": goal_radius,
    }
    if robot_type != "differential_drive":
        # Bicycle and other types: only the radius is comparable; limits stay unchecked.
        for key in ("max_angular_speed", "max_linear_accel", "max_linear_decel"):
            limits[key] = None
        limits["max_angular_accel"] = None
        limits["max_linear_speed"] = None
    limits.update(overrides or {})
    return DriveLimits(source=f"row.scenario_params:{robot_type}", **limits)  # type: ignore[arg-type]


def _trace_of(row: dict[str, Any]) -> dict[str, Any] | None:
    meta = row.get("algorithm_metadata")
    trace = meta.get("simulation_step_trace") if isinstance(meta, dict) else None
    return trace if isinstance(trace, dict) and trace.get("steps") else None


def _steps_xy(trace: dict[str, Any]) -> list[tuple[float, float] | None]:
    out: list[tuple[float, float] | None] = []
    for step in trace["steps"]:
        pos = (step.get("robot") or {}).get("position")
        x, y = (
            (_num(pos[0]), _num(pos[1]))
            if isinstance(pos, list) and len(pos) == 2
            else (None, None)
        )
        out.append((x, y) if x is not None and y is not None else None)
    return out


def _goal_xy(step: dict[str, Any], key: str) -> tuple[float, float] | None:
    goal = step.get("goal")
    pos = goal.get(key) if isinstance(goal, dict) else None
    if isinstance(pos, list) and len(pos) == 2:
        x, y = _num(pos[0]), _num(pos[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def check_goal_heading(  # noqa: C901
    episode_id: str, trace: dict[str, Any], tol: Tolerances
) -> list[Violation]:
    """Flag sustained motion towards ``goal.next`` or the origin instead of ``goal.current``.

    Returns:
        Violations, at most one per leg.
    """
    steps = trace["steps"]
    pos = _steps_xy(trace)
    # legs: maximal runs of an identical pre-step goal.current
    legs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(steps) + 1):
        cur = _goal_xy(steps[i - 1], "current")
        nxt = _goal_xy(steps[i], "current") if i < len(steps) else None
        if i == len(steps) or nxt != cur:
            legs.append((start, i))
            start = i
    out: list[Violation] = []
    w = tol.window_steps
    for leg_idx, (a, b) in enumerate(legs):
        cur = _goal_xy(steps[a], "current")
        if cur is None:
            continue
        nxt = _goal_xy(steps[a], "next")
        alt_name, alt = (
            ("next", nxt) if nxt is not None and nxt != (0.0, 0.0) else ("origin", (0.0, 0.0))
        )
        if math.dist(cur, alt) < 1e-6:
            continue
        lo = a + tol.leg_settle_steps
        flagged = 0
        first_flag = None
        for s in range(lo + w, b):
            p0, p1 = pos[s - w], pos[s]
            if p0 is None or p1 is None:
                continue
            d = (p1[0] - p0[0], p1[1] - p0[1])
            dn = math.hypot(*d)
            if dn < tol.min_window_displacement:
                continue
            tc = (cur[0] - p0[0], cur[1] - p0[1])
            ta = (alt[0] - p0[0], alt[1] - p0[1])
            nc, na = math.hypot(*tc), math.hypot(*ta)
            if nc < 1e-6 or na < 1e-6:
                continue
            cos_c = (d[0] * tc[0] + d[1] * tc[1]) / (dn * nc)
            cos_a = (d[0] * ta[0] + d[1] * ta[1]) / (dn * na)
            # distance to current must actually grow, and distance to the alternative shrink
            grows = math.dist(p1, cur) > math.dist(p0, cur) + 0.25 * dn
            shrinks = math.dist(p1, alt) < math.dist(p0, alt) - 0.25 * dn
            if (
                cos_c <= tol.max_cos_to_current
                and cos_a >= tol.min_cos_to_alt
                and grows
                and shrinks
            ):
                flagged += 1
                if first_flag is None:
                    first_flag = s
        if flagged >= tol.min_flagged_steps:
            out.append(
                Violation(
                    "a_goal_heading",
                    f"towards_{alt_name}_not_current",
                    episode_id,
                    first_flag,
                    b - 1,
                    {
                        "leg": leg_idx,
                        "flagged_window_ends": flagged,
                        "current": list(cur),
                        "alt": list(alt),
                    },
                )
            )
    return out


def check_drive_limits(  # noqa: C901
    episode_id: str, trace: dict[str, Any], limits: DriveLimits, tol: Tolerances
) -> list[Violation]:
    """Check speed, linear acceleration, yaw rate and yaw acceleration against the limits.

    Returns:
        One violation per kind with the worst step as the example.
    """
    dt = _num(trace.get("dt"))
    if not dt or dt <= 0:
        return []
    steps = trace["steps"]
    pos = _steps_xy(trace)
    reset_robot = (trace.get("reset") or {}).get("robot") or {}
    rp = reset_robot.get("position")
    prev_p = (float(rp[0]), float(rp[1])) if isinstance(rp, list) and len(rp) == 2 else None
    prev_h = _num(reset_robot.get("heading"))
    prev_v: float | None = None
    prev_w: float | None = None
    worst: dict[str, tuple[float, int, float]] = {}

    def note(kind: str, excess: float, step: int, limit: float) -> None:
        if kind not in worst or excess > worst[kind][0]:
            worst[kind] = (excess, step, limit)

    for k, step in enumerate(steps):
        p = pos[k]
        h = _num((step.get("robot") or {}).get("heading"))
        v = w = None
        if p is not None and prev_p is not None:
            v = math.dist(p, prev_p) / dt
        if h is not None and prev_h is not None:
            w = _wrap(h - prev_h) / dt
        if v is not None and limits.max_linear_speed is not None:
            lim = limits.max_linear_speed * (1 + tol.speed_rel) + tol.speed_abs
            if v > lim:
                note("speed", v - lim, k, limits.max_linear_speed)
        if w is not None and limits.max_angular_speed is not None:
            lim = limits.max_angular_speed * (1 + tol.yaw_rel) + tol.yaw_abs
            if abs(w) > lim:
                note("yaw_rate", abs(w) - lim, k, limits.max_angular_speed)
        if v is not None and prev_v is not None:
            acc = (v - prev_v) / dt
            up = limits.max_linear_accel
            dn = limits.max_linear_decel if limits.max_linear_decel is not None else up
            if up is not None and acc > up * (1 + tol.accel_rel) + tol.accel_abs:
                note("accel", acc - up, k, up)
            if dn is not None and -acc > dn * (1 + tol.accel_rel) + tol.accel_abs:
                note("decel", -acc - dn, k, dn)
        if w is not None and prev_w is not None and limits.max_angular_accel is not None:
            wacc = abs(w - prev_w) / dt
            lim = limits.max_angular_accel * (1 + tol.yaw_accel_rel) + tol.yaw_accel_abs
            if wacc > lim:
                note("yaw_accel", wacc - lim, k, limits.max_angular_accel)
        if p is not None:
            prev_p = p
        if h is not None:
            prev_h = h
        prev_v, prev_w = v, w
    return [
        Violation("b_drive_limits", kind, episode_id, s, s, {"excess": round(x, 4), "limit": lim})
        for kind, (x, s, lim) in worst.items()
    ]


def check_position_jumps(
    episode_id: str, trace: dict[str, Any], limits: DriveLimits, tol: Tolerances
) -> list[Violation]:
    """Flag step-to-step robot displacement above ``v_max * dt`` plus tolerance.

    The first step is compared with the reset position, so a respawn after reset shows up.

    Returns:
        Violations for the worst jump.
    """
    dt = _num(trace.get("dt"))
    if not dt or dt <= 0 or limits.max_linear_speed is None:
        return []
    pos = _steps_xy(trace)
    rp = ((trace.get("reset") or {}).get("robot") or {}).get("position")
    prev = (float(rp[0]), float(rp[1])) if isinstance(rp, list) and len(rp) == 2 else None
    lim = limits.max_linear_speed * dt + tol.jump_abs
    worst: tuple[float, int] | None = None
    count = 0
    for k, p in enumerate(pos):
        if p is None:
            continue
        if prev is not None:
            jump = math.dist(p, prev)
            if jump > lim:
                count += 1
                if worst is None or jump > worst[0]:
                    worst = (jump, k)
        prev = p
    if worst is None:
        return []
    return [
        Violation(
            "d_position_jump",
            "jump",
            episode_id,
            worst[1],
            worst[1],
            {"jump_m": round(worst[0], 4), "limit_m": round(lim, 4), "count": count},
        )
    ]


def check_clearance_contact(  # noqa: C901
    episode_id: str, trace: dict[str, Any], limits: DriveLimits, tol: Tolerances
) -> list[Violation]:
    """Recompute surface clearance with the real radii and compare contact with flags.

    Returns:
        Violations of kind ``clearance_mismatch``, ``contact_without_flag``, ``flag_without_contact``.
    """
    steps = trace["steps"]
    pos = _steps_xy(trace)
    r_sum = limits.robot_radius + limits.ped_radius
    mismatch: tuple[float, int] | None = None
    flagless: tuple[float, int] | None = None
    phantom: tuple[float, int] | None = None
    for k, step in enumerate(steps):
        p = pos[k]
        if p is None:
            continue
        min_clear = None
        for ped in step.get("pedestrians") or []:
            pp = ped.get("position")
            if not (isinstance(pp, list) and len(pp) == 2):
                continue
            x, y = _num(pp[0]), _num(pp[1])
            if x is None or y is None:
                continue
            clear = math.dist(p, (x, y)) - r_sum
            min_clear = clear if min_clear is None else min(min_clear, clear)
            rec = _num(ped.get("surface_clearance_m"))
            if rec is not None and abs(rec - clear) > tol.clearance_abs:
                gap = abs(rec - clear)
                if mismatch is None or gap > mismatch[0]:
                    mismatch = (gap, k)
        coll = step.get("collision")
        if not isinstance(coll, dict) or min_clear is None:
            continue
        ped_flag = bool(coll.get("pedestrian"))
        if min_clear <= tol.flagless_contact_clearance and not ped_flag:
            if flagless is None or -min_clear > flagless[0]:
                flagless = (-min_clear, k)
        if ped_flag and min_clear > tol.phantom_contact_clearance:
            if phantom is None or min_clear > phantom[0]:
                phantom = (min_clear, k)
    out = []
    if mismatch:
        out.append(
            Violation(
                "c_clearance_contact",
                "clearance_mismatch",
                episode_id,
                mismatch[1],
                mismatch[1],
                {"abs_gap_m": round(mismatch[0], 5), "radii_sum_m": r_sum},
            )
        )
    if flagless:
        out.append(
            Violation(
                "c_clearance_contact",
                "contact_without_flag",
                episode_id,
                flagless[1],
                flagless[1],
                {"penetration_m": round(flagless[0], 4)},
            )
        )
    if phantom:
        out.append(
            Violation(
                "c_clearance_contact",
                "flag_without_contact",
                episode_id,
                phantom[1],
                phantom[1],
                {"clearance_m": round(phantom[0], 4)},
            )
        )
    return out


def _expected_timeout_steps(row: dict[str, Any]) -> int | None:
    params = row.get("scenario_params") or {}
    caps = []
    horizon = _num(row.get("horizon"))
    if horizon:
        caps.append(int(horizon))
    max_ep = _num((params.get("simulation_config") or {}).get("max_episode_steps"))
    if max_ep:
        caps.append(int(max_ep))
    return min(caps) if caps else None


def check_termination_row(episode_id: str, row: dict[str, Any]) -> list[Violation]:
    """Row-level termination consistency (needs no step trace).

    Returns:
        Violations of the reason versus outcome flags and step count.
    """
    reason = row.get("termination_reason")
    ledger = (row.get("event_ledger") or {}).get("exact_events") or {}
    outcome = row.get("outcome") or {}
    steps = _num(row.get("steps"))
    out: list[Violation] = []

    def bad(kind: str, **detail: Any) -> None:
        out.append(
            Violation("e_termination", kind, episode_id, None, None, {"reason": reason, **detail})
        )

    collided = bool(ledger.get("collision", outcome.get("collision_event", False)))
    reached = bool(ledger.get("goal_reached", outcome.get("route_complete", False)))
    if reason == "success" and (not reached or collided):
        bad("success_without_goal_or_with_collision", reached=reached, collided=collided)
    elif reason == "collision" and not collided:
        bad("collision_without_collision_event")
    elif reason in _TIMEOUT_REASONS:
        if collided or reached:
            bad("timeout_reason_with_collision_or_goal", collided=collided, reached=reached)
        expected = _expected_timeout_steps(row)
        if steps is not None and expected is not None and steps < expected:
            bad("timeout_before_step_cap", steps=int(steps), cap=expected)
    elif reason not in {"success", "collision", "error"}:
        bad("unknown_reason")
    if collided and reached:
        bad("goal_and_collision_both_set")
    if steps is not None and expected_cap_exceeded(row, steps):
        bad("steps_above_cap", steps=int(steps))
    return out


def expected_cap_exceeded(row: dict[str, Any], steps: float) -> bool:
    """Whether the episode ran more steps than the horizon allows.

    Returns:
        True when ``steps`` exceeds the row horizon.
    """
    horizon = _num(row.get("horizon"))
    return bool(horizon and steps > horizon)


def check_termination_trace(
    episode_id: str,
    row: dict[str, Any],
    trace: dict[str, Any],
    limits: DriveLimits,
    tol: Tolerances,
) -> list[Violation]:
    """Compare the termination reason with the final traced state.

    Returns:
        Violations of the reason versus final position and collision flags.
    """
    steps = trace["steps"]
    reason = row.get("termination_reason")
    out: list[Violation] = []
    last = len(steps) - 1
    if last < 0:
        return out
    row_steps = _num(row.get("steps"))
    if row_steps is not None and int(row_steps) != len(steps):
        out.append(
            Violation(
                "e_termination",
                "trace_length_differs_from_steps",
                episode_id,
                last,
                last,
                {"trace_steps": len(steps), "row_steps": int(row_steps)},
            )
        )
    final = steps[last]
    coll = final.get("collision") if isinstance(final.get("collision"), dict) else None
    final_flag = bool(coll and any(coll.values()))
    any_flag_tail = bool(
        any(
            isinstance(s.get("collision"), dict) and any(s["collision"].values())
            for s in steps[-3:]
        )
    )
    pos = _steps_xy(trace)[last]
    goal = _goal_xy(final, "current")
    reach_r = limits.goal_radius + limits.robot_radius + tol.goal_slack
    if reason == "success":
        if final_flag:
            out.append(
                Violation(
                    "e_termination", "success_with_final_collision_flag", episode_id, last, last, {}
                )
            )
        if pos is not None and goal is not None and math.dist(pos, goal) > reach_r:
            out.append(
                Violation(
                    "e_termination",
                    "success_far_from_goal",
                    episode_id,
                    last,
                    last,
                    {"dist_m": round(math.dist(pos, goal), 3), "reach_m": round(reach_r, 3)},
                )
            )
    elif reason == "collision":
        if coll is not None and not any_flag_tail:
            out.append(
                Violation(
                    "e_termination", "collision_without_flag_in_tail", episode_id, last, last, {}
                )
            )
    elif reason in _TIMEOUT_REASONS:
        if final_flag:
            out.append(
                Violation(
                    "e_termination", "timeout_with_final_collision_flag", episode_id, last, last, {}
                )
            )
    return out


def check_drive_limits_row(
    episode_id: str, row: dict[str, Any], limits: DriveLimits
) -> list[Violation]:
    """Row-level speed check from aggregate metrics (used when no step trace exists).

    Returns:
        A violation when the mean speed exceeds the linear speed limit.
    """
    avg = _num((row.get("metrics") or {}).get("avg_speed"))
    lim = limits.max_linear_speed
    if avg is not None and lim is not None and avg > lim * 1.01 + 0.01:
        return [
            Violation(
                "b_drive_limits",
                "avg_speed_above_limit_row",
                episode_id,
                None,
                None,
                {"avg_speed": round(avg, 4), "limit": lim},
            )
        ]
    return []


def check_episode(
    row: dict[str, Any],
    *,
    tol: Tolerances | None = None,
    overrides: dict[str, float] | None = None,
    enabled: Iterable[str] = INVARIANTS,
) -> tuple[list[Violation], bool]:
    """Run all enabled checks on one episode row.

    Returns:
        (violations, has_trace).
    """
    tol = tol or Tolerances()
    enabled = set(enabled)
    episode_id = str(row.get("episode_id", "?"))
    limits = resolve_limits(row, overrides)
    trace = _trace_of(row)
    out: list[Violation] = []
    if trace is not None:
        if "a_goal_heading" in enabled:
            out += check_goal_heading(episode_id, trace, tol)
        if "b_drive_limits" in enabled:
            out += check_drive_limits(episode_id, trace, limits, tol)
        if "c_clearance_contact" in enabled:
            out += check_clearance_contact(episode_id, trace, limits, tol)
        if "d_position_jump" in enabled:
            out += check_position_jumps(episode_id, trace, limits, tol)
        if "e_termination" in enabled:
            out += check_termination_trace(episode_id, row, trace, limits, tol)
    elif "b_drive_limits" in enabled:
        out += check_drive_limits_row(episode_id, row, limits)
    if "e_termination" in enabled:
        out += check_termination_row(episode_id, row)
    return out, trace is not None


def summarize(
    results: Iterable[tuple[str, str, str, list[Violation], bool]], max_examples: int = 5
) -> dict[str, Any]:
    """Aggregate per-episode results into an arm x scenario x invariant table.

    Args:
        results: ``(arm, scenario_id, episode_id, violations, has_trace)`` tuples.
        max_examples: Example episode ids kept per cell and invariant.

    Returns:
        JSON-serialisable summary.
    """
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    arm_tot: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "with_trace": 0, "flagged": dict.fromkeys(INVARIANTS, 0)}
    )
    for arm, scen, eid, viols, has_trace in results:
        cell = cells.setdefault(
            (arm, scen),
            {
                "episodes": 0,
                "with_trace": 0,
                "invariants": {i: {"episodes": 0, "examples": []} for i in INVARIANTS},
            },
        )
        cell["episodes"] += 1
        cell["with_trace"] += int(has_trace)
        arm_tot[arm]["episodes"] += 1
        arm_tot[arm]["with_trace"] += int(has_trace)
        for inv in {v.invariant for v in viols}:
            inv_cell = cell["invariants"][inv]
            inv_cell["episodes"] += 1
            arm_tot[arm]["flagged"][inv] += 1
            if len(inv_cell["examples"]) < max_examples:
                first = next(v for v in viols if v.invariant == inv)
                inv_cell["examples"].append(
                    {
                        "episode_id": eid,
                        "kind": first.kind,
                        "step": first.step_start,
                        "detail": first.detail,
                    }
                )
    table = defaultdict(dict)
    for (arm, scen), cell in sorted(cells.items()):
        table[arm][scen] = cell
    return {"arms": dict(arm_tot), "table": dict(table)}


def to_markdown(summary: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    """Render the summary as Markdown: an arm totals table, then per-cell hits.

    Returns:
        Markdown text.
    """
    lines = ["# Step-trace invariant report", ""]
    if meta:
        for k, v in meta.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines += ["## Flagged episodes per arm", ""]
    head = "| arm | episodes | with trace | " + " | ".join(INVARIANTS) + " |"
    lines += [head, "|" + "---|" * (3 + len(INVARIANTS))]
    for arm, tot in sorted(summary["arms"].items()):
        lines.append(
            f"| {arm} | {tot['episodes']} | {tot['with_trace']} | "
            + " | ".join(str(tot["flagged"][i]) for i in INVARIANTS)
            + " |"
        )
    lines += ["", "## Violations per arm x scenario (examples: episode ids)", ""]
    lines += [
        "| arm | scenario | invariant | episodes flagged | examples |",
        "|---|---|---|---|---|",
    ]
    for arm, scens in sorted(summary["table"].items()):
        for scen, cell in sorted(scens.items()):
            for inv in INVARIANTS:
                ic = cell["invariants"][inv]
                if ic["episodes"]:
                    ex = ", ".join(f"{e['episode_id']} ({e['kind']})" for e in ic["examples"])
                    lines.append(
                        f"| {arm} | {scen} | {inv} | {ic['episodes']}/{cell['episodes']} | {ex} |"
                    )
    return "\n".join(lines) + "\n"
