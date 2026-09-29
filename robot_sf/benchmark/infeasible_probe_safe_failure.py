"""Safe-failure metric for infeasible-by-design probe scenarios (issue #9974).

The 2.0 m doorway probe is proven infeasible for the 2.0 m robot, so the desired
outcome of every episode is a timeout with no robot-attributable contact. This
module computes the declared ``safe_failure_metric`` from episode rows:

    numerator:   timeout_without_robot_attributable_contact
    denominator: every probe episode row (the declaration says 420)
    unresolved:  any row whose class cannot be established fails admission

Robot-attributable contact reuses the existing attribution and adds none:

* ``outcome.collision_event`` (canonical flag, see ``termination_reason``);
* ``metrics.collisions`` above zero (the same collision metric the outcome
  contradiction check ties to that flag).

Obstacle and wall collisions count as contact, because ``collision_event``
includes ``is_obstacle_collision``. The ``robot_force_*`` values are
social-force repulsion, nonzero whenever a pedestrian is within a few metres,
so they are not contact. They are reported only as a descriptive side column
(``robot_force_descriptive``) and never classify a row.

A planner exception writes no episode row; the batch runner records it in the
``failures`` list of the run ``summary.json``. ``classify_probe_slots`` therefore
classifies an expected slot with no row but a matching failure record as
``error`` (crash). A slot with neither a row nor a failure record is
unresolved and fails admission. No row carries ``termination_reason == "error"``
in practice, so that value gets no crash branch here.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

PROBE_SCENARIO_IDS = frozenset({"francis2023_narrow_doorway"})
DECLARED_PROBE_EPISODES = 420
TIMEOUT_TERMINATIONS = frozenset({"max_steps", "truncated"})
ROBOT_FORCE_DESCRIPTIVE_KEYS = (
    "robot_force_impulse_total",
    "robot_force_peak",
    "robot_force_exposed_ped_count",
)

SAFE_FAILURE = "safe_failure"
CONTACT = "robot_attributable_contact"
CRASH = "error"
DEFECT_SUCCESS = "defect_success_on_infeasible_probe"
OTHER_TERMINATION = "other_termination"
UNRESOLVED = "unresolved"


def _number(value: Any) -> float | None:
    if type(value) is bool or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flag(outcome: Mapping[str, Any], key: str) -> bool | None:
    value = outcome.get(key)
    return value if type(value) is bool else None


def _contact(outcome: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool | None:
    """Return robot-attributable contact, or None when it cannot be established."""
    collision = _flag(outcome, "collision_event")
    if collision is None:
        return None
    collisions = _number(metrics.get("collisions", 0))
    if collisions is None:
        return None
    return collision or collisions > 0


def classify_probe_row(row: Mapping[str, Any]) -> str:
    """Classify one probe episode row.

    Returns:
        One of the module class names; ``unresolved`` when the row lacks the
        fields, or contradicts itself, so no class can be established.
    """
    outcome, metrics = row.get("outcome"), row.get("metrics")
    termination = row.get("termination_reason")
    integrity = row.get("integrity")
    if not isinstance(outcome, dict) or not isinstance(metrics, dict):
        return UNRESOLVED
    if not isinstance(termination, str):
        return UNRESOLVED
    if isinstance(integrity, dict) and integrity.get("contradictions"):
        return UNRESOLVED
    route_complete = _flag(outcome, "route_complete")
    timeout = _flag(outcome, "timeout_event")
    contact = _contact(outcome, metrics)
    if route_complete is None or timeout is None or contact is None:
        return UNRESOLVED
    if route_complete or termination == "success":
        return DEFECT_SUCCESS
    if contact or termination == "collision":
        return CONTACT
    if termination in TIMEOUT_TERMINATIONS:
        return SAFE_FAILURE if timeout else UNRESOLVED
    return OTHER_TERMINATION


def robot_force_descriptive(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Describe the robot-force values per key without classifying anything.

    Returns:
        Per key: rows carrying a finite value, rows with a value above zero, and the maximum.
    """
    stats = {
        key: {"rows_with_value": 0, "rows_above_zero": 0, "max": None}
        for key in ROBOT_FORCE_DESCRIPTIVE_KEYS
    }
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, entry in stats.items():
            value = _number(metrics.get(key))
            if value is None:
                continue
            entry["rows_with_value"] += 1
            entry["rows_above_zero"] += value > 0
            entry["max"] = value if entry["max"] is None else max(entry["max"], value)
    return stats


def _summarize(
    classes: Counter[str], expected_rows: int, force: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    total = sum(classes.values())
    if classes[UNRESOLVED] or total != expected_rows or total == 0:
        status = "fail_admission"
    elif classes[DEFECT_SUCCESS]:
        status = "defect"
    else:
        status = "computed"
    return {
        "schema_version": "infeasible-probe-safe-failure.v1",
        "numerator": "timeout_without_robot_attributable_contact",
        "denominator": total,
        "expected_rows": expected_rows,
        "safe_failure_episodes": classes[SAFE_FAILURE],
        "safe_failure_rate": classes[SAFE_FAILURE] / total if total else None,
        "class_counts": dict(sorted(classes.items())),
        "robot_force_descriptive": force,
        "status": status,
    }


def safe_failure_summary(
    rows: Iterable[Mapping[str, Any]], *, expected_rows: int = DECLARED_PROBE_EPISODES
) -> dict[str, Any]:
    """Compute the declared safe-failure rate over probe episode rows.

    Status is ``computed`` only when every row is resolved and the row count
    equals ``expected_rows``; ``defect`` when computed but a probe episode
    succeeded; otherwise ``fail_admission``.

    Returns:
        Counts per class, the rate (numerator over all rows) and the status.
    """
    rows = list(rows)
    return _summarize(
        Counter(classify_probe_row(row) for row in rows),
        expected_rows,
        robot_force_descriptive(rows),
    )


def classify_probe_slots(
    expected_slots: Iterable[tuple[Any, ...]],
    rows_by_slot: Mapping[tuple[Any, ...], Mapping[str, Any]],
    failure_slots: Iterable[tuple[Any, ...]] = (),
) -> dict[str, Any]:
    """Summarize the exact expected probe slots.

    A slot with a row is classified from the row. A slot without a row but with
    a matching batch failure record (a prefix of the slot, see below) is a
    crash. A slot with neither is unresolved. ``failure_slots`` holds the
    failure identity as a tuple that is a prefix of the slot tuple, for example
    ``(planner, kinematics, scenario_id, seed)``.

    Returns:
        The summary of ``safe_failure_summary`` over the expected slots.
    """
    expected = set(expected_slots)
    failed = {tuple(item) for item in failure_slots}
    classes: Counter[str] = Counter()
    for slot in expected:
        row = rows_by_slot.get(slot)
        if row is not None:
            classes[classify_probe_row(row)] += 1
        elif any(slot[: len(item)] == item for item in failed):
            classes[CRASH] += 1
        else:
            classes[UNRESOLVED] += 1
    force = robot_force_descriptive(rows_by_slot[slot] for slot in expected if slot in rows_by_slot)
    return _summarize(classes, len(expected), force)
