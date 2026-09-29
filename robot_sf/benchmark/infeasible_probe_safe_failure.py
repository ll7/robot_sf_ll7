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
  contradiction check ties to that flag);
* the 0.0.8 robot-force metrics (``robot_force_impulse_total``,
  ``robot_force_peak``, ``robot_force_exposed_ped_count``) above zero, when
  the row carries them.
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
ROBOT_FORCE_CONTACT_KEYS = (
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
    contact = collision or collisions > 0
    for key in ROBOT_FORCE_CONTACT_KEYS:
        if key not in metrics:
            continue
        value = _number(metrics[key])
        if value is None:
            return None
        contact = contact or value > 0
    return contact


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
    if termination == "error":
        return CRASH
    if contact or termination == "collision":
        return CONTACT
    if termination in TIMEOUT_TERMINATIONS:
        return SAFE_FAILURE if timeout else UNRESOLVED
    return OTHER_TERMINATION


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
    classes = Counter(classify_probe_row(row) for row in rows)
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
        "status": status,
    }
