"""Replay organization helpers for manual-control recordings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot_sf.manual_control.session import AttemptKey

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.manual_control.recording import ManualControlRecord


@dataclass(frozen=True)
class ManualAttemptReplay:
    """Ordered records for one manual-control scenario/seed attempt."""

    key: AttemptKey
    attempt_id: int
    records: tuple[ManualControlRecord, ...]


def group_records_by_attempt(records: Iterable[ManualControlRecord]) -> list[ManualAttemptReplay]:
    """Group manual-control records by scenario, seed, and attempt id.

    Returns
    -------
    list[ManualAttemptReplay]
        Attempt replays sorted by scenario, seed, and attempt id.
    """
    grouped: dict[tuple[str, int, int], list[ManualControlRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.scenario_id, record.seed, record.attempt_id)].append(record)

    replays: list[ManualAttemptReplay] = []
    for (scenario_id, seed, attempt_id), attempt_records in sorted(grouped.items()):
        ordered = tuple(sorted(attempt_records, key=lambda record: record.step_idx))
        replays.append(
            ManualAttemptReplay(
                key=AttemptKey(scenario_id=scenario_id, seed=seed),
                attempt_id=attempt_id,
                records=ordered,
            )
        )
    return replays
