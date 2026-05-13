"""Replay organization helpers for manual-control recordings."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.manual_control.recording import _json_compatible
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


@dataclass(frozen=True)
class ManualReplayEvent:
    """Serializable replay event derived from one manual-control record."""

    event: str
    scenario_id: str
    seed: int
    attempt_id: int
    step_idx: int
    session_id: str
    input_keys: tuple[str, ...]
    mapped_action: tuple[float, ...] | None
    metrics: dict[str, object]
    training_sample: bool

    @classmethod
    def from_record(cls, record: ManualControlRecord) -> ManualReplayEvent:
        """Build a replay event from one manual-control record.

        Returns
        -------
        ManualReplayEvent
            Serializable replay event preserving the original record semantics.
        """
        return cls(
            event=record.event,
            scenario_id=record.scenario_id,
            seed=record.seed,
            attempt_id=record.attempt_id,
            step_idx=record.step_idx,
            session_id=record.session.session_id,
            input_keys=tuple(record.input_keys),
            mapped_action=record.mapped_action,
            metrics=dict(record.metrics),
            training_sample=record.training_sample,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible replay event.

        Returns
        -------
        dict[str, object]
            Serializable replay event.
        """
        return _json_compatible(
            {
                "event": self.event,
                "scenario_id": self.scenario_id,
                "seed": self.seed,
                "attempt_id": self.attempt_id,
                "step_idx": self.step_idx,
                "session_id": self.session_id,
                "input_keys": list(self.input_keys),
                "mapped_action": (
                    list(self.mapped_action) if self.mapped_action is not None else None
                ),
                "metrics": self.metrics,
                "training_sample": self.training_sample,
            }
        )


def group_records_by_attempt(records: Iterable[ManualControlRecord]) -> list[ManualAttemptReplay]:
    """Group manual-control records by scenario, seed, and attempt id.

    Returns
    -------
    list[ManualAttemptReplay]
        Attempt replays sorted by scenario, seed, and attempt id. Records within each
        attempt preserve source-stream order.
    """
    grouped: dict[tuple[str, int, int], list[tuple[int, ManualControlRecord]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        grouped[(record.scenario_id, record.seed, record.attempt_id)].append((record_index, record))

    replays: list[ManualAttemptReplay] = []
    for (scenario_id, seed, attempt_id), indexed_records in sorted(grouped.items()):
        ordered = tuple(record for _, record in indexed_records)
        replays.append(
            ManualAttemptReplay(
                key=AttemptKey(scenario_id=scenario_id, seed=seed),
                attempt_id=attempt_id,
                records=ordered,
            )
        )
    return replays


def iter_replay_events(replay: ManualAttemptReplay) -> tuple[ManualReplayEvent, ...]:
    """Return serializable events for one completed-attempt replay.

    Returns
    -------
    tuple[ManualReplayEvent, ...]
        Replay events in deterministic record order.
    """
    return tuple(ManualReplayEvent.from_record(record) for record in replay.records)


def write_attempt_replay_json(
    replays: Iterable[ManualAttemptReplay],
    path: str | Path,
) -> Path:
    """Write grouped completed-attempt replay events as JSON.

    Returns
    -------
    Path
        Output JSON path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "replay_schema": "manual_control_attempt_replay_v1",
        "attempts": [
            {
                "scenario_id": replay.key.scenario_id,
                "seed": replay.key.seed,
                "attempt_id": replay.attempt_id,
                "events": [event.to_json_dict() for event in iter_replay_events(replay)],
            }
            for replay in replays
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
