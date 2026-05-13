"""Bounded replay-to-step rewind helpers for manual-control sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robot_sf.manual_control.recording import ManualControlRecord, ManualRewindMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.manual_control.replay import ManualAttemptReplay


REPLAY_TO_STEP_REWIND_STRATEGY = "replay_to_step_v1"
"""Manual-control rewind strategy implemented by the first bounded slice."""


@dataclass(frozen=True)
class ManualRewindPlan:
    """Pure rewind plan for one manual-control attempt."""

    replay: ManualAttemptReplay
    from_step_idx: int
    to_step_idx: int
    restored_records: tuple[ManualControlRecord, ...]
    invalidated_record_indexes: tuple[int, ...]
    reason: str | None = None
    strategy: str = REPLAY_TO_STEP_REWIND_STRATEGY

    def to_rewind_record(self) -> ManualControlRecord:
        """Build the append-only rewind event record for this plan.

        Returns
        -------
        ManualControlRecord
            Explicit rewind boundary record for the original session stream.
        """
        if not self.replay.records:
            raise ValueError("cannot build a rewind record without source records")
        source = self.replay.records[-1]
        return ManualControlRecord.for_attempt(
            event="rewind",
            attempt_key=self.replay.key,
            attempt_id=self.replay.attempt_id,
            step_idx=self.to_step_idx,
            session=source.session,
            metrics={
                "rewind_strategy": self.strategy,
                "from_step_idx": self.from_step_idx,
                "to_step_idx": self.to_step_idx,
            },
            rewind=ManualRewindMetadata(
                strategy=self.strategy,
                from_step_idx=self.from_step_idx,
                to_step_idx=self.to_step_idx,
                invalidates_samples_after_step=self.to_step_idx,
                reason=self.reason,
            ),
            training_sample=False,
        )


def plan_replay_to_step_rewind(
    replay: ManualAttemptReplay,
    *,
    target_step_idx: int,
    reason: str | None = None,
) -> ManualRewindPlan:
    """Plan a bounded rewind by replaying the attempt prefix through a target step.

    Returns
    -------
    ManualRewindPlan
        Deterministic rewind plan containing the restored prefix and invalidated samples.
    """
    if target_step_idx < 0:
        raise ValueError("target_step_idx must be non-negative")
    if not replay.records:
        raise ValueError("cannot rewind an empty manual-control attempt")

    from_step_idx = max(record.step_idx for record in replay.records)
    if target_step_idx > from_step_idx:
        raise ValueError(
            f"cannot rewind from step {from_step_idx} to future step {target_step_idx}"
        )

    restored_records = tuple(
        record for record in replay.records if record.step_idx <= target_step_idx
    )
    invalidated_record_indexes = tuple(
        index
        for index, record in enumerate(replay.records)
        if record.training_sample and target_step_idx < record.step_idx <= from_step_idx
    )
    return ManualRewindPlan(
        replay=replay,
        from_step_idx=from_step_idx,
        to_step_idx=target_step_idx,
        restored_records=restored_records,
        invalidated_record_indexes=invalidated_record_indexes,
        reason=reason,
    )


def compute_rewind_invalidated_record_indexes(
    records: Iterable[ManualControlRecord],
) -> frozenset[int]:
    """Return record indexes invalidated by later append-only rewind events.

    Returns
    -------
    frozenset[int]
        Indexes of training-sample records that should be excluded from BC export.
    """
    ordered_records = tuple(records)
    invalidated: set[int] = set()
    for rewind_index, rewind_record in enumerate(ordered_records):
        if rewind_record.rewind is None:
            continue
        rewind = rewind_record.rewind
        for record_index, record in enumerate(ordered_records[:rewind_index]):
            if not record.training_sample:
                continue
            if not _same_attempt(record, rewind_record):
                continue
            if rewind.to_step_idx < record.step_idx <= rewind.from_step_idx:
                invalidated.add(record_index)
    return frozenset(invalidated)


def _same_attempt(left: ManualControlRecord, right: ManualControlRecord) -> bool:
    """Return whether two records belong to the same manual-control attempt."""
    return (
        left.scenario_id == right.scenario_id
        and left.seed == right.seed
        and left.attempt_id == right.attempt_id
    )
