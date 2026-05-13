"""Demonstration export helpers for manual-control recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.manual_control.recording import load_manual_jsonl_records
from robot_sf.manual_control.rewind import compute_rewind_invalidated_record_indexes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.manual_control.recording import ManualControlRecord


@dataclass(frozen=True)
class DemonstrationSample:
    """Compact behavior-cloning sample extracted from a manual-control record."""

    session_id: str
    scenario_id: str
    seed: int
    attempt_id: int
    step_idx: int
    observation: dict[str, Any]
    action: tuple[float, ...]
    input_keys: tuple[str, ...]
    source_record_index: int
    rewind_segment_id: int = 0
    source_record_schema: str = "manual_control_v1"
    source_path: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible demonstration sample.

        Returns
        -------
        dict[str, Any]
            Serializable behavior-cloning sample.
        """
        return {
            "sample_schema": "manual_control_bc_v1",
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "attempt_id": self.attempt_id,
            "step_idx": self.step_idx,
            "observation": self.observation,
            "action": list(self.action),
            "input_keys": list(self.input_keys),
            "rewind_segment_id": self.rewind_segment_id,
            "source": {
                "record_schema": self.source_record_schema,
                "record_index": self.source_record_index,
                "path": self.source_path,
            },
        }


def export_demonstration_samples(
    records: Iterable[ManualControlRecord],
    *,
    source_path: str | Path | None = None,
) -> list[DemonstrationSample]:
    """Extract behavior-cloning samples from training-marked manual records.

    Returns
    -------
    list[DemonstrationSample]
        Extracted demonstration samples in record order.
    """
    samples: list[DemonstrationSample] = []
    normalized_source_path = str(source_path) if source_path is not None else None
    ordered_records = tuple(records)
    invalidated_record_indexes = compute_rewind_invalidated_record_indexes(ordered_records)
    rewind_segment_id = 0
    for record_index, record in enumerate(ordered_records):
        if record.rewind is not None:
            rewind_segment_id += 1
            continue
        if record_index in invalidated_record_indexes:
            continue
        if not record.training_sample:
            continue
        if record.observation is None or record.mapped_action is None:
            raise ValueError(
                "training_sample records must include both observation and mapped_action"
            )
        samples.append(
            DemonstrationSample(
                session_id=record.session.session_id,
                scenario_id=record.scenario_id,
                seed=record.seed,
                attempt_id=record.attempt_id,
                step_idx=record.step_idx,
                observation=record.observation,
                action=tuple(record.mapped_action),
                input_keys=tuple(record.input_keys),
                source_record_index=record_index,
                rewind_segment_id=rewind_segment_id,
                source_path=normalized_source_path,
            )
        )
    return samples


def export_demonstration_samples_from_jsonl(path: str | Path) -> list[DemonstrationSample]:
    """Load a manual-control JSONL stream and extract BC samples.

    Returns
    -------
    list[DemonstrationSample]
        Extracted demonstration samples from the JSONL recording.
    """
    return export_demonstration_samples(load_manual_jsonl_records(path), source_path=path)


def write_demonstration_samples_jsonl(
    samples: Iterable[DemonstrationSample],
    path: str | Path,
) -> Path:
    """Write compact behavior-cloning samples as JSONL.

    Returns
    -------
    Path
        Output JSONL path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json_dict(), sort_keys=True) + "\n")
    return output_path
