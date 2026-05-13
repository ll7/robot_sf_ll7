"""Append-only recording primitives for manual-control sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.manual_control.modes import ManualViewMode

if TYPE_CHECKING:
    from robot_sf.manual_control.session import AttemptKey


@dataclass(frozen=True)
class ManualSessionMetadata:
    """Static metadata written into every manual-control session record."""

    session_id: str
    input_mapping_version: str
    view_mode: str = ManualViewMode.FIXED_MAP.value
    policy_to_beat: str | None = None
    policy_to_beat_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualRewindMetadata:
    """Explicit rewind boundary metadata embedded in a manual-control record."""

    strategy: str
    from_step_idx: int
    to_step_idx: int
    invalidates_samples_after_step: int
    reason: str | None = None


@dataclass(frozen=True)
class ManualControlRecord:
    """One append-only manual-control event or training sample."""

    event: str
    scenario_id: str
    seed: int
    attempt_id: int
    step_idx: int
    session: ManualSessionMetadata
    input_keys: list[str] = field(default_factory=list)
    mapped_action: tuple[float, ...] | None = None
    observation: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    rewind: ManualRewindMetadata | None = None
    training_sample: bool = False

    @classmethod
    def for_attempt(  # noqa: PLR0913
        cls,
        *,
        event: str,
        attempt_key: AttemptKey,
        attempt_id: int,
        step_idx: int,
        session: ManualSessionMetadata,
        input_keys: list[str] | None = None,
        mapped_action: tuple[float, ...] | None = None,
        observation: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        rewind: ManualRewindMetadata | None = None,
        training_sample: bool = False,
    ) -> ManualControlRecord:
        """Build a record tied to one scenario/seed attempt.

        Returns
        -------
        ManualControlRecord
            Record populated from the attempt key and supplied event data.
        """
        return cls(
            event=event,
            scenario_id=attempt_key.scenario_id,
            seed=attempt_key.seed,
            attempt_id=attempt_id,
            step_idx=step_idx,
            session=session,
            input_keys=input_keys or [],
            mapped_action=mapped_action,
            observation=observation,
            metrics=metrics or {},
            rewind=rewind,
            training_sample=training_sample,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Serializable manual-control record.
        """
        payload = asdict(self)
        payload["record_schema"] = "manual_control_v1"
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ManualControlRecord:
        """Reconstruct a record from a JSONL payload.

        Returns
        -------
        ManualControlRecord
            Parsed manual-control record.
        """
        schema = payload.get("record_schema")
        if schema != "manual_control_v1":
            raise ValueError(f"unsupported manual-control record schema: {schema!r}")
        session_payload = payload.get("session")
        if not isinstance(session_payload, dict):
            raise ValueError("manual-control record is missing session metadata")
        rewind_payload = payload.get("rewind")
        return cls(
            event=str(payload["event"]),
            scenario_id=str(payload["scenario_id"]),
            seed=int(payload["seed"]),
            attempt_id=int(payload["attempt_id"]),
            step_idx=int(payload["step_idx"]),
            session=ManualSessionMetadata(
                session_id=str(session_payload["session_id"]),
                input_mapping_version=str(session_payload["input_mapping_version"]),
                view_mode=str(session_payload.get("view_mode", "fixed_map")),
                policy_to_beat=session_payload.get("policy_to_beat"),
                policy_to_beat_source=session_payload.get("policy_to_beat_source"),
                extra=dict(session_payload.get("extra") or {}),
            ),
            input_keys=list(payload.get("input_keys") or []),
            mapped_action=(
                tuple(float(value) for value in payload["mapped_action"])
                if payload.get("mapped_action") is not None
                else None
            ),
            observation=payload.get("observation"),
            metrics=dict(payload.get("metrics") or {}),
            rewind=(
                ManualRewindMetadata(
                    strategy=str(rewind_payload["strategy"]),
                    from_step_idx=int(rewind_payload["from_step_idx"]),
                    to_step_idx=int(rewind_payload["to_step_idx"]),
                    invalidates_samples_after_step=int(
                        rewind_payload["invalidates_samples_after_step"]
                    ),
                    reason=rewind_payload.get("reason"),
                )
                if isinstance(rewind_payload, dict)
                else None
            ),
            training_sample=bool(payload.get("training_sample", False)),
        )


class ManualJsonlRecorder:
    """Append-only JSONL recorder for manual-control sessions."""

    def __init__(self, path: str | Path):
        """Open a manual-control JSONL stream."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: ManualControlRecord) -> None:
        """Append one sorted-key JSON object and flush it."""
        self._file.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying JSONL file."""
        self._file.close()

    def __enter__(self) -> ManualJsonlRecorder:
        """Return the recorder for context-manager use.

        Returns
        -------
        ManualJsonlRecorder
            Active recorder instance.
        """
        return self

    def __exit__(self, *_exc_info) -> None:
        """Close the recorder at context-manager exit."""
        self.close()


def load_manual_jsonl_records(path: str | Path) -> list[ManualControlRecord]:
    """Load manual-control records from a JSONL file.

    Returns
    -------
    list[ManualControlRecord]
        Parsed records in file order.
    """
    records: list[ManualControlRecord] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on manual-control line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"manual-control line {line_number} is not a JSON object")
        records.append(ManualControlRecord.from_json_dict(payload))
    return records
