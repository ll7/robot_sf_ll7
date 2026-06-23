"""JSONL record writing helpers for map-based benchmark runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.event_ledger import validate_record_event_ledger
from robot_sf.benchmark.schema_validator import validate_episode
from robot_sf.benchmark.utils import validate_episode_success_integrity

if TYPE_CHECKING:
    from typing import TextIO


def write_validated_to_handle(
    handle: TextIO,
    schema: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Validate one episode record and append it to an open JSONL handle."""
    violations = validate_episode_success_integrity(record)
    violations.extend(validate_record_event_ledger(record))
    if violations:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(violations))
    validate_episode(record, schema)
    handle.write(json.dumps(record, sort_keys=True) + "\n")
