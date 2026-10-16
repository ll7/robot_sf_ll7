"""Consume the complete slot tuples emitted by the existing native reconciler.

No runtime exports or receipt-owned axis inventory are required. Missing rows
provide the native encoding; native summary columns provide its public labels.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.validation import run_empty_world_sweep as native

# Reconciliation only scans this private empty directory, never source archives.
_EMPTY = tempfile.TemporaryDirectory(prefix="receipt-native-slots-")


def _reconcile(cfg: Any, scenarios: list[dict], rows: list[dict]) -> tuple[list[dict], dict]:
    return native.reconcile_execution(cfg, scenarios, rows, Path(_EMPTY.name), {}, step_trace=False)


@lru_cache(maxsize=1)
def _layout() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Match emitted tuple positions to native row and summary declarations."""
    from robot_sf.benchmark.camera_ready._config import load_campaign_config

    template = next(iter(native.SUITES.values()))
    cfg = load_campaign_config(native.REPO_ROOT / template)
    cfg = SimpleNamespace(
        **{
            **vars(cfg),
            "planners": [SimpleNamespace(key="receipt_probe_arm")],
            "kinematics_matrix": ["receipt_probe_kinematics"],
        }
    )
    rows, axes = _reconcile(cfg, [{"name": "receipt_probe_scenario", "seeds": [1001]}], [])
    slot = axes["missing_slots"][0]
    row = rows[0]
    summary = native.flatten(row)
    fields, encoding = [], []
    for value in slot:
        matches = [key for key, item in row.items() if type(item) is type(value) and item == value]
        labels = [
            key for key, item in summary.items() if type(item) is type(value) and item == value
        ]
        if len(matches) != 1 or len(labels) != 1:
            raise ValueError("receipt identity refused: ambiguous native slot encoding")
        encoding.append(matches[0])
        fields.append(labels[0])
    if len(set(fields)) != len(slot) or len(set(encoding)) != len(slot):
        raise ValueError("receipt identity refused: ambiguous native slot fields")
    return tuple(fields), tuple(encoding)


class EpisodeSlot(tuple):
    """An immutable complete native tuple, labelled by native emitted columns."""

    __slots__ = ()
    _fields, _row_fields = _layout()

    def __new__(cls, *values: Any, **declared: Any) -> EpisodeSlot:
        """Construct a complete positional tuple or exact native declarations."""
        if declared:
            if values or set(declared) != set(cls._fields):
                raise TypeError("invalid native slot declarations")
            values = tuple(declared[field] for field in cls._fields)
        if len(values) != len(cls._fields):
            raise TypeError("incomplete native slot tuple")
        return super().__new__(cls, values)

    def __getattr__(self, field: str) -> Any:
        """Expose native summary labels without a receipt-owned field list."""
        if field not in self._fields:
            raise AttributeError(field)
        return self[self._fields.index(field)]

    def __getnewargs__(self) -> tuple:
        """Preserve the entire tuple during copying or serialization."""
        return tuple(self)

    def _asdict(self) -> dict[str, Any]:
        return dict(zip(self._fields, self, strict=True))

    def _replace(self, **changes: Any) -> EpisodeSlot:
        if not set(changes).issubset(self._fields):
            raise ValueError("unknown native slot field")
        return EpisodeSlot(**{**self._asdict(), **changes})


def episode_slot(row: dict) -> EpisodeSlot:
    """Ask native reconciliation for the entire actual row's unexpected tuple."""
    cfg = SimpleNamespace(planners=[], kinematics_matrix=[])
    _, axes = _reconcile(cfg, [], [row])
    slots = axes["unexpected_slots"]
    if len(slots) != 1:
        raise ValueError("receipt identity refused: incomplete native observed slot")
    return EpisodeSlot(*slots[0])


def episode_slot_row(slot: EpisodeSlot) -> dict[str, Any]:
    """Encode every tuple position using the runtime's materialized-row keys."""
    return dict(zip(EpisodeSlot._row_fields, slot, strict=True))


def expected_episode_slots(cfg: Any, scenarios: list[dict]) -> set[EpisodeSlot]:
    """Obtain the full reviewed tuple set from native missing-slot output."""
    _, axes = _reconcile(cfg, scenarios, [])
    return {EpisodeSlot(*slot) for slot in axes["missing_slots"]}


def directory_episode_identity(source: str) -> dict[str, Any]:
    """Use main's native extractors for independent directory declarations."""
    context = {"_source_file": source}
    return {"arm": native._arm_key(context), "kinematics": native._kinematics(context)}
