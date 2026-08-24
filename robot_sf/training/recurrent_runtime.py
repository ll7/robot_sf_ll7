"""Recurrent-state-aware runtime helpers for the RecurrentPPO training lane.

This module owns the parts of :mod:`scripts.training.train_recurrent_ppo` that
must understand recurrent state: deterministic evaluation with explicit
``lstm_states``/``episode_start`` handling, fail-closed state-integrity guards,
per-index state isolation, reset accounting, and the checkpoint index.

The scientific comparison contract is frozen by
``configs/training/comparison_matrix/issue_7846_ppo_rppo_contract_v1.yaml``;
nothing here may change that contract. This module is diagnostic/runtime
infrastructure only and produces no benchmark or paper-facing claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

RESET_REASONS = ("episode_start", "terminated", "truncated", "env_reset")


class RecurrentStateError(RuntimeError):
    """Raised when recurrent state violates the fail-closed integrity contract."""


@dataclass(slots=True)
class ResetAccounting:
    """Count of recurrent-state resets grouped by declared reason."""

    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(RESET_REASONS, 0),
    )

    def record(self, reason: str) -> None:
        """Record one recurrent-state reset under its declared reason.

        Args:
            reason: One of the declared reasons in ``RESET_REASONS``.

        Raises:
            RecurrentStateError: If ``reason`` is not a declared reset reason.
        """
        if reason not in self.counts:
            raise RecurrentStateError(f"Unknown reset reason: {reason!r}")
        self.counts[reason] += 1

    def as_dict(self) -> dict[str, int]:
        """Return the reset counts keyed by reason."""
        return dict(self.counts)


def _validate_state_array(state: Any, *, where: str) -> np.ndarray:
    """Validate one hidden/cell state array for finiteness and numeric dtype.

    Args:
        state: Candidate hidden or cell state array.
        where: Human-readable location used in error messages.

    Returns:
        The validated array.

    Raises:
        RecurrentStateError: On non-floating dtype or non-finite values.
    """
    array = np.asarray(state)
    if not np.issubdtype(array.dtype, np.floating):
        raise RecurrentStateError(f"{where} must be a floating array, received {array.dtype!r}")
    if not np.all(np.isfinite(array)):
        non_finite = int(np.count_nonzero(~np.isfinite(array)))
        raise RecurrentStateError(f"{where} contains {non_finite} non-finite values")
    return array


def validate_recurrent_state(
    lstm_states: Any,
    *,
    num_envs: int,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fail closed on missing, malformed, non-finite, or shape-drifted state.

    Args:
        lstm_states: Recurrent state as ``(hidden, cell)`` from an SB3 policy.
        num_envs: Required batch dimension of the recurrent state.
        expected_shape: Optional exact shape both arrays must match.

    Returns:
        The validated ``(hidden_state, cell_state)`` tuple.

    Raises:
        RecurrentStateError: On missing, malformed, non-finite, or drifted state.
    """
    if lstm_states is None:
        raise RecurrentStateError("Recurrent evaluation requires lstm_states; received None")
    try:
        hidden, cell = lstm_states
    except (TypeError, ValueError) as exc:
        raise RecurrentStateError(
            "lstm_states must unpack into (hidden_state, cell_state)",
        ) from exc
    hidden_arr = _validate_state_array(hidden, where="lstm_states.hidden")
    cell_arr = _validate_state_array(cell, where="lstm_states.cell")
    if hidden_arr.shape != cell_arr.shape:
        raise RecurrentStateError(
            f"lstm_states hidden/cell shape mismatch: {hidden_arr.shape} vs {cell_arr.shape}",
        )
    if hidden_arr.ndim != 3 or hidden_arr.shape[1] != num_envs:
        raise RecurrentStateError(
            f"lstm_states batch dimension (axis 1 of (n_layers, n_envs, hidden)) must equal "
            f"num_envs={num_envs}, received shape {hidden_arr.shape}",
        )
    if expected_shape is not None and hidden_arr.shape != expected_shape:
        raise RecurrentStateError(
            f"Recurrent state shape drift: expected {expected_shape}, received {hidden_arr.shape}",
        )
    return hidden_arr, cell_arr


@dataclass(slots=True)
class EpisodeBoundary:
    """Terminal flags for one vector-environment step."""

    terminated: bool
    truncated: bool

    @property
    def any_terminal(self) -> bool:
        """Whether either terminal flag is set."""
        return self.terminated or self.truncated


def reset_reasons_for(boundaries: Sequence[EpisodeBoundary]) -> list[str | None]:
    """Return the reset reason per environment index (None when no boundary).

    Args:
        boundaries: Terminal flags per vector-environment index.

    Returns:
        One entry per index: ``"terminated"``, ``"truncated"``, or None.
    """
    reasons: list[str | None] = []
    for boundary in boundaries:
        if boundary.terminated:
            reasons.append("terminated")
        elif boundary.truncated:
            reasons.append("truncated")
        else:
            reasons.append(None)
    return reasons


def summarize_state_norms(hidden: np.ndarray, cell: np.ndarray) -> dict[str, float]:
    """Compact diagnostic norms for hidden/cell states (not quality evidence).

    Args:
        hidden: Hidden-state array with leading batch dimension.
        cell: Cell-state array with matching shape.

    Returns:
        Max and mean norms per environment for hidden and cell states.
    """
    hidden_norms = np.linalg.norm(hidden.reshape(hidden.shape[0], -1), axis=1)
    cell_norms = np.linalg.norm(cell.reshape(cell.shape[0], -1), axis=1)
    return {
        "hidden_norm_max": float(np.max(hidden_norms)),
        "hidden_norm_mean": float(np.mean(hidden_norms)),
        "cell_norm_max": float(np.max(cell_norms)),
        "cell_norm_mean": float(np.mean(cell_norms)),
    }


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to JSON-safe values.

    Args:
        value: Arbitrary nested payload value.

    Returns:
        A JSON-serializable copy of ``value``.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Atomically write ``payload`` as JSON via a temp file plus rename.

    Args:
        path: Destination file path.
        payload: JSON-serializable payload.

    Returns:
        The destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)
    return path


def append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one compact monitoring record as a JSON line.

    Args:
        path: Destination JSONL file.
        record: One monitoring record with scalar values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(dict(record)), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file streamed in chunks.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_index_entry(
    *,
    kind: str,
    checkpoint_path: Path,
    eval_step: int | None,
    score: float | None,
    metric_name: str | None,
    source_sha: str,
    seed: int | None,
) -> dict[str, Any]:
    """Build one immutable row of ``checkpoint_index.json``.

    Args:
        kind: Checkpoint role such as ``latest``, ``best``, or ``final``.
        checkpoint_path: Path of the saved model archive.
        eval_step: Evaluation step that produced the checkpoint, if any.
        score: Selection-metric score, when applicable.
        metric_name: Selection metric name, when applicable.
        source_sha: Git commit SHA the run was produced at.
        seed: Training seed owning this checkpoint.

    Returns:
        The index entry mapping.
    """
    entry: dict[str, Any] = {
        "kind": kind,
        "checkpoint": checkpoint_path.name,
        "sha256": sha256_file(checkpoint_path),
        "recorded_at": utc_now_iso(),
        "source_sha": source_sha,
    }
    if eval_step is not None:
        entry["eval_step"] = int(eval_step)
    if score is not None:
        entry["score"] = float(score)
    if metric_name is not None:
        entry["metric"] = metric_name
    if seed is not None:
        entry["seed"] = int(seed)
    return entry


def write_checkpoint_index(path: Path, entries: list[dict[str, Any]]) -> Path:
    """Persist the checkpoint index atomically and fail closed on duplicates.

    Args:
        path: Destination ``checkpoint_index.json`` path.
        entries: Index entries previously built by
            :func:`build_checkpoint_index_entry`.

    Returns:
        The written index path.

    Raises:
        RecurrentStateError: When two entries share kind and checkpoint name.
    """
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        identity = (entry["kind"], entry["checkpoint"])
        if identity in seen:
            raise RecurrentStateError(
                f"Duplicate checkpoint identity in index: kind={entry['kind']!r} "
                f"checkpoint={entry['checkpoint']!r}",
            )
        seen.add(identity)
    payload = {
        "schema_version": "recurrent-checkpoint-index.v1",
        "checkpoints": entries,
    }
    return write_json_atomic(path, payload)
