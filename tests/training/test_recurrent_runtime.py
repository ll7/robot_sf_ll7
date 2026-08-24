"""Fail-closed unit tests for the recurrent runtime helpers (issue #7847)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.training.recurrent_runtime import (
    EpisodeBoundary,
    RecurrentStateError,
    ResetAccounting,
    build_checkpoint_index_entry,
    reset_reasons_for,
    summarize_state_norms,
    validate_recurrent_state,
    write_checkpoint_index,
    write_json_atomic,
)

if TYPE_CHECKING:
    from pathlib import Path


def _states(num_envs: int = 2, hidden: int = 4, layers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    shape = (layers, num_envs, hidden)
    return np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


class TestValidateRecurrentState:
    """Fail-closed validation of SB3 recurrent state tuples."""

    def test_accepts_valid_states(self) -> None:
        hidden, cell = _states()
        out_hidden, out_cell = validate_recurrent_state((hidden, cell), num_envs=2)
        assert out_hidden.shape == hidden.shape
        assert out_cell.shape == cell.shape

    def test_rejects_none_state(self) -> None:
        with pytest.raises(RecurrentStateError, match="requires lstm_states"):
            validate_recurrent_state(None, num_envs=2)

    def test_rejects_non_finite_values(self) -> None:
        hidden, cell = _states()
        hidden[0, 0, 0] = float("nan")
        with pytest.raises(RecurrentStateError, match="non-finite"):
            validate_recurrent_state((hidden, cell), num_envs=2)

    def test_rejects_batch_mismatch(self) -> None:
        hidden, cell = _states()
        with pytest.raises(RecurrentStateError, match="num_envs"):
            validate_recurrent_state((hidden, cell), num_envs=3)

    def test_rejects_shape_drift_against_expected_shape(self) -> None:
        hidden, cell = _states()
        with pytest.raises(RecurrentStateError, match="shape drift"):
            validate_recurrent_state(
                (hidden, cell),
                num_envs=2,
                expected_shape=(1, 2, 8),
            )

    def test_rejects_hidden_cell_shape_mismatch(self) -> None:
        hidden, _ = _states()
        cell, _ = _states(hidden=8)
        with pytest.raises(RecurrentStateError, match="mismatch"):
            validate_recurrent_state((hidden, cell), num_envs=2)

    def test_rejects_integer_dtype(self) -> None:
        hidden = np.ones((1, 2, 4), dtype=np.int32)
        cell = np.zeros_like(hidden)
        with pytest.raises(RecurrentStateError, match="floating array"):
            validate_recurrent_state((hidden, cell), num_envs=2)


class TestResetReasons:
    """Reset-reason mapping and accounting."""

    def test_terminated_takes_precedence_over_truncated(self) -> None:
        reasons = reset_reasons_for(
            [
                EpisodeBoundary(terminated=True, truncated=True),
                EpisodeBoundary(terminated=False, truncated=True),
                EpisodeBoundary(terminated=False, truncated=False),
            ]
        )
        assert reasons == ["terminated", "truncated", None]

    def test_accounting_records_declared_reasons_only(self) -> None:
        accounting = ResetAccounting()
        accounting.record("terminated")
        accounting.record("terminated")
        accounting.record("env_reset")
        assert accounting.as_dict() == {
            "episode_start": 0,
            "terminated": 2,
            "truncated": 0,
            "env_reset": 1,
        }

    def test_accounting_rejects_unknown_reason(self) -> None:
        with pytest.raises(RecurrentStateError, match="Unknown reset reason"):
            ResetAccounting().record("spontaneous")


class TestCheckpointIndex:
    """Immutable checkpoint index rows and duplicate rejection."""

    def test_entry_contains_sha_and_provenance(self, tmp_path: Path) -> None:
        model = tmp_path / "best.zip"
        model.write_bytes(b"checkpoint-bytes")
        entry = build_checkpoint_index_entry(
            kind="best",
            checkpoint_path=model,
            eval_step=1024,
            score=0.75,
            metric_name="success_rate",
            source_sha="abc123",
            seed=4014,
        )
        import hashlib

        assert entry["sha256"] == hashlib.sha256(b"checkpoint-bytes").hexdigest()
        assert entry["eval_step"] == 1024
        assert entry["score"] == 0.75
        assert entry["metric"] == "success_rate"
        assert entry["seed"] == 4014

    def test_index_writes_atomically_and_rejects_duplicates(self, tmp_path: Path) -> None:
        model_a = tmp_path / "a.zip"
        model_b = tmp_path / "b.zip"
        model_a.write_bytes(b"a")
        model_b.write_bytes(b"b")
        entries = [
            build_checkpoint_index_entry(
                kind="latest",
                checkpoint_path=model_a,
                eval_step=None,
                score=None,
                metric_name=None,
                source_sha="abc123",
                seed=None,
            ),
            build_checkpoint_index_entry(
                kind="final",
                checkpoint_path=model_b,
                eval_step=None,
                score=None,
                metric_name=None,
                source_sha="abc123",
                seed=None,
            ),
        ]
        index_path = write_checkpoint_index(tmp_path / "checkpoint_index.json", entries)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "recurrent-checkpoint-index.v1"
        assert len(payload["checkpoints"]) == 2
        assert not (tmp_path / "checkpoint_index.json.tmp").exists()

        duplicate = dict(entries[0])
        with pytest.raises(RecurrentStateError, match="Duplicate checkpoint identity"):
            write_checkpoint_index(tmp_path / "checkpoint_index.json", [entries[0], duplicate])


class TestAtomicJsonWrites:
    """Atomic manifest-style writes are JSON-safe."""

    def test_write_json_atomic_serializes_numpy_and_non_finite(self, tmp_path: Path) -> None:
        payload = {
            "score": np.float64(0.5),
            "count": np.int64(3),
            "nan_value": float("nan"),
            "nested": {"items": [np.float32(1.5)]},
        }
        path = write_json_atomic(tmp_path / "manifest.json", payload)
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["score"] == 0.5
        assert parsed["count"] == 3
        assert parsed["nan_value"] is None
        assert parsed["nested"]["items"] == [1.5]
        assert not (tmp_path / "manifest.json.tmp").exists()


def test_state_norm_summary_shapes() -> None:
    """Norm summaries expose max/mean per state kind."""
    hidden, cell = _states(num_envs=3, hidden=8)
    summary = summarize_state_norms(hidden, cell)
    assert set(summary) == {
        "hidden_norm_max",
        "hidden_norm_mean",
        "cell_norm_max",
        "cell_norm_mean",
    }
    assert summary["hidden_norm_max"] >= summary["hidden_norm_mean"]
