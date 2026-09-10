"""Deterministic preparation tests for relevance-window selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.relevance_windows import (
    ExcerptContractError,
    RelevanceContractError,
    RelevanceThresholds,
    compute_parent_rows_sha256,
    select_relevance_windows,
    validate_excerpt_manifest,
    write_selection_manifest,
)


def _parent_rows(count: int = 12) -> list[dict[str, object]]:
    """Return a complete synthetic parent with every candidate signal available."""
    rows: list[dict[str, object]] = []
    for step in range(count):
        rows.append(
            {
                "step": step,
                "time_s": step * 0.2,
                "actor_ids": ["robot:0", "ped:0", "ped:1"],
                "clearance_m": 3.0,
                "closing_velocity_m_s": 0.0,
                "ttc_s": 10.0,
                "closest_approach_m": 3.0,
                "braking_margin_m": 2.0,
                "visibility_latency_s": 0.0,
                "path_conflict": False,
                "fallback_or_saturation": False,
                "progress_m": step * 0.1,
                "stall_s": 0.0,
                "discomfort": 0.0,
                "collision": False,
            }
        )
    return rows


def _parent_digest() -> str:
    """Return an external parent artifact identity for tests."""
    return hashlib.sha256(b"parent-artifact").hexdigest()


def test_no_event_timeline_has_no_windows_and_retains_parent() -> None:
    """A complete no-event timeline yields no excerpt but keeps all rows/actors."""
    rows = _parent_rows()
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    assert selection.windows == ()
    assert len(selection.parent_rows) == len(rows)
    assert selection.manifest.selected_step_indices == ()
    assert selection.manifest.actor_ids == ("ped:0", "ped:1", "robot:0")
    rows[0]["clearance_m"] = 0.0
    assert selection.parent_rows[0]["clearance_m"] == 3.0


def test_single_interaction_uses_preroll_postroll_and_hysteresis() -> None:
    """A doorway interaction includes a marked precursor and deterministic roll."""
    rows = _parent_rows()
    rows[2].update({"event_id": "doorway", "precursor": True})
    rows[4].update({"event_id": "doorway", "clearance_m": 0.8})
    rows[5].update({"event_id": "doorway", "clearance_m": 0.7})
    rows[6].update({"event_id": "doorway"})
    selection = select_relevance_windows(
        rows,
        parent_digest=_parent_digest(),
        thresholds=RelevanceThresholds(pre_roll_steps=2, post_roll_steps=2),
    )
    assert len(selection.windows) == 1
    window = selection.windows[0]
    assert window.start_step == 2
    assert window.end_step == 8
    assert window.trigger_steps == (4, 5, 6)
    assert window.precursor_steps == (2,)
    assert "clearance_m" in window.reasons
    assert 2 in selection.manifest.required_precursor_steps
    assert selection.vectors[6].active_reasons == ("clearance_m",)
    assert selection.vectors[7].active_reasons == ()


def test_multistage_events_merge_overlapping_intervals() -> None:
    """Two nearby stages become one explicit merged interval with all triggers."""
    rows = _parent_rows()
    rows[1].update({"event_id": "multi", "precursor": True})
    rows[3].update({"event_id": "multi", "clearance_m": 0.8})
    rows[5].update({"event_id": "multi", "path_conflict": True})
    rows[8].update({"event_id": "multi", "collision": True})
    selection = select_relevance_windows(
        rows,
        parent_digest=_parent_digest(),
        thresholds=RelevanceThresholds(pre_roll_steps=2, post_roll_steps=2, merge_gap_steps=1),
    )
    assert len(selection.windows) == 1
    window = selection.windows[0]
    assert window.start_step == 1
    assert window.end_step == 11
    assert window.trigger_steps == (3, 4, 5, 6, 8, 9)
    assert set(window.reasons) >= {"clearance_m", "path_conflict", "collision"}
    assert set(window.precursor_steps) == {1}


def test_disjoint_event_precursors_do_not_expand_or_merge_each_other() -> None:
    """Separate event identities retain separate windows and precursor ownership."""
    rows = _parent_rows()
    rows[1].update({"event_id": "first", "precursor": True})
    rows[3].update({"event_id": "first", "clearance_m": 0.8})
    rows[5].update({"event_id": "second", "precursor": True})
    rows[7].update({"event_id": "second", "clearance_m": 0.8})
    selection = select_relevance_windows(
        rows,
        parent_digest=_parent_digest(),
        thresholds=RelevanceThresholds(pre_roll_steps=2, post_roll_steps=2),
    )
    assert len(selection.windows) == 2
    assert selection.windows[0].precursor_steps == (1,)
    assert selection.windows[1].precursor_steps == (5,)
    assert selection.windows[0].start_step == 1
    assert selection.windows[1].start_step == 5


def test_too_late_crop_is_rejected_as_unsafe() -> None:
    """Dropping a required precursor cannot be presented as a valid excerpt."""
    rows = _parent_rows()
    rows[1].update({"event_id": "doorway", "precursor": True})
    rows[4].update({"event_id": "doorway", "clearance_m": 0.5})
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    late = replace(
        selection.manifest,
        selected_step_indices=tuple(
            step for step in selection.manifest.selected_step_indices if step != 1
        ),
    )
    with pytest.raises(ExcerptContractError, match="unsafe crop"):
        validate_excerpt_manifest(late, selection.parent_rows)


def test_missing_signal_is_unknown_not_a_safe_zero() -> None:
    """Unavailable TTC remains in the vector and cannot trigger or clear a latch."""
    rows = _parent_rows(3)
    rows[1]["ttc_s"] = None
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    vector = selection.vectors[1]
    assert "ttc_s" in vector.unknown_signals
    assert "ttc_s" in selection.manifest.missing_signals
    assert "ttc_s" not in vector.active_reasons


def test_boolean_signal_rejects_numeric_truthy_values() -> None:
    """Numeric values for typed boolean signals remain invalid and non-triggering."""
    rows = _parent_rows(1)
    rows[0]["path_conflict"] = 1

    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    signal = next(
        signal for signal in selection.vectors[0].signals if signal.name == "path_conflict"
    )

    assert signal.value is None
    assert signal.missingness == "invalid"
    assert "path_conflict" in selection.vectors[0].unknown_signals
    assert "path_conflict" not in selection.vectors[0].active_reasons


@pytest.mark.parametrize("approved_thresholds", [None, {}])
def test_approved_status_requires_nonempty_thresholds(approved_thresholds) -> None:
    """Approved status cannot be serialized without an approved rule set."""
    with pytest.raises(RelevanceContractError, match="nonempty approved_thresholds"):
        RelevanceThresholds(
            approval_status="approved",
            approved_thresholds=approved_thresholds,
        )


def test_fractional_signal_availability_is_rejected() -> None:
    """Timing provenance must not silently truncate a fractional step."""
    rows = _parent_rows(1)
    rows[0]["signal_metadata"] = {"ttc_s": {"available_at_step": 1.5}}
    with pytest.raises(RelevanceContractError, match="available_at_step must be integer"):
        select_relevance_windows(rows, parent_digest=_parent_digest())


def test_future_signal_is_unknown_until_declared_availability_step() -> None:
    """A later-computed risk signal cannot trigger an earlier window."""
    rows = _parent_rows(3)
    rows[0]["signals"] = {
        "ttc_s": {"value": 1.0, "available_at_step": 1},
    }
    rows[1]["signals"] = {
        "ttc_s": {"value": 1.0, "available_at_step": 1},
    }
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    assert "ttc_s" in selection.vectors[0].unknown_signals
    assert selection.vectors[0].active_reasons == ()
    assert selection.vectors[1].active_reasons == ("ttc_s",)


def test_future_signal_remains_unknown_even_when_row_has_flat_value() -> None:
    """Flat row fields obey the same causal availability boundary as metadata."""
    rows = _parent_rows(2)
    rows[0].update(
        {
            "ttc_s": 1.0,
            "signal_metadata": {"ttc_s": {"available_at_step": 1}},
        }
    )
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    assert "ttc_s" in selection.vectors[0].unknown_signals
    assert selection.vectors[0].active_reasons == ()


def test_manifest_writer_preserves_parent_rows_and_digest(tmp_path: Path) -> None:
    """The proposal writer emits full parent rows and a recomputable digest."""
    rows = _parent_rows(4)
    rows[2]["clearance_m"] = 0.8
    selection = select_relevance_windows(rows, parent_digest=_parent_digest())
    path = tmp_path / "manifest.json"
    write_selection_manifest(selection, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest"]["schema_version"] == "scenario_relevance_windows.v1"
    assert payload["manifest"]["full_parent_retained"] is True
    assert payload["manifest"]["parent_rows_sha256"] == compute_parent_rows_sha256(rows)
    assert len(payload["parent_rows"]) == 4
    assert payload["manifest"]["selector_config"]["thresholds"]["approval_status"] == "proposed"
