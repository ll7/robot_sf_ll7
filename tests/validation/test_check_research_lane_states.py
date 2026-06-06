"""Tests for the research-lane state table validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.validation import check_research_lane_states

if TYPE_CHECKING:
    from pathlib import Path


def _write_note(root: Path, table_rows: str) -> Path:
    """Write a minimal research-lane state note and linked evidence file."""
    context_dir = root / "docs" / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "evidence.md").write_text("# Evidence\n", encoding="utf-8")
    note = context_dir / "research_lane_states.md"
    note.write_text(
        "\n".join(
            [
                "# Active Research Lane Scientific States",
                "",
                "## Active Lane Table",
                "",
                "| lane | issue | current_state | last_evidence_source | next_discriminating_experiment | stop_condition |",
                "| --- | --- | --- | --- | --- | --- |",
                table_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return note


def test_valid_lane_table_passes(tmp_path: Path) -> None:
    """A complete table with a known state and real local link should pass."""
    note = _write_note(
        tmp_path,
        "| Topology | #1 | `primary_route_overselected` | [evidence.md](evidence.md) | Revise scorer | Stop retuning |",
    )

    assert check_research_lane_states.validate_research_lane_states(note) == []


def test_missing_required_field_fails(tmp_path: Path) -> None:
    """Placeholder fields should fail instead of silently passing."""
    note = _write_note(
        tmp_path,
        "| Topology | #1 | `diagnostic_signal` | [evidence.md](evidence.md) | TBD | Stop retuning |",
    )

    errors = check_research_lane_states.validate_research_lane_states(note)

    assert errors == ["Topology: next_discriminating_experiment is missing or placeholder text"]


def test_unknown_state_and_missing_link_fail(tmp_path: Path) -> None:
    """Unknown scientific states and dead local links should both be reported."""
    note = _write_note(
        tmp_path,
        "| Topology | #1 | `ready` | [missing.md](missing.md) | Revise scorer | Stop retuning |",
    )

    errors = check_research_lane_states.validate_research_lane_states(note)

    assert "Topology: unknown current_state 'ready'" in errors
    assert "Topology: missing linked evidence source missing.md" in errors


def test_misaligned_table_row_fails(tmp_path: Path) -> None:
    """Rows with too many or too few cells should not be silently accepted."""
    note = _write_note(
        tmp_path,
        "| Topology | #1 | `candidate` | [evidence.md](evidence.md) | Revise scorer |",
    )

    errors = check_research_lane_states.validate_research_lane_states(note)

    assert errors == ["row 1: expected 6 cells, found 5"]


def test_cli_returns_nonzero_for_invalid_table(tmp_path: Path, capsys) -> None:
    """CLI should expose validation errors through its exit code."""
    note = _write_note(
        tmp_path,
        "| Topology | #1 | `candidate` | [evidence.md](evidence.md) | Revise scorer | TBD |",
    )

    exit_code = check_research_lane_states.main(["--path", str(note)])

    assert exit_code == 1
    assert "stop_condition is missing" in capsys.readouterr().err
