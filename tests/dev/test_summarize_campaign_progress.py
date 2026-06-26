"""Tests compact campaign progress summaries."""

from __future__ import annotations

import json
import os
from itertools import count
from pathlib import Path  # noqa: TC003

from scripts.dev.summarize_campaign_progress import main, summarize_campaign_progress

# Step by 10s so distinct writes stay strictly ordered even on filesystems with
# coarse mtime resolution (e.g. FAT/exFAT round to 2-second granularity).
_MTIME_EPOCHS = count(1_700_000_000, 10)


def _write_lines(path: Path, count: int) -> None:
    """Write deterministic JSONL-ish rows."""

    path.write_text(
        "".join(f'{{"episode": {index}}}\n' for index in range(count)), encoding="utf-8"
    )
    mtime_epoch = next(_MTIME_EPOCHS)
    os.utime(path, (mtime_epoch, mtime_epoch))


def test_summarize_campaign_progress_reports_active_variant(tmp_path: Path) -> None:
    """Newest JSONL artifact should identify active suite/variant compactly."""

    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    _write_lines(output_dir / "hard__baseline__ckpt.jsonl", 7)
    _write_lines(output_dir / "global__baseline__ckpt.jsonl", 69)
    active = output_dir / "global__dense_lattice__ckpt.jsonl"
    _write_lines(active, 12)

    summary = summarize_campaign_progress(output_dir, write_state=False)

    assert summary.exists is True
    assert summary.active_variant == "dense_lattice"
    assert summary.active_suite == "global"
    assert summary.completed_variants == ["baseline"]
    assert summary.completed_variant_count == 1
    assert summary.jsonl_artifact_count == 3
    assert summary.jsonl_line_count == 88
    assert summary.summary_exists is False
    assert summary.report_exists is False
    assert summary.newest_artifact is not None
    assert summary.newest_artifact.path == str(active)


def test_summarize_campaign_progress_tracks_deltas_with_state_file(tmp_path: Path) -> None:
    """Polling state should produce bounded size and line deltas."""

    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    state_file = tmp_path / "state.json"
    active = output_dir / "global__combined_max_authority__ckpt.jsonl"
    _write_lines(active, 2)

    first = summarize_campaign_progress(output_dir, state_file=state_file)
    _write_lines(active, 5)
    second = summarize_campaign_progress(output_dir, state_file=state_file)

    assert first.newest_artifact is not None
    assert first.newest_artifact.line_delta is None
    assert second.newest_artifact is not None
    assert second.newest_artifact.line_count == 5
    assert second.newest_artifact.line_delta == 3
    assert second.newest_artifact.size_delta_bytes is not None
    assert second.newest_artifact.size_delta_bytes > 0


def test_summarize_campaign_progress_ignores_default_state_file(tmp_path: Path) -> None:
    """Default state file should not become newest artifact on repeated polls."""

    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    active = output_dir / "global__dense_lattice__ckpt.jsonl"
    _write_lines(active, 2)

    summarize_campaign_progress(output_dir)
    second = summarize_campaign_progress(output_dir)

    assert second.newest_artifact is not None
    assert second.newest_artifact.path == str(active)
    assert second.active_variant == "dense_lattice"


def test_summarize_campaign_progress_detects_final_artifacts_and_failures(tmp_path: Path) -> None:
    """Final report aliases and failure marker paths should be bounded."""

    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    (output_dir / "campaign_summary.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "campaign_report.md").write_text("# Report\n", encoding="utf-8")
    for index in range(4):
        (output_dir / f"failure_{index}.json").write_text("{}\n", encoding="utf-8")

    summary = summarize_campaign_progress(output_dir, write_state=False, failure_limit=2)

    assert summary.summary_exists is True
    assert summary.report_exists is True
    assert summary.recommended_next_poll_seconds == 0
    assert len(summary.failure_markers) == 2
    assert summary.failure_markers_truncated is True


def test_summarize_campaign_progress_cli_outputs_json(tmp_path: Path, capsys) -> None:
    """CLI output should be machine-readable JSON for repeated Codex polling."""

    output_dir = tmp_path / "campaign"
    output_dir.mkdir()
    _write_lines(output_dir / "hard__baseline__ckpt.jsonl", 1)

    rc = main(["--output-dir", str(output_dir), "--no-state"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "campaign_progress_summary.v1"
    assert payload["output_dir"] == str(output_dir.resolve())
    assert payload["jsonl_artifact_count"] == 1
    assert payload["tracked_artifacts"][0]["line_count"] == 1
