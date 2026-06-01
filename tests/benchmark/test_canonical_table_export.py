"""Tests for canonical benchmark table export contracts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.canonical_table_export import export_canonical_table
from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def test_export_planner_outcome_table_preserves_degraded_rows_and_metadata(
    tmp_path: Path,
) -> None:
    """Canonical planner tables should keep degraded rows and provenance metadata."""
    source_path = tmp_path / "campaign_summary.json"
    source_path.write_text('{"schema": "fixture"}\n', encoding="utf-8")
    rows = [
        {
            "planner_key": "zed",
            "execution_mode": "degraded",
            "success_mean": 0.0,
            "collisions_mean": 0.0,
            "near_misses_mean": 1.25,
            "snqi_mean": None,
            "runtime_mean_sec": 12.34567,
            "row_status": "degraded",
        },
        {
            "planner_key": "alpha|planner",
            "execution_mode": "native",
            "success_mean": 1.0,
            "collisions_mean": 0.0,
            "near_misses_mean": 0.0,
            "snqi_mean": 0.87543,
            "runtime_mean_sec": 1.2,
            "row_status": "ok",
        },
    ]

    result = export_canonical_table(
        rows,
        table_id="planner_outcome_summary",
        output_dir=tmp_path / "tables",
        source_paths=[source_path],
        command="robot_sf_bench export-canonical-table ...",
    )

    csv_text = result.output_paths["csv"].read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == (
        "planner_key,execution_mode,success_mean,collisions_mean,near_misses_mean,"
        "snqi_mean,runtime_mean_sec,row_status"
    )
    assert csv_text.splitlines()[1].startswith("alpha|planner,native,1.0000")
    assert "zed,degraded,0.0000,0.0000,1.2500,,12.3457,degraded" in csv_text

    md_text = result.output_paths["md"].read_text(encoding="utf-8")
    assert "| planner_key | execution_mode |" in md_text
    assert "alpha\\|planner" in md_text
    assert "| zed | degraded | 0.0000 | 0.0000 | 1.2500 |  | 12.3457 | degraded |" in md_text

    tex_text = result.output_paths["tex"].read_text(encoding="utf-8")
    assert "\\begin{tabular}" in tex_text
    assert "planner\\_key" in tex_text
    assert "alpha|planner & native" in tex_text
    assert "degraded" in tex_text

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "benchmark_canonical_table_export.v1"
    assert metadata["table_id"] == "planner_outcome_summary"
    assert metadata["row_count"] == 2
    assert metadata["source_files"][0]["sha256"]
    assert metadata["command"] == "robot_sf_bench export-canonical-table ..."


def test_cli_export_canonical_table_writes_execution_mode_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    """CLI export should write named table outputs and a metadata sidecar."""
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            [
                {
                    "planner_key": "fallback_planner",
                    "availability_status": "not_available",
                    "execution_mode": "fallback",
                    "readiness_status": "fallback",
                    "exclusion_reason": "missing optional dependency",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "canonical"

    rc = cli_main(
        [
            "export-canonical-table",
            "--table-id",
            "execution_mode",
            "--rows",
            str(rows_path),
            "--out-dir",
            str(out_dir),
            "--formats",
            "csv,md,tex",
            "--source",
            str(rows_path),
        ],
    )

    cap = capsys.readouterr()
    assert rc == 0, cap.err
    payload = json.loads(cap.out)
    assert payload["table_id"] == "execution_mode"
    assert (out_dir / "execution_mode.csv").is_file()
    assert (out_dir / "execution_mode.md").is_file()
    assert (out_dir / "execution_mode.tex").is_file()
    metadata = json.loads((out_dir / "execution_mode.metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_files"][0]["path"] == rows_path.as_posix()
    assert "fallback" in (out_dir / "execution_mode.csv").read_text(encoding="utf-8")
