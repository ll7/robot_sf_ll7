"""Tests for the benchmark artifact compiler CLI."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.artifact_catalog import validate_artifact_catalog
from scripts.tools import compile_benchmark_artifacts as compiler

if TYPE_CHECKING:
    from pathlib import Path


def test_compile_benchmark_artifacts_emits_catalog_tables_figures_and_gaps(
    tmp_path: Path,
) -> None:
    """A tiny campaign fixture should compile into a provenance-rich artifact pack."""
    campaign_root = _write_campaign_fixture(tmp_path / "campaign")
    output_dir = tmp_path / "compiled"

    exit_code = compiler.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--output",
            str(output_dir),
            "--catalog-id",
            "fixture_campaign_artifacts",
        ]
    )

    assert exit_code == 0
    catalog_path = output_dir / "artifact_catalog.yaml"
    assert validate_artifact_catalog(catalog_path) == []

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    artifact_ids = {entry["artifact_id"] for entry in catalog["artifacts"]}
    assert "tab_campaign_table" in artifact_ids
    assert "fig_planner_status_summary" in artifact_ids
    assert "tab_not_available_inputs" in artifact_ids

    table_csv = output_dir / "tables" / "campaign_table.csv"
    table_md = output_dir / "tables" / "campaign_table.md"
    table_tex = output_dir / "tables" / "campaign_table.tex"
    assert table_csv.exists()
    assert table_md.exists()
    assert table_tex.exists()
    assert b"\r" not in table_csv.read_bytes()
    assert b"\r" not in (output_dir / "sources" / "reports" / "campaign_table.csv").read_bytes()
    assert "'-0.25" not in table_csv.read_text(encoding="utf-8")
    assert "'-0.25" not in table_md.read_text(encoding="utf-8")
    assert "'-0.25" not in table_tex.read_text(encoding="utf-8")
    assert "-0.25" in table_md.read_text(encoding="utf-8")
    assert "fallback" in table_md.read_text(encoding="utf-8")
    assert "not_available" in table_md.read_text(encoding="utf-8")
    assert "None" not in table_md.read_text(encoding="utf-8")

    assert (output_dir / "figures" / "planner_status_summary.png").exists()
    assert (output_dir / "figures" / "planner_status_summary.pdf").exists()
    assert "diagnostic-only" in (output_dir / "captions.md").read_text(encoding="utf-8")

    missing_records = json.loads((output_dir / "not_available_inputs.json").read_text())
    assert missing_records["schema_version"] == "benchmark_artifact_compiler.not_available.v1"
    assert {
        "input": "reports/snqi_diagnostics.json",
        "status": "not_available",
        "reason": "optional input missing",
    } in missing_records["records"]

    checksums = (output_dir / "checksums.sha256").read_text(encoding="utf-8")
    assert "tables/campaign_table.md" in checksums
    assert "figures/planner_status_summary.png" in checksums


def _write_campaign_fixture(campaign_root: Path) -> Path:
    """Create a tiny benchmark campaign reports tree."""
    reports = campaign_root / "reports"
    reports.mkdir(parents=True)
    (reports / "matrix_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture.matrix_summary.v1",
                "campaign_id": "fixture_campaign",
                "planners": ["baseline", "fallback_planner", "missing_planner"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / "statistical_sufficiency.json").write_text(
        json.dumps({"schema_version": "fixture.sufficiency.v1", "status": "diagnostic"}) + "\n",
        encoding="utf-8",
    )
    with (reports / "campaign_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["planner", "status", "success_rate", "snqi_mean"]
        )
        writer.writeheader()
        writer.writerow(
            {"planner": "baseline", "status": "native", "success_rate": "0.80", "snqi_mean": "0.10"}
        )
        writer.writerow(
            {
                "planner": "fallback_planner",
                "status": "fallback",
                "success_rate": "0.20",
                "snqi_mean": "'-0.25",
            }
        )
        writer.writerow(
            {
                "planner": "missing_planner",
                "status": "not_available",
                "success_rate": "",
                "snqi_mean": "",
            }
        )
    with (reports / "campaign_table.csv").open("a", encoding="utf-8") as handle:
        handle.write("partial_row_planner,native\n")
    return campaign_root
