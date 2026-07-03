"""Tests for issue #4165 proxemic ablation reporting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.proxemic_ablation_report import (
    build_proxemic_ablation_report,
    load_records,
    write_report_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_builds_paired_proxemic_delta_report_with_parameter_provenance(tmp_path: Path) -> None:
    """Report includes paired deltas and proxemic parameter provenance."""

    smoke_config, proxemic_config = _write_configs(tmp_path)
    baseline_rows = [
        _row("crossing_easy", 1, intrusion_rate=0.30, clearance=0.55, efficiency=0.80, runtime=1.0),
        _row("crossing_hard", 2, intrusion_rate=0.50, clearance=0.45, efficiency=0.70, runtime=1.2),
    ]
    proxemic_rows = [
        _row("crossing_easy", 1, intrusion_rate=0.20, clearance=0.70, efficiency=0.78, runtime=1.1),
        _row("crossing_hard", 2, intrusion_rate=0.40, clearance=0.50, efficiency=0.68, runtime=1.4),
    ]

    report = build_proxemic_ablation_report(
        baseline_records=baseline_rows,
        proxemic_records=proxemic_rows,
        smoke_config_path=smoke_config.relative_to(tmp_path),
        proxemic_config_path=proxemic_config.relative_to(tmp_path),
        repo_root=tmp_path,
    )

    assert report["report_status"] == "ready"
    assert report["paired_rows"] == 2
    assert report["deltas"]["intrusion_rate_delta"] == pytest.approx(-0.1)
    assert report["deltas"]["minimum_clearance_delta"] == pytest.approx(0.1)
    assert report["deltas"]["path_efficiency_delta"] == pytest.approx(-0.02)
    assert report["deltas"]["runtime_overhead_seconds"] == pytest.approx(0.15)
    assert report["parameter_provenance"]["proxemic_config"]["parameters"]["enabled"] is True
    assert len(report["parameter_provenance"]["proxemic_config"]["sha256"]) == 64


def test_blocks_fallback_rows_and_missing_delta_fields(tmp_path: Path) -> None:
    """Fallback rows and missing required metrics block readiness."""

    smoke_config, proxemic_config = _write_configs(tmp_path)
    baseline_rows = [
        _row("crossing_easy", 1, intrusion_rate=0.30, clearance=0.55, efficiency=0.80, runtime=1.0)
    ]
    proxemic_rows = [
        {
            "scenario_id": "crossing_easy",
            "seed": 1,
            "row_status": "fallback",
            "metrics": {"path_efficiency": 0.78, "episode_sec": 1.1},
        }
    ]

    report = build_proxemic_ablation_report(
        baseline_records=baseline_rows,
        proxemic_records=proxemic_rows,
        smoke_config_path=smoke_config.relative_to(tmp_path),
        proxemic_config_path=proxemic_config.relative_to(tmp_path),
        repo_root=tmp_path,
    )

    assert report["report_status"] == "blocked"
    assert any("row_status='fallback'" in reason for reason in report["blocked_reasons"])
    assert any("missing metrics" in reason for reason in report["blocked_reasons"])


def test_report_blocks_invalid_caveats_and_nonpositive_runtime_ratio(tmp_path: Path) -> None:
    """Bad caveat text and non-positive baseline runtime block ratio interpretation."""

    smoke_config, proxemic_config = _write_configs(tmp_path)
    baseline_rows = [
        _row("crossing_easy", 1, intrusion_rate=0.30, clearance=0.55, efficiency=0.80, runtime=0.0)
    ]
    proxemic_rows = [
        _row("crossing_easy", 1, intrusion_rate=0.20, clearance=0.70, efficiency=0.78, runtime=1.0)
    ]
    proxemic_rows[0]["metrics"]["success"] = "maybe"

    report = build_proxemic_ablation_report(
        baseline_records=baseline_rows,
        proxemic_records=proxemic_rows,
        smoke_config_path=smoke_config.relative_to(tmp_path),
        proxemic_config_path=proxemic_config.relative_to(tmp_path),
        repo_root=tmp_path,
    )

    assert report["report_status"] == "blocked"
    assert any("invalid caveat values: success" in reason for reason in report["blocked_reasons"])
    assert any("runtime_seconds must be positive" in reason for reason in report["blocked_reasons"])


def test_load_records_accepts_json_and_jsonl_and_writes_report(tmp_path: Path) -> None:
    """Rows load from JSON/JSONL and report artifacts are written."""

    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    rows = [
        _row("crossing_easy", 1, intrusion_rate=0.30, clearance=0.55, efficiency=0.80, runtime=1.0)
    ]
    json_path.write_text(json.dumps({"episodes": rows}), encoding="utf-8")
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_records(json_path) == rows
    assert load_records(jsonl_path) == rows

    smoke_config, proxemic_config = _write_configs(tmp_path)
    report = build_proxemic_ablation_report(
        baseline_records=rows,
        proxemic_records=rows,
        smoke_config_path=smoke_config.relative_to(tmp_path),
        proxemic_config_path=proxemic_config.relative_to(tmp_path),
        repo_root=tmp_path,
    )
    output_dir = tmp_path / "report"
    write_report_artifacts(report, output_dir)

    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["issue"] == 4165
    assert "intrusion_rate_delta" in (output_dir / "README.md").read_text(encoding="utf-8")


def _row(
    scenario_id: str,
    seed: int,
    *,
    intrusion_rate: float,
    clearance: float,
    efficiency: float,
    runtime: float,
) -> dict[str, object]:
    """Build one fixture row."""

    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "row_status": "native",
        "metrics": {
            "proxemic_intrusion_rate": intrusion_rate,
            "min_distance": clearance,
            "path_efficiency": efficiency,
            "episode_sec": runtime,
            "success": True,
            "collisions": 0,
        },
    }


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Write fixture configs."""

    smoke_config = tmp_path / "smoke.yaml"
    proxemic_config = tmp_path / "proxemic.yaml"
    smoke_config.write_text(
        "issue: 4165\narms:\n  baseline_classical: {}\n  proxemic_costmap_on: {}\n",
        encoding="utf-8",
    )
    proxemic_config.write_text(
        "enabled: true\npersonal_radius: 0.45\nsocial_radius: 1.2\n",
        encoding="utf-8",
    )
    return smoke_config, proxemic_config
