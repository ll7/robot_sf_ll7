"""Tests for the static benchmark dashboard generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import generate_benchmark_dashboard

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    """Write an indented JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_campaign_bundle(tmp_path: Path) -> Path:
    """Create a minimal camera-ready campaign bundle."""
    root = tmp_path / "output" / "benchmarks" / "camera_ready" / "campaign"
    reports = root / "reports"
    _write_json(
        reports / "campaign_summary.json",
        {
            "campaign": {
                "campaign_id": "campaign",
                "name": "smoke",
                "benchmark_success": True,
                "total_episodes": 6,
                "successful_runs": 2,
                "total_runs": 2,
            },
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "algo": "goal",
                    "status": "ok",
                    "benchmark_success": "true",
                    "readiness_status": "native",
                    "success_mean": "1.0000",
                    "collisions_mean": "0.0000",
                    "near_misses_mean": "0.0000",
                    "snqi_mean": "0.7500",
                },
                {
                    "planner_key": "social_force",
                    "algo": "social_force",
                    "status": "ok",
                    "benchmark_success": "true",
                    "readiness_status": "adapter",
                    "success_mean": "0.5000",
                    "collisions_mean": "0.0000",
                    "near_misses_mean": "1.0000",
                    "snqi_mean": "0.4200",
                },
            ],
            "warnings": ["example warning"],
            "artifacts": {
                "campaign_summary_json": str(reports / "campaign_summary.json"),
                "campaign_table_csv": str(reports / "campaign_table.csv"),
            },
        },
    )
    (reports / "campaign_table.csv").write_text(
        "planner_key,success_mean\ngoal,1.0\n", encoding="utf-8"
    )
    return root


def test_dashboard_generator_writes_self_contained_site(tmp_path: Path, monkeypatch) -> None:
    """Generator writes index, data, CSS, downloads, and per-planner pages."""
    bundle = _make_campaign_bundle(tmp_path)
    monkeypatch.setattr(generate_benchmark_dashboard, "get_repository_root", lambda: tmp_path)

    out_dir = tmp_path / "dashboard"
    exit_code = generate_benchmark_dashboard.main(
        ["--bundle-root", str(bundle), "--out", str(out_dir), "--title", "Smoke Dashboard"]
    )

    assert exit_code == 0
    manifest = json.loads((out_dir / "dashboard_manifest.json").read_text(encoding="utf-8"))
    assert manifest["self_contained"] is True
    assert manifest["entrypoint"] == "index.html"
    assert "planners/goal.html" in manifest["planner_pages"]
    assert (out_dir / "index.html").exists()
    assert (out_dir / "assets" / "dashboard.css").exists()
    assert (out_dir / "data" / "dashboard_data.json").exists()
    assert (out_dir / "downloads" / "campaign_summary_json_campaign_summary.json").exists()
    assert (out_dir / "downloads" / "campaign_table_csv_campaign_table.csv").exists()
    index = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Smoke Dashboard" in index
    assert "planners/goal.html" in index
    assert "https://" not in index


def test_dashboard_generator_rejects_missing_campaign_summary(tmp_path: Path) -> None:
    """Generator fails clearly when the bundle lacks the supported summary."""
    with pytest.raises(FileNotFoundError, match="campaign_summary.json"):
        generate_benchmark_dashboard.load_dashboard_bundle(tmp_path / "missing")


def test_dashboard_generator_rejects_repo_root_output_dir(tmp_path: Path, monkeypatch) -> None:
    """Generator should refuse to delete the repository root or one of its parents."""

    bundle = _make_campaign_bundle(tmp_path)
    monkeypatch.setattr(generate_benchmark_dashboard, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="repository root or a parent"):
        generate_benchmark_dashboard.main(["--bundle-root", str(bundle), "--out", str(tmp_path)])


def test_dashboard_generator_rejects_case_insensitive_planner_slug_collisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Planner pages should fail closed when normalized slugs collide."""

    bundle = _make_campaign_bundle(tmp_path)
    monkeypatch.setattr(generate_benchmark_dashboard, "get_repository_root", lambda: tmp_path)

    summary_path = bundle / "reports" / "campaign_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["planner_rows"] = [
        {**payload["planner_rows"][0], "planner_key": "Goal"},
        {**payload["planner_rows"][1], "planner_key": "goal"},
    ]
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Planner slug collision"):
        generate_benchmark_dashboard.main(
            ["--bundle-root", str(bundle), "--out", str(tmp_path / "dashboard")]
        )
