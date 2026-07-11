"""End-to-end contract coverage for the currently merged issue #3574 tooling.

The fixture exercises the pre-run manifest, trace readiness bridge, and report
writer together.  It is deliberately synthetic: it proves integration only and
does not supply benchmark evidence about heterogeneous populations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.heterogeneous_population_ablation import (
    build_mean_matched_harness_manifest,
)
from robot_sf.benchmark.pedestrian_control_trace import PEDESTRIAN_CONTROL_TRACE_LABELS_KEY

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml"
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/build_heterogeneous_population_ablation_report.py"


def _load_report_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_heterogeneous_population_ablation_report", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct complete records matching the tracked pre-run manifest."""

    records: list[dict[str, Any]] = []
    for row in manifest["manifest_rows"]:
        pedestrians = []
        for label in row["arm_population"][PEDESTRIAN_CONTROL_TRACE_LABELS_KEY]:
            pedestrians.append(
                {
                    **label,
                    "steps": [
                        {
                            "step": 0,
                            "clearance_m": 0.8,
                            "near_field_exposure_s": 0.1,
                        },
                        {
                            "step": 1,
                            "clearance_m": 1.2,
                            "near_field_exposure_s": 0.0,
                        },
                    ],
                }
            )
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "planner": row["planner"],
                "seed": row["seed"],
                "population_arm": row["population_arm"],
                "metrics": {"mean_clearance": 1.0},
                "algorithm_metadata": {
                    "pedestrian_control_trace": {
                        "schema_version": "pedestrian-control-trace.v1",
                        "near_field_clearance_threshold_m": 1.0,
                        "pedestrian_count": len(pedestrians),
                        "pedestrians": pedestrians,
                    }
                },
            }
        )
    return records


def _run_cli(module: ModuleType, argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = argv
    try:
        return int(module.main())
    finally:
        sys.argv = old_argv


def test_tracked_manifest_metrics_flow_into_per_archetype_report(tmp_path: Path) -> None:
    """Every declared trace metric receives an aligned per-archetype report."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _fixture_records(manifest)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    durable_dir = tmp_path / "durable"

    code = _run_cli(
        _load_report_cli(),
        [
            "build_heterogeneous_population_ablation_report.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--durable-dir",
            str(durable_dir),
        ],
    )

    assert code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["integration_readiness"]["ready"] is True
    reports = summary["per_archetype_metric_reports"]
    assert sorted(reports) == ["clearance_m", "near_field_exposure_s"]
    for metric_key in reports:
        assert len(reports[metric_key]) == 9
        first_report = next(iter(reports[metric_key].values()))
        assert first_report["metric_key"] == metric_key
        assert first_report["arms"]["heterogeneous"]["ready"] is True
        assert first_report["arms"]["mean_matched_homogeneous"]["ready"] is True

    assert summary["ablation_reports"] == reports["clearance_m"]
    assert (durable_dir / "summary.json").exists()
