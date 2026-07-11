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

import pytest
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


def _fixture_records(
    manifest: dict[str, Any], *, null_first_response_law_fraction: bool = False
) -> list[dict[str, Any]]:
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
        response_law_fraction = row.get("response_law_fraction")
        if response_law_fraction is None:
            response_law_fraction = 0.0
        if null_first_response_law_fraction and not records:
            response_law_fraction = None
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "planner": row["planner"],
                "seed": row["seed"],
                "population_arm": row["population_arm"],
                "response_law_fraction": response_law_fraction,
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


def test_report_metric_direction_contract_is_explicit_and_fail_closed() -> None:
    """Each supported trace metric has a declared safety direction."""

    module = _load_report_cli()
    assert module.metric_higher_is_safer("clearance_m") is True
    assert module.metric_higher_is_safer("near_field_exposure_s") is False
    with pytest.raises(ValueError, match="No higher_is_safer direction"):
        module.metric_higher_is_safer("unknown_safety_metric")


def test_tracked_manifest_metrics_flow_into_per_archetype_report(tmp_path: Path) -> None:
    """Every declared trace metric receives an aligned per-archetype report."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    manifest = build_mean_matched_harness_manifest(config, config_path=str(CONFIG_PATH))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _fixture_records(manifest, null_first_response_law_fraction=True)
        ),
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
        assert len(reports[metric_key]) == 36
        first_report = next(iter(reports[metric_key].values()))
        assert first_report["metric_key"] == metric_key
        assert first_report["higher_is_safer"] is (metric_key == "clearance_m"), (
            f"Expected {metric_key!r} to declare the correct CVaR safety direction; "
            f"got {first_report['higher_is_safer']!r}"
        )
        assert first_report["arms"]["heterogeneous"]["ready"] is True
        assert first_report["arms"]["mean_matched_homogeneous"]["ready"] is True

    assert summary["ablation_reports"] == reports["clearance_m"]
    assert (durable_dir / "summary.json").exists()


def test_tracked_manifest_declares_required_response_law_sweep() -> None:
    """The committed #3574 matrix makes all required fractions runnable."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["response_law_fractions"] == [0.0, 0.1, 0.25, 0.5]
