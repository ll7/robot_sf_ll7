"""Tests for issue #4165 proxemic ablation reporting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.proxemic_ablation_report import (
    build_proxemic_ablation_report,
    build_proxemic_ablation_report_from_map_runner_records,
    load_records,
    map_runner_records_to_ablation_rows,
    write_report_artifacts,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_checked_in_fixture_matches_report_contract() -> None:
    """Tracked fixture rows keep issue #4165 report contract executable."""
    fixture_dir = _REPO_ROOT / "tests/benchmark/fixtures/proxemic_ablation_issue_4165"
    report = build_proxemic_ablation_report(
        baseline_records=load_records(fixture_dir / "baseline_classical.jsonl"),
        proxemic_records=load_records(fixture_dir / "proxemic_costmap_on.jsonl"),
        smoke_config_path=_REPO_ROOT / "configs/benchmarks/issue_4165_proxemic_costmap_smoke.yaml",
        proxemic_config_path=_REPO_ROOT / "configs/planners/proxemic_costmap_v1.yaml",
        repo_root=_REPO_ROOT,
    )

    assert report["report_status"] == "ready"
    assert report["claim_boundary"] == "paired_cpu_smoke_or_fixture_report_only"
    assert report["paired_rows"] == 2
    assert report["deltas"]["intrusion_rate_delta"] == pytest.approx(-0.1)
    assert report["deltas"]["minimum_clearance_delta"] == pytest.approx(0.1)
    assert report["deltas"]["path_efficiency_delta"] == pytest.approx(-0.02)
    assert report["deltas"]["runtime_overhead_seconds"] == pytest.approx(0.15)
    assert report["parameter_provenance"]["proxemic_config"]["sha256"]


def test_builds_report_from_map_runner_episode_records(tmp_path: Path) -> None:
    """Map-runner episode rows feed paired delta report with fail-closed provenance."""

    smoke_config, proxemic_config = _write_configs(tmp_path)
    baseline_records = [
        _map_runner_record(
            "crossing_easy",
            1,
            proxemic_enabled=False,
            intrusion_rate=0.30,
            clearance=0.55,
            efficiency=0.80,
            runtime=1.0,
        )
    ]
    proxemic_records = [
        _map_runner_record(
            "crossing_easy",
            1,
            proxemic_enabled=True,
            intrusion_rate=0.20,
            clearance=0.70,
            efficiency=0.78,
            runtime=1.1,
        )
    ]

    report = build_proxemic_ablation_report_from_map_runner_records(
        baseline_records=baseline_records,
        proxemic_records=proxemic_records,
        smoke_config_path=smoke_config.relative_to(tmp_path),
        proxemic_config_path=proxemic_config.relative_to(tmp_path),
        repo_root=tmp_path,
    )

    assert report["report_status"] == "ready"
    assert report["claim_boundary"] == "paired_cpu_smoke_report_only"
    assert report["input_source"]["source"] == "map_runner_episode_records"
    assert report["deltas"]["intrusion_rate_delta"] == pytest.approx(-0.1)


def test_map_runner_rows_fail_closed_on_wrong_layer_state() -> None:
    """Baseline/proxemic arm state mismatches are blocked before reporting."""

    with pytest.raises(ValueError, match="expected False"):
        map_runner_records_to_ablation_rows(
            [_map_runner_record("crossing_easy", 1, proxemic_enabled=True)],
            arm="baseline_classical",
            expected_proxemic_enabled=False,
        )


def test_map_runner_rows_fail_closed_on_malformed_records() -> None:
    """Malformed map-runner rows report all source-contract blockers."""

    fallback = _map_runner_record("crossing_easy", 1, proxemic_enabled=True)
    fallback["algorithm_metadata"]["last_decision"]["proxemic_costmap"]["fallback_or_degraded"] = (
        True
    )

    with pytest.raises(ValueError) as excinfo:
        map_runner_records_to_ablation_rows(
            [
                {"scenario_id": "missing-everything", "seed": 1},
                {
                    "scenario_id": "bad-enabled",
                    "seed": 2,
                    "metrics": {},
                    "algorithm_metadata": {"proxemic_costmap": {"enabled": "yes"}},
                },
                fallback,
            ],
            arm="proxemic_costmap_on",
            expected_proxemic_enabled=True,
        )

    message = str(excinfo.value)
    assert "missing map-runner metrics mapping" in message
    assert "missing algorithm_metadata proxemic_costmap" in message
    assert "proxemic_costmap.enabled must be boolean" in message
    assert "fallback/degraded" in message


def test_map_runner_rows_accept_direct_and_runtime_proxemic_metadata() -> None:
    """Map-runner adapter accepts known proxemic metadata locations."""

    direct = _map_runner_record("crossing_easy", 1, proxemic_enabled=True)
    direct["algorithm_metadata"]["proxemic_costmap"] = direct["algorithm_metadata"].pop(
        "last_decision"
    )["proxemic_costmap"]
    runtime = _map_runner_record("crossing_hard", 2, proxemic_enabled=True)
    runtime["algorithm_metadata"]["planner_runtime"] = runtime["algorithm_metadata"].pop(
        "last_decision"
    )

    rows = map_runner_records_to_ablation_rows(
        [direct, runtime],
        arm="proxemic_costmap_on",
        expected_proxemic_enabled=True,
    )

    assert [row["source"] for row in rows] == [
        "map_runner_episode_records",
        "map_runner_episode_records",
    ]
    assert rows[0]["proxemic_costmap"]["enabled"] is True
    assert rows[1]["proxemic_costmap"]["enabled"] is True


def test_cli_builds_map_runner_input_report(tmp_path: Path) -> None:
    """Issue CLI supports real map-runner episode JSONL input."""

    smoke_config, proxemic_config = _write_configs(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    proxemic_path = tmp_path / "proxemic.jsonl"
    baseline_path.write_text(
        json.dumps(_map_runner_record("crossing_easy", 1, proxemic_enabled=False)) + "\n",
        encoding="utf-8",
    )
    proxemic_path.write_text(
        json.dumps(_map_runner_record("crossing_easy", 1, proxemic_enabled=True)) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"

    exit_code = _report_main(
        [
            "--baseline-episodes",
            str(baseline_path),
            "--proxemic-episodes",
            str(proxemic_path),
            "--input-format",
            "map-runner",
            "--smoke-config",
            str(smoke_config.relative_to(tmp_path)),
            "--proxemic-config",
            str(proxemic_config.relative_to(tmp_path)),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_source"]["fail_closed_layer_state"] is True


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


def _map_runner_record(
    scenario_id: str,
    seed: int,
    *,
    proxemic_enabled: bool,
    intrusion_rate: float = 0.30,
    clearance: float = 0.55,
    efficiency: float = 0.80,
    runtime: float = 1.0,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "status": "completed",
        "algo": "hybrid_rule_local_planner",
        "metrics": {
            "social_space_intrusion_rate": intrusion_rate,
            "min_clearance": clearance,
            "path_efficiency": efficiency,
            "episode_sec": runtime,
            "success": True,
            "collision": False,
        },
        "algorithm_metadata": {
            "last_decision": {
                "proxemic_costmap": {
                    "enabled": proxemic_enabled,
                    "status": "ok" if proxemic_enabled else "disabled",
                    "config_hash": "unit-test-hash",
                    "fallback_or_degraded": False,
                    "soft_cost_only": True,
                }
            }
        },
    }


def _report_main(argv: list[str]) -> int:
    script_path = (
        _REPO_ROOT / "scripts/benchmark/build_proxemic_layer_ablation_report_issue_4165.py"
    )
    spec = importlib.util.spec_from_file_location("issue_4165_proxemic_report_cli", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main(argv)


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
