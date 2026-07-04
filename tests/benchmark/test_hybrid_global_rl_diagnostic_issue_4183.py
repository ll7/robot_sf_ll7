"""Tests for issue #4183 hybrid_global_rl diagnostic packet contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.hybrid_global_rl_diagnostic import (
    BASELINE_ARM,
    ROUTE_ARM,
    HybridGlobalRLDiagnosticError,
    build_diagnostic_report,
    load_jsonl_records,
    preflight_configs,
)

ROUTE_CONFIG = Path("configs/benchmarks/issue_4183_hybrid_global_rl_route_conditioned.yaml")
BASELINE_CONFIG = Path("configs/benchmarks/issue_4183_learned_local_unconditioned.yaml")


def _record(  # noqa: PLR0913 - test fixture helper keeps row variations explicit.
    *,
    scenario_id: str = "francis2023_intersection_wait",
    seed: int = 4183,
    algo: str,
    checkpoint: str = "model/ppo_model_retrained_10m_2025-02-01.zip",
    route_conditioning_status: str = "conditioned",
    waypoint_status: str = "ok",
    execution_mode: str = "adapter",
    local_policy_status: str = "ok",
    route_progress: float = 0.8,
) -> dict[str, object]:
    diagnostics = {
        "route_conditioning_status": route_conditioning_status,
        "waypoint_status": waypoint_status,
        "waypoint_reason": "selected_route_waypoint",
        "local_policy_metadata": {
            "status": local_policy_status,
            "config": {"model_path": checkpoint},
        },
    }
    metadata: dict[str, object] = {
        "execution_mode": execution_mode,
        "config": {"model_path": checkpoint},
    }
    if algo == "hybrid_global_rl":
        metadata["hybrid_global_rl_diagnostics"] = diagnostics
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": algo,
        "metrics": {
            "route_progress": route_progress,
            "success": route_progress >= 1.0,
            "total_collision_count": 0,
            "near_misses": 0,
        },
        "algorithm_metadata": metadata,
    }


def test_preflight_configs_require_matching_checkpoint_and_existing_paths() -> None:
    """The two issue configs declare the same scenario, seeds, horizon, and checkpoint."""

    report = preflight_configs(ROUTE_CONFIG, BASELINE_CONFIG)
    assert report["status"] == "valid"
    assert report["route_config"] == str(ROUTE_CONFIG)
    assert report["baseline_config"] == str(BASELINE_CONFIG)
    assert report["learned_policy_checkpoint"] == "model/ppo_model_retrained_10m_2025-02-01.zip"
    assert report["checkpoint_sha256"]


def test_preflight_blocks_missing_checkpoint(tmp_path: Path) -> None:
    """Missing learned checkpoints fail closed before any diagnostic run."""

    route_payload = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))
    baseline_payload = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    route_payload["issue_4183_diagnostic"]["learned_policy_checkpoint"] = "model/missing.zip"
    baseline_payload["issue_4183_diagnostic"]["learned_policy_checkpoint"] = "model/missing.zip"
    route_path = tmp_path / "route.yaml"
    baseline_path = tmp_path / "baseline.yaml"
    route_path.write_text(yaml.safe_dump(route_payload), encoding="utf-8")
    baseline_path.write_text(yaml.safe_dump(baseline_payload), encoding="utf-8")

    report = preflight_configs(route_path, baseline_path)

    assert report["status"] == "blocked_missing_learned_checkpoint"
    assert any("blocked_missing_learned_checkpoint" in error for error in report["errors"])


def test_build_report_pairs_rows_and_excludes_fallback(tmp_path: Path) -> None:
    """Report copies adapter diagnostics and excludes fallback rows from effect evidence."""

    route_records = [
        _record(algo="hybrid_global_rl", seed=4183, route_progress=1.0),
        _record(
            algo="hybrid_global_rl",
            seed=4184,
            execution_mode="fallback",
            local_policy_status="fallback",
            route_progress=0.4,
        ),
    ]
    baseline_records = [
        _record(algo="ppo", seed=4183, route_progress=0.7),
        _record(algo="ppo", seed=4184, route_progress=0.5),
    ]

    summary = build_diagnostic_report(
        route_records=route_records,
        baseline_records=baseline_records,
        route_config_path=ROUTE_CONFIG,
        baseline_config_path=BASELINE_CONFIG,
        output_dir=tmp_path,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert summary["row_count"] == 2
    assert summary["run_status"] == "completed"
    assert summary["included_diagnostic_rows"] == 1
    assert summary["fallback_or_degraded_excluded_rows"] == 1
    assert summary["route_conditioned_effect_claim_rows"] == 1
    csv_text = (tmp_path / "paired_rows.csv").read_text(encoding="utf-8")
    assert "included_diagnostic" in csv_text
    assert "excluded_fallback_or_degraded" in csv_text
    assert "selected_route_waypoint" in csv_text
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "diagnostic-only" in readme
    assert "#4161 and #4015" in readme
    assert "## Integration Report" in readme
    assert "New Blockers" in readme
    assert "- none" in readme


def test_build_report_rejects_unpaired_rows(tmp_path: Path) -> None:
    """Rows without same scenario/seed/checkpoint partner are excluded as pairing errors."""

    summary = build_diagnostic_report(
        route_records=[_record(algo="hybrid_global_rl", seed=4183)],
        baseline_records=[_record(algo="ppo", seed=4184)],
        route_config_path=ROUTE_CONFIG,
        baseline_config_path=BASELINE_CONFIG,
        output_dir=tmp_path,
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert summary["included_diagnostic_rows"] == 0
    assert summary["invalid_pair_rows"] == 2
    assert "excluded_pairing_error" in (tmp_path / "paired_rows.csv").read_text(encoding="utf-8")


def test_build_report_records_blocked_fail_closed_run(tmp_path: Path) -> None:
    """A run that produces no episode rows records blocker reasons instead of evidence."""

    summary = build_diagnostic_report(
        route_records=[],
        baseline_records=[],
        route_config_path=ROUTE_CONFIG,
        baseline_config_path=BASELINE_CONFIG,
        output_dir=tmp_path,
        generated_at="2026-07-03T00:00:00+00:00",
        run_failures=[
            {
                "arm": ROUTE_ARM,
                "scenario_id": "francis2023_intersection_wait",
                "seed": 4183,
                "reason": "hybrid_global_rl route waypoint unavailable",
            }
        ],
    )

    assert summary["run_status"] == "blocked_no_valid_episode_rows"
    assert summary["included_diagnostic_rows"] == 0
    assert summary["run_failures"][0]["reason"] == "hybrid_global_rl route waypoint unavailable"
    assert summary["integration_report"]["blockers_remaining"][0]["blocker"] == (
        "no_valid_episode_rows"
    )
    assert summary["integration_report"]["blockers_new"] == []
    assert (
        "rerun the same paired route/occupancy diagnostic builder"
        in summary["integration_report"]["next_empirical_action"]
    )
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "no_valid_episode_rows" in readme
    assert "fail_closed_route_conditioned_hybrid_global_rl" in readme


def test_jsonl_loader_rejects_bad_rows(tmp_path: Path) -> None:
    """The CLI loader rejects malformed JSONL before report construction."""

    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text(json.dumps(_record(algo="ppo")) + "\n", encoding="utf-8")
    assert load_jsonl_records(valid_path)[0]["algo"] == "ppo"

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(HybridGlobalRLDiagnosticError):
        load_jsonl_records(bad_path)


def test_issue_configs_declare_only_route_conditioning_delta() -> None:
    """The issue configs share run controls and differ only by route conditioning arm."""

    route_payload = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))
    baseline_payload = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))

    for key in ("scenario_matrix", "seed_policy", "horizon", "dt", "record_forces"):
        assert route_payload[key] == baseline_payload[key]
    assert route_payload["issue_4183_diagnostic"]["arm"] == ROUTE_ARM
    assert baseline_payload["issue_4183_diagnostic"]["arm"] == BASELINE_ARM
    assert route_payload["issue_4183_diagnostic"]["route_conditioning_enabled"] is True
    assert baseline_payload["issue_4183_diagnostic"]["route_conditioning_enabled"] is False
