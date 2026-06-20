"""Tests for minimal multi-AMV benchmark helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.aggregate import compute_aggregates, flatten_metrics
from robot_sf.benchmark.multi_amv import (
    MultiAmvSettings,
    ensure_multi_amv_planner_supported,
    inter_robot_metrics,
    multi_amv_episode_extension,
    multi_amv_planner_support,
    multi_amv_planner_support_inventory,
    multi_amv_settings_from_scenario,
)
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.gym_env.unified_config import MultiRobotConfig, RobotSimulationConfig
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_multi_amv_smoke import (
    _apply_campaign_status,
    _multi_robot_config_from_scenario,
    _read_jsonl,
    build_multi_amv_episode_record,
)


def test_multi_amv_settings_from_scenario_validates_thresholds() -> None:
    """Scenario multi-AMV settings should fail closed for inconsistent thresholds."""
    settings = multi_amv_settings_from_scenario(
        {
            "multi_amv": {
                "num_robots": 2,
                "collision_distance_m": 0.4,
                "near_miss_distance_m": 1.2,
                "deadlock_speed_mps": 0.05,
                "deadlock_window_steps": 8,
            }
        }
    )

    assert settings.num_robots == 2
    assert settings.near_miss_distance_m == pytest.approx(1.2)

    with pytest.raises(ValueError, match="near_miss_distance_m"):
        multi_amv_settings_from_scenario(
            {"multi_amv": {"collision_distance_m": 1.0, "near_miss_distance_m": 0.5}}
        )


def test_inter_robot_metrics_counts_near_miss_collision_and_deadlock() -> None:
    """Inter-robot metric block should expose discrete pairwise encounters."""
    positions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.8, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
            [[0.0, 0.0], [0.3, 0.0]],
        ],
        dtype=float,
    )
    metrics = inter_robot_metrics(
        positions,
        dt=1.0,
        settings=MultiAmvSettings(
            num_robots=2,
            collision_distance_m=0.4,
            near_miss_distance_m=1.0,
            deadlock_speed_mps=0.05,
            deadlock_window_steps=2,
        ),
    )

    assert metrics["robot_count"] == pytest.approx(2.0)
    assert metrics["pair_count"] == pytest.approx(1.0)
    assert metrics["min_inter_robot_distance_m"] == pytest.approx(0.3)
    assert metrics["inter_robot_near_miss_events"] == pytest.approx(1.0)
    assert metrics["inter_robot_collision_events"] == pytest.approx(1.0)
    assert metrics["deadlock_detected"] is True


def test_inter_robot_metrics_handles_empty_trajectory() -> None:
    """Empty multi-robot trajectories should fail closed without crashing."""

    metrics = inter_robot_metrics(
        np.zeros((0, 2, 2), dtype=float),
        dt=1.0,
        settings=MultiAmvSettings(num_robots=2),
    )

    assert metrics["robot_count"] == pytest.approx(2.0)
    assert metrics["pair_count"] == pytest.approx(1.0)
    assert np.isnan(metrics["min_inter_robot_distance_m"])
    assert metrics["inter_robot_collision_events"] == pytest.approx(0.0)
    assert metrics["inter_robot_near_miss_events"] == pytest.approx(0.0)
    assert metrics["deadlock_detected"] is False


def test_multi_amv_episode_extension_is_additive_and_namespaced() -> None:
    """Multi-AMV episode data should live in a namespaced optional block."""
    settings = MultiAmvSettings(num_robots=2)
    metrics = {"robot_count": 2.0, "pair_count": 1.0, "deadlock_detected": False}

    block = multi_amv_episode_extension(
        settings=settings,
        inter_robot=metrics,
        planner_family="goal_controller_smoke",
        planner_status="goal_controller_smoke",
        planner_note="first-slice smoke planner",
    )

    assert set(block) == {"multi_amv"}
    assert block["multi_amv"]["enabled"] is True
    assert block["multi_amv"]["settings"]["num_robots"] == 2
    assert block["multi_amv"]["settings"]["collision_distance_m"] == pytest.approx(0.4)
    assert block["multi_amv"]["planner_family"] == "goal_controller_smoke"
    assert block["multi_amv"]["planner_status"] == "goal_controller_smoke"
    assert block["multi_amv"]["planner_support"]["support_status"] == "native"
    assert block["multi_amv"]["planner_support"]["contract_kind"] == "goal_controller_smoke"
    assert "metrics" not in block["multi_amv"]


def test_multi_amv_episode_extension_requires_multi_robot_metrics() -> None:
    """The extension should fail closed when called outside multi-robot scope."""
    with pytest.raises(ValueError, match="at least two robots"):
        multi_amv_episode_extension(
            settings=MultiAmvSettings(num_robots=1),
            inter_robot={"robot_count": 1.0},
            planner_family="goal_controller_smoke",
            planner_status="test",
        )


def test_multi_amv_episode_extension_requires_non_empty_metrics() -> None:
    """The extension should fail closed when called with empty metrics."""
    with pytest.raises(ValueError, match="metrics must be non-empty"):
        multi_amv_episode_extension(
            settings=MultiAmvSettings(num_robots=2),
            inter_robot={},
            planner_family="goal_controller_smoke",
            planner_status="test",
        )


def test_multi_amv_episode_extension_omits_optional_planner_note() -> None:
    """Planner note should stay absent when callers do not provide one."""
    block = multi_amv_episode_extension(
        settings=MultiAmvSettings(num_robots=2),
        inter_robot={"robot_count": 2.0, "pair_count": 1.0},
        planner_family="goal_controller_smoke",
        planner_status="explicit-status",
    )

    assert block["multi_amv"]["planner_status"] == "explicit-status"
    assert "planner_note" not in block["multi_amv"]


def test_multi_amv_smoke_scenario_passes_schema() -> None:
    """Tracked minimal multi-AMV scenario should be accepted by the benchmark schema."""
    scenario_path = Path("configs/scenarios/single/multi_amv_minimal_smoke.yaml")
    scenarios = [dict(scenario) for scenario in load_scenarios(scenario_path)]

    assert not validate_scenario_list(scenarios)
    assert scenarios[0]["multi_amv"]["num_robots"] == 2


def test_multi_robot_config_from_scenario_preserves_base_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Smoke config conversion should copy base config fields explicitly and override robots."""

    base = RobotSimulationConfig()
    base.map_id = "multi-amv-test-map"
    base.use_planner = True
    base.planner_clearance_margin = 0.75

    monkeypatch.setattr(
        "scripts.validation.run_multi_amv_smoke.build_robot_config_from_scenario",
        lambda scenario, scenario_path: base,
    )

    config = _multi_robot_config_from_scenario(
        {"multi_amv": {"num_robots": 3}},
        tmp_path / "scenario.yaml",
    )

    assert isinstance(config, MultiRobotConfig)
    assert config.map_id == "multi-amv-test-map"
    assert config.use_planner is True
    assert config.planner_clearance_margin == pytest.approx(0.75)
    assert config.num_robots == 3


def test_multi_amv_planner_support_inventory_classifies_known_families() -> None:
    """Planner support inventory should distinguish smoke from real planner support."""
    inventory = multi_amv_planner_support_inventory()

    assert inventory["goal_controller_smoke"]["support_status"] == "native"
    assert inventory["goal_controller_smoke"]["contract_kind"] == "goal_controller_smoke"
    assert inventory["orca"]["support_status"] == "not_available"
    assert inventory["ppo"]["support_status"] == "not_available"
    assert inventory["teb"]["support_status"] == "research_only"


def test_ensure_multi_amv_planner_supported_fails_closed_for_single_robot_planners() -> None:
    """Unsupported planner/multi-AMV combinations should fail before benchmark execution."""
    with pytest.raises(ValueError, match="not_available.*multi-AMV"):
        ensure_multi_amv_planner_supported("orca")


def test_ensure_multi_amv_planner_supported_can_reject_smoke_for_non_trivial_support() -> None:
    """Goal-controller smoke should not be mistaken for non-trivial planner-family support."""
    with pytest.raises(ValueError, match="not non-trivial"):
        ensure_multi_amv_planner_supported("goal_controller_smoke", require_non_smoke=True)


def test_multi_amv_planner_support_rejects_unknown_family() -> None:
    """Unknown planner families should fail closed with the known inventory named."""
    with pytest.raises(ValueError, match="unknown multi-AMV planner family"):
        multi_amv_planner_support("not-a-planner")


def test_multi_amv_episode_record_uses_canonical_metrics_block() -> None:
    """Canonical multi-AMV records should keep metrics reportable at the root."""
    settings = MultiAmvSettings(num_robots=2)
    inter_robot = {
        "robot_count": 2.0,
        "pair_count": 1.0,
        "min_inter_robot_distance_m": 0.75,
        "inter_robot_collision_events": 0.0,
        "deadlock_detected": False,
    }

    record = build_multi_amv_episode_record(
        scenario_id="multi_amv_minimal_smoke",
        seed=0,
        horizon=4,
        steps_recorded=5,
        settings=settings,
        inter_robot=inter_robot,
        planner_family="goal_controller_smoke",
        planner_status="goal_controller_smoke",
        planner_note="first-slice smoke planner",
        wall_time_sec=0.25,
        start_timestamp=datetime(2026, 5, 14, 8, 0, tzinfo=UTC),
    )

    assert record["version"] == "v1"
    assert record["algo"] == "goal_controller_smoke"
    assert record["metrics"]["inter_robot"] == inter_robot
    assert record["multi_amv"]["settings"]["num_robots"] == 2
    assert "metrics" not in record["multi_amv"]
    assert record["termination_reason"] == "terminated"
    assert record["outcome"]["collision_event"] is False
    assert record["timestamps"]["start"] == "2026-05-14T08:00:00+00:00"
    assert "end" in record["timestamps"]


def test_multi_amv_inter_robot_metrics_flatten_for_aggregates() -> None:
    """Aggregate/report helpers should include canonical inter-robot metrics."""
    record = {
        "episode_id": "episode-a",
        "scenario_id": "multi_amv_minimal_smoke",
        "seed": 0,
        "algo": "goal_controller_smoke",
        "metrics": {
            "success": 1.0,
            "inter_robot": {
                "robot_count": 2.0,
                "pair_count": 1.0,
                "min_inter_robot_distance_m": 0.75,
                "inter_robot_collision_events": 0.0,
                "deadlock_detected": False,
            },
        },
    }

    flat = flatten_metrics(record)
    aggregates = compute_aggregates([record], group_by="algo")

    assert flat["min_inter_robot_distance_m"] == pytest.approx(0.75)
    assert flat["inter_robot_collision_events"] == pytest.approx(0.0)
    assert flat["deadlock_detected"] is False
    assert aggregates["goal_controller_smoke"]["min_inter_robot_distance_m"][
        "mean"
    ] == pytest.approx(0.75)
    assert aggregates["goal_controller_smoke"]["inter_robot_collision_events"][
        "mean"
    ] == pytest.approx(0.0)


def _actuation_record(
    *,
    scenario_id: str,
    seed: int,
    variant: str,
    clip: float,
    success: float = 0.0,
    readiness_status: str = "native",
    availability_status: str = "available",
    execution_mode: str = "native",
) -> dict[str, object]:
    """Build a compact actuation-ranking fixture record."""
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": variant,
        "readiness_status": readiness_status,
        "availability_status": availability_status,
        "execution_mode": execution_mode,
        "termination_reason": "timeout_low_progress",
        "metrics": {
            "success": success,
            "collisions": 0.0,
            "command_clip_fraction": clip,
            "yaw_rate_saturation_fraction": 0.0,
            "signed_braking_peak_m_s2": -2.0,
        },
        "final_route_progress_m": 10.0,
        "final_distance_to_goal_m": 3.0,
    }


def test_paired_actuation_feasibility_ranking_keeps_diagnostic_boundary() -> None:
    """Ranking summary should expose paired mechanism fields without overclaiming."""
    from robot_sf.benchmark.multi_amv import paired_actuation_feasibility_ranking

    records = []
    for scenario_id in ("classic_cross_trap_high", "classic_bottleneck_high"):
        for seed in (101, 102):
            records.extend(
                [
                    _actuation_record(
                        scenario_id=scenario_id,
                        seed=seed,
                        variant="hybrid_rule_v3_fast_progress",
                        clip=0.30,
                    ),
                    _actuation_record(
                        scenario_id=scenario_id,
                        seed=seed,
                        variant="actuation_aware_hybrid_rule_v0",
                        clip=0.20,
                    ),
                ]
            )

    summary = paired_actuation_feasibility_ranking(
        records,
        baseline_variant="hybrid_rule_v3_fast_progress",
        intervention_variant="actuation_aware_hybrid_rule_v0",
    )

    assert summary["schema_version"] == "paired-amv-actuation-feasibility-ranking.v1"
    assert summary["classification"] == "bounded_diagnostic_feasibility_direction"
    assert summary["ranking_supported"] is True
    assert summary["claim_boundary"].startswith("diagnostic-only")
    assert summary["uncertainty"]["paired_rows"] == 4
    assert summary["uncertainty"]["scenario_count"] == 2
    assert summary["uncertainty"]["seed_count"] == 2
    assert summary["uncertainty"]["command_clip_delta_mean"] == pytest.approx(-0.10)
    assert summary["pairs"][0]["baseline"]["command_clip_fraction"] == pytest.approx(0.30)
    assert summary["pairs"][0]["intervention"]["command_clip_fraction"] == pytest.approx(0.20)
    assert summary["pairs"][0]["disagreement_cases"][0]["type"] == "feasibility_success_divergence"


def test_paired_actuation_feasibility_ranking_excludes_fallback_rows() -> None:
    """Fallback/degraded rows should fail closed and keep the result inconclusive."""
    from robot_sf.benchmark.multi_amv import paired_actuation_feasibility_ranking

    summary = paired_actuation_feasibility_ranking(
        [
            _actuation_record(
                scenario_id="classic_cross_trap_high",
                seed=101,
                variant="hybrid_rule_v3_fast_progress",
                clip=0.30,
            ),
            _actuation_record(
                scenario_id="classic_cross_trap_high",
                seed=101,
                variant="actuation_aware_hybrid_rule_v0",
                clip=0.20,
                readiness_status="fallback",
            ),
        ],
        baseline_variant="hybrid_rule_v3_fast_progress",
        intervention_variant="actuation_aware_hybrid_rule_v0",
    )

    assert summary["classification"] == "diagnostic_only_inconclusive"
    assert summary["ranking_supported"] is False
    assert summary["excluded_rows"] == [
        {
            "scenario_id": "classic_cross_trap_high",
            "seed": 101,
            "variant": "actuation_aware_hybrid_rule_v0",
            "reason": "readiness_status=fallback",
        }
    ]
    assert summary["incomplete_pairs"][0]["missing_variants"] == ["actuation_aware_hybrid_rule_v0"]


def test_paired_actuation_feasibility_ranking_excludes_partial_failure_alias() -> None:
    """Partial-failure status aliases should be excluded from diagnostic ranking support."""
    from robot_sf.benchmark.multi_amv import paired_actuation_feasibility_ranking

    summary = paired_actuation_feasibility_ranking(
        [
            _actuation_record(
                scenario_id="classic_cross_trap_high",
                seed=101,
                variant="hybrid_rule_v3_fast_progress",
                clip=0.30,
            ),
            {
                **_actuation_record(
                    scenario_id="classic_cross_trap_high",
                    seed=101,
                    variant="actuation_aware_hybrid_rule_v0",
                    clip=0.20,
                ),
                "status": "partial_failure",
            },
        ],
        baseline_variant="hybrid_rule_v3_fast_progress",
        intervention_variant="actuation_aware_hybrid_rule_v0",
    )

    assert summary["classification"] == "diagnostic_only_inconclusive"
    assert summary["ranking_supported"] is False
    assert summary["excluded_rows"][0]["reason"] == "status=partial_failure"


def test_paired_actuation_feasibility_ranking_excludes_missing_status_fields() -> None:
    """Raw episode rows without campaign status metadata should fail closed."""
    from robot_sf.benchmark.multi_amv import paired_actuation_feasibility_ranking

    raw_baseline = _actuation_record(
        scenario_id="classic_cross_trap_high",
        seed=101,
        variant="hybrid_rule_v3_fast_progress",
        clip=0.30,
    )
    for key in ("readiness_status", "availability_status", "execution_mode"):
        raw_baseline.pop(key)
    raw_baseline["algorithm_metadata"] = {"planner_kinematics": {"execution_mode": "adapter"}}

    summary = paired_actuation_feasibility_ranking(
        [
            raw_baseline,
            _actuation_record(
                scenario_id="classic_cross_trap_high",
                seed=101,
                variant="actuation_aware_hybrid_rule_v0",
                clip=0.20,
            ),
        ],
        baseline_variant="hybrid_rule_v3_fast_progress",
        intervention_variant="actuation_aware_hybrid_rule_v0",
    )

    assert summary["classification"] == "diagnostic_only_inconclusive"
    assert summary["ranking_supported"] is False
    assert summary["excluded_rows"][0]["reason"] == "readiness_status="


def test_actuation_ranking_campaign_status_enrichment_restores_explicit_statuses() -> None:
    """Campaign planner rows should annotate raw episode rows before ranking."""
    from robot_sf.benchmark.multi_amv import paired_actuation_feasibility_ranking

    raw_records = []
    for scenario_id in ("classic_cross_trap_high", "classic_bottleneck_high"):
        for seed in (101, 102):
            baseline = _actuation_record(
                scenario_id=scenario_id,
                seed=seed,
                variant="hybrid_rule_local_planner",
                clip=0.30,
            )
            baseline["algorithm_metadata"] = {
                "algorithm": "hybrid_rule_local_planner",
                "config": {
                    "planner_variant": "hybrid_rule_v3_teb_like_rollout",
                    "max_linear_speed": 3.0,
                    "max_linear_accel": 3.0,
                },
                "planner_kinematics": {"execution_mode": "adapter"},
            }
            intervention = _actuation_record(
                scenario_id=scenario_id,
                seed=seed,
                variant="actuation_aware_hybrid_rule_v0",
                clip=0.20,
            )
            for record in (baseline, intervention):
                for key in ("readiness_status", "availability_status", "execution_mode"):
                    record.pop(key)
            raw_records.extend([baseline, intervention])

    enriched = _apply_campaign_status(
        raw_records,
        {
            "planner_rows": [
                {
                    "planner_key": "hybrid_rule_v3_fast_progress",
                    "algo": "hybrid_rule_local_planner",
                    "status": "ok",
                    "readiness_status": "adapter",
                    "availability_status": "available",
                    "execution_mode": "adapter",
                },
                {
                    "planner_key": "actuation_aware_hybrid_rule_v0",
                    "algo": "actuation_aware_hybrid_rule_v0",
                    "status": "ok",
                    "readiness_status": "adapter",
                    "availability_status": "available",
                    "execution_mode": "adapter",
                },
            ]
        },
    )
    summary = paired_actuation_feasibility_ranking(
        enriched,
        baseline_variant="hybrid_rule_v3_fast_progress",
        intervention_variant="actuation_aware_hybrid_rule_v0",
    )

    assert summary["classification"] == "bounded_diagnostic_feasibility_direction"
    assert summary["ranking_supported"] is True
    assert summary["excluded_rows"] == []


def test_read_jsonl_rejects_non_object_records(tmp_path: Path) -> None:
    """Ranking CLI JSONL reader should reject malformed non-object rows."""
    path = tmp_path / "records.jsonl"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        _read_jsonl(path)
