"""Tests for the bounded planner configuration optimizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.validation.planner_optimizer import (
    ParameterSpec,
    _apply_params,
    load_optimizer_config,
    run_planner_optimization,
    score_records,
    select_best_trial,
)
from scripts.validation.run_policy_search_candidate import load_candidate_definition

_ROOT = Path(__file__).parents[2]


def _episode(
    termination: str = "success",
    *,
    scenario_id: str = "fixture",
    seed: int = 7,
    collisions: int = 0,
    near_misses: int = 0,
    time_ratio: float = 0.5,
    force_q95: float = 1.0,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "termination_reason": termination,
        "metrics": {
            "collisions": collisions,
            "near_misses": near_misses,
            "time_to_goal_ideal_ratio": time_ratio,
            "ped_force_q95": force_q95,
        },
        "outcome": {
            "route_complete": termination == "success",
            "collision_event": termination == "collision" or collisions > 0,
            "timeout_event": termination in {"truncated", "max_steps"},
        },
        "integrity": {"contradictions": []},
    }


def test_lexicographic_score_orders_validity_safety_completion_efficiency_and_comfort() -> None:
    all_valid = score_records([_episode(), _episode()], expected_episodes=2)
    partial = score_records([_episode(), _episode("error")], expected_episodes=2)
    assert tuple(all_valid["selection_tuple"][:1]) > tuple(partial["selection_tuple"][:1])

    collision = score_records(
        [_episode("collision", collisions=1), _episode()], expected_episodes=2
    )
    no_collision = score_records([_episode(), _episode("truncated")], expected_episodes=2)
    assert collision["selection_tuple"][:2] < no_collision["selection_tuple"][:2]

    near_miss = score_records([_episode(near_misses=1), _episode()], expected_episodes=2)
    no_near_miss = score_records([_episode(), _episode("truncated")], expected_episodes=2)
    assert near_miss["selection_tuple"][:3] < no_near_miss["selection_tuple"][:3]

    slower_completion = score_records(
        [_episode(time_ratio=0.9), _episode(time_ratio=0.9)], expected_episodes=2
    )
    fast_incomplete = score_records(
        [_episode(time_ratio=0.1), _episode("truncated")], expected_episodes=2
    )
    assert slower_completion["selection_tuple"][:4] > fast_incomplete["selection_tuple"][:4]

    fast_uncomfortable = score_records(
        [_episode(time_ratio=0.2, force_q95=100.0)], expected_episodes=1
    )
    slow_comfortable = score_records([_episode(time_ratio=0.8, force_q95=0.0)], expected_episodes=1)
    assert fast_uncomfortable["selection_tuple"][:5] > slow_comfortable["selection_tuple"][:5]
    assert fast_uncomfortable["selection_tuple"][5] < slow_comfortable["selection_tuple"][5]


def test_parameter_bounds_reject_invalid_types_and_inverted_ranges() -> None:
    assert (
        ParameterSpec.from_mapping(
            {"name": "max_linear_speed", "type": "float", "low": 1, "high": 3}
        ).low
        == 1.0
    )
    with pytest.raises(TypeError):
        ParameterSpec.from_mapping({"name": "enabled", "type": "float", "low": True, "high": 1})
    with pytest.raises(ValueError):
        ParameterSpec.from_mapping({"name": "samples", "type": "int", "low": 4, "high": 2})
    with pytest.raises(ValueError):
        ParameterSpec.from_mapping({"name": "nested.field", "type": "float", "low": 0, "high": 1})


def test_trial_selection_ignores_invalid_evaluations() -> None:
    valid = {"status": "complete", "trial_index": 1, "score": score_records([_episode()], 1)}
    invalid = {
        "status": "invalid",
        "trial_index": 0,
        "score": score_records([_episode("error")], 1),
    }
    assert select_best_trial([invalid, valid]) is valid


def test_score_rejects_missing_duplicate_and_unexpected_episode_identities() -> None:
    expected = [
        {"scenario_id": "crossing", "seed": 1},
        {"scenario_id": "head_on", "seed": 2},
    ]
    score = score_records(
        [
            _episode(scenario_id="crossing", seed=1),
            _episode(scenario_id="other", seed=2),
        ],
        expected_episodes=2,
        expected_identities=expected,
    )
    assert score["evaluation_complete"] is False
    assert score["valid_episode_count"] == 1
    assert score["invalid_reason_counts"]["unexpected_or_duplicate_episode_identity"] == 1
    assert score["missing_episode_identities"] == {"head_on::seed2": 1}


def test_missing_secondary_metrics_remain_unknown_instead_of_looking_optimal() -> None:
    row = _episode()
    row["metrics"].pop("ped_force_q95")
    row["metrics"].pop("near_misses")
    score = score_records([row], expected_episodes=1)
    assert score["near_miss_free_fraction"] is None
    assert score["comfort_ped_force_q95_mean"] is None
    assert score["near_miss_metric_missing_episode_count"] == 1
    assert score["comfort_metric_missing_valid_episode_count"] == 1
    assert score["sampler_objectives"][2] == -1.0e9


def test_malformed_canonical_outcome_and_integrity_are_invalid() -> None:
    row = _episode()
    row.pop("outcome")
    row.pop("integrity")
    score = score_records([row], expected_episodes=1)
    assert score["valid_episode_count"] == 0
    assert score["invalid_reason_counts"]["episode_outcome_missing_or_malformed"] == 1
    assert score["invalid_reason_counts"]["episode_integrity_missing_or_malformed"] == 1


def test_candidate_parameter_export_round_trips_through_policy_search_loader(
    tmp_path: Path,
) -> None:
    _entry, payload, _merged, _path = load_candidate_definition(
        _ROOT / "docs/context/policy_search/candidate_registry.yaml",
        "hybrid_rule_v3_fast_progress",
    )
    exported = _apply_params(
        payload,
        {"max_linear_speed": 2.6, "goal_progress_weight": 5.0},
        name="optimizer_export_fixture",
    )
    config_path = tmp_path / "optimizer_export.yaml"
    registry_path = tmp_path / "registry.yaml"
    config_path.write_text(yaml.safe_dump(exported, sort_keys=False), encoding="utf-8")
    registry_path.write_text(
        yaml.safe_dump(
            {
                "candidates": {
                    "optimizer_export_fixture": {"candidate_config_path": config_path.name}
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _entry, loaded_payload, loaded_config, loaded_path = load_candidate_definition(
        registry_path, "optimizer_export_fixture"
    )
    assert loaded_payload["algo"] == "hybrid_rule_local_planner"
    assert loaded_config["max_linear_speed"] == pytest.approx(2.6)
    assert loaded_config["goal_progress_weight"] == pytest.approx(5.0)
    assert loaded_path == config_path.resolve()


def test_config_uses_fixed_disjoint_train_and_heldout_scenario_seed_identities() -> None:
    config = load_optimizer_config(_ROOT / "configs/policy_search/planner_optimizer_issue9650.yaml")
    train = config.train.identities(config.train.load_scenarios())
    heldout = config.heldout.identities(config.heldout.load_scenarios())
    assert train == [
        {"scenario_id": "classic_cross_trap_high", "seed": 101},
        {"scenario_id": "classic_cross_trap_high", "seed": 102},
    ]
    assert heldout == [
        {"scenario_id": "classic_head_on_corridor_low", "seed": 111},
        {"scenario_id": "classic_head_on_corridor_low", "seed": 112},
    ]
    train_ids = {(row["scenario_id"], row["seed"]) for row in train}
    heldout_ids = {(row["scenario_id"], row["seed"]) for row in heldout}
    assert not (train_ids & heldout_ids)


def test_tiny_optimizer_smoke_persists_invalid_trials_and_export(tmp_path: Path) -> None:
    config_path = tmp_path / "smoke.yaml"
    config = yaml.safe_load(
        (_ROOT / "configs/policy_search/planner_optimizer_issue9650.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["run_id"] = "optimizer_unit_smoke"
    config["search"]["trials_per_method"] = 2
    config["candidate_registry"] = "docs/context/policy_search/candidate_registry.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "run"

    calls = 0

    def fake_evaluator(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if kwargs["tag"] == "random_trial_000":
            return {"records": [], "summary": {}, "jsonl_path": None}
        scenario_rows = kwargs["scenarios_or_path"]
        records = [
            _episode(scenario_id=str(scenario["name"]), seed=int(seed))
            for scenario in scenario_rows
            for seed in scenario["seeds"]
        ]
        return {"records": records, "summary": {}, "jsonl_path": None}

    manifest = run_planner_optimization(config_path, output, evaluator=fake_evaluator)

    assert manifest["schema"] == "planner_optimizer_run.v1"
    assert manifest["status"] == "complete"
    assert manifest["budget"]["evaluation_budget_per_method"] == 4
    assert manifest["suite_split"]["overlap"] == []
    assert manifest["heldout_comparison"]["selection_was_frozen_before_heldout"] is True
    assert manifest["method_results"]["random"]["invalid_trial_count"] == 1
    assert manifest["method_results"]["random"]["best_trial"]["trial_index"] == 1
    assert manifest["selected"]["method"] == "baseline"
    assert (
        calls == 6
    )  # train baseline + four trials + one held-out config (selected aliases baseline)
    assert (output / "trials.jsonl").read_text(encoding="utf-8").count("\n") == 4
    assert (output / "run_manifest.json").is_file()
    assert (output / "best_candidate.yaml").is_file()
    assert (output / "trials/random/trial_001/stage_summary.json").is_file()
    registry = yaml.safe_load((output / "candidate_registry.yaml").read_text(encoding="utf-8"))
    name = manifest["selected"]["candidate_name"]
    _entry, payload, effective, candidate_path = load_candidate_definition(
        output / "candidate_registry.yaml", name
    )
    assert payload["algo"] == "hybrid_rule_local_planner"
    assert candidate_path == (output / "best_candidate.yaml").resolve()
    assert effective["max_linear_speed"] == pytest.approx(3.0)
    assert registry["candidates"][name]["candidate_config_path"] == "best_candidate.yaml"
    assert (
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["run_id"]
        == "optimizer_unit_smoke"
    )
    with pytest.raises(FileExistsError):
        run_planner_optimization(config_path, output, evaluator=fake_evaluator)


def test_unexpected_evaluator_bug_propagates_and_preserves_partial_run(tmp_path: Path) -> None:
    config_path = tmp_path / "unexpected-error.yaml"
    config = yaml.safe_load(
        (_ROOT / "configs/policy_search/planner_optimizer_issue9650.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["run_id"] = "optimizer_unexpected_error"
    config["candidate_registry"] = "docs/context/policy_search/candidate_registry.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "partial-run"

    def broken_evaluator(**kwargs: Any) -> dict[str, Any]:
        raise AttributeError("unexpected evaluator programming error")

    with pytest.raises(AttributeError, match="unexpected evaluator programming error"):
        run_planner_optimization(config_path, output, evaluator=broken_evaluator)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert (output / ".run.lock").is_file()
