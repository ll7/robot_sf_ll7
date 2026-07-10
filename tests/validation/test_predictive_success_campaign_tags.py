"""Regression tests for predictive success campaign artifact naming."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.validation import predictive_eval_common
from scripts.validation import run_predictive_success_campaign as campaign


def test_checkpoint_token_is_path_sensitive() -> None:
    """Checkpoint token should differ for same filename under different directories."""
    a = campaign._checkpoint_token("output/a/predictive_model.pt")
    b = campaign._checkpoint_token("output/b/predictive_model.pt")
    assert a != b
    assert a.startswith("predictive_model_")
    assert b.startswith("predictive_model_")


def test_load_planner_variants_rejects_duplicate_names(tmp_path: Path) -> None:
    """Authority grids should not silently collapse duplicate variant names."""
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "variants": [
                    {"name": "baseline", "params": {}},
                    {"name": "baseline", "params": {"max_angular_speed": 1.8}},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate variant name"):
        campaign._load_planner_variants(grid)


def test_main_check_only_writes_manifest_summary(monkeypatch, tmp_path: Path) -> None:
    """Check-only mode validates toy manifests without checkpoint evaluation."""
    matrix = tmp_path / "matrix.yaml"
    map_path = tmp_path / "maps" / "open.svg"
    map_path.parent.mkdir()
    map_path.write_text("<svg />", encoding="utf-8")
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "classic_cross_trap_high",
                        "map_file": "maps/open.svg",
                        "seeds": [1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    hard_manifest = tmp_path / "hard.yaml"
    hard_manifest.write_text(
        yaml.safe_dump({"classic_cross_trap_high": [101, 102]}),
        encoding="utf-8",
    )
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "variants": [
                    {"name": "baseline", "params": {}},
                    {"name": "high_angular", "params": {"max_angular_speed": 1.8}},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        campaign,
        "parse_args",
        lambda: campaign.argparse.Namespace(
            checkpoints=None,
            check_only=True,
            scenario_matrix=matrix,
            hard_seed_manifest=hard_manifest,
            planner_grid=grid,
            output_dir=output_dir,
        ),
    )

    assert campaign.main() == 0
    summary = json.loads((output_dir / "manifest_check_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "checked_no_evaluation_run"
    assert summary["matched_hard_scenarios"] == 1
    assert summary["planner_variants"] == ["baseline", "high_angular"]
    assert "no checkpoint evaluation" in summary["scope_note"]


def test_rank_key_prefers_global_success_before_clearance_when_hard_tied() -> None:
    """Campaign ranking should not pick the safest failure over the more successful variant."""
    hard_a = campaign.EvalResult(
        checkpoint="a",
        variant="a",
        suite="hard",
        episodes=7,
        success_rate=0.0,
        success_ci_low=0.0,
        success_ci_high=0.0,
        mean_min_distance=3.0,
        mean_avg_speed=0.2,
        jsonl_path="a.jsonl",
    )
    hard_b = campaign.EvalResult(
        checkpoint="b",
        variant="b",
        suite="hard",
        episodes=7,
        success_rate=0.0,
        success_ci_low=0.0,
        success_ci_high=0.0,
        mean_min_distance=2.0,
        mean_avg_speed=0.2,
        jsonl_path="b.jsonl",
    )
    global_a = campaign.EvalResult(
        checkpoint="a",
        variant="a",
        suite="global",
        episodes=66,
        success_rate=0.05,
        success_ci_low=0.0,
        success_ci_high=0.1,
        mean_min_distance=3.0,
        mean_avg_speed=0.2,
        jsonl_path="a.jsonl",
    )
    global_b = campaign.EvalResult(
        checkpoint="b",
        variant="b",
        suite="global",
        episodes=66,
        success_rate=0.08,
        success_ci_low=0.0,
        success_ci_high=0.1,
        mean_min_distance=2.0,
        mean_avg_speed=0.2,
        jsonl_path="b.jsonl",
    )
    assert campaign._rank_key(hard_b, global_b) > campaign._rank_key(hard_a, global_a)


def test_closed_loop_gate_rejects_success_neutral_clearance_candidate() -> None:
    """#1856 gate should reject candidates that only improve clearance signals."""
    ranked = [
        {
            "variant": "clearance_only",
            "hard": {"success_rate": 0.0},
            "global": {"success_rate": 0.10, "mean_min_distance": 2.4},
        },
        {
            "variant": "baseline_like",
            "hard": {"success_rate": 0.0},
            "global": {"success_rate": 0.10, "mean_min_distance": 2.1},
        },
    ]

    gate = campaign._closed_loop_gate_result(
        ranked,
        baseline_variant="baseline_like",
        min_global_success_delta=0.02,
        min_hard_success_delta=0.0,
        max_min_distance_regression=0.10,
    )

    assert gate is not None
    assert gate["passed"] is False
    assert "global_success_delta_below_gate" in str(gate["reason"])


def test_closed_loop_gate_allows_success_gain_with_bounded_clearance_regression() -> None:
    """#1856 gate should permit a candidate that improves closed-loop success."""
    ranked = [
        {
            "variant": "phase_coupled_sequence_gate",
            "hard": {"success_rate": 0.1},
            "global": {"success_rate": 0.14, "mean_min_distance": 2.02},
        },
        {
            "variant": "baseline_like",
            "hard": {"success_rate": 0.0},
            "global": {"success_rate": 0.10, "mean_min_distance": 2.1},
        },
    ]

    gate = campaign._closed_loop_gate_result(
        ranked,
        baseline_variant="baseline_like",
        min_global_success_delta=0.02,
        min_hard_success_delta=0.0,
        max_min_distance_regression=0.10,
    )

    assert gate is not None
    assert gate["passed"] is True
    assert gate["deltas"]["global_success"] == pytest.approx(0.04)
    assert gate["deltas"]["global_mean_min_distance"] == pytest.approx(-0.08)


def test_main_writes_gate_failure_summary_for_missing_baseline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A missing baseline variant should fail closed without crashing report rendering."""
    output_dir = tmp_path / "campaign"

    monkeypatch.setattr(
        campaign,
        "_load_planner_variants",
        lambda _path: [{"name": "candidate", "params": {}}],
    )
    monkeypatch.setattr(campaign, "load_seed_manifest", lambda _path: {"scenario": [1]})
    monkeypatch.setattr(
        campaign,
        "make_subset_scenarios",
        lambda _matrix, _manifest: [{"name": "scenario"}],
    )
    monkeypatch.setattr(
        campaign,
        "_run_campaign_results",
        lambda **_kwargs: [
            {
                "checkpoint": "predictive_model.pt",
                "variant": "candidate",
                "config": {},
                "hard": {
                    "success_rate": 0.1,
                    "success_ci_low": 0.0,
                    "success_ci_high": 0.2,
                    "mean_min_distance": 1.0,
                    "mean_avg_speed": 0.3,
                },
                "global": {
                    "success_rate": 0.2,
                    "success_ci_low": 0.1,
                    "success_ci_high": 0.3,
                    "mean_min_distance": 1.2,
                    "mean_avg_speed": 0.4,
                },
                "ranking_key": [0.1, 0.2, 1.0, 1.2, 0.4],
            }
        ],
    )
    monkeypatch.setattr(
        campaign,
        "parse_args",
        lambda: campaign.argparse.Namespace(
            checkpoints=["predictive_model.pt"],
            scenario_matrix=tmp_path / "scenarios.yaml",
            hard_seed_manifest=tmp_path / "hard.yaml",
            planner_grid=tmp_path / "grid.yaml",
            horizon=12,
            dt=0.1,
            workers=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
            closed_loop_gate_baseline_variant="missing_baseline",
            closed_loop_gate_min_global_success_delta=0.02,
            closed_loop_gate_min_hard_success_delta=0.0,
            closed_loop_gate_max_min_distance_regression=0.10,
            output_dir=output_dir,
        ),
    )

    assert campaign.main() == 2
    summary = json.loads((output_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed_closed_loop_gate"
    assert summary["closed_loop_gate"]["reason"] == "baseline_variant_missing"
    report = (output_dir / "campaign_report.md").read_text(encoding="utf-8")
    assert "baseline_variant_missing" in report
    assert "not_available" in report


def test_issue_1856_coupling_grid_touches_planner_objective_not_checkpoint() -> None:
    """The #1856 grid should revise planner coupling, not pick a new model row."""
    path = (
        Path(__file__).parents[2]
        / "configs"
        / "benchmarks"
        / "predictive_sweep_planner_grid_v2_coupling_gate.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = {item["name"]: item["params"] for item in payload["variants"]}

    assert set(variants) == {"baseline_like", "phase_coupled_sequence_gate"}
    revised = variants["phase_coupled_sequence_gate"]
    assert revised["predictive_sequence_search_enabled"] is True
    assert revised["predictive_phase_logic_enabled"] is True
    assert revised["predictive_progress_risk_weight"] > variants["baseline_like"].get(
        "predictive_progress_risk_weight",
        0.0,
    )
    assert "predictive_checkpoint_path" not in revised
    assert "predictive_model_id" not in revised


def test_default_hard_seed_manifest_matches_default_scenario_matrix() -> None:
    """The success campaign defaults should produce a non-empty hard-case subset."""
    root = Path(__file__).parents[2]
    manifest = predictive_eval_common.load_seed_manifest(
        root / "configs" / "benchmarks" / "predictive_hard_seeds_v1.yaml"
    )
    scenarios = predictive_eval_common.make_subset_scenarios(
        root / "configs" / "scenarios" / "classic_interactions.yaml",
        manifest,
    )

    scenario_ids = {scenario["name"] for scenario in scenarios}
    assert scenario_ids == {
        "classic_cross_trap_high",
        "classic_cross_trap_medium",
        "classic_cross_trap_low",
        "classic_group_crossing_high",
    }
    assert sum(len(scenario["seeds"]) for scenario in scenarios) == 7


def test_run_eval_fails_when_jsonl_artifact_missing(monkeypatch, tmp_path: Path) -> None:
    """Campaign evaluation should fail closed when no episode JSONL is produced."""

    def _fake_run_map_batch(*_args, **_kwargs) -> None:
        """Simulate a runner failure that returns without materializing the JSONL."""
        return None

    monkeypatch.setattr(campaign, "run_map_batch", _fake_run_map_batch)
    args = campaign.argparse.Namespace(
        horizon=12,
        dt=0.1,
        workers=1,
        bootstrap_samples=10,
        bootstrap_seed=1,
    )

    with pytest.raises(
        campaign.MissingCampaignArtifactError,
        match="expected JSONL artifact was not usable",
    ):
        campaign._run_eval(
            scenarios_or_path=[],
            suite_name="hard",
            checkpoint=str(tmp_path / "predictive_model.pt"),
            variant_name="baseline_like",
            algo_cfg={},
            args=args,
            output_dir=tmp_path,
        )


def test_main_writes_failure_summary_for_missing_jsonl(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Campaign main should leave a structured failure summary before exiting nonzero."""

    checkpoint = tmp_path / "predictive_model.pt"
    torch.save({}, checkpoint)
    output_dir = tmp_path / "campaign"

    def _fake_run_map_batch(*_args, **_kwargs) -> None:
        """Simulate a runner failure that returns without materializing the JSONL."""
        return None

    monkeypatch.setattr(campaign, "run_map_batch", _fake_run_map_batch)
    monkeypatch.setattr(
        campaign,
        "_load_planner_variants",
        lambda _path: [{"name": "baseline_like", "params": {}}],
    )
    monkeypatch.setattr(campaign, "load_seed_manifest", lambda _path: {"scenario": [1]})
    monkeypatch.setattr(
        campaign,
        "make_subset_scenarios",
        lambda _matrix, _manifest: [{"name": "scenario"}],
    )
    monkeypatch.setattr(
        campaign,
        "parse_args",
        lambda: campaign.argparse.Namespace(
            checkpoints=[str(checkpoint)],
            scenario_matrix=tmp_path / "scenarios.yaml",
            hard_seed_manifest=tmp_path / "hard.yaml",
            planner_grid=tmp_path / "grid.yaml",
            horizon=12,
            dt=0.1,
            workers=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
            output_dir=output_dir,
        ),
    )

    assert campaign.main() == 2
    summary = json.loads((output_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure"]["type"] == "MissingCampaignArtifactError"
    assert summary["failure"]["reason"] == "missing_jsonl"


def test_main_writes_failure_summary_for_malformed_jsonl(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Malformed runner JSONL should still leave a structured failure summary."""

    checkpoint = tmp_path / "predictive_model.pt"
    torch.save({}, checkpoint)
    output_dir = tmp_path / "campaign"

    def _fake_run_map_batch(_scenarios_or_path, jsonl_path: Path, **_kwargs) -> None:
        """Simulate a runner that leaves malformed JSONL before failing downstream."""
        jsonl_path.write_text("{not-json}\n", encoding="utf-8")

    monkeypatch.setattr(campaign, "run_map_batch", _fake_run_map_batch)
    monkeypatch.setattr(
        campaign,
        "_load_planner_variants",
        lambda _path: [{"name": "baseline_like", "params": {}}],
    )
    monkeypatch.setattr(campaign, "load_seed_manifest", lambda _path: {"scenario": [1]})
    monkeypatch.setattr(
        campaign,
        "make_subset_scenarios",
        lambda _matrix, _manifest: [{"name": "scenario"}],
    )
    monkeypatch.setattr(
        campaign,
        "parse_args",
        lambda: campaign.argparse.Namespace(
            checkpoints=[str(checkpoint)],
            scenario_matrix=tmp_path / "scenarios.yaml",
            hard_seed_manifest=tmp_path / "hard.yaml",
            planner_grid=tmp_path / "grid.yaml",
            horizon=12,
            dt=0.1,
            workers=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
            output_dir=output_dir,
        ),
    )

    assert campaign.main() == 2
    summary = json.loads((output_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure"]["type"] == "SystemExit"
    assert "Malformed JSON" in summary["failure"]["message"]
