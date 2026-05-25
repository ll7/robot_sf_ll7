"""Regression tests for predictive success campaign artifact naming."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.validation import run_predictive_success_campaign as campaign

if TYPE_CHECKING:
    from pathlib import Path


def test_checkpoint_token_is_path_sensitive() -> None:
    """Checkpoint token should differ for same filename under different directories."""
    a = campaign._checkpoint_token("output/a/predictive_model.pt")
    b = campaign._checkpoint_token("output/b/predictive_model.pt")
    assert a != b
    assert a.startswith("predictive_model_")
    assert b.startswith("predictive_model_")


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
    checkpoint.write_text("stub", encoding="utf-8")
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
    checkpoint.write_text("stub", encoding="utf-8")
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
