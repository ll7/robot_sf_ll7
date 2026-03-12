"""Regression tests for predictive success campaign artifact naming."""

from __future__ import annotations

from scripts.validation import run_predictive_success_campaign as campaign


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
