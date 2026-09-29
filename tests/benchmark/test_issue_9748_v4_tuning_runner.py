"""Focused guards for the #9748 deterministic v4 development search."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from scripts.benchmark import run_issue_9748_v4_tuning as runner
from scripts.validation.check_issue_9748_dev_split import EXPECTED_PLANNER_CONFIGS


def test_search_is_fixed_baseline_plus_one_factor_trials() -> None:
    trials = runner._load_search()
    assert len(trials) == 6
    assert trials[0] == {"id": "baseline", "params": {}}
    assert all(len(trial["params"]) == 1 for trial in trials[1:])
    assert {next(iter(trial["params"])) for trial in trials[1:]} == runner.TUNABLE


@pytest.mark.parametrize("seed", [101, 111, 140, 1031])
def test_release_and_non_dev_seeds_are_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="frozen development set"):
        runner._choose([seed], list(range(1001, 1031)), label="seed")


def test_every_trial_resolves_to_v4_on_all_four_dev_identities() -> None:
    campaign = load_campaign_config(runner.CAMPAIGN_PATH)
    scenarios = _load_campaign_scenarios(campaign)
    for path in EXPECTED_PLANNER_CONFIGS.values():
        for trial in runner._load_search():
            manifest = runner._candidate_manifest(path, trial["params"])
            hashes = {
                runner._effective_config_hash(
                    manifest=manifest, candidate_path=path, scenario=scenario
                )
                for scenario in scenarios
            }
            assert len(hashes) == 1  # release-keyed overrides never fire on dev identities


def test_ranking_excludes_degraded_and_prioritizes_collisions() -> None:
    def group(trial_id: str, *, collisions: int, completions: int, degraded: int = 0) -> dict:
        return {
            "candidate": "v4",
            "trial_id": trial_id,
            "seeds": [1001],
            "counts": {
                "written": 1,
                "errors": 0,
                "degraded": degraded,
                "collisions": collisions,
                "completions": completions,
            },
            "completed_times": [10.0] if completions else [],
        }

    groups = [
        group("fast", collisions=1, completions=1),
        group("safe", collisions=0, completions=0),
        group("fallback", collisions=0, completions=1, degraded=1),
    ]
    assert [item["trial_id"] for item in runner._rank_trials(groups, ["v4"])["v4"]] == [
        "safe",
        "fast",
        "fallback",
    ]
