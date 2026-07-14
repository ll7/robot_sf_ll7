"""Tests for the #5446 -> #5447 trace re-export list builder.

Checks the join logic in isolation with a small synthetic candidate manifest
(one seed_flip and one planner_upset candidate, one selected and one not) and
a matching mining-rows table, since the real bundle's candidate manifest is
too large to fixture directly.
"""

from __future__ import annotations

from scripts.analysis.build_issue_5446_trace_reexport_list import build_reexport_list


def _mining_row(scenario_id: str, algo: str, seed: int) -> dict:
    return {
        "episode_id": f"{scenario_id}--{seed}--cfg",
        "scenario_id": scenario_id,
        "algo": algo,
        "seed": seed,
    }


def test_selected_seed_flip_candidate_expands_to_per_seed_tuples() -> None:
    """A selected seed_flip candidate's raw_seed_outcomes become tuples with real episode ids."""
    candidate_manifest = {
        "summary": {"n_candidates": 1},
        "candidates": [
            {
                "candidate_id": "seed_flip::s1::plannerA",
                "archetype": "seed_flip",
                "scenario_id": "s1",
                "planner": "plannerA",
                "selected": True,
                "reproducibility": {"raw_seed_outcomes": {"111": 1, "112": 0}},
            }
        ],
    }
    mining_rows = [_mining_row("s1", "plannerA", 111), _mining_row("s1", "plannerA", 112)]

    manifest = build_reexport_list(candidate_manifest, mining_rows)

    assert manifest["n_tuples"] == 2
    assert manifest["n_episode_id_found"] == 2
    tuples = {(t["scenario_id"], t["planner"], t["seed"]): t for t in manifest["tuples"]}
    assert tuples[("s1", "plannerA", "111")]["episode_id"] == "s1--111--cfg"
    assert tuples[("s1", "plannerA", "111")]["requested_by_candidates"] == [
        "seed_flip::s1::plannerA"
    ]


def test_unselected_candidate_is_not_expanded() -> None:
    """Only Pareto-selected candidates request a trace re-export."""
    candidate_manifest = {
        "summary": {},
        "candidates": [
            {
                "candidate_id": "seed_flip::s1::plannerA",
                "archetype": "seed_flip",
                "scenario_id": "s1",
                "planner": "plannerA",
                "selected": False,
                "reproducibility": {"raw_seed_outcomes": {"111": 1}},
            }
        ],
    }
    manifest = build_reexport_list(candidate_manifest, [_mining_row("s1", "plannerA", 111)])
    assert manifest["n_tuples"] == 0


def test_planner_upset_candidate_expands_both_planners() -> None:
    """A selected planner_upset candidate expands raw_paired_outcomes for both planners."""
    candidate_manifest = {
        "summary": {},
        "candidates": [
            {
                "candidate_id": "planner_upset::s1::weak>strong",
                "archetype": "planner_upset",
                "scenario_id": "s1",
                "planner": "weak",
                "selected": True,
                "upset_outcome": {
                    "raw_paired_outcomes": {
                        "weak": {"111": 1},
                        "strong": {"111": 0},
                    }
                },
            }
        ],
    }
    mining_rows = [_mining_row("s1", "weak", 111), _mining_row("s1", "strong", 111)]
    manifest = build_reexport_list(candidate_manifest, mining_rows)
    assert manifest["n_tuples"] == 2
    planners = {t["planner"] for t in manifest["tuples"]}
    assert planners == {"weak", "strong"}


def test_missing_episode_id_is_reported_not_fabricated() -> None:
    """A seed with no matching mining row reports not_found rather than guessing an id."""
    candidate_manifest = {
        "summary": {},
        "candidates": [
            {
                "candidate_id": "seed_flip::s1::plannerA",
                "archetype": "seed_flip",
                "scenario_id": "s1",
                "planner": "plannerA",
                "selected": True,
                "reproducibility": {"raw_seed_outcomes": {"999": 1}},
            }
        ],
    }
    manifest = build_reexport_list(candidate_manifest, [])
    assert manifest["n_tuples"] == 1
    assert manifest["tuples"][0]["episode_id"] is None
    assert manifest["tuples"][0]["episode_id_status"] == "not_found_in_mining_rows"
