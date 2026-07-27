"""Tests for disjoint fit/evaluation splitting, overlap provenance, and null tests.

These cover the issue #3275 machinery in isolation. The module imports no
simulation/torch surfaces, so these tests run standalone.
"""

from __future__ import annotations

import pytest

from robot_sf.adversarial.disjoint_evaluation import (
    DisjointSplit,
    archive_sha256,
    classify_held_out_evidence,
    compute_overlap_provenance,
    disjoint_family_split,
    permutation_test_mean_difference,
    ranking_permutation_test,
    scenario_family_key,
    shuffled_outcome_null_test,
)


def _entry(family: str, archive_id: str, seed: int) -> dict:
    """Build a minimal archive entry with a string cluster_key family."""
    return {
        "archive_id": archive_id,
        "cluster_key": family,
        "candidate": {"scenario_seed": seed},
    }


def test_scenario_family_key_sources() -> None:
    """Family key prefers cluster_key, then failure/manifest, then a fallback."""
    assert scenario_family_key({"cluster_key": "goal_collision"}) == "goal_collision"

    dict_key = scenario_family_key(
        {"cluster_key": {"policy": "orca", "primary_failure": "collision"}}
    )
    # dict cluster keys serialize deterministically (sorted keys).
    assert dict_key == '{"policy":"orca","primary_failure":"collision"}'

    fallback = scenario_family_key(
        {"failure_attribution": {"primary_failure": "timeout"}, "source_manifest": "m.json"}
    )
    assert fallback == "failure=timeout|manifest=m.json"

    assert scenario_family_key({}) == "unknown_family"


def test_disjoint_family_split_partitions_by_family() -> None:
    """Two families split into non-overlapping fit/eval sides."""
    entries = [
        _entry("A", "a0", 1),
        _entry("A", "a1", 2),
        _entry("B", "b0", 3),
        _entry("B", "b1", 4),
    ]
    split = disjoint_family_split(entries, eval_fraction=0.5, seed=0)
    assert isinstance(split, DisjointSplit)
    assert split.is_disjoint_split is True
    assert set(split.fit_families).isdisjoint(split.eval_families)
    assert split.fit_entries and split.eval_entries
    # Every entry lands on exactly one side.
    assert len(split.fit_entries) + len(split.eval_entries) == len(entries)


def test_disjoint_family_split_is_deterministic() -> None:
    """Same seed yields the same family assignment."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2), _entry("C", "c0", 3)]
    first = disjoint_family_split(entries, seed=7)
    second = disjoint_family_split(entries, seed=7)
    assert first.fit_families == second.fit_families
    assert first.eval_families == second.eval_families


def test_disjoint_family_split_single_family_cannot_split() -> None:
    """A single family cannot form a disjoint split; all entries go to fit."""
    entries = [_entry("A", "a0", 1), _entry("A", "a1", 2)]
    split = disjoint_family_split(entries, seed=0)
    assert split.is_disjoint_split is False
    assert split.eval_entries == []
    assert len(split.fit_entries) == 2


def test_disjoint_family_split_rejects_degenerate_fraction() -> None:
    """eval_fraction must be strictly inside (0, 1)."""
    entries = [_entry("A", "a0", 1), _entry("B", "b0", 2)]
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="eval_fraction"):
            disjoint_family_split(entries, eval_fraction=bad)


def test_overlap_provenance_disjoint() -> None:
    """Disjoint families, seeds, and archive ids pass the disjointness check."""
    fit = [_entry("A", "a0", 1), _entry("A", "a1", 2)]
    eval_ = [_entry("B", "b0", 3), _entry("B", "b1", 4)]
    prov = compute_overlap_provenance(fit, eval_)
    assert prov["disjointness_checks_passed"] is True
    assert prov["scenario_family_overlap"] == []
    assert prov["seed_overlap"] == []
    assert prov["archive_id_overlap"] == []
    assert prov["disjointness_failure_reasons"] == []
    assert prov["split_policy"] == "disjoint_scenario_family"
    assert prov["fit_size"] == 2
    assert prov["eval_size"] == 2


def test_overlap_provenance_detects_family_and_id_overlap() -> None:
    """Shared family or archive id fails the disjointness check."""
    fit = [_entry("A", "a0", 1)]
    eval_ = [_entry("A", "a1", 2)]  # same family A
    prov = compute_overlap_provenance(fit, eval_)
    assert prov["disjointness_checks_passed"] is False
    assert prov["scenario_family_overlap"] == ["A"]

    shared_id = compute_overlap_provenance([_entry("A", "x", 1)], [_entry("B", "x", 2)])
    assert shared_id["disjointness_checks_passed"] is False
    assert shared_id["archive_id_overlap"] == ["x"]


def test_overlap_provenance_seed_overlap_fails_held_out_disjointness() -> None:
    """Seed overlap is recorded and fails the held-out disjointness gate."""
    fit = [_entry("A", "a0", 5)]
    eval_ = [_entry("B", "b0", 5)]  # shared seed 5, disjoint family/id
    prov = compute_overlap_provenance(fit, eval_)
    assert prov["seed_overlap"] == [5]
    assert prov["seed_overlap_count"] == 1
    assert prov["disjointness_checks_passed"] is False
    assert prov["disjointness_failure_reasons"] == ["seed_overlap"]
    assert prov["seed_overlap_invalidates_held_out_evidence"] is True


def test_overlap_provenance_empty_eval_not_disjoint() -> None:
    """An empty eval side cannot be a disjoint split."""
    prov = compute_overlap_provenance([_entry("A", "a0", 1)], [])
    assert prov["disjointness_checks_passed"] is False
    assert prov["disjointness_failure_reasons"] == ["empty_eval"]


def test_archive_sha256_is_deterministic_and_sensitive() -> None:
    """Hash is stable for equal data and changes when data changes."""
    a = [_entry("A", "a0", 1)]
    assert archive_sha256(a) == archive_sha256([_entry("A", "a0", 1)])
    assert archive_sha256(a) != archive_sha256([_entry("A", "a0", 2)])


def test_permutation_test_separates_signal_from_noise() -> None:
    """Clear group separation yields a small p-value; identical groups ~1.0."""
    separated = permutation_test_mean_difference(
        [10.0, 10.0, 10.0], [0.0, 0.0, 0.0], n_permutations=200, seed=1
    )
    assert separated["status"] == "complete"
    assert separated["observed_difference"] == 10.0
    assert separated["p_value"] < 0.3

    identical = permutation_test_mean_difference(
        [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], n_permutations=200, seed=1
    )
    assert identical["observed_difference"] == 0.0
    assert identical["p_value"] == 1.0


def test_permutation_test_empty_group_fail_closed() -> None:
    """An empty group returns a not_available status, not a misleading p-value."""
    result = permutation_test_mean_difference([], [1.0], n_permutations=10, seed=0)
    assert result["status"] == "not_available_empty_group"
    assert result["p_value"] is None


def test_permutation_test_rejects_zero_permutations() -> None:
    """n_permutations must be at least one."""
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_test_mean_difference([1.0], [2.0], n_permutations=0)


def test_shuffled_outcome_null_test_tags_test_name() -> None:
    """The shuffled-outcome null test delegates and tags the test name."""
    result = shuffled_outcome_null_test([3.0, 3.0], [0.0, 0.0], n_permutations=100, seed=2)
    assert result["test"] == "shuffled_outcome_label_permutation"
    assert result["status"] == "complete"


def test_ranking_permutation_test_detects_ranking_signal() -> None:
    """A descending ranking concentrates high outcomes in its top-k (small p)."""
    descending = [float(v) for v in range(20, 0, -1)]
    result = ranking_permutation_test(descending, selection_size=3, n_permutations=500, seed=3)
    assert result["status"] == "complete"
    assert result["observed_top_mean"] == 19.0
    assert result["p_value"] < 0.1


def test_ranking_permutation_test_flat_outcomes_no_signal() -> None:
    """Flat outcomes give the maximum p-value (no ranking signal)."""
    result = ranking_permutation_test(
        [5.0, 5.0, 5.0, 5.0], selection_size=2, n_permutations=100, seed=0
    )
    assert result["p_value"] == 1.0


def test_ranking_permutation_test_invalid_selection() -> None:
    """Invalid selection sizes fail closed."""
    assert ranking_permutation_test([1.0, 2.0], selection_size=0)["status"] == (
        "not_available_invalid_selection"
    )
    assert ranking_permutation_test([1.0, 2.0], selection_size=5)["status"] == (
        "not_available_invalid_selection"
    )
    assert ranking_permutation_test([], selection_size=1)["status"] == (
        "not_available_invalid_selection"
    )


def test_classify_held_out_evidence_fail_closed() -> None:
    """Held-out evidence is eligible only when every precondition holds."""
    assert (
        classify_held_out_evidence(
            disjointness_checks_passed=False,
            independent_outcomes_available=True,
            certification_available=True,
            null_tests_reject_null=True,
        )
        == "not_available_no_disjoint_split"
    )
    assert (
        classify_held_out_evidence(
            disjointness_checks_passed=True,
            independent_outcomes_available=False,
            certification_available=True,
            null_tests_reject_null=True,
        )
        == "not_available_requires_independent_planner_outcomes"
    )
    assert (
        classify_held_out_evidence(
            disjointness_checks_passed=True,
            independent_outcomes_available=True,
            certification_available=False,
            null_tests_reject_null=True,
        )
        == "not_available_requires_candidate_certification"
    )
    assert (
        classify_held_out_evidence(
            disjointness_checks_passed=True,
            independent_outcomes_available=True,
            certification_available=True,
            null_tests_reject_null=False,
        )
        == "not_available_null_tests_not_rejected"
    )
    assert (
        classify_held_out_evidence(
            disjointness_checks_passed=True,
            independent_outcomes_available=True,
            certification_available=True,
            null_tests_reject_null=True,
        )
        == "eligible_held_out_diagnostic"
    )


# --- Issue #3275 frozen contract primitive tests ------------------------------

from robot_sf.adversarial.disjoint_evaluation import (  # noqa: E402
    FAMILY_INVARIANT_FEATURE_NAMES,
    ISSUE_3275_DECISION_VOCABULARY,
    ArmAssignment,
    assign_arms_disjoint_by_candidate,
    binary_yield_min_detectable_difference,
    classify_issue_3275_decision,
    family_invariant_distance,
    family_invariant_features,
    fisher_exact_two_sided,
    fisher_exact_two_sided_table,
    frozen_held_out_family_split,
)


def test_family_invariant_features_are_robot_path_relative() -> None:
    """Lateral/longitudinal features are computed relative to the robot path."""
    candidate = {
        "start": {"x": 2.0, "y": 4.0},
        "goal": {"x": 8.0, "y": 4.0},
        "spawn_time_s": 1.0,
        "pedestrian_speed_mps": 1.2,
        "pedestrian_delay_s": 0.5,
    }
    # Robot walks straight along x from (0,0) to (10,0): path length 10.
    feats = family_invariant_features(candidate, (0.0, 0.0), (10.0, 0.0))
    assert set(feats) == set(FAMILY_INVARIANT_FEATURE_NAMES)
    # Pedestrian spawns at (2,4): longitudinal fraction 0.2, lateral 4/10 = 0.4.
    assert feats["longitudinal_spawn_fraction"] == pytest.approx(0.2)
    assert feats["lateral_spawn_path_fraction"] == pytest.approx(0.4)
    assert feats["pedestrian_speed_mps"] == pytest.approx(1.2)


def test_family_invariant_features_keep_longitudinal_positions_outside_path() -> None:
    """Longitudinal fractions retain before-start and beyond-goal positions."""
    candidate = {
        "start": {"x": -2.0, "y": 0.0},
        "goal": {"x": 13.0, "y": 0.0},
        "spawn_time_s": 1.0,
        "pedestrian_speed_mps": 1.2,
        "pedestrian_delay_s": 0.5,
    }

    feats = family_invariant_features(candidate, (0.0, 0.0), (10.0, 0.0))

    assert feats["longitudinal_spawn_fraction"] == pytest.approx(-0.2)
    assert feats["longitudinal_goal_fraction"] == pytest.approx(1.3)


def test_family_invariant_features_coincident_robot_path_fails_closed() -> None:
    """A zero-length robot path has no family-invariant projection."""
    candidate = {
        "start": {"x": 1.0, "y": 1.0},
        "goal": {"x": 2.0, "y": 2.0},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
    }
    with pytest.raises(ValueError, match="path length is zero"):
        family_invariant_features(candidate, (5.0, 5.0), (5.0, 5.0))


def test_family_invariant_features_same_meaning_both_families() -> None:
    """Same RELATIVE geometry yields the same features under either family's robot path."""
    # Identical relative geometry: candidate offset (2,4)->(8,4) from the robot
    # start, on a 10-unit eastward path. Path A lives at (0,0); path B is the same
    # path translated to (5,5), with the candidate translated identically.
    cand_a = {
        "start": {"x": 2.0, "y": 4.0},
        "goal": {"x": 8.0, "y": 4.0},
        "spawn_time_s": 1.0,
        "pedestrian_speed_mps": 1.2,
        "pedestrian_delay_s": 0.5,
    }
    cand_b = {
        "start": {"x": 7.0, "y": 9.0},
        "goal": {"x": 13.0, "y": 9.0},
        "spawn_time_s": 1.0,
        "pedestrian_speed_mps": 1.2,
        "pedestrian_delay_s": 0.5,
    }
    a = family_invariant_features(cand_a, (0.0, 0.0), (10.0, 0.0))
    b = family_invariant_features(cand_b, (5.0, 5.0), (15.0, 5.0))
    assert a == b


def test_family_invariant_distance_zero_for_identical_geometry() -> None:
    """Identical relative geometry yields zero family-invariant distance."""
    cand = {
        "start": {"x": 2.0, "y": 4.0},
        "goal": {"x": 8.0, "y": 4.0},
        "spawn_time_s": 1.0,
        "pedestrian_speed_mps": 1.2,
        "pedestrian_delay_s": 0.5,
    }
    dist = family_invariant_distance(cand, cand, (0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    assert dist == pytest.approx(0.0)


def test_frozen_held_out_family_split_is_deterministic_and_disjoint() -> None:
    """The frozen split assigns families explicitly and keeps them disjoint."""
    entries = [
        {"archive_id": "g0", "scenario_family": "classic_group_crossing_medium"},
        {"archive_id": "g1", "scenario_family": "classic_group_crossing_medium"},
        {"archive_id": "t0", "scenario_family": "classic_cross_trap_medium"},
    ]
    split = frozen_held_out_family_split(
        entries,
        fit_family="classic_group_crossing_medium",
        eval_family="classic_cross_trap_medium",
    )
    assert split.is_disjoint_split is True
    assert [e["archive_id"] for e in split.fit_entries] == ["g0", "g1"]
    assert [e["archive_id"] for e in split.eval_entries] == ["t0"]


def test_frozen_held_out_family_split_rejects_same_family() -> None:
    """Fit and evaluation families must differ for a held-out split."""
    with pytest.raises(ValueError, match="must differ"):
        frozen_held_out_family_split(
            [],
            fit_family="classic_group_crossing_medium",
            eval_family="classic_group_crossing_medium",
        )


def test_assign_arms_disjoint_by_candidate_has_no_overlap() -> None:
    """The frozen arm-overlap policy never assigns a candidate to both arms."""
    pool = [f"c{i}" for i in range(10)]
    ranked = list(reversed(pool))  # rank order differs from pool order
    arms = assign_arms_disjoint_by_candidate(ranked, pool, budget_per_arm=3, rng_seed=7)
    assert isinstance(arms, ArmAssignment)
    assert arms.policy == "disjoint_by_candidate"
    assert len(arms.proposal_ids) == 3
    assert len(arms.random_ids) == 3
    assert arms.overlap_ids == []
    assert set(arms.proposal_ids).isdisjoint(arms.random_ids)
    # Proposal arm takes the top of the ranking.
    assert arms.proposal_ids == ranked[:3]


def test_assign_arms_disjoint_rejects_negative_budget() -> None:
    """Negative budgets are a contract violation."""
    with pytest.raises(ValueError, match="budget_per_arm"):
        assign_arms_disjoint_by_candidate(["a"], ["a"], budget_per_arm=-1, rng_seed=0)


def test_assign_arms_disjoint_rejects_pool_too_small_for_both_arms() -> None:
    """Arm assignment cannot silently return unequal candidate budgets."""
    pool = ["pool_0", "pool_1", "pool_2"]

    with pytest.raises(ValueError, match="two disjoint arm budgets"):
        assign_arms_disjoint_by_candidate(pool, pool, budget_per_arm=2, rng_seed=0)


def test_assign_arms_disjoint_rejects_non_pool_or_partial_rank_ids() -> None:
    """Arm assignment fails closed unless rank IDs are a full pool-ID permutation."""
    with pytest.raises(ValueError, match="absent from pool_ids"):
        assign_arms_disjoint_by_candidate(
            ["candidate-object"], ["pool_0"], budget_per_arm=1, rng_seed=0
        )
    with pytest.raises(ValueError, match="omits pool IDs"):
        assign_arms_disjoint_by_candidate(
            ["pool_0"], ["pool_0", "pool_1"], budget_per_arm=1, rng_seed=0
        )


def test_fisher_exact_extremes_and_symmetry() -> None:
    """Disjoint counts reject; identical counts do not."""
    assert fisher_exact_two_sided(0, 4, 4) <= 0.05
    assert fisher_exact_two_sided(2, 2, 4) > 0.05
    # Symmetric in its two arms.
    assert fisher_exact_two_sided_table(3, 1, 0, 4) == pytest.approx(
        fisher_exact_two_sided_table(0, 4, 3, 1)
    )


def test_binary_yield_min_detectable_matches_recorded_power_table() -> None:
    """The recorded power table values are reproducible from the helper."""
    assert binary_yield_min_detectable_difference(10, alpha=0.05) == pytest.approx(0.5, abs=1e-6)
    assert binary_yield_min_detectable_difference(12, alpha=0.05) == pytest.approx(0.417, abs=1e-3)
    assert binary_yield_min_detectable_difference(20, alpha=0.05) == pytest.approx(0.25, abs=1e-6)


def test_classify_issue_3275_decision_vocabulary_is_frozen() -> None:
    """The decision vocabulary is exactly continue|stop|inconclusive."""
    assert ISSUE_3275_DECISION_VOCABULARY == ("continue", "stop", "inconclusive")


def test_classify_issue_3275_decision_inconclusive_without_outcomes() -> None:
    """No independent outcomes -> inconclusive (never continue/stop)."""
    decision = classify_issue_3275_decision(
        proposal_yield=1.0,
        random_yield=0.0,
        minimally_important=0.2,
        null_rejected=True,
        powered=True,
        independent_available=False,
    )
    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "independent_outcomes_unavailable_or_fail_closed"


def test_classify_issue_3275_decision_stop_when_random_better() -> None:
    """A powered, significant random-favoring result stops the proposal lane."""
    decision = classify_issue_3275_decision(
        proposal_yield=0.1,
        random_yield=0.5,
        minimally_important=0.2,
        null_rejected=True,
        powered=True,
        independent_available=True,
    )
    assert decision["status"] == "stop"
    assert decision["reason"] == "proposal_does_not_beat_random"


def test_classify_issue_3275_decision_non_significant_random_better_is_inconclusive() -> None:
    """A non-rejected null takes precedence over a random-favoring point estimate."""
    decision = classify_issue_3275_decision(
        proposal_yield=0.1,
        random_yield=0.5,
        minimally_important=0.2,
        null_rejected=False,
        powered=True,
        independent_available=True,
    )

    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "null_not_rejected"


def test_classify_issue_3275_decision_underpowered_random_better_is_inconclusive() -> None:
    """Underpowered evidence is inconclusive even when random has the better yield."""
    decision = classify_issue_3275_decision(
        proposal_yield=0.1,
        random_yield=0.5,
        minimally_important=0.2,
        null_rejected=False,
        powered=False,
        independent_available=True,
    )
    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "underpowered_for_minimally_important_effect"


def test_classify_issue_3275_decision_inconclusive_when_underpowered() -> None:
    """Positive delta but underpowered -> inconclusive."""
    decision = classify_issue_3275_decision(
        proposal_yield=1.0,
        random_yield=0.0,
        minimally_important=0.2,
        null_rejected=True,
        powered=False,
        independent_available=True,
    )
    assert decision["status"] == "inconclusive"
    assert decision["reason"] == "underpowered_for_minimally_important_effect"


def test_classify_issue_3275_decision_continue_when_powered_and_significant() -> None:
    """Powered, significant, beyond-minimally-important delta -> continue."""
    decision = classify_issue_3275_decision(
        proposal_yield=0.9,
        random_yield=0.1,
        minimally_important=0.2,
        null_rejected=True,
        powered=True,
        independent_available=True,
    )
    assert decision["status"] == "continue"
