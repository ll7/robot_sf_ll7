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
