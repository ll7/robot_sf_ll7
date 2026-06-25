"""Tests for the constraints-first scoring layer (issue #3572)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.constraints_first_scoring import (
    CONSTRAINTS_FIRST_SCHEMA,
    AdmissibilityGates,
    collision_upper_confidence_bound,
    constraints_first_planner_summary,
    is_episode_admissible,
    ranking_inversion,
    survivorship_aware_metric,
)


def test_collision_ucb_reproduces_rule_of_three() -> None:
    """Zero observed collisions must still yield a non-trivial upper bound (~3/N)."""
    ub = collision_upper_confidence_bound(0, 100)

    assert ub == pytest.approx(1 - 0.05 ** (1 / 100))
    assert ub == pytest.approx(0.03, abs=0.005)


def test_collision_ucb_is_one_when_all_collide() -> None:
    """If every episode collides the upper bound is 1.0."""
    assert collision_upper_confidence_bound(5, 5) == 1.0


def test_collision_ucb_decreases_with_more_episodes() -> None:
    """The same zero-collision rate must tighten as N grows."""
    assert collision_upper_confidence_bound(0, 1000) < collision_upper_confidence_bound(0, 10)


@pytest.mark.parametrize(
    ("n_events", "n_episodes", "confidence"),
    [(-1, 10, 0.95), (5, 4, 0.95), (0, 0, 0.95), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_collision_ucb_rejects_invalid_inputs(
    n_events: int, n_episodes: int, confidence: float
) -> None:
    """Out-of-range counts or confidence must fail closed."""
    with pytest.raises(ValueError):
        collision_upper_confidence_bound(n_events, n_episodes, confidence=confidence)


def test_admissibility_gates_collision_first() -> None:
    """A collision makes an episode inadmissible regardless of comfort/efficiency."""
    assert not is_episode_admissible({"collisions": 1, "comfort": 1.0})
    assert is_episode_admissible({"collisions": 0, "comfort": 0.0})


def test_admissibility_respects_near_miss_timeout_deadlock() -> None:
    """Near-miss severity, timeout, and deadlock gates must each block admissibility."""
    gates = AdmissibilityGates(max_near_miss_severity=0.5)

    assert not is_episode_admissible({"collisions": 0, "near_miss_severity": 0.9}, gates)
    assert is_episode_admissible({"collisions": 0, "near_miss_severity": 0.3}, gates)
    assert not is_episode_admissible({"collisions": 0, "timeout": True})
    assert not is_episode_admissible({"collisions": 0, "deadlock": True})


def test_survivorship_delta_exposes_conditioning_bias() -> None:
    """Comfort over only-successful episodes must differ from the unconditional mean."""
    episodes = [
        {"comfort": 1.0, "safe_success": True},
        {"comfort": 1.0, "safe_success": True},
        {"comfort": 0.0, "safe_success": False},
    ]
    report = survivorship_aware_metric(episodes, "comfort")

    assert report["unconditional_mean"] == pytest.approx(2 / 3)
    assert report["conditioned_on_safe_success_mean"] == pytest.approx(1.0)
    assert report["survivorship_delta"] == pytest.approx(1 / 3)
    assert report["n_all"] == 3
    assert report["n_safe_success"] == 2


def test_survivorship_sample_sizes_exclude_non_numeric_values() -> None:
    """A non-numeric metric value must be dropped from both the mean and the sample count."""
    episodes = [
        {"comfort": 1.0, "safe_success": True},
        {"comfort": "n/a", "safe_success": True},  # unparseable -> excluded everywhere
        {"comfort": 0.0, "safe_success": False},
    ]
    report = survivorship_aware_metric(episodes, "comfort")

    # The non-numeric value is excluded from the mean, so it must not inflate n_all/n_safe_success.
    assert report["n_all"] == 2
    assert report["n_safe_success"] == 1
    assert report["unconditional_mean"] == pytest.approx(0.5)
    assert report["conditioned_on_safe_success_mean"] == pytest.approx(1.0)


def test_planner_summary_is_versioned_and_complete() -> None:
    """The planner summary must expose admissibility, collision UCB, and survivorship."""
    episodes = [
        {"collisions": 0, "comfort": 0.9, "efficiency": 0.8, "safe_success": True},
        {"collisions": 1, "comfort": 0.2, "efficiency": 0.9, "safe_success": False},
        {"collisions": 0, "comfort": 0.7, "efficiency": 0.6, "safe_success": True},
    ]
    summary = constraints_first_planner_summary(episodes)

    assert summary["schema_version"] == CONSTRAINTS_FIRST_SCHEMA
    assert summary["n_episodes"] == 3
    assert summary["admissible_rate"] == pytest.approx(2 / 3)
    assert summary["collision_rate"] == pytest.approx(1 / 3)
    assert 0.0 < summary["collision_upper_confidence_bound"] <= 1.0
    assert summary["comfort"]["survivorship_delta"] is not None


def test_planner_summary_rejects_empty() -> None:
    """A planner with no episodes cannot be summarized."""
    with pytest.raises(ValueError):
        constraints_first_planner_summary([])


def test_ranking_inversion_detects_order_change() -> None:
    """A planner that looks good only under the soft composite must surface as inverted."""
    compensatory = {"A": 0.9, "B": 0.8, "C": 0.5}
    # Under constraints-first, B (a frequent collider) drops below C.
    constraints_first = {"A": 0.9, "B": 0.3, "C": 0.6}
    result = ranking_inversion(compensatory, constraints_first)

    assert result["any_inversion"] is True
    assert set(result["inverted_planners"]) == {"B", "C"}
    assert result["per_planner"]["B"]["compensatory_rank"] == 2
    assert result["per_planner"]["B"]["constraints_first_rank"] == 3


def test_ranking_inversion_none_when_orders_match() -> None:
    """Identical orderings must report no inversion."""
    scores = {"A": 0.9, "B": 0.5}
    result = ranking_inversion(scores, {"A": 0.8, "B": 0.1})

    assert result["any_inversion"] is False
    assert result["inverted_planners"] == []


def test_ranking_inversion_requires_same_planner_set() -> None:
    """Mismatched planner sets must fail closed."""
    with pytest.raises(ValueError):
        ranking_inversion({"A": 1.0}, {"B": 1.0})


# --- end-to-end report builder -----------------------------------------------


def _planner_episodes() -> dict[str, list[dict[str, object]]]:
    """Two planners: A is admissible-heavy, B looks good but collides often."""
    return {
        "A": [
            {"collisions": 0, "comfort": 0.8, "efficiency": 0.7, "safe_success": True},
            {"collisions": 0, "comfort": 0.9, "efficiency": 0.6, "safe_success": True},
        ],
        "B": [
            {"collisions": 1, "comfort": 1.0, "efficiency": 1.0, "safe_success": False},
            {"collisions": 0, "comfort": 0.9, "efficiency": 0.9, "safe_success": True},
        ],
    }


def test_report_builds_per_planner_summaries() -> None:
    """The report must include a constraints-first summary for every planner."""
    from robot_sf.benchmark.constraints_first_scoring import (
        CONSTRAINTS_FIRST_SCHEMA,
        build_constraints_first_report,
    )

    report = build_constraints_first_report(_planner_episodes())

    assert report["schema_version"] == CONSTRAINTS_FIRST_SCHEMA
    assert set(report["per_planner"]) == {"A", "B"}
    assert report["per_planner"]["A"]["admissible_rate"] == pytest.approx(1.0)
    assert report["per_planner"]["B"]["admissible_rate"] == pytest.approx(0.5)
    assert "ranking_inversion" not in report


def test_report_surfaces_ranking_inversion_against_composite() -> None:
    """When B leads the soft composite but fails the gate, the report must flag inversion."""
    from robot_sf.benchmark.constraints_first_scoring import build_constraints_first_report

    # Compensatory composite ranks B above A (B's comfort/efficiency are higher).
    report = build_constraints_first_report(
        _planner_episodes(), compensatory_scores={"A": 0.78, "B": 0.95}
    )

    inversion = report["ranking_inversion"]
    assert inversion["any_inversion"] is True
    # Constraints-first ranks A (admissible_rate 1.0) above B (0.5).
    assert inversion["per_planner"]["A"]["constraints_first_rank"] == 1
    assert inversion["per_planner"]["B"]["constraints_first_rank"] == 2
    assert inversion["per_planner"]["B"]["compensatory_rank"] == 1


def test_report_rejects_empty_input() -> None:
    """A report with no planners cannot be built."""
    from robot_sf.benchmark.constraints_first_scoring import build_constraints_first_report

    with pytest.raises(ValueError):
        build_constraints_first_report({})


# --- CLI ----------------------------------------------------------------------


def test_cli_writes_report_from_jsonl(tmp_path) -> None:
    """The CLI must group a flat episode JSONL by planner and write the report."""
    import json

    from robot_sf.benchmark.constraints_first_scoring import main

    episodes = tmp_path / "episodes.jsonl"
    lines = [
        {"planner": "A", "collisions": 0, "comfort": 0.8, "efficiency": 0.7, "safe_success": True},
        {"planner": "B", "collisions": 1, "comfort": 1.0, "efficiency": 1.0, "safe_success": False},
        {"planner": "B", "collisions": 0, "comfort": 0.9, "efficiency": 0.9, "safe_success": True},
    ]
    episodes.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    compensatory = tmp_path / "comp.json"
    compensatory.write_text(json.dumps({"A": 0.78, "B": 0.95}), encoding="utf-8")
    out = tmp_path / "report.json"

    code = main(
        [
            "--episodes",
            str(episodes),
            "--compensatory",
            str(compensatory),
            "--output",
            str(out),
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["per_planner"]) == {"A", "B"}
    assert report["ranking_inversion"]["any_inversion"] is True


def test_cli_reports_missing_planner_key_without_traceback(tmp_path, capsys) -> None:
    """A record missing the planner key must fail closed with exit code 2."""
    import json

    from robot_sf.benchmark.constraints_first_scoring import main

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"collisions": 0}) + "\n", encoding="utf-8")

    code = main(["--episodes", str(episodes)])

    assert code == 2
    assert "planner" in capsys.readouterr().err
