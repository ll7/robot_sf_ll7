"""Tests for issue #3207 fidelity-sensitivity launch-packet helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.fidelity_sensitivity import (
    build_launch_packet,
    build_rank_stability_summary,
    kendall_tau_from_orders,
    load_fidelity_sensitivity_config,
    metric_drift,
    rank_flip_count,
    validate_fidelity_sensitivity_config,
)


def test_load_repo_config_validates_three_or_more_axes() -> None:
    """The tracked issue #3207 config should satisfy the launch-packet contract."""
    config = load_fidelity_sensitivity_config("configs/research/fidelity_sensitivity_v1.yaml")

    assert config["schema_version"] == "fidelity-sensitivity.v1"
    assert len(config["axes"]) >= 3
    assert config["ranking"]["metric"] == "snqi"


def test_validate_config_rejects_too_few_axes() -> None:
    """Issue #3207 requires at least three fidelity axes."""
    config = load_fidelity_sensitivity_config("configs/research/fidelity_sensitivity_v1.yaml")
    config["axes"] = config["axes"][:2]

    with pytest.raises(ValueError, match="at least three axes"):
        validate_fidelity_sensitivity_config(config)


def test_kendall_tau_and_flip_count_detect_reversed_ranking() -> None:
    """Rank-stability helpers should flag a fully reversed three-planner ordering."""
    baseline = ["orca", "social_force", "hybrid_rule"]
    variant = ["hybrid_rule", "social_force", "orca"]

    assert kendall_tau_from_orders(baseline, baseline) == pytest.approx(1.0)
    assert kendall_tau_from_orders(baseline, variant) == pytest.approx(-1.0)
    assert rank_flip_count(baseline, variant) == 2


def test_build_rank_stability_summary_identifies_ranking_flip() -> None:
    """A variant that changes planner order should become a caveat candidate."""
    summary = build_rank_stability_summary(
        {"orca": 0.8, "social_force": 0.6, "hybrid_rule": 0.3},
        {"orca": 0.4, "social_force": 0.6, "hybrid_rule": 0.9},
        higher_is_better=True,
        min_tau=0.8,
    )

    assert summary["baseline_order"] == ["orca", "social_force", "hybrid_rule"]
    assert summary["variant_order"] == ["hybrid_rule", "social_force", "orca"]
    assert summary["ranking_flipped"] is True
    assert summary["stable_by_tau_threshold"] is False


def test_metric_drift_reports_absolute_and_relative_deltas() -> None:
    """Metric drift should retain both absolute and relative deltas."""
    drift = metric_drift(
        {"snqi": 0.5, "collisions": 0.0},
        {"snqi": 0.4, "collisions": 1.0},
    )

    assert drift["snqi"]["absolute_delta"] == pytest.approx(-0.1)
    assert drift["snqi"]["relative_delta"] == pytest.approx(-0.2)
    assert drift["collisions"]["absolute_delta"] == pytest.approx(1.0)
    assert drift["collisions"]["relative_delta"] is None


def test_build_launch_packet_preserves_no_evidence_boundary() -> None:
    """Generated packets should be explicit launch packets, not result evidence."""
    config = load_fidelity_sensitivity_config("configs/research/fidelity_sensitivity_v1.yaml")
    packet = build_launch_packet(
        config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc1234",
    )

    assert packet["status"] == "launch_packet_only"
    assert packet["axis_count"] >= 3
    assert "not benchmark evidence" in packet["claim_boundary"]
