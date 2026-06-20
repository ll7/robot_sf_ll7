"""Tests for issue #3207 fidelity-sensitivity launch-packet helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.fidelity_sensitivity import (
    build_launch_packet,
    build_rank_stability_summary,
    format_launch_packet_markdown,
    kendall_tau_from_orders,
    load_fidelity_sensitivity_config,
    metric_drift,
    rank_flip_count,
    validate_fidelity_sensitivity_config,
)

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_launch_packet_builder() -> ModuleType:
    module_path = (
        REPO_ROOT / "scripts" / "benchmark" / "build_fidelity_sensitivity_launch_packet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fidelity_sensitivity_launch_packet_builder", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load builder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_sensitivity_smoke_report_builder"] = module
    spec.loader.exec_module(module)
    return module


def _load_smoke_report_builder() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "benchmark" / "build_fidelity_sensitivity_smoke_report.py"
    spec = importlib.util.spec_from_file_location(
        "fidelity_sensitivity_smoke_report_builder", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load builder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch_packet_builder = _load_launch_packet_builder()
smoke_report_builder = _load_smoke_report_builder()


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
        date="2026-06-20",
    )

    assert packet["status"] == "launch_packet_only"
    assert packet["axis_count"] >= 3
    assert packet["date"] == "2026-06-20"
    assert "not benchmark evidence" in packet["claim_boundary"]


def test_build_launch_packet_preserves_empty_claim_boundary() -> None:
    """Explicit falsy config values should not be replaced by defaults."""
    config = load_fidelity_sensitivity_config("configs/research/fidelity_sensitivity_v1.yaml")
    config["claim_boundary"] = ""

    packet = build_launch_packet(
        config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc1234",
    )

    assert packet["claim_boundary"] == ""


def test_format_launch_packet_markdown_uses_issue_and_date_heading() -> None:
    """Markdown headings should reflect packet metadata instead of hardcoded issue text."""
    config = load_fidelity_sensitivity_config("configs/research/fidelity_sensitivity_v1.yaml")
    packet = build_launch_packet(
        config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="abc1234",
        date="2026-06-20",
    )
    packet["issue"] = 9999

    markdown = format_launch_packet_markdown(packet)

    assert markdown.startswith("# Issue #9999 Fidelity Sensitivity Launch Packet 2026-06-20")


def test_builder_extracts_iso_date_from_output_dir() -> None:
    """The builder should preserve ISO dates embedded in output paths."""
    assert (
        launch_packet_builder._iso_date_from_output_dir(
            "docs/context/evidence/issue_3207_fidelity_sensitivity_launch_packet_2026-06-20"
        )
        == "2026-06-20"
    )
    assert launch_packet_builder._iso_date_from_output_dir("output/fidelity_sensitivity") is None


def test_builder_git_head_handles_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal environments without git on PATH should still build a packet."""

    def missing_git(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(launch_packet_builder.subprocess, "run", missing_git)

    assert launch_packet_builder._git_head() == "unknown"


def test_smoke_report_builder_detects_rank_flip() -> None:
    """Diagnostic smoke summaries should reuse the rank-stability contract."""
    result_sets = {
        "dt_0_10": {
            "simple_policy": [{"metrics": {"min_distance": 3.0}, "seed": 101}],
            "social_force": [{"metrics": {"min_distance": 2.0}, "seed": 101}],
        },
        "dt_0_20": {
            "simple_policy": [{"metrics": {"min_distance": 1.0}, "seed": 101}],
            "social_force": [{"metrics": {"min_distance": 4.0}, "seed": 101}],
        },
    }

    report = smoke_report_builder.build_report(
        result_sets,
        inputs={},
        options=smoke_report_builder.SmokeReportOptions(
            baseline_variant="dt_0_10",
            ranking_metric="min_distance",
            higher_is_better=True,
            git_head="abc1234",
            date="2026-06-20",
        ),
    )

    stability = report["comparisons_vs_baseline"]["dt_0_20"]["rank_stability"]
    assert report["status"] == "diagnostic_smoke"
    assert stability["ranking_flipped"] is True
    assert stability["kendall_tau_vs_baseline"] == pytest.approx(-1.0)


def test_smoke_report_builder_requires_matching_planners() -> None:
    """Each variant must compare the same planner set as the baseline."""
    result_sets = {
        "dt_0_10": {"simple_policy": [{"metrics": {"min_distance": 3.0}}]},
        "dt_0_20": {"social_force": [{"metrics": {"min_distance": 4.0}}]},
    }

    with pytest.raises(ValueError, match="do not match baseline planners"):
        smoke_report_builder.build_report(
            result_sets,
            inputs={},
            options=smoke_report_builder.SmokeReportOptions(
                baseline_variant="dt_0_10",
                ranking_metric="min_distance",
                higher_is_better=True,
            ),
        )
