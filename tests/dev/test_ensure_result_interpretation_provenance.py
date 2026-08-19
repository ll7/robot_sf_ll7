"""Tests for CI hydration of result-interpretation fixture provenance."""

from __future__ import annotations

from scripts.dev import ensure_result_interpretation_provenance as hydration


def test_required_commits_match_source_and_catalog_provenance() -> None:
    """Only validator-required source and catalog commits are hydrated."""
    commits = set(hydration.collect_required_commits())

    assert "2fc4498cc5499bd3569eb1ac941a3029e0f51040" in commits
    assert "8f9438632e794f084db72bb016a14b539bbca648" in commits
    assert "4e513ebbbc3b11ef580cea76888fd6de43836c66" in commits
    assert "54ed835669192dd22974ff4a68acbc83ddfe5148" not in commits
    assert all(len(commit) == 40 for commit in commits)
