"""Regression checks for external review-bot routing configuration."""

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HIGH_RISK_PATH_FILTERS = [
    "robot_sf/**",
    "fast-pysf/**",
    "scripts/**",
    "tests/**",
    ".github/workflows/**",
]


def test_coderabbit_routes_only_labeled_reviews_to_high_risk_paths() -> None:
    """Keep CodeRabbit's filters and label gate fail-closed."""
    config = yaml.safe_load((REPOSITORY_ROOT / ".coderabbit.yaml").read_text())

    assert config["reviews"]["path_filters"] == HIGH_RISK_PATH_FILTERS
    assert config["reviews"]["auto_review"] == {
        "enabled": False,
        "labels": ["review-bot-auto", "review-bot"],
    }


def test_routing_workflow_tracks_every_coderabbit_high_risk_path() -> None:
    """Keep the workflow's label-routing and security contracts aligned with CodeRabbit."""
    workflow = yaml.load(
        (REPOSITORY_ROOT / ".github/workflows/review-bot-routing.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["route-coderabbit"]["steps"][0]["with"]["script"]

    assert workflow["on"]["pull_request_target"]["types"] == [
        "opened",
        "reopened",
        "synchronize",
        "ready_for_review",
    ]
    assert workflow["permissions"] == {
        "issues": "write",
        "pull-requests": "write",
    }
    assert "review-bot-auto" in script
    assert all(path_filter.removesuffix("**") in script for path_filter in HIGH_RISK_PATH_FILTERS)
    assert "actions/checkout" not in str(workflow["jobs"]["route-coderabbit"])
