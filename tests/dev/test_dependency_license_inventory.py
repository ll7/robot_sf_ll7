"""Tests for the committed dependency-license control-plane inputs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import validate


def test_profile_and_policy_manifests_match_their_schemas() -> None:
    """The checked-in manifests remain machine-validatable."""
    root = Path(__file__).resolve().parents[2]
    profiles = json.loads(
        (root / "scripts/validation/dependency_license_profiles.v1.json").read_text(
            encoding="utf-8"
        )
    )
    profile_schema = json.loads(
        (root / "scripts/validation/dependency_license_profiles.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    policy_schema = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    validate(profiles, profile_schema)
    validate(policy, policy_schema)


def test_dependency_review_runs_inventory_and_freshness_checks() -> None:
    """The advisory workflow publishes blocked evidence and checks its digests."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github" / "workflows" / "dependency-review.yml").read_text(encoding="utf-8")
    )
    trigger = workflow.get("on") or workflow[True]
    paths = set(trigger["pull_request"]["paths"])
    assert {
        "scripts/tools/check_dependency_license_inventory.py",
        "scripts/validation/dependency_license_profiles.v1.json",
        "scripts/validation/dependency_license_policy.v1.json",
        "docs/context/dependency_license_inventory.md",
    } <= paths
    steps = workflow["jobs"]["review"]["steps"]
    run_commands = [step.get("run", "") for step in steps]
    assert any(
        "--output output/validation/dependency-license-inventory.json" in run
        for run in run_commands
    )
    assert any("--check-freshness" in run for run in run_commands)
