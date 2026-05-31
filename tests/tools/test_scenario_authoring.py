"""Tests for the v1 scenario authoring generator and validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from scripts.tools import create_scenario, validate_scenario
from scripts.tools.scenario_authoring import (
    build_scenario_payload,
    dump_scenario_yaml,
    validate_scenario_file,
    write_scenario_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_bottleneck_template_generation_is_deterministic_and_draft_only() -> None:
    """The v1 generator should produce stable YAML without benchmark-evidence claims."""

    payload = build_scenario_payload(
        template="bottleneck",
        name="draft_bottleneck_review",
        seeds=(7, 8),
        source_issue="#1891",
    )

    first = dump_scenario_yaml(payload)
    second = dump_scenario_yaml(payload)
    parsed = yaml.safe_load(first)

    assert first == second
    assert parsed["schema_version"] == "robot_sf.scenario_matrix.v1"
    scenario = parsed["scenarios"][0]
    assert scenario["name"] == "draft_bottleneck_review"
    assert scenario["map_id"] == "classic_bottleneck"
    assert scenario["seeds"] == [7, 8]
    assert scenario["metadata"]["authoring"]["status"] == "draft"
    assert scenario["metadata"]["authoring"]["benchmark_evidence"] is False


def test_generated_bottleneck_template_validates_with_existing_loader(tmp_path: Path) -> None:
    """Generated YAML should pass authoring checks and the repository scenario loader."""

    scenario_path = tmp_path / "draft_bottleneck.yaml"
    write_scenario_yaml(
        scenario_path,
        build_scenario_payload(template="bottleneck", name="draft_bottleneck_review"),
    )

    report = validate_scenario_file(scenario_path)

    assert report.ok is True
    assert report.scenario_count == 1
    assert report.issues == ()


def test_validator_reports_missing_required_fields_with_actionable_paths(tmp_path: Path) -> None:
    """Authoring validation should catch malformed drafts before benchmark tools run."""

    scenario_path = tmp_path / "broken.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "broken_missing_seed_metadata",
                        "map_id": "classic_bottleneck",
                        "simulation_config": {"max_episode_steps": 100},
                        "robot_config": {},
                        "metadata": {"purpose": "Broken draft for validation test."},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = validate_scenario_file(scenario_path)
    messages = [issue.format() for issue in report.issues]

    assert report.ok is False
    assert any(
        "/scenarios/0/seeds" in message and "deterministic seeds" in message for message in messages
    )
    assert any(
        "/scenarios/0/metadata/authoring" in message and "metadata.authoring" in message
        for message in messages
    )


def test_validator_reports_unknown_map_id_from_loader(tmp_path: Path) -> None:
    """Map-id mistakes should surface as loader-backed, actionable validation errors."""

    scenario_path = tmp_path / "missing_map.yaml"
    payload = build_scenario_payload(template="bottleneck", name="missing_map_review")
    payload["scenarios"][0]["map_id"] = "not_a_registered_map"
    write_scenario_yaml(scenario_path, payload)

    report = validate_scenario_file(scenario_path)

    assert report.ok is False
    assert len(report.issues) == 1
    assert report.issues[0].path == "$"
    assert "Unknown map_id 'not_a_registered_map'" in report.issues[0].message


def test_create_and_validate_clis_return_success_for_generated_template(tmp_path: Path) -> None:
    """The documented v1 CLI path should create and validate a draft YAML file."""

    scenario_path = tmp_path / "cli_draft.yaml"

    create_result = create_scenario.main(
        [
            "--template",
            "bottleneck",
            "--name",
            "cli_draft_bottleneck",
            "--seeds",
            "11,12",
            "--output",
            str(scenario_path),
        ]
    )
    validate_result = validate_scenario.main([str(scenario_path)])

    assert create_result == 0
    assert validate_result == 0
    assert scenario_path.exists()
