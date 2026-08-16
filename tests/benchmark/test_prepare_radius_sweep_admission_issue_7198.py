"""Contract tests for the issue #7198 Gate 2 admission packet."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from robot_sf.benchmark.radius_sweep_manifest import (
    EXPECTED_DT,
    EXPECTED_HORIZON,
    EXPECTED_SCENARIO_COUNT,
    EXPECTED_SEED_RANGE,
    RELEASE_PLANNER_KEYS,
)
from scripts.benchmark.prepare_radius_sweep_admission_issue_7198 import (
    GATE1_SURFACES,
    _parse_queue_summary,
    _parse_route_output,
    _preflight_command,
    validate_gate1_report,
    validate_preflight_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_7198_radius_sweep_admission_v1.yaml"


def _packet_config() -> dict:
    return yaml.safe_load(PACKET_CONFIG_PATH.read_text(encoding="utf-8"))


def _gate1_report() -> dict:
    config = _packet_config()
    verdicts = []
    for radius in config["gate1"]["required_radii_m"]:
        verdicts.append(
            {
                "target_radius_m": radius,
                "go": True,
                "surfaces": [{"surface": name, "bound": True} for name in GATE1_SURFACES],
            }
        )
    return {
        "schema": "radius_binding_canary_report.v1",
        "canary_schema": "radius_binding_canary.v1",
        "issue": 6641,
        "parent_issue": 6600,
        "scenario_id": "francis2023_narrow_doorway",
        "radii_m": [0.5, 0.8, 1.0],
        "go": True,
        "verdicts": verdicts,
    }


def _preflight_payload() -> dict:
    return {
        "schema_version": "benchmark-preflight-validate-config.v1",
        "campaign_id": "issue7198-r0p5",
        "config_path": "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml",
        "config_sha256": "a" * 64,
        "radius_binding": {
            "issue": 6642,
            "parent_issue": 6600,
            "arm_key": "r0p5",
            "radius_m": 0.5,
            "status": "bound_runtime",
        },
        "scenario_count": EXPECTED_SCENARIO_COUNT,
        "planner_count": len(RELEASE_PLANNER_KEYS),
        "horizon": EXPECTED_HORIZON,
        "dt": EXPECTED_DT,
        "seed_policy": {
            "resolved_seeds": list(range(EXPECTED_SEED_RANGE[0], EXPECTED_SEED_RANGE[1] + 1))
        },
        "checkpoint_preflight": {
            "mode": "metadata_only",
            "stage": False,
            "checked": 5,
            "resolved": 5,
            "submit_safe": False,
        },
        "episodes": 0,
    }


def test_gate1_report_requires_all_fifteen_binding_surfaces() -> None:
    summary, errors = validate_gate1_report(_gate1_report(), packet_config=_packet_config())

    assert errors == []
    assert summary["status"] == "valid"
    assert [item["bound_surface_count"] for item in summary["verdicts"]] == [5, 5, 5]


def test_gate1_report_rejects_surface_roster_drift() -> None:
    report = _gate1_report()
    report["verdicts"][1]["surfaces"].pop()

    summary, errors = validate_gate1_report(report, packet_config=_packet_config())

    assert summary["status"] == "blocked"
    assert any("five surfaces at 0.8" in error for error in errors)


def test_preflight_passes_structure_but_metadata_only_is_not_submit_safe() -> None:
    result = validate_preflight_payload(
        _preflight_payload(),
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "passed"
    assert result["checkpoint_preflight"]["submit_safe"] is False
    assert result["episodes"] == 0


def test_preflight_rejects_any_episode_count() -> None:
    payload = _preflight_payload()
    payload["episodes"] = 1

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("nonzero episodes" in error for error in result["errors"])


def test_preflight_rejects_config_checksum_drift() -> None:
    payload = _preflight_payload()

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="b" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("config_sha256" in error for error in result["errors"])


def test_queue_summary_parser_captures_readiness_counts() -> None:
    summary = _parse_queue_summary(
        "\n".join(
            [
                "- queue_entries: 128",
                "- ready_entries: 0",
                "- blocked_or_inactive_entries: 128",
                "- active_ledger_jobs: 0",
            ]
        )
    )

    assert summary == {
        "queue_entries": 128,
        "ready_entries": 0,
        "blocked_or_inactive_entries": 128,
        "active_ledger_jobs": 0,
    }


def test_route_parser_preserves_static_estimate() -> None:
    route = _parse_route_output(
        "explain:\n"
        " selected: imech192:a30-cpu\n"
        " why:\n"
        " - estimated elapsed 45474s\n"
        " - score 27.20\n"
    )

    assert route == {
        "selected_route": "imech192:a30-cpu",
        "estimated_elapsed_sec": 45474,
        "score": 27.2,
        "status": "parsed",
    }


def test_preflight_command_is_check_only_and_zero_episode() -> None:
    command = _preflight_command(
        "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml",
        Path("output/preflight/r0p5"),
        "issue7198-r0p5",
    )

    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "preflight"
    assert "--skip-publication-bundle" in command
    assert command[command.index("--checkpoint-preflight-mode") + 1] == "metadata_only"
    assert "run" not in command[command.index("--mode") + 1 :]


def test_gate1_negative_control_does_not_mutate_fixture() -> None:
    report = _gate1_report()
    changed = copy.deepcopy(report)
    changed["verdicts"][0]["surfaces"][0]["bound"] = False

    _summary, errors = validate_gate1_report(changed, packet_config=_packet_config())

    assert errors
    assert report["verdicts"][0]["surfaces"][0]["bound"] is True
