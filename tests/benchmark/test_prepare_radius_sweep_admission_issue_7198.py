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
    _submission_command,
    validate_gate1_report,
    validate_preflight_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_CONFIG_PATH = REPO_ROOT / "configs/benchmarks/issue_7198_radius_sweep_admission_v1.yaml"


def _packet_config() -> dict:
    return yaml.safe_load(PACKET_CONFIG_PATH.read_text(encoding="utf-8"))


def _gate1_report() -> dict:
    config = _packet_config()
    radii = [float(value) for value in config["gate1"]["required_radii_m"]]
    verdicts = []
    for radius in radii:
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
        "radii_m": radii,
        "go": True,
        "verdicts": verdicts,
    }


def _preflight_payload(
    *, mode: str = "metadata_only", stage: bool = False, submit_safe: bool = False
) -> dict:
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
            "mode": mode,
            "stage": stage,
            "checked": 5,
            "resolved": 5,
            "submit_safe": submit_safe,
        },
        "episodes": 0,
    }


def test_gate1_report_requires_all_fifteen_binding_surfaces() -> None:
    """Verify every Gate 1 radius has all five bindings before Gate 2 admission."""
    summary, errors = validate_gate1_report(_gate1_report(), packet_config=_packet_config())

    assert errors == []
    assert summary["status"] == "valid"
    assert [item["bound_surface_count"] for item in summary["verdicts"]] == [5, 5, 5]


def test_gate1_report_rejects_surface_roster_drift() -> None:
    """Verify a missing binding surface blocks the radius-sweep admission packet."""
    report = _gate1_report()
    report["verdicts"][1]["surfaces"].pop()

    summary, errors = validate_gate1_report(report, packet_config=_packet_config())

    assert summary["status"] == "blocked"
    assert any("five surfaces at 0.8" in error for error in errors)


def test_gate1_report_rejects_non_mapping_surface_entries() -> None:
    """Verify malformed surface entries block admission without raising an attribute error."""
    report = _gate1_report()
    report["verdicts"][0]["surfaces"][0] = "not-a-surface-mapping"

    summary, errors = validate_gate1_report(report, packet_config=_packet_config())

    assert summary["status"] == "blocked"
    assert any("non-mapping entries" in error for error in errors)


def test_preflight_passes_structure_but_metadata_only_is_not_submit_safe() -> None:
    """Verify structural preflight success does not promote metadata-only checkpoints."""
    result = validate_preflight_payload(
        _preflight_payload(),
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "passed"
    assert result["checkpoint_preflight"]["submit_safe"] is False
    assert result["episodes"] == 0


def test_enforced_staged_preflight_is_submit_safe() -> None:
    """Verify only staged, checksum-verified checkpoint preparation can pass the submit gate."""
    result = validate_preflight_payload(
        _preflight_payload(mode="enforced_staged", stage=True, submit_safe=True),
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
        expected_checkpoint_mode="enforced_staged",
    )

    assert result["structural_status"] == "passed"
    assert result["checkpoint_preflight"]["mode"] == "enforced_staged"
    assert result["checkpoint_preflight"]["submit_safe"] is True


def test_enforced_staged_preflight_rejects_unstaged_checkpoint() -> None:
    """Verify a mislabeled staged receipt cannot silently authorize submission."""
    result = validate_preflight_payload(
        _preflight_payload(mode="enforced_staged", stage=False, submit_safe=False),
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
        expected_checkpoint_mode="enforced_staged",
    )

    assert result["structural_status"] == "blocked"
    assert any("did not stage" in error for error in result["errors"])


def test_preflight_rejects_any_episode_count() -> None:
    """Verify any emitted episode count blocks a preparation-only preflight."""
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
    """Verify a changed arm checksum is rejected before production submission."""
    payload = _preflight_payload()

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="b" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("config_sha256" in error for error in result["errors"])


def test_preflight_rejects_non_mapping_seed_policy() -> None:
    """Verify malformed seed metadata blocks admission rather than raising an attribute error."""
    payload = _preflight_payload()
    payload["seed_policy"] = []

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("seed_policy must be a mapping" in error for error in result["errors"])


def test_preflight_rejects_non_mapping_checkpoint_metadata() -> None:
    """Verify malformed checkpoint metadata cannot pass the preflight admission gate."""
    payload = _preflight_payload()
    payload["checkpoint_preflight"] = []

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("checkpoint_preflight must be a mapping" in error for error in result["errors"])


def test_queue_summary_parser_captures_readiness_counts() -> None:
    """Verify private queue readiness counts are parsed into structured evidence."""
    summary = _parse_queue_summary(
        "\n".join(
            [
                "- queue_entries: 128",
                "- ready_entries: 0",
                "- submit_eligible_entries: 0",
                "- ready_but_submit_blocked: 0",
                "- blocked_or_inactive_entries: 128",
                "- active_ledger_jobs: 0",
            ]
        )
    )

    assert summary == {
        "queue_entries": 128,
        "ready_entries": 0,
        "submit_eligible_entries": 0,
        "ready_but_submit_blocked": 0,
        "blocked_or_inactive_entries": 128,
        "active_ledger_jobs": 0,
    }


def test_route_parser_preserves_static_estimate() -> None:
    """Verify route parsing tolerates indentation while retaining the selected route."""
    route = _parse_route_output(
        "explain:\n"
        "rank\troute_id\tcluster\tpartition\tscore\test_elapsed_sec\tcpus\tgpus\tmem_gb\treasons\tsbatch_args\n"
        "1\timech192:a30-cpu\timech192\ta30\t27.20\t45474\t40\t0\t155\t\t--partition=a30 --qos=a30-cpu\n"
        "   selected: imech192:a30-cpu\n"
        " why:\n"
        " - estimated elapsed 45474s\n"
        " - score 27.20\n"
    )

    assert route == {
        "selected_route": "imech192:a30-cpu",
        "partition": "a30",
        "sbatch_args": "--partition=a30 --qos=a30-cpu",
        "estimated_elapsed_sec": 45474,
        "score": 27.2,
        "status": "parsed",
    }


def test_route_parser_accepts_negative_fallback_score() -> None:
    """Verify a fallback route's negative score remains observable for fail-closed admission."""
    route = _parse_route_output(
        "   selected: licca:test\n"
        " - estimated elapsed 43200s\n"
        " - score -10065.69\n"
        "1\tlicca:test\tlicca\ttest\t-10065.69\t43200\t40\t0\t155\tnot_allowed_for_job_class\t--partition=test\n"
    )

    assert route["selected_route"] == "licca:test"
    assert route["partition"] == "test"
    assert route["score"] == -10065.69


def test_preflight_command_is_check_only_and_zero_episode() -> None:
    """Verify the recorded arm command uses preflight mode and skips production output."""
    command = _preflight_command(
        "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml",
        Path("output/preflight/r0p5"),
        "issue7198-r0p5",
    )

    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "preflight"
    assert "--skip-publication-bundle" in command
    assert command[command.index("--checkpoint-preflight-mode") + 1] == "metadata_only"


def test_enforced_staged_preflight_command_selects_submit_safe_mode() -> None:
    """Verify the packet can render the enforced staged checkpoint command."""
    command = _preflight_command(
        "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml",
        Path("output/preflight/r0p5"),
        "issue7198-r0p5",
        "enforced_staged",
    )

    assert command[command.index("--checkpoint-preflight-mode") + 1] == "enforced_staged"


def test_submission_command_preserves_expansion_and_custom_manifest_path() -> None:
    """Verify the copied submission template expands results and names its output packet."""
    command = _submission_command(
        _packet_config(),
        {"route": {"selected_route": "imech192:a30-cpu"}},
        artifact_manifest="output/custom-admission/packet.json",
    )

    assert "${ROBOT_SF_RADIUS_SWEEP_RESULTS_URI}/{job_id}" in command
    assert "'${ROBOT_SF_RADIUS_SWEEP_RESULTS_URI}/{job_id}'" not in command
    assert "output/custom-admission/packet.json" in command


def test_gate1_negative_control_does_not_mutate_fixture() -> None:
    """Verify a negative binding control is isolated from the reusable Gate 1 fixture."""
    report = _gate1_report()
    changed = copy.deepcopy(report)
    changed["verdicts"][0]["surfaces"][0]["bound"] = False

    _summary, errors = validate_gate1_report(changed, packet_config=_packet_config())

    assert errors
    assert report["verdicts"][0]["surfaces"][0]["bound"] is True
