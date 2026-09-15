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
    _is_preparation_only_packet,
    _parse_queue_summary,
    _parse_route_output,
    _preflight_command,
    _private_ops_snapshot,
    _queue_summary_blockers,
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
            "mode": "enforced_staged",
            "stage": True,
            "checked": 5,
            "resolved": 5,
            "submit_safe": True,
            "arms": [
                {
                    "planner_key": f"planner_{index}",
                    "algo": "learned",
                    "kind": "model_id",
                    "value": f"model_{index}",
                    "status": "staged",
                    "checkpoint_sha256": "a" * 64,
                }
                for index in range(5)
            ],
        },
        "episodes": 0,
    }


def _queue_summary_text(
    *,
    state: str,
    queue_entries: int = 1,
    active_ledger_jobs: int = 0,
    scope_issue: int | None = 6642,
) -> str:
    ready_entries = 1 if state == "ready" else 0
    lines = [
        f"- queue_entries: {queue_entries}",
        f"- ready_entries: {ready_entries}",
        f"- submit_eligible_entries: {ready_entries}",
        "- ready_but_submit_blocked: 0",
        f"- blocked_or_inactive_entries: {queue_entries - ready_entries}",
        f"- active_ledger_jobs: {active_ledger_jobs}",
        "## States",
        f"- {state}: {queue_entries}",
    ]
    if scope_issue is not None:
        lines.insert(0, f"- scope_issue: #{scope_issue}")
    return "\n".join(lines)


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


def test_preflight_requires_enforced_staged_checkpoint_evidence() -> None:
    """Verify submit-safe preflight retains staged checkpoint identity and checksums."""
    result = validate_preflight_payload(
        _preflight_payload(),
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "passed"
    assert result["checkpoint_preflight"]["submit_safe"] is True
    assert len(result["checkpoint_preflight"]["arms"]) == 5
    assert result["checkpoint_preflight"]["arms"][0]["checkpoint_sha256"] == "a" * 64
    assert result["episodes"] == 0


def test_preflight_rejects_metadata_only_checkpoint_evidence() -> None:
    """Verify metadata-only checkpoint resolution remains blocked for submission."""
    payload = _preflight_payload()
    payload["checkpoint_preflight"].update(
        {"mode": "metadata_only", "stage": False, "submit_safe": False}
    )

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("enforced_staged" in error for error in result["errors"])
    assert any("submit_safe=true" in error for error in result["errors"])


def test_preflight_rejects_unverified_staged_checkpoint() -> None:
    """Verify staged admission cannot omit a checkpoint checksum."""
    payload = _preflight_payload()
    payload["checkpoint_preflight"]["arms"][0]["checkpoint_sha256"] = "not-a-sha"

    result = validate_preflight_payload(
        payload,
        arm_key="r0p5",
        radius_m=0.5,
        config_sha256="a" * 64,
    )

    assert result["structural_status"] == "blocked"
    assert any("verified SHA-256" in error for error in result["errors"])


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
    summary = _parse_queue_summary(_queue_summary_text(state="blocked", queue_entries=128))

    assert summary == {
        "scope_issue": 6642,
        "queue_entries": 128,
        "ready_entries": 0,
        "submit_eligible_entries": 0,
        "ready_but_submit_blocked": 0,
        "blocked_or_inactive_entries": 128,
        "active_ledger_jobs": 0,
        "state_counts": {"blocked": 128},
    }


def test_preparation_contract_accepts_exact_blocked_campaign_row() -> None:
    """Verify preparation may inspect the blocked campaign row without authorizing dispatch."""
    config = _packet_config()
    assert _is_preparation_only_packet(config) is True

    summary = _parse_queue_summary(_queue_summary_text(state="blocked"))

    assert (
        _queue_summary_blockers(
            summary,
            expected_ready=1,
            preparation_only=True,
            expected_scope_issue=6642,
        )
        == []
    )


def test_preparation_contract_is_explicit_and_fail_closed() -> None:
    """Verify queue relaxation is disabled when either no-dispatch marker changes."""
    config = _packet_config()

    config["preflight"]["production_submission_authorized"] = True
    assert _is_preparation_only_packet(config) is False

    config = _packet_config()
    config["verdict"]["never_authorizes_submission"] = False
    assert _is_preparation_only_packet(config) is False


def test_production_queue_validation_rejects_blocked_preparation_row() -> None:
    """Verify a blocked row cannot satisfy the ready/submit-eligible production gate."""
    summary = _parse_queue_summary(_queue_summary_text(state="blocked"))

    blockers = _queue_summary_blockers(
        summary,
        expected_ready=1,
        preparation_only=False,
        expected_scope_issue=6642,
    )

    assert any("ready entries" in blocker for blocker in blockers)
    assert any("submit-eligible entries" in blocker for blocker in blockers)


def test_production_queue_validation_accepts_ready_submit_eligible_row() -> None:
    """Verify the unchanged production path still accepts a fully admissible row."""
    summary = _parse_queue_summary(_queue_summary_text(state="ready"))

    assert (
        _queue_summary_blockers(
            summary,
            expected_ready=1,
            preparation_only=False,
            expected_scope_issue=6642,
        )
        == []
    )


def test_preparation_contract_rejects_unscoped_or_nonblocked_queue_state() -> None:
    """Verify preparation cannot turn aggregate or unexpectedly ready evidence into a pass."""
    summary = _parse_queue_summary(
        _queue_summary_text(state="ready", active_ledger_jobs=1, scope_issue=None)
    )

    blockers = _queue_summary_blockers(
        summary,
        expected_ready=1,
        preparation_only=True,
        expected_scope_issue=6642,
    )

    assert any("issue-scoped" in blocker for blocker in blockers)
    assert any("ready_entries=0" in blocker for blocker in blockers)
    assert any("active_ledger_jobs=0" in blocker for blocker in blockers)
    assert any("exactly one blocked row" in blocker for blocker in blockers)


def test_preparation_snapshot_scopes_queue_query_without_dispatch(tmp_path, monkeypatch) -> None:
    """Verify the packet path queries only #6642 and records no submission operation."""
    config = _packet_config()
    private_root = tmp_path / "private-ops"
    for key in (
        "queue_summary_script",
        "route_script",
        "preflight_script",
        "submission_entrypoint",
    ):
        path = private_root / config["private_ops"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_PRIVATE_OPS", str(private_root))
    queue_output = _queue_summary_text(state="blocked")
    calls = []

    def fake_run(command, *, cwd, timeout):
        calls.append(command)
        if command[:2] == ["git", "-C"]:
            return {"returncode": 0, "stdout": "", "stderr": ""}
        if command[0] == "bash":
            return {"returncode": 0, "stdout": queue_output, "stderr": ""}
        if command[0] == "python3":
            return {
                "returncode": 0,
                "stdout": "explain:\n   selected: imech192:a30-cpu\n",
                "stderr": "",
            }
        raise AssertionError(f"unexpected command: {command!r}")

    monkeypatch.setattr(
        "scripts.benchmark.prepare_radius_sweep_admission_issue_7198._run", fake_run
    )
    blockers = []
    snapshot = _private_ops_snapshot(
        REPO_ROOT,
        config,
        tmp_path / "packet",
        blockers,
        preparation_only=True,
        campaign_issue=6642,
    )

    queue_calls = [command for command in calls if command[0] == "bash"]
    assert len(queue_calls) == 1
    assert queue_calls[0][-2:] == ["--issue", "6642"]
    assert snapshot["queue_summary"]["preparation_only"] is True
    assert snapshot["queue_summary"]["summary"]["state_counts"] == {"blocked": 1}
    assert blockers == []


def test_route_parser_preserves_static_estimate() -> None:
    """Verify route parsing tolerates indentation while retaining the selected route."""
    route = _parse_route_output(
        "explain:\n"
        "   selected: imech192:a30-cpu\n"
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
    """Verify the recorded arm command uses preflight mode and skips production output."""
    command = _preflight_command(
        "configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml",
        Path("output/preflight/r0p5"),
        "issue7198-r0p5",
    )

    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "preflight"
    assert "--skip-publication-bundle" in command
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
