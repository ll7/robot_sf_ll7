"""Tests for the issue #3810 no-submit long-horizon launch packet."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_3810_long_horizon_launch_packet.py"

_SPEC = importlib.util.spec_from_file_location("_issue_3810_packet_check", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_packet() -> dict:
    return yaml.safe_load(PACKET.read_text(encoding="utf-8"))


def test_issue_3810_packet_passes_fail_closed_contract() -> None:
    """The checked-in packet satisfies the issue #3810 launch contract."""
    summary = _MODULE.validate_packet(_load_packet())

    assert summary["ok"] is True
    assert summary["issue"] == 3810
    assert summary["max_episode_steps"] == 600
    assert summary["seed_count"] == 30
    assert summary["planner_count"] >= 10
    assert summary["compute_submit_authorized"] is False
    assert summary["slurm_job_id"] == "not_submitted"
    assert summary["target_host"] == "imech156-u"
    assert summary["blocking_jobs"] == [13175]
    assert summary["job_13175_state"] == "requires_submit_host_refresh"
    assert summary["issue_3810_duplicate_status"] == "requires_submit_host_refresh"
    assert summary["live_issue_state"] == "state:running"
    assert summary["go_no_go"] == "blocked_pending_submit_host_route_and_reconciliation"
    assert summary["private_ops_dry_run"] == "route_unverified"
    assert summary["interpretation_gate"] == "blocked_pending_active_run_retention_reconciliation"


def test_issue_3810_packet_rejects_authorized_submit() -> None:
    """The packet validator rejects any direct compute authorization."""
    packet = _load_packet()
    packet["launch_packet"]["compute_submit_authorized"] = True

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "compute submit must be false" in str(exc)
    else:
        raise AssertionError("packet should reject compute submission authorization")


def test_issue_3810_packet_rejects_missing_job_13175_blocker() -> None:
    """Job 13175 stays a blocker until submit-host reconciliation is recorded."""
    packet = _load_packet()
    packet["launch_packet"]["blocking_jobs"] = []

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "job 13175 must block submit" in str(exc)
    else:
        raise AssertionError("packet should reject missing job 13175 blocker")


def test_issue_3810_packet_rejects_stale_job_13175_reconciliation() -> None:
    """Stale analyzed state cannot unlock the public launch packet."""
    packet = _load_packet()
    packet["launch_packet"]["ledger_reconciliation"]["job_13175_state"] = "analyzed"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "job 13175 reconciliation must require submit-host refresh" in str(exc)
    else:
        raise AssertionError("packet should reject stale job 13175 reconciliation")


def test_issue_3810_packet_rejects_nonblocking_live_issue_state() -> None:
    """The live state:running label remains a submit blocker."""
    packet = _load_packet()
    packet["launch_packet"]["live_issue_state"]["submit_blocker"] = False

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "live issue state must block submit while running" in str(exc)
    else:
        raise AssertionError("packet should reject a nonblocking live issue state")


def test_issue_3810_packet_rejects_stale_target_host() -> None:
    """The launch-packet target host must match the requested Slurm decision host."""
    packet = _load_packet()
    packet["launch_packet"]["target_host"] = "imech036"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "target host must be imech156-u" in str(exc)
    else:
        raise AssertionError("packet should reject stale target host")


def test_issue_3810_packet_rejects_stale_dry_run_target_host() -> None:
    """The private-ops dry-run target host is guarded independently of the packet host."""
    packet = _load_packet()
    packet["launch_packet"]["go_no_go"]["private_ops_dry_run"]["target_host"] = "imech036"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "private-ops dry run host mismatch" in str(exc)
    else:
        raise AssertionError("packet should reject a stale dry-run target host")


def test_issue_3810_packet_rejects_decision_policy_without_target_host_gate() -> None:
    """The dry-run decision policy must still gate on the requested host support."""
    packet = _load_packet()
    packet["launch_packet"]["go_no_go"]["private_ops_dry_run"]["decision_policy"] = (
        "submission remains blocked until route support is proven."
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "private-ops dry run must gate imech156-u support" in str(exc)
    else:
        raise AssertionError("packet should reject a decision policy missing the host gate")


def test_issue_3810_packet_rejects_stale_live_issue_label() -> None:
    """A packet with changed live issue state must be refreshed before submit."""
    packet = _load_packet()
    packet["launch_packet"]["live_issue_state"]["required_label"] = "state:ready"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "live issue state must record state:running blocker" in str(exc)
    else:
        raise AssertionError("packet should reject stale live issue label")


def test_issue_3810_packet_rejects_missing_private_ops_dry_run() -> None:
    """The packet must name the submit-host dry-run evidence contract."""
    packet = _load_packet()
    packet["launch_packet"]["go_no_go"].pop("private_ops_dry_run")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "private_ops_dry_run must be a mapping" in str(exc)
    else:
        raise AssertionError("packet should reject missing private-ops dry run")


def test_issue_3810_packet_rejects_unblocked_interpretation_gate() -> None:
    """The active run must stay analysis-blocking until retention reconciles."""
    packet = _load_packet()
    packet["launch_packet"]["interpretation_gate"]["status"] = "ready_for_claims"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "interpretation gate must stay blocked" in str(exc)
    else:
        raise AssertionError("packet should reject an unblocked interpretation gate")


def test_issue_3810_packet_rejects_missing_interpretation_evidence() -> None:
    """Report interpretation requires retained evidence and analysis inputs."""
    packet = _load_packet()
    packet["launch_packet"]["interpretation_gate"]["required_evidence"].remove(
        "external_artifact_pointer"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "interpretation gate evidence missing" in str(exc)
    else:
        raise AssertionError("packet should reject missing interpretation evidence")


def test_issue_3810_packet_rejects_claim_promotion_policy() -> None:
    """Passing the public packet must not imply benchmark claim promotion."""
    packet = _load_packet()
    packet["launch_packet"]["interpretation_gate"]["decision_policy"] = (
        "Reports may be promoted after the local public packet passes."
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "interpretation gate must block claim promotion" in str(exc)
    else:
        raise AssertionError("packet should reject weak interpretation policy")


def test_issue_3810_packet_rejects_missing_go_no_go_status() -> None:
    """The packet must say what local decision command is safe to run."""
    packet = _load_packet()
    packet["launch_packet"].pop("go_no_go")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "go_no_go" in str(exc)
    else:
        raise AssertionError("packet should require go/no-go status")


def test_issue_3810_packet_rejects_null_go_no_go_command() -> None:
    """An explicit null decision command must fail closed, not coerce to 'None'."""
    packet = _load_packet()
    packet["launch_packet"]["go_no_go"]["exact_local_decision_command"] = None

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "exact local decision command missing" in str(exc)
    else:
        raise AssertionError("packet should reject a null local decision command")


def test_issue_3810_packet_rejects_horizon_only_claim_boundary() -> None:
    """The packet keeps the multi-factor comparison caveat mandatory."""
    packet = _load_packet()
    packet["campaign"]["claim_boundary"] = "Launch packet only. It does not run a benchmark."

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "horizon-only" in str(exc)
    else:
        raise AssertionError("packet should reject missing comparison caveat")


def test_issue_3810_packet_rejects_low_exposure_success_evidence() -> None:
    """Low-exposure successes stay diagnostic instead of success evidence."""
    packet = _load_packet()
    packet["launch_packet"]["wait_it_out_guard"]["low_exposure_success_status"] = (
        "successful_evidence"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "low exposure must be diagnostic" in str(exc)
    else:
        raise AssertionError("packet should reject low-exposure success evidence")


def test_issue_3810_packet_rejects_null_list_fields_cleanly() -> None:
    """Explicit-null list fields fail closed with PacketError, not TypeError."""
    packet = _load_packet()
    packet["metrics"]["ids"] = None

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "SNQI metric required" in str(exc)
    else:
        raise AssertionError("packet should reject a null metrics.ids list")


def test_issue_3810_packet_rejects_blank_matrix_path() -> None:
    """A blank scenario matrix path must not pass the repo-relative check."""
    packet = _load_packet()
    packet["scenario_suite"]["matrix_path"] = ""

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "scenario matrix must be repo-relative" in str(exc)
    else:
        raise AssertionError("packet should reject a blank scenario matrix path")


def test_issue_3810_packet_rejects_extra_seed() -> None:
    """The fixed S30 denominator must not silently expand."""
    packet = _load_packet()
    packet["seed_policy"]["seeds"].append(141)

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "exactly 30 S30 seeds" in str(exc)
    else:
        raise AssertionError("packet should reject extra S30 seed")


def test_issue_3810_packet_rejects_duplicate_seed() -> None:
    """The fixed S30 denominator must not contain duplicate seeds."""
    packet = _load_packet()
    packet["seed_policy"]["seeds"][-1] = packet["seed_policy"]["seeds"][0]

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "seed list must not contain duplicates" in str(exc)
    else:
        raise AssertionError("packet should reject duplicate S30 seed")


def test_issue_3810_packet_rejects_missing_retention_paths() -> None:
    """The packet must keep durable evidence retention paths reviewable."""
    packet = _load_packet()
    packet["durable_evidence"].pop("retention_paths")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "retention_paths must be a mapping" in str(exc)
    else:
        raise AssertionError("packet should reject missing retention paths")


def test_issue_3810_packet_rejects_untracked_raw_output_policy() -> None:
    """Raw benchmark outputs must not become the durable evidence plan."""
    packet = _load_packet()
    packet["durable_evidence"]["local_output_boundary"] = (
        "Keep generated outputs as durable benchmark evidence."
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "local output boundary must mention raw JSONL" in str(exc)
    else:
        raise AssertionError("packet should reject weak local output boundary")


def test_issue_3810_packet_rejects_missing_campaign_id() -> None:
    """A missing campaign id fails closed instead of escaping as KeyError."""
    packet = _load_packet()
    packet["campaign"].pop("id", None)

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "campaign.id required" in str(exc)
    else:
        raise AssertionError("packet should reject a missing campaign id")


def test_issue_3810_packet_cli_returns_2_on_malformed_packet(tmp_path) -> None:
    """The CLI converts a malformed packet into a clean exit 2, not a traceback."""
    bad_packet = tmp_path / "bad_packet.yaml"
    bad_packet.write_text("schema_version: research-campaign-manifest.v0.1\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(bad_packet), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "campaign must be a mapping" in payload["error"]
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


def test_issue_3810_packet_cli_json() -> None:
    """The command-line validator emits a compact JSON pass summary."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packet", str(PACKET), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"compute_submit_authorized": false' in completed.stdout
    assert '"max_episode_steps": 600' in completed.stdout


def test_issue_3810_packet_rejects_missing_validation_commands() -> None:
    """Packet validation must include all required command markers."""
    packet = _load_packet()
    packet["validation"]["commands"] = [
        "echo noop",
    ]

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "validation.commands missing required marker" in str(exc)
    else:
        raise AssertionError("packet should reject missing validation command markers")


def test_issue_3810_packet_rejects_route_path_regression() -> None:
    """Route commands must remain private-ops-backed until submit-host refresh."""
    packet = _load_packet()
    packet["launch_packet"]["route"]["queue_summary_command"] = "python scripts/dev/queue_summary.py"

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "route.queue_summary_command must be absolute" in str(exc)
    else:
        raise AssertionError("packet should reject non-private-ops queue summary command")


def test_issue_3810_packet_rejects_missing_snqi_inputs() -> None:
    """SNQI recalibration inputs and scripts must be explicit and locked."""
    packet = _load_packet()
    packet["launch_packet"]["snqi_recalibration"].pop("inputs")

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "inputs must be a mapping" in str(exc)
    else:
        raise AssertionError("packet should reject missing SNQI inputs")


def test_issue_3810_packet_rejects_weak_horizon_report_command() -> None:
    """Horizon sensitivity report command must include the long/short campaign refs."""
    packet = _load_packet()
    packet["launch_packet"]["horizon_sensitivity_report"]["command"] = (
        "uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py "
        "--long-campaign output/benchmarks/issue_3810_comprehensive_h600_snqi/reports"
    )

    try:
        _MODULE.validate_packet(packet)
    except _MODULE.PacketError as exc:
        assert "horizon report command missing --short-campaign-reference" in str(exc)
    else:
        raise AssertionError("packet should reject incomplete horizon command")
