#!/usr/bin/env python3
"""Fail-closed readiness check for the issue #5348 S30 attempt-6 PPO-resume launch packet.

This checker validates the resume packet contract only. It never resumes the PPO arm, runs a
benchmark, submits SLURM/GPU, aggregates results, or promotes evidence. It fails closed if the
packet would let a GPU operator restart PPO from zero, re-run the five clean arms, lose the 562
preserved PPO episodes, skip the #5347 gate, change the report-builder contract, or self-authorize
compute submission.

Exit codes:
- 0: ``ready`` — packet is a valid, fail-closed resume contract.
- 1: ``blocked`` — packet is well-formed but a contract requirement is unmet.
- 2: ``malformed`` — packet is missing or not a YAML mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5348_s30_attempt6_resume_launch_packet.yaml"
SCHEMA_VERSION = "s30-attempt6-resume-launch-packet.v1"
EXPECTED_CAMPAIGN_CONFIG = (
    "configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml"
)
EXPECTED_PUBLICATION_FILES = (
    "reports/campaign_table.csv",
    "campaign_manifest.json",
    "run_meta.json",
)
EPISODES_PER_ARM = 1440
PRESERVED_PPO_EPISODES = 562
REMAINING_PPO_EPISODES = 878
CLEAN_ARM_COUNT = 5
ARM_COUNT = 6
EXPECTED_CLEAN_ARMS = frozenset(
    {
        "scenario_adaptive_hybrid_orca_v1",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "orca",
    }
)
REQUIRED_FAIL_CLOSED_STATUSES = frozenset(
    {"fallback", "degraded", "unavailable", "failed", "partial", "not_available", "diagnostic_only"}
)

READY = "ready"
BLOCKED = "blocked"
MALFORMED = "malformed"
EXIT_CODES = {READY: 0, BLOCKED: 1, MALFORMED: 2}


class PacketError(ValueError):
    """Raised when the launch packet would not fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PacketError(f"{key} must be a mapping")
    return value


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    """Return a compact validation summary for a fail-closed resume packet."""
    _require(packet.get("schema_version") == SCHEMA_VERSION, "unexpected schema_version")
    _require(packet.get("parent_issue") == 5348, "parent_issue must be 5348")
    _require(packet.get("no_benchmark_result_claim") is True, "no_benchmark_result_claim required")
    claim_boundary = str(packet.get("claim_boundary", ""))
    _require(
        "no SLURM/GPU submission" in claim_boundary,
        "claim_boundary must forbid SLURM/GPU submission",
    )
    _require(
        "does not exist until the PPO resume completes" in claim_boundary,
        "claim_boundary must state the six-arm result does not yet exist",
    )

    # --- Step-1 gate on #5347 / #5360 must be satisfied before resume is legitimate.
    gating = _require_mapping(packet, "gating")
    _require(gating.get("blocking_issue") == 5347, "gating.blocking_issue must be 5347")
    _require(gating.get("blocking_pr") == 5360, "gating.blocking_pr must be 5360")
    _require(
        gating.get("gate_status") == "satisfied", "gate_status must be satisfied (#5360 merged)"
    )

    campaign = _require_mapping(packet, "campaign")
    _require(str(campaign.get("job_id")) == "13376", "campaign.job_id must be 13376")
    _require(campaign.get("attempt") == 6, "campaign.attempt must be 6")
    _require(campaign.get("episodes_per_arm") == EPISODES_PER_ARM, "episodes_per_arm must be 1440")
    _require(campaign.get("arm_count") == ARM_COUNT, "arm_count must be 6")
    config_path = campaign.get("config")
    _require(
        config_path == EXPECTED_CAMPAIGN_CONFIG,
        f"campaign.config must be {EXPECTED_CAMPAIGN_CONFIG}",
    )
    config_sha256 = campaign.get("config_sha256")
    config_file = REPO_ROOT / EXPECTED_CAMPAIGN_CONFIG
    _require(config_file.is_file(), f"campaign config is missing: {EXPECTED_CAMPAIGN_CONFIG}")
    actual_config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest()
    _require(
        config_sha256 == actual_config_sha256,
        "campaign.config_sha256 does not match the checked-in campaign config",
    )

    # --- Step-1 preservation: five clean arms complete, 562 PPO rows preserved, no restart.
    preservation = _require_mapping(packet, "preservation")
    clean_arms = preservation.get("clean_arms")
    _require(isinstance(clean_arms, list), "preservation.clean_arms must be a list")
    _require(len(clean_arms) == CLEAN_ARM_COUNT, "there must be exactly 5 clean arms")
    clean_keys = set()
    clean_episode_total = 0
    for arm in clean_arms:
        _require(isinstance(arm, dict), "each clean arm must be a mapping")
        _require(
            arm.get("episodes_complete") == EPISODES_PER_ARM,
            f"clean arm {arm.get('key')!r} must have 1440 complete episodes",
        )
        clean_keys.add(str(arm.get("key")))
        clean_episode_total += EPISODES_PER_ARM
    _require(clean_keys == set(EXPECTED_CLEAN_ARMS), "clean arm keys do not match the S30 roster")

    resume_arm = _require_mapping(preservation, "resume_arm")
    _require(resume_arm.get("key") == "ppo", "resume_arm.key must be ppo")
    preserved = resume_arm.get("episodes_preserved")
    remaining = resume_arm.get("episodes_remaining")
    _require(preserved == PRESERVED_PPO_EPISODES, "PPO must preserve exactly 562 episodes")
    _require(remaining == REMAINING_PPO_EPISODES, "PPO must have exactly 878 remaining episodes")
    _require(
        preserved + remaining == EPISODES_PER_ARM,
        "PPO preserved + remaining must equal 1440",
    )
    _require(
        resume_arm.get("episodes_total_target") == EPISODES_PER_ARM,
        "PPO episodes_total_target must be 1440",
    )
    _require(
        resume_arm.get("resume_after_episode") == PRESERVED_PPO_EPISODES,
        "resume must start after episode 562",
    )
    _require(resume_arm.get("restart_from_zero") is False, "PPO must not restart from zero")
    expected_preserved_total = clean_episode_total + PRESERVED_PPO_EPISODES
    _require(
        preservation.get("total_preserved_episodes") == expected_preserved_total,
        f"total_preserved_episodes must be {expected_preserved_total}",
    )

    # --- Step-2 resume: PPO only, resume=True, no self-authorized compute.
    resume_execution = _require_mapping(packet, "resume_execution")
    _require(resume_execution.get("resume") is True, "resume_execution.resume must be true")
    _require(resume_execution.get("arms_to_run") == ["ppo"], "only the ppo arm may be resumed")
    _require(
        set(resume_execution.get("arms_to_skip") or []) == set(EXPECTED_CLEAN_ARMS),
        "the five clean arms must be skipped, not re-run",
    )
    _require(
        str(resume_execution.get("entry_point", "")).endswith("run_camera_ready_benchmark.py"),
        "resume entry_point must be run_camera_ready_benchmark.py",
    )
    _require(
        "campaign_id" in (resume_execution.get("operator_supplied") or []),
        "campaign_id must be operator-supplied (fail-closed until known)",
    )
    _require(
        resume_execution.get("submit_slurm_from_this_issue") is False,
        "packet must not submit SLURM from this issue",
    )
    _require(
        resume_execution.get("compute_submit_authorized") is False,
        "compute_submit_authorized must be false",
    )

    # --- Step-3 aggregation: unchanged report-builder contract, six rows.
    aggregation = _require_mapping(packet, "aggregation")
    _require(
        aggregation.get("report_builder_contract_changed") is False,
        "report-builder contract must be unchanged",
    )
    _require(aggregation.get("expected_rows") == ARM_COUNT, "aggregation must expect 6 rows")
    _require(
        aggregation.get("expected_episodes_per_row") == EPISODES_PER_ARM,
        "aggregation must expect 1440 episodes per row",
    )
    _require(
        str(aggregation.get("expected_output", "")).endswith("campaign_table.csv"),
        "aggregation output must be campaign_table.csv",
    )

    # --- Step-4 publication: durable evidence with provenance, no raw artifacts in git.
    publication = _require_mapping(packet, "publication")
    required_files = publication.get("required_files")
    _require(
        required_files == list(EXPECTED_PUBLICATION_FILES),
        "publication.required_files must declare the complete campaign output contract",
    )
    _require(
        str(publication.get("evidence_dir", "")).startswith("docs/context/evidence/"),
        "publication evidence_dir must be under docs/context/evidence/",
    )
    _require(
        "sha256_manifest" in (publication.get("provenance") or []),
        "publication provenance must include sha256_manifest",
    )
    _require(
        publication.get("raw_episode_jsonl_in_git") is False,
        "raw episode JSONL must stay out of git",
    )

    # --- Fail-closed row policy.
    fail_closed = _require_mapping(packet, "fail_closed_policy")
    _require(
        set(fail_closed.get("valid_row_statuses") or []) == {"native", "adapter"},
        "only native/adapter rows may count as benchmark evidence",
    )
    _require(
        REQUIRED_FAIL_CLOSED_STATUSES <= set(fail_closed.get("fail_closed_statuses") or []),
        "fail_closed_statuses missing required weak-row statuses",
    )

    # --- Execution boundary.
    boundary = _require_mapping(packet, "execution_boundary")
    _require(
        boundary.get("full_campaign_in_this_issue") is False,
        "full_campaign_in_this_issue must be false",
    )
    _require(
        boundary.get("submit_slurm_from_this_issue") is False,
        "execution_boundary must not submit SLURM from this issue",
    )
    _require(
        boundary.get("status_until_run") == "ready_for_resume_submission",
        "status_until_run must be ready_for_resume_submission",
    )

    return {
        "ok": True,
        "issue": 5348,
        "schema_version": SCHEMA_VERSION,
        "gate_status": gating.get("gate_status"),
        "job_id": str(campaign.get("job_id")),
        "attempt": campaign.get("attempt"),
        "clean_arm_count": len(clean_arms),
        "ppo_preserved": preserved,
        "ppo_remaining": remaining,
        "resume_after_episode": resume_arm.get("resume_after_episode"),
        "restart_from_zero": resume_arm.get("restart_from_zero"),
        "total_preserved_episodes": preservation.get("total_preserved_episodes"),
        "arms_to_run": resume_execution.get("arms_to_run"),
        "report_builder_contract_changed": aggregation.get("report_builder_contract_changed"),
        "compute_submit_authorized": resume_execution.get("compute_submit_authorized"),
        "status_until_run": boundary.get("status_until_run"),
    }


def _load_packet(path: Path) -> dict[str, Any]:
    resolved_path = path if path.is_absolute() else REPO_ROOT / path
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError(f"{resolved_path} must contain a YAML mapping")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load the packet, validate it, and print a status summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", type=Path, default=DEFAULT_PACKET, help="Path to the resume launch packet YAML."
    )
    parser.add_argument("--json", action="store_true", help="Emit a compact JSON summary.")
    args = parser.parse_args(argv)

    try:
        packet = _load_packet(args.packet)
    except (FileNotFoundError, PacketError) as exc:
        summary = {"ok": False, "status": MALFORMED, "issue": 5348, "error": str(exc)}
        print(json.dumps(summary) if args.json else f"malformed: {exc}")
        return EXIT_CODES[MALFORMED]

    try:
        summary = validate_packet(packet)
    except PacketError as exc:
        summary = {"ok": False, "status": BLOCKED, "issue": 5348, "error": str(exc)}
        print(json.dumps(summary) if args.json else f"blocked: {exc}")
        return EXIT_CODES[BLOCKED]

    summary["status"] = READY
    if args.json:
        print(json.dumps(summary))
    else:
        print(
            "ready: issue #5348 S30 attempt-6 resume packet is a valid fail-closed contract "
            f"(PPO {summary['ppo_preserved']}+{summary['ppo_remaining']}=1440, "
            f"resume after {summary['resume_after_episode']}, gate {summary['gate_status']})"
        )
    return EXIT_CODES[READY]


if __name__ == "__main__":
    raise SystemExit(main())
