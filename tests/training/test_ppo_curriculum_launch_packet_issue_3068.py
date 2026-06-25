"""Tests for the issue #3068 PPO density/complexity curriculum launch packet.

This packet is a PRE-LAUNCH specification. The tests assert that the checked-in packet
is structurally complete, references real configs with matching sha256 checksums, encodes
the three competing explanations as discriminating checks, matches the curriculum and
baseline training budget, and carries an explicit no-training-result-claim status.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKET_PATH = _REPO_ROOT / "configs/training/ppo_curriculum_issue_3068_launch_packet.yaml"
_DOC_PATH = _REPO_ROOT / "docs/context/issue_3068_ppo_curriculum_launch_packet.md"


@pytest.fixture(scope="module")
def packet() -> dict[str, object]:
    """Load the checked-in #3068 curriculum launch packet."""
    return yaml.safe_load(_PACKET_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_parses_and_has_required_top_level_keys(packet: dict[str, object]) -> None:
    """The packet must parse and expose every required top-level section."""
    required = {
        "schema_version",
        "campaign_id",
        "generating_commit",
        "evidence_status",
        "no_training_result_claim",
        "hypothesis",
        "discriminating_checks",
        "training_starting_points",
        "curriculum",
        "baseline_comparator",
        "seeds",
        "run_budget",
        "stop_rule",
        "metrics",
        "expected_artifacts",
        "validation_command",
        "execution_boundary",
        "durable_storage_plan",
    }
    assert required.issubset(packet.keys())
    assert packet["schema_version"] == "ppo-curriculum-launch-packet.v1"
    assert packet["campaign_id"] == "issue_3068_ppo_curriculum_v1"


def test_no_training_result_claim(packet: dict[str, object]) -> None:
    """The pre-launch packet must explicitly carry no training-result claim."""
    assert packet["no_training_result_claim"] is True
    assert packet["evidence_status"] == "proposal"
    assert "no_claim_statement" in packet
    assert packet["execution_boundary"]["full_training_in_this_issue"] is False
    assert packet["execution_boundary"]["submit_slurm_from_this_issue"] is True


def test_referenced_configs_exist_and_checksums_match(packet: dict[str, object]) -> None:
    """Every referenced config path must exist with a matching sha256 checksum."""
    checksums = packet["training_starting_points"]["checksums"]
    assert checksums, "expected non-empty checksum map"
    for rel_path, expected in checksums.items():
        target = _REPO_ROOT / rel_path
        assert target.is_file(), f"referenced config missing: {rel_path}"
        assert _sha256(target) == expected, f"checksum mismatch for {rel_path}"


def test_referenced_config_paths_are_listed_in_checksums(packet: dict[str, object]) -> None:
    """The concrete config pointers must also exist on disk."""
    starts = packet["training_starting_points"]
    checksums = starts["checksums"]
    for key in (
        "base_ppo_training_config",
        "ppo_algo_config",
        "guarded_algo_config",
        "scenario_config",
        "seed_manifest",
    ):
        rel_path = starts[key]
        assert (_REPO_ROOT / rel_path).is_file(), f"{key} missing: {rel_path}"
        assert rel_path in checksums, f"{key} not checksummed: {rel_path}"


def test_competing_explanations_have_discriminating_checks(packet: dict[str, object]) -> None:
    """The three competing explanations must each map to a discriminating check."""
    explanations = packet["hypothesis"]["competing_explanations"]
    expected_ids = {
        "extra_budget_not_curriculum",
        "train_curve_not_final_benchmark",
        "insufficient_provenance",
    }
    assert {e["id"] for e in explanations} == expected_ids

    checks = packet["discriminating_checks"]
    for explanation in explanations:
        assert explanation["discriminating_check"] in checks


def test_matched_budget_check(packet: dict[str, object]) -> None:
    """Curriculum and baseline budgets must be matched (rules out the extra-budget story)."""
    matched = packet["discriminating_checks"]["matched_budget"]
    assert matched["budget_matched"] is True
    assert matched["curriculum_total_timesteps"] == matched["baseline_total_timesteps"]

    curriculum_total = packet["curriculum"]["total_timesteps"]
    baseline_total = packet["baseline_comparator"]["total_timesteps"]
    assert curriculum_total == baseline_total == matched["curriculum_total_timesteps"]

    # Curriculum stage timesteps must sum to the declared total.
    stage_sum = sum(stage["timesteps"] for stage in packet["curriculum"]["stages"])
    assert stage_sum == curriculum_total


def test_train_curve_and_final_benchmark_both_present(packet: dict[str, object]) -> None:
    """Both train-curve AND final-benchmark metrics must be declared."""
    check = packet["discriminating_checks"]["train_curve_and_final_benchmark"]
    assert check["train_curve_metrics"]
    assert check["final_benchmark_metrics"]

    metrics = packet["metrics"]
    assert metrics["train_curve"]
    assert metrics["final_benchmark"]
    assert metrics["primary"] in metrics["final_benchmark"]


def test_curriculum_schedule_is_density_complexity(packet: dict[str, object]) -> None:
    """The curriculum must vary density/complexity across ordered stages."""
    curriculum = packet["curriculum"]
    assert "ped_density" in curriculum["variable"]
    stages = curriculum["stages"]
    assert len(stages) >= 2
    densities = [stage["ped_density"] for stage in stages]
    assert densities == sorted(densities), "ped_density should be non-decreasing"
    for stage in stages:
        assert {"id", "ped_density", "max_peds_per_group", "timesteps"}.issubset(stage)


def test_baseline_comparator_is_fixed_difficulty(packet: dict[str, object]) -> None:
    """The baseline comparator must be a fixed-difficulty PPO arm with a real base config."""
    baseline = packet["baseline_comparator"]
    assert baseline["id"]
    assert "ped_density" in baseline
    assert baseline["base_ppo_training_config"] in packet["training_starting_points"]["checksums"]


def test_seeds_budget_stop_rule_present(packet: dict[str, object]) -> None:
    """Seeds, run budget (matched between arms), and a stop rule must all be declared."""
    seeds = packet["seeds"]
    assert seeds["training"]
    assert seeds["evaluation"]["eval"]

    budget = packet["run_budget"]
    assert budget["total_training_runs"] == len(seeds["training"]) * len(budget["arms"])
    assert budget["budget_matched_between_arms"] is True

    stop_rule = packet["stop_rule"]
    assert "max_timesteps_per_run" in stop_rule
    assert "convergence_success_rate" in stop_rule


def test_expected_artifacts_and_durable_policy(packet: dict[str, object]) -> None:
    """Expected artifacts must be listed and the durable-storage policy must keep them out of git."""
    artifacts = packet["expected_artifacts"]["items"]
    assert isinstance(artifacts, list)
    assert artifacts

    durable = packet["durable_storage_plan"]
    assert durable["checkpoints_in_git"] is False
    assert durable["raw_logs_in_git"] is False
    assert durable["required_before_training"]


def test_validation_command_present(packet: dict[str, object]) -> None:
    """A validation command referencing the packet must be declared."""
    assert "ppo_curriculum_issue_3068_launch_packet.yaml" in packet["validation_command"]


def test_context_doc_links_packet() -> None:
    """The context doc must exist, link the packet, and state the no-claim boundary."""
    assert _DOC_PATH.is_file()
    text = _DOC_PATH.read_text(encoding="utf-8")
    assert "configs/training/ppo_curriculum_issue_3068_launch_packet.yaml" in text
    assert "No training-result" in text or "no training-result" in text


def test_queue_hint_points_to_launchable_issue_3068_scout(packet: dict[str, object]) -> None:
    """Ready queue hint must reference a real #3068 PPO training config."""
    hint = packet["queue_hint"]
    config_path = _REPO_ROOT / hint["config"]
    assert hint["public_issue"] == "ll7/robot_sf_ll7#3068"
    assert hint["submit_ready"] is True
    assert hint["job_class"] == "robot_sf_gpu_training_small"
    assert config_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["policy_id"] == "ppo_expert_issue_3068_curriculum_full_surface_scout_12m_env22"
    assert config["scenario_config"].endswith("ppo_all_available_training_v1_h500_schedule.yaml")
    assert config["total_timesteps"] == 12000000
    assert config["tracking"]["wandb"]["enabled"] is True
    assert "issue-3068" in config["tracking"]["wandb"]["tags"]
