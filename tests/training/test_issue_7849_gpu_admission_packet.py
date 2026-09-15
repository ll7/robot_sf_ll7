"""Focused checks for the issue #7849 GPU admission candidate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.research_answerability import evaluate_answerability
from scripts.dev.preflight_launch_packet import preflight_launch_packet
from scripts.training import train_ppo, train_recurrent_ppo

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = (
    REPO_ROOT / "configs/training/comparison_matrix/issue_7849_ppo_rppo_gpu_admission_v1.yaml"
)
INVENTORY_PATH = (
    REPO_ROOT / "configs/training/comparison_matrix/issue_7849_ppo_rppo_source_inventory_v1.yaml"
)
FULL_PPO_PATH = REPO_ROOT / "configs/training/ppo/issue_7849_ppo_full_v1.yaml"
FULL_RECURRENT_PATH = REPO_ROOT / "configs/training/ppo/issue_7849_recurrent_ppo_full_v1.yaml"
CANARY_PPO_PATH = REPO_ROOT / "configs/training/ppo/issue_7849_ppo_gate1_canary_v1.yaml"
CANARY_RECURRENT_PATH = (
    REPO_ROOT / "configs/training/ppo/issue_7849_recurrent_ppo_gate1_canary_v1.yaml"
)


def _load(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_is_exact_source_bound_and_fail_closed() -> None:
    packet = _load(PACKET_PATH)
    report = preflight_launch_packet(PACKET_PATH, repo_root=REPO_ROOT)

    assert report["ready"] is True, report["reasons"]
    assert packet["issue"] == 7849
    assert packet["execution_authorized"] is False
    assert packet["claim_eligible"] is False
    assert packet["source"]["base_commit"] == "1bb17787b6000a2aa7b5149d201a408e3eedcef5"
    assert packet["source"]["inventory_sha256"] == _sha256(INVENTORY_PATH)
    assert packet["full_budget"]["independent_run_count"] == 10
    assert packet["full_budget"]["total_environment_steps"] == 150_000_000
    assert packet["execution_boundary"]["submit_slurm_from_this_issue"] is False
    assert packet["execution_boundary"]["full_training_in_this_pr"] is False
    assert packet["arms"]["ppo"]["current_seed_execution"] == "per_seed_cli_override_available"
    seed_commands = packet["full_budget"]["command_shapes"]["ppo_per_seed"]
    assert [entry["seed"] for entry in seed_commands] == [123, 231, 777, 992, 1337]
    for entry in seed_commands:
        assert f"--seed {entry['seed']}" in entry["command"]
        assert entry["run_id"] in entry["command"]
        assert entry["artifact_root"] in entry["command"]
    assert packet["checkpoint_evaluation"]["recurrent_metric_producer"]["status"] == (
        "available_native"
    )


def test_inventory_rehashes_every_declared_source_and_runtime_input() -> None:
    inventory = _load(INVENTORY_PATH)
    assert inventory["source_base_commit"] == "1bb17787b6000a2aa7b5149d201a408e3eedcef5"

    entries = []
    for section in ("files", "scenario_inputs", "map_inputs"):
        section_entries = inventory.get(section)
        assert isinstance(section_entries, list)
        entries.extend(section_entries)

    assert len(entries) >= 80
    paths = set()
    for entry in entries:
        assert isinstance(entry, dict)
        raw_path = entry.get("path")
        expected_sha = entry.get("sha256")
        assert isinstance(raw_path, str)
        assert isinstance(expected_sha, str)
        path = REPO_ROOT / raw_path
        assert path.is_file(), raw_path
        assert _sha256(path) == expected_sha, raw_path
        paths.add(raw_path)

    assert "maps/registry.yaml" in paths
    assert "maps/svg_maps/classic_crossing.svg" in paths
    assert "configs/training/ppo/issue_7849_ppo_full_v1.yaml" in paths
    assert "configs/training/ppo/issue_7849_recurrent_ppo_full_v1.yaml" in paths


def test_full_and_canary_configs_resolve_to_registered_contract() -> None:
    full_ppo = train_ppo.load_expert_training_config(FULL_PPO_PATH)
    full_recurrent = train_recurrent_ppo.load_recurrent_ppo_config(FULL_RECURRENT_PATH)
    canary_ppo = train_ppo.load_expert_training_config(CANARY_PPO_PATH)
    canary_recurrent = train_recurrent_ppo.load_recurrent_ppo_config(CANARY_RECURRENT_PATH)

    assert full_ppo.total_timesteps == 15_000_000
    assert full_recurrent.base.total_timesteps == 15_000_000
    assert full_ppo.seeds == (123, 231, 777, 992, 1337)
    assert full_recurrent.base.seeds == full_ppo.seeds
    assert full_ppo.evaluation.evaluation_episodes == 100
    assert full_recurrent.base.evaluation.evaluation_episodes == 100
    assert train_ppo._build_eval_steps(15_000_000, full_ppo.evaluation.step_schedule) == list(
        range(1_000_000, 15_000_001, 1_000_000)
    )

    assert full_ppo.env_overrides["observation_mode"] == "default_gym"
    assert full_recurrent.base.env_overrides["observation_mode"] == "default_gym"
    assert full_ppo.env_factory_kwargs["reward_name"] == "route_completion_v2"
    assert full_recurrent.base.env_factory_kwargs["reward_name"] == "route_completion_v2"
    assert full_recurrent.algorithm == "recurrent_ppo"
    assert full_recurrent.recurrent_policy == "MultiInputLstmPolicy"
    assert full_recurrent.recurrent_ppo_hyperparams["device"] == "cuda"

    assert canary_ppo.total_timesteps == 2_048
    assert canary_recurrent.base.total_timesteps == 2_048
    assert canary_ppo.seeds == (4014,)
    assert canary_recurrent.base.seeds == (4014,)
    assert set(canary_ppo.seeds).isdisjoint(full_ppo.seeds)
    assert set(canary_recurrent.base.seeds).isdisjoint(full_recurrent.base.seeds)
    assert canary_ppo.evaluation.evaluation_episodes == 3
    assert canary_recurrent.base.evaluation.evaluation_episodes == 3
    assert train_ppo._build_eval_steps(2_048, canary_ppo.evaluation.step_schedule) == [1_024, 2_048]


def test_feed_forward_seed_override_isolated_and_declared(tmp_path: Path, monkeypatch) -> None:
    """One CLI-equivalent seed override creates a distinct artifact identity."""
    config = train_ppo.load_expert_training_config(FULL_PPO_PATH)
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    parsed = train_ppo.build_arg_parser().parse_args(
        [
            "--config",
            str(FULL_PPO_PATH),
            "--seed",
            "231",
            "--run-id",
            "issue7849-ppo-seed-231",
        ],
    )
    assert parsed.training_seed == 231
    assert parsed.run_id == "issue7849-ppo-seed-231"

    result = train_ppo.run_expert_training(
        config,
        config_path=FULL_PPO_PATH,
        config_sha256=_sha256(FULL_PPO_PATH),
        dry_run=True,
        training_seed=231,
        run_id="issue7849-ppo-seed-231",
    )

    assert result.config.seeds == (231,)
    assert result.config.policy_id == "ppo_issue_7849_full_v1_seed_231"
    assert result.expert_artifact.seeds == (231,)
    assert result.training_run_artifact.seeds == (231,)
    assert result.training_run_artifact.run_id == "issue7849-ppo-seed-231"
    assert result.checkpoint_path.name == "ppo_issue_7849_full_v1_seed_231.zip"
    assert "training_seed_override=231" in result.training_run_artifact.notes

    with pytest.raises(ValueError, match="not declared in config.seeds"):
        train_ppo.run_expert_training(config, dry_run=True, training_seed=999)
    with pytest.raises(ValueError, match="single path component"):
        train_ppo.run_expert_training(
            config,
            dry_run=True,
            training_seed=231,
            run_id="../unsafe",
        )


def test_canary_inputs_and_answerability_are_explicitly_blocked() -> None:
    packet = _load(PACKET_PATH)
    canary_inputs = packet["gate_1_canary"]["canary_inputs"]
    assert _sha256(CANARY_PPO_PATH) == canary_inputs["ppo"]["config_sha256"]
    assert _sha256(CANARY_RECURRENT_PATH) == canary_inputs["recurrent_ppo"]["config_sha256"]

    canary = packet["gate_1_canary"]
    assert canary["status"] == "not_run"
    assert canary["outcome_use"] == "forbidden"
    assert "sbatch" not in canary["ppo_command"]
    assert "sbatch" not in canary["recurrent_ppo_command"]
    assert "train_ppo.py" in canary["ppo_command"]
    assert "train_recurrent_ppo.py" in canary["recurrent_ppo_command"]

    answerability = evaluate_answerability(packet["answerability"])
    assert answerability.state == "blocked_missing_producer"
    assert "full_matrix_episode_rows" in answerability.reasons[0]
    assert packet["domain_aware_approval"]["approved"] is False
    assert packet["compute_authorization"]["authorized"] is False
    preserved = packet["preserved_vs_executable"]
    assert (
        preserved["preserved_outputs"][0]["status"] == "preserved_but_insufficient_for_issue_1496"
    )
    assert preserved["executable_training_packets"]["issue_7849_full_matrix"] == "none"
