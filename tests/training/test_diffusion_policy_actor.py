"""CPU smoke tests for issue #4010 diffusion-policy training artifacts."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.training.diffusion_policy import (
    DIAGNOSTIC_PACKET_SCHEMA_VERSION,
    MULTIMODAL_PROBE_SCHEMA_VERSION,
    REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION,
    SMOKE_MANIFEST_SCHEMA_VERSION,
    DiffusionPolicyTrainingSmokeConfig,
    build_diagnostic_packet,
    build_multimodal_probe,
    build_representative_rollout,
    load_smoke_config,
    run_training_smoke,
)
from robot_sf.training.diffusion_policy import (
    main as diffusion_training_main,
)


def _map_runner_obs() -> dict[str, object]:
    return {
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "heading": [0.0],
            "radius": [0.3],
        },
        "pedestrians": {
            "positions": [[0.7, 0.1], [1.1, -0.2]],
            "velocities": [[-0.1, 0.0], [0.0, 0.1]],
            "radii": [0.25, 0.25],
            "count": [2],
        },
        "dt": [0.1],
    }


def test_smoke_config_rejects_non_cpu_scale_values() -> None:
    """The smoke trainer validates the narrow first-slice action contract."""
    with pytest.raises(ValueError, match="training_steps"):
        DiffusionPolicyTrainingSmokeConfig(training_steps=0)
    with pytest.raises(ValueError, match="action_dim=2"):
        DiffusionPolicyTrainingSmokeConfig(action_dim=3)


def test_load_smoke_config_accepts_nested_yaml(tmp_path) -> None:
    """Tracked YAML config shape maps to the smoke-training dataclass."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n",
        encoding="utf-8",
    )

    config = load_smoke_config(config_path)

    assert config.training_steps == 1
    assert config.batch_size == 2


def test_training_smoke_writes_checkpoint_normalizer_manifest(tmp_path) -> None:
    """CPU smoke writes the three artifacts required by the issue #4010 contract."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        artifact_prefix="unit_smoke",
    )

    artifacts = run_training_smoke(config, output_dir=tmp_path)

    assert artifacts.checkpoint_path.is_file()
    assert artifacts.normalizer_path.is_file()
    assert artifacts.manifest_path.is_file()
    normalizer = json.loads(artifacts.normalizer_path.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert normalizer["schema_version"] == "diffusion_policy_normalizer.v1"
    assert normalizer["sample_count"] > 0
    assert manifest["schema_version"] == SMOKE_MANIFEST_SCHEMA_VERSION
    assert manifest["issue"] == 4010
    assert manifest["artifacts"]["checkpoint_path"] == artifacts.checkpoint_path.name
    assert manifest["artifacts"]["normalizer_path"] == artifacts.normalizer_path.name
    assert manifest["training"]["status"] == "completed"
    assert manifest["evidence_tier"] == "diagnostic-only"
    load_contract = manifest["artifacts"]["map_runner_load_contract"]
    assert load_contract["allow_untrained_smoke"] is False
    assert load_contract["requires_checkpoint_path"] is True
    assert load_contract["requires_normalizer_path"] is True
    assert (
        load_contract["algo_config_fragment"]["checkpoint_path"] == artifacts.checkpoint_path.name
    )
    assert (
        load_contract["algo_config_fragment"]["normalizer_path"] == artifacts.normalizer_path.name
    )


def test_training_script_entrypoint_writes_manifest_path(tmp_path) -> None:
    """Canonical script command writes smoke manifest for issue #4010."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: cli_smoke\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_diffusion_policy.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest_path = output_dir / "cli_smoke.manifest.json"
    assert payload == {"manifest_path": str(manifest_path)}
    assert manifest_path.exists()


def test_training_script_can_write_diagnostic_packet(tmp_path) -> None:
    """Canonical command writes smoke-only integration packet issue #4010."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: cli_packet\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_diffusion_policy.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--write-diagnostic-packet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    packet_path = output_dir / "cli_packet.manifest.diagnostic_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert payload == {
        "diagnostic_packet_path": str(packet_path),
        "manifest_path": str(output_dir / "cli_packet.manifest.json"),
    }
    assert packet["schema_version"] == DIAGNOSTIC_PACKET_SCHEMA_VERSION
    assert packet["evidence_tier"] == "diagnostic-only"
    assert packet["acceptance_status"]["benchmark_campaign_run"] is False
    assert packet["acceptance_status"]["slurm_or_gpu_submission"] is False
    assert packet["acceptance_status"]["paper_or_dissertation_claim"] is False


def test_multimodal_probe_records_fixed_conflict_modes(tmp_path) -> None:
    """Fixed-conflict probe records diagnostic action-mode diversity."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=4,
        max_pedestrians=2,
        artifact_prefix="multimodal_probe",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    probe = build_multimodal_probe(
        artifacts.manifest,
        artifact_dir=artifacts.manifest_path.parent,
        sample_count=48,
    )
    packet = build_diagnostic_packet(artifacts.manifest, multimodal_probe=probe)

    assert probe["schema_version"] == MULTIMODAL_PROBE_SCHEMA_VERSION
    assert probe["evidence_tier"] == "diagnostic-only"
    assert probe["sample_count"] == 48
    assert probe["passed"] is True
    assert probe["distinct_core_mode_count"] >= 2
    assert packet["acceptance_status"]["multimodal_action_probe"] is True
    assert "multimodal_action_probe" not in {item["id"] for item in packet["remaining_blockers"]}


def test_representative_rollout_records_checkpoint_backed_trajectory(tmp_path) -> None:
    """Representative rollout records finite bounded diagnostic scenario commands."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=4,
        max_pedestrians=2,
        artifact_prefix="representative_rollout",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    rollout = build_representative_rollout(
        artifacts.manifest,
        artifact_dir=artifacts.manifest_path.parent,
        step_count=6,
    )
    packet = build_diagnostic_packet(artifacts.manifest, representative_rollout=rollout)

    assert rollout["schema_version"] == REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION
    assert rollout["evidence_tier"] == "diagnostic-only"
    assert rollout["scenario_family"] == "crossing"
    assert rollout["passed"] is True
    assert rollout["finite_command_count"] == 6
    assert rollout["commands_within_limits"] is True
    assert rollout["runtime_loaded_checkpoint"] is True
    assert len(rollout["trajectory"]) == 6
    assert packet["acceptance_status"]["representative_rollout"] is True
    assert "representative_rollout" not in {item["id"] for item in packet["remaining_blockers"]}


def test_training_script_can_write_multimodal_probe_and_packet(tmp_path) -> None:
    """Canonical command writes fixed-conflict probe and packet together."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 2\n"
        "  batch_size: 4\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: cli_multimodal\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_diffusion_policy.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--write-multimodal-probe",
            "--multimodal-samples",
            "48",
            "--write-diagnostic-packet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    probe_path = output_dir / "cli_multimodal.manifest.multimodal_probe.json"
    packet_path = output_dir / "cli_multimodal.manifest.diagnostic_packet.json"
    assert payload["multimodal_probe_path"] == str(probe_path)
    assert payload["diagnostic_packet_path"] == str(packet_path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert probe["schema_version"] == MULTIMODAL_PROBE_SCHEMA_VERSION
    assert probe["passed"] is True
    assert packet["multimodal_probe"]["passed"] is True


def test_training_script_can_write_representative_rollout_and_packet(tmp_path) -> None:
    """Canonical command writes representative rollout into packet."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 2\n"
        "  batch_size: 4\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: cli_rollout\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_diffusion_policy.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--write-representative-rollout",
            "--rollout-steps",
            "6",
            "--write-diagnostic-packet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    rollout_path = output_dir / "cli_rollout.manifest.representative_rollout.json"
    packet_path = output_dir / "cli_rollout.manifest.diagnostic_packet.json"
    assert payload["representative_rollout_path"] == str(rollout_path)
    assert payload["diagnostic_packet_path"] == str(packet_path)
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert rollout["schema_version"] == REPRESENTATIVE_ROLLOUT_SCHEMA_VERSION
    assert rollout["passed"] is True
    assert packet["representative_rollout"]["passed"] is True


def test_diagnostic_packet_records_checkpoint_backed_integration(tmp_path) -> None:
    """Packet records loaded checkpoint/normalizer and residual blockers together."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        artifact_prefix="diagnostic_packet",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)
    _policy, meta = _build_policy(
        "diffusion_policy",
        {
            "checkpoint_path": str(artifacts.checkpoint_path),
            "normalizer_path": str(artifacts.normalizer_path),
            "deterministic": True,
            "seed": 4010,
            "max_pedestrians": config.max_pedestrians,
            "max_linear_speed": config.max_linear_speed,
            "max_angular_speed": config.max_angular_speed,
            "num_action_samples": config.num_action_samples,
            "denoising_steps": config.denoising_steps,
        },
        robot_kinematics="differential_drive",
    )

    packet = build_diagnostic_packet(artifacts.manifest, map_runner_metadata=meta)

    assert packet["schema_version"] == DIAGNOSTIC_PACKET_SCHEMA_VERSION
    assert (
        packet["new_capability"]
        == "checkpoint-backed diffusion-policy diagnostic integration packet"
    )
    assert packet["acceptance_status"]["checkpoint_backed_map_runner_load"] is True
    assert packet["runtime_metadata"]["checkpoint_status"] == "checkpoint_loaded"
    assert packet["runtime_metadata"]["normalizer_status"] == "loaded"
    assert {item["id"] for item in packet["remaining_blockers"]} == {
        "representative_rollout",
        "multimodal_action_probe",
        "paper_grade_benchmark_claim",
    }


def test_diagnostic_packet_fails_closed_for_wrong_manifest_schema() -> None:
    """Packet builder rejects unrelated manifests instead of emitting weak evidence."""
    with pytest.raises(ValueError, match=SMOKE_MANIFEST_SCHEMA_VERSION):
        build_diagnostic_packet({"schema_version": "other"})


def test_diagnostic_packet_fails_closed_for_missing_artifacts_mapping() -> None:
    """Packet builder rejects manifests without artifact provenance."""
    with pytest.raises(ValueError, match="artifacts mapping"):
        build_diagnostic_packet({"schema_version": SMOKE_MANIFEST_SCHEMA_VERSION})


def test_diagnostic_packet_fails_closed_for_missing_required_artifact_path() -> None:
    """Packet builder requires both checkpoint and normalizer paths."""
    with pytest.raises(ValueError, match="normalizer_path"):
        build_diagnostic_packet(
            {
                "schema_version": SMOKE_MANIFEST_SCHEMA_VERSION,
                "artifacts": {"checkpoint_path": "checkpoint.pt", "normalizer_path": ""},
            }
        )


def test_diagnostic_packet_without_runtime_metadata_marks_load_blocked(tmp_path) -> None:
    """Packet without map-runner metadata keeps checkpoint-backed load blocked."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=1,
        batch_size=2,
        max_pedestrians=2,
        artifact_prefix="metadata_absent",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    packet = build_diagnostic_packet(artifacts.manifest)

    assert packet["acceptance_status"]["checkpoint_backed_map_runner_load"] is False
    assert packet["remaining_blockers"][0]["id"] == "checkpoint_backed_map_runner_load"
    assert packet["runtime_metadata"]["checkpoint_status"] == "unknown"
    assert packet["runtime_metadata"]["normalizer_status"] == "unknown"


def test_training_main_returns_zero_and_writes_packet(tmp_path, capsys) -> None:
    """Direct main path covers CLI packet branch without spawning a subprocess."""
    config_path = tmp_path / "smoke.yaml"
    output_dir = tmp_path / "artifacts"
    config_path.write_text(
        "diffusion_policy_training_smoke:\n"
        "  schema_version: diffusion_policy_training_smoke.v1\n"
        "  training_steps: 1\n"
        "  batch_size: 2\n"
        "  max_pedestrians: 2\n"
        "  artifact_prefix: direct_main\n",
        encoding="utf-8",
    )

    exit_code = diffusion_training_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--write-diagnostic-packet",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["manifest_path"] == str(output_dir / "direct_main.manifest.json")
    assert payload["diagnostic_packet_path"] == str(
        output_dir / "direct_main.manifest.diagnostic_packet.json"
    )


def test_checkpoint_status_uses_loaded_artifacts_even_when_smoke_flag_true(tmp_path) -> None:
    """Diagnostics report loaded artifacts whenever checkpoint-backed config is used."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        max_linear_speed=0.6,
        max_angular_speed=0.5,
        artifact_prefix="map_runner_smoke_flag",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    policy, meta = _build_policy(
        "diffusion_policy",
        {
            "allow_untrained_smoke": True,
            "checkpoint_path": str(artifacts.checkpoint_path),
            "normalizer_path": str(artifacts.normalizer_path),
            "deterministic": True,
            "seed": 4010,
            "max_pedestrians": config.max_pedestrians,
            "max_linear_speed": config.max_linear_speed,
            "max_angular_speed": config.max_angular_speed,
            "num_action_samples": config.num_action_samples,
            "denoising_steps": config.denoising_steps,
        },
        robot_kinematics="differential_drive",
    )
    policy(_map_runner_obs())
    stats = policy._planner_stats()

    assert meta["diffusion_policy"]["allow_untrained_smoke"] is True
    assert meta["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert meta["diffusion_policy"]["normalizer_status"] == "loaded"
    assert stats["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert stats["diffusion_policy"]["normalizer_status"] == "loaded"


def test_map_runner_loads_smoke_checkpoint_not_untrained_weights(tmp_path) -> None:
    """Map-runner can build diffusion policy from the trained smoke checkpoint."""
    config = DiffusionPolicyTrainingSmokeConfig(
        training_steps=2,
        batch_size=3,
        max_pedestrians=2,
        max_linear_speed=0.6,
        max_angular_speed=0.5,
        artifact_prefix="map_runner_smoke",
    )
    artifacts = run_training_smoke(config, output_dir=tmp_path)

    policy, meta = _build_policy(
        "diffusion_policy",
        {
            "checkpoint_path": str(artifacts.checkpoint_path),
            "normalizer_path": str(artifacts.normalizer_path),
            "deterministic": True,
            "seed": 4010,
            "max_pedestrians": config.max_pedestrians,
            "max_linear_speed": config.max_linear_speed,
            "max_angular_speed": config.max_angular_speed,
            "num_action_samples": config.num_action_samples,
            "denoising_steps": config.denoising_steps,
        },
        robot_kinematics="differential_drive",
    )

    command = policy(_map_runner_obs())
    stats = policy._planner_stats()

    assert 0.0 <= command[0] <= config.max_linear_speed
    assert -config.max_angular_speed <= command[1] <= config.max_angular_speed
    assert meta["diffusion_policy"]["allow_untrained_smoke"] is False
    assert meta["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert meta["diffusion_policy"]["normalizer_status"] == "loaded"
    assert stats["diffusion_policy"]["allow_untrained_smoke"] is False
    assert stats["diffusion_policy"]["checkpoint_status"] == "checkpoint_loaded"
    assert stats["diffusion_policy"]["normalizer_status"] == "loaded"
    assert stats["diffusion_policy"]["evidence_tier"] == "diagnostic-only"
