#!/usr/bin/env python3
"""Imitation Learning Pipeline - End-to-End Example.

This example demonstrates the complete imitation learning workflow by calling
the training scripts in sequence:

1. Train an expert PPO policy (or use existing)
2. Collect expert trajectories
3. Pre-train a new policy via behavioral cloning
4. Fine-tune with PPO
5. Compare baseline vs pre-trained performance

**Purpose**: Show complete workflow for accelerating PPO training with expert demonstrations

**Prerequisites**:
- uv sync --all-extras
- Sufficient disk space for models and trajectories (~500MB)
- 30-60 minutes runtime for full pipeline (less in demo mode)

**Usage**:
    # Run full pipeline (expert training takes longest - 30-60 min)
    uv run python examples/advanced/16_imitation_learning_pipeline.py

    # Skip expert training if you already have a trained policy
    uv run python examples/advanced/16_imitation_learning_pipeline.py --skip-expert --policy-id ppo_expert_v1

    # Quick demo mode (reduced episodes/timesteps)
    uv run python examples/advanced/16_imitation_learning_pipeline.py --demo-mode

**Note**: Only expert_ppo.yaml config exists. BC and fine-tuning steps use CLI arguments.

**Output**:
- Expert policy: output/benchmarks/expert_policies/
- Trajectories: output/benchmarks/expert_trajectories/
- Pre-trained policy: output/benchmarks/expert_policies/
- Comparison report: output/imitation_reports/comparisons/

**Related**:
- Full documentation: docs/imitation_learning_pipeline.md
- Quickstart guide: specs/001-ppo-imitation-pretrain/quickstart.md
- Individual scripts: scripts/training/train_expert_ppo.py, collect_expert_trajectories.py, etc.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


def _run_command(cmd: list[str], step_name: str) -> int:
    """Run a subprocess command and log results.

    Args:
        cmd: Command and arguments to run
        step_name: Human-readable step name for logging

    Returns:
        Exit code (0 for success)
    """
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(f"{step_name} failed with exit code {result.returncode}")
        return result.returncode

    logger.success(f"{step_name} completed successfully")
    return 0


def _load_policy_id_from_config(config_path: Path) -> str:
    """Read the policy_id from the expert training YAML config."""

    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "policy_id" not in data:
        raise ValueError(
            "expert_ppo.yaml must define a top-level 'policy_id' key to identify the checkpoint.",
        )

    policy_id = data["policy_id"]
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("policy_id in expert_ppo.yaml must be a non-empty string.")

    return policy_id


def main():  # noqa: C901 - Sequential workflow orchestration; complexity is intentional
    """Run complete imitation learning pipeline by calling training scripts."""
    parser = argparse.ArgumentParser(description="Imitation Learning Pipeline - End-to-End Example")
    parser.add_argument(
        "--skip-expert",
        action="store_true",
        help="Skip expert training (use existing policy)",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="ppo_expert_demo",
        help="Expert policy ID to use/create",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="trajectories_demo",
        help="Trajectory dataset ID to create",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Quick demo mode (reduced timesteps/episodes)",
    )
    args = parser.parse_args()

    dataset_id = args.dataset_id

    logger.info("=" * 70)
    logger.info("IMITATION LEARNING PIPELINE - END-TO-END EXAMPLE")
    logger.info("=" * 70)

    # Configuration file path (only expert_ppo.yaml exists; others use CLI args)
    expert_config = Path("configs/training/ppo_imitation/expert_ppo.yaml")

    # Verify expert config exists
    if not expert_config.exists():
        logger.error(f"Configuration file not found: {expert_config}")
        logger.info("Please ensure you're running from repository root")
        return 1

    configured_policy_id = _load_policy_id_from_config(expert_config)

    if args.skip_expert:
        expert_policy_id = args.policy_id
    else:
        expert_policy_id = configured_policy_id
        if args.policy_id != expert_policy_id:
            logger.warning(
                "Ignoring --policy-id override (expert training uses policy_id={} from {})",
                expert_policy_id,
                expert_config,
            )

    bc_policy_id = f"bc_{expert_policy_id}"
    finetuned_policy_id = f"finetuned_{expert_policy_id}"

    logger.info(f"Expert policy ID: {expert_policy_id}")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Demo mode: {args.demo_mode}")
    logger.info("")

    try:
        # Step 1: Train expert policy (or use existing)
        if args.skip_expert:
            logger.info("=" * 70)
            logger.info("STEP 1: Using Existing Expert Policy")
            logger.info("=" * 70)
            logger.info(f"Policy ID: {expert_policy_id}")

            # Verify policy exists
            from robot_sf import common

            policy_path = common.get_expert_policy_dir() / f"{expert_policy_id}.zip"
            if not policy_path.exists():
                logger.error(f"Expert policy not found: {policy_path}")
                logger.info("Run without --skip-expert to train a new policy")
                return 1
            logger.info(f"Found policy: {policy_path}")
        else:
            logger.info("=" * 70)
            logger.info("STEP 1: Training Expert PPO Policy")
            logger.info("=" * 70)

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/training/train_expert_ppo.py",
                "--config",
                str(expert_config),
            ]
            if args.demo_mode:
                cmd.append("--dry-run")  # Use dry-run for quick demo

            exit_code = _run_command(cmd, "Expert training")
            if exit_code != 0:
                return exit_code

        # Step 2: Collect trajectories
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Collecting Expert Trajectories")
        logger.info("=" * 70)

        num_episodes = 20 if args.demo_mode else 100
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/collect_expert_trajectories.py",
            "--dataset-id",
            dataset_id,
            "--policy-id",
            expert_policy_id,
            "--episodes",
            str(num_episodes),
        ]

        exit_code = _run_command(cmd, "Trajectory collection")
        if exit_code != 0:
            return exit_code

        # Step 3: BC pre-training
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 3: Behavioral Cloning Pre-training")
        logger.info("=" * 70)

        # BC script uses programmatic config (no YAML file needed)
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/pretrain_from_expert.py",
            "--dataset-id",
            dataset_id,
            "--policy-id",
            bc_policy_id,
        ]
        if args.demo_mode:
            cmd.extend(["--epochs", "5"])  # Reduced epochs for demo

        exit_code = _run_command(cmd, "BC pre-training")
        if exit_code != 0:
            return exit_code

        # Step 4: PPO fine-tuning
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 4: PPO Fine-tuning")
        logger.info("=" * 70)

        # Fine-tuning script uses programmatic config (no YAML file needed)
        timesteps = "30000" if args.demo_mode else "200000"
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/training/train_ppo_with_pretrained_policy.py",
            "--pretrained-id",
            bc_policy_id,
            "--policy-id",
            finetuned_policy_id,
            "--timesteps",
            timesteps,
        ]

        exit_code = _run_command(cmd, "PPO fine-tuning")
        if exit_code != 0:
            return exit_code

        # Step 5: Generate comparison (optional - script may not exist yet)
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 5: Performance Comparison")
        logger.info("=" * 70)

        comparison_script = Path("scripts/training/compare_training_runs.py")
        if comparison_script.exists():
            cmd = [
                "uv",
                "run",
                "python",
                str(comparison_script),
                "--baseline-id",
                expert_policy_id,
                "--pretrained-id",
                bc_policy_id,
            ]
            _run_command(cmd, "Comparison generation")
        else:
            logger.warning(f"Comparison script not found: {comparison_script}")
            logger.info("Skipping comparison step")

        # Success summary
        logger.info("")
        logger.info("=" * 70)
        logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Artifacts created:")
        logger.info(f"  - Expert policy: {expert_policy_id}")
        logger.info(f"  - Trajectory dataset: {dataset_id}")
        logger.info(f"  - BC pre-trained policy: {bc_policy_id}")
        logger.info(f"  - Fine-tuned policy: {finetuned_policy_id}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - View artifacts: output/benchmarks/expert_policies/")
        logger.info("  - View trajectories: output/benchmarks/expert_trajectories/")
        logger.info("  - Full docs: docs/imitation_learning_pipeline.md")
        logger.info("  - Quickstart: specs/001-ppo-imitation-pretrain/quickstart.md")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
