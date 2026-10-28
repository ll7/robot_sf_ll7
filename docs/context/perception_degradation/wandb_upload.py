"""Upload one raw stage root to W&B as a campaign artifact (credentials in-process only).

    uv run python docs/context/perception_degradation/wandb_upload.py \
        --raw-root <stage run root> --stage main --commit <execution commit>

Follows the repository's campaign-artifact route (``ll7/robot_sf``, type ``campaign``);
prints the durable ``wandb://`` URI and the artifact digest, never the key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def main() -> None:
    """Upload and read back the artifact digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    name = f"campaign-xperc_perception_degradation_{args.stage}_{args.commit[:9]}"
    run = wandb.init(
        entity="ll7",
        project="robot_sf",
        job_type="campaign-preservation",
        name=f"{name}-upload",
        config={"stage": args.stage, "execution_commit": args.commit},
    )
    artifact = wandb.Artifact(
        name,
        type="campaign",
        metadata={
            "stage": args.stage,
            "execution_commit": args.commit,
            "evidence_tier": "development-seed diagnostic; new current-version execution",
        },
    )
    artifact.add_dir(str(args.raw_root))
    logged = run.log_artifact(artifact)
    logged.wait()
    run.finish()
    readback = wandb.Api().artifact(f"ll7/robot_sf/{name}:latest")
    print(f"wandb://ll7/robot_sf/{name}:{readback.version} digest={readback.digest}")


if __name__ == "__main__":
    main()
