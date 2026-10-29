"""Upload the lane's raw archive to W&B as one campaign artifact (credentials in-process only).

    python wandb_upload.py --raw-root <archive dir> --commit <execution commit>

Follows the repository's campaign-artifact route (``ll7/robot_sf``, type ``campaign``);
prints the durable ``wandb://`` URI and the artifact digest, never the key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def main() -> None:
    """Upload and read back the artifact digest and file count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    name = f"campaign-exsel_trace_example_screening_{args.commit[:9]}"
    run = wandb.init(
        entity="ll7",
        project="robot_sf",
        job_type="campaign-preservation",
        name=f"{name}-upload",
        config={
            "execution_commit": args.commit,
            "claims": ["EXSEL-6.4-MEASURE-WALKTHROUGH", "EXSEL-7.2-NARROW-DOORWAY"],
        },
    )
    artifact = wandb.Artifact(
        name,
        type="campaign",
        metadata={
            "execution_commit": args.commit,
            "evidence_tier": "development-seed diagnostic; new current-version execution",
        },
    )
    artifact.add_dir(str(args.raw_root))
    logged = run.log_artifact(artifact)
    logged.wait()
    run.finish()
    readback = wandb.Api().artifact(f"ll7/robot_sf/{name}:latest")
    files = list(readback.manifest.entries)
    print(
        f"wandb://ll7/robot_sf/{name}:{readback.version} digest={readback.digest} "
        f"state={readback.state} files={len(files)}"
    )


if __name__ == "__main__":
    main()
