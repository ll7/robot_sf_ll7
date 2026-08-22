"""Contract tests for the immutable single-node release entrypoint."""

from pathlib import Path


def test_single_node_release_requires_checkpoint_receipt() -> None:
    """The cluster entrypoint must forward an enforced staging receipt."""
    script = Path("SLURM/submit_release_single_node.sbatch").read_text(encoding="utf-8")
    assert 'CHECKPOINT_RECEIPT="${4:?staged-checkpoint receipt path required}"' in script
    assert '--checkpoint-receipt "$CHECKPOINT_RECEIPT"' in script
    assert 'RUNTIME_SMOKE_RECEIPT="${5:?exact-source runtime-smoke result path required}"' in script
    assert '--runtime-smoke-receipt "$RUNTIME_SMOKE_RECEIPT"' in script
    assert 'RESUME_RECEIPT="${6:-}"' in script
    assert 'runner_args+=(--resume-receipt "$RESUME_RECEIPT")' in script
    assert "#SBATCH --requeue" not in script
    assert "#SBATCH --cpus-per-task=36" in script
    assert "#SBATCH --mem=256G" in script
    assert "#SBATCH --time=36:00:00" in script
