"""Resume budgets and receipt lifetimes cannot replay already-consumed layouts."""

import json

import numpy as np
import pytest

from robot_sf.training.pilot_checkpoint_recovery import receipt_offsets, remaining_targets


def test_remaining_chunks_reach_original_final_budget():
    """Actual checkpoint offsets avoid nominal-step rounding on resume."""
    assert remaining_targets(3_010_560) == (4_014_080, 5_017_600)
    assert remaining_targets(5_017_600) == ()
    with pytest.raises(ValueError):
        remaining_targets(3_000_000)


def test_receipt_lifetimes_are_contiguous_and_match_the_master(tmp_path):
    """A replacement worker continues the same shuffled stream without reuse."""
    permutation = np.random.default_rng(100000).permutation(100000) + 100000
    for worker in range(20):
        for episode in range(2):
            row = {
                "worker_index": worker,
                "episode_index": episode,
                "seed": int(permutation[worker + 20 * episode]),
            }
            (tmp_path / f"worker-{worker}-lifetime-{episode}.jsonl").write_text(
                json.dumps(row) + "\n"
            )
    assert receipt_offsets(tmp_path, policy="B", master=100000) == dict.fromkeys(range(20), 2)
    path = tmp_path / "worker-0-lifetime-1.jsonl"
    row = json.loads(path.read_text())
    row["episode_index"] = 0
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="contiguous"):
        receipt_offsets(tmp_path, policy="B", master=100000)
