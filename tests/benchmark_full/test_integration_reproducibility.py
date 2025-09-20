"""Integration test T021: reproducibility of episode_ids.

Two consecutive runs with identical seed and config should yield identical
sets of episode_ids (order not enforced). Uses the synthetic episode record
logic which is deterministic w.r.t scenario planning seeds and job creation.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def _read_episode_ids(root: Path) -> set[str]:
    p = root / "episodes" / "episodes.jsonl"
    ids: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ids.add(rec["episode_id"])
    return ids


def test_reproducibility_same_seed(config_factory):
    cfg = config_factory(smoke=True, master_seed=123)
    manifest1 = run_full_benchmark(cfg)
    ids1 = _read_episode_ids(Path(manifest1.output_root))
    # Second run (resume) should not add new episodes; IDs must match exactly
    manifest2 = run_full_benchmark(cfg)
    ids2 = _read_episode_ids(Path(manifest2.output_root))
    assert ids1 == ids2 and len(ids1) >= 1
