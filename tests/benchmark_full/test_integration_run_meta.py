"""Integration tests for run-level benchmark metadata export."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


@pytest.mark.timeout(60)
def test_smoke_run_exports_traceable_run_metadata(config_factory) -> None:
    """Verify smoke runs emit canonical run metadata with key traceability fields.

    This guards reproducibility contracts for benchmark artifacts by asserting
    `run_meta.json` is generated and contains provenance fields required by
    downstream paper tooling.
    """

    cfg = config_factory(smoke=True, workers=1)
    manifest = run_full_benchmark(cfg)
    root = manifest.output_root

    run_meta_path = root / "run_meta.json"
    assert run_meta_path.exists(), "Expected run_meta.json next to manifest.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    assert run_meta["run_id"] == root.resolve().name
    created_at = run_meta["created_at_utc"]
    datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    repo = run_meta["repo"]
    for key in ("name", "remote", "branch", "commit"):
        assert key in repo
        assert isinstance(repo[key], str)

    cli = run_meta["cli"]
    assert isinstance(cli["command"], str)
    assert isinstance(cli["args"], list)

    assert isinstance(run_meta["matrix_path"], str)
    assert run_meta["matrix_path"]

    seed_plan = run_meta["seed_plan"]
    assert seed_plan["base_seed"] == int(cfg.master_seed)
    assert seed_plan["repeats"] == int(cfg.initial_episodes)

    env = run_meta["environment"]
    assert isinstance(env["python_version"], str)
    assert isinstance(env["platform"], str)

    mirror_path = root / "artifacts" / run_meta["run_id"] / "run_meta.json"
    assert mirror_path.exists(), "Expected paper-contract mirror path for run metadata"
    mirror_meta = json.loads(mirror_path.read_text(encoding="utf-8"))
    assert mirror_meta == run_meta
