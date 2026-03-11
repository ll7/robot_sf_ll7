"""Integration tests for freeze-manifest validation in full classic benchmark runs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from robot_sf.benchmark.full_classic.planning import load_scenario_matrix


def _scenario_matrix_hash(path: str | Path) -> str:
    """Compute scenario matrix hash exactly as the orchestrator does.

    Returns:
        12-character SHA1 prefix of canonical matrix payload.
    """

    raw = load_scenario_matrix(str(path))
    matrix_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(matrix_bytes).hexdigest()[:12]


def _write_freeze_manifest(path: Path, *, cfg, matrix_hash: str, base_seed: int) -> None:
    """Write a minimal freeze manifest fixture.

    Args:
        path: Target freeze-manifest path.
        cfg: Benchmark config object.
        matrix_hash: Expected scenario matrix hash.
        base_seed: Seed used for freeze `seed_plan.base_seed`.
    """

    payload = {
        "scenario": {
            "matrix_path": str(Path(cfg.scenario_matrix_path).resolve()),
            "matrix_hash": matrix_hash,
        },
        "baselines": {
            "algorithms": [cfg.algo],
            "planner_configs": [{"planner_backend": "default", "planner_classic_config": None}],
        },
        "seed_plan": {"base_seed": int(base_seed), "repeats": int(cfg.initial_episodes)},
        "metrics": {
            "subset": [
                "success_rate",
                "collision_rate",
                "snqi",
                "time_to_goal",
                "path_efficiency",
            ]
        },
        "bootstrap": {
            "samples": int(cfg.bootstrap_samples),
            "confidence": float(cfg.bootstrap_confidence),
            "seed": int(cfg.master_seed),
        },
        "software": {
            "identifiers": {
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                ),
                "platform": platform.platform(),
            }
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


@pytest.mark.timeout(60)
def test_freeze_manifest_match_is_recorded_in_run_meta(config_factory) -> None:
    """Verify matching freeze manifests are recorded as `match` in run metadata."""

    cfg = config_factory(smoke=True, workers=1)
    matrix_hash = _scenario_matrix_hash(cfg.scenario_matrix_path)
    freeze_path = Path(cfg.output_root) / "freeze_match.yaml"
    _write_freeze_manifest(freeze_path, cfg=cfg, matrix_hash=matrix_hash, base_seed=cfg.master_seed)
    cfg.freeze_manifest_path = str(freeze_path)

    manifest = run_full_benchmark(cfg)
    run_meta = json.loads((manifest.output_root / "run_meta.json").read_text(encoding="utf-8"))
    freeze_meta = run_meta["freeze_manifest"]
    assert freeze_meta["status"] == "match"
    assert freeze_meta["mismatches"] == []


@pytest.mark.timeout(60)
def test_freeze_manifest_mismatch_is_recorded_in_run_meta(config_factory) -> None:
    """Verify mismatched freeze manifests produce structured mismatch metadata."""

    cfg = config_factory(smoke=True, workers=1)
    matrix_hash = _scenario_matrix_hash(cfg.scenario_matrix_path)
    freeze_path = Path(cfg.output_root) / "freeze_mismatch.yaml"
    _write_freeze_manifest(
        freeze_path,
        cfg=cfg,
        matrix_hash=matrix_hash,
        base_seed=int(cfg.master_seed) + 7,
    )
    cfg.freeze_manifest_path = str(freeze_path)

    manifest = run_full_benchmark(cfg)
    run_meta = json.loads((manifest.output_root / "run_meta.json").read_text(encoding="utf-8"))
    freeze_meta = run_meta["freeze_manifest"]
    assert freeze_meta["status"] == "mismatch"
    mismatch_paths = {entry["path"] for entry in freeze_meta["mismatches"]}
    assert "seed_plan.base_seed" in mismatch_paths
