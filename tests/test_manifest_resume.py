from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.manifest import load_manifest, manifest_path_for
from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _scenarios(repeats: int = 2):
    return [
        {
            "id": "manifest-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        }
    ]


def test_manifest_created_and_used_for_resume(tmp_path: Path):
    out_file = tmp_path / "episodes.jsonl"
    sc = _scenarios(repeats=2)
    # First run writes 2 lines and should produce a manifest
    summary1 = run_batch(
        sc,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=6,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=True,
    )
    assert summary1["written"] == 2
    sidecar = manifest_path_for(out_file)
    assert sidecar.exists()
    ids1 = load_manifest(out_file)
    assert ids1 is not None and len(ids1) == 2

    # Second run with resume must skip via manifest
    summary2 = run_batch(
        sc,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=6,
        dt=0.1,
        record_forces=False,
        append=True,
        workers=1,
        resume=True,
    )
    assert summary2["written"] == 0


def test_manifest_stale_fallbacks_to_scan(tmp_path: Path):
    out_file = tmp_path / "episodes.jsonl"
    sc = _scenarios(repeats=1)
    # Write one episode
    run_batch(
        sc,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=True,
    )
    # Corrupt the manifest: wrong size triggers fallback
    sidecar = manifest_path_for(out_file)
    with sidecar.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["stat"]["size"] = int(data["stat"]["size"]) + 999
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    # Add a new scenario repeat and expect it to write 1 more using scan fallback
    sc2 = _scenarios(repeats=2)
    summary2 = run_batch(
        sc2,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=True,
        workers=1,
        resume=True,
    )
    assert summary2["written"] == 1


def test_manifest_identity_hash_mismatch_forces_scan(tmp_path: Path):
    out_file = tmp_path / "episodes.jsonl"
    sc = _scenarios(repeats=1)
    # First run writes the file and manifest
    run_batch(
        sc,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=1,
        resume=True,
    )
    # Directly read manifest and blank out identity hash to simulate older writer
    sidecar = manifest_path_for(out_file)
    with sidecar.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Overwrite with a wrong identity hash
    data["identity_hash"] = "deadbeef"
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    # Add another repeat; with mismatch, load_manifest should return None and
    # runner should fall back to scanning and write the missing one.
    sc2 = _scenarios(repeats=2)
    summary2 = run_batch(
        sc2,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=True,
        workers=1,
        resume=True,
    )
    assert summary2["written"] == 1
