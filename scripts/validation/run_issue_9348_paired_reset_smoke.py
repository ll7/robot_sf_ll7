#!/usr/bin/env python3
"""Verify doorway pair receipts through the real episode runner at one step.

This is a diagnostic custody smoke, not the H400 confirmation campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.benchmark.three_width_doorway_application import (
    DEFAULT_MANIFEST_PATH,
    DoorwayPairingSession,
    build_pair_manifest,
    load_three_width_manifest,
    non_width_config_sha256,
    run_three_width_preflight,
)
from robot_sf.evidence.writers import write_json
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Run 18 one-step cells and write the complete pre-command receipt manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_REPO_ROOT / DEFAULT_MANIFEST_PATH)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--variants-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_three_width_manifest(manifest_path)
    preflight = run_three_width_preflight(manifest_path, output_dir=args.variants_dir)
    if not preflight["go"]:
        raise ValueError("doorway geometry/oracle preflight did not admit diagnostic continuation")
    assets = [
        record["assets"]
        | {"variant_id": record["variant_id"], "gap_width_m": record["geometry"]["gap_width_m"]}
        for record in preflight["variants"]
    ]
    session = DoorwayPairingSession()
    rows = []
    sf_path = Path(manifest["_resolved"]["social_force_config_path"])
    sf_config = yaml.safe_load(sf_path.read_text(encoding="utf-8"))
    for planner in manifest["_resolved"]["planner_roster"]:
        config_sha = _sha256(sf_path) if planner == "social_force" else None
        for seed in manifest["_resolved"]["planner_seeds"]:
            for asset in assets:
                scenario_path = Path(asset["scenario_path"])
                scenario = dict(load_scenarios(scenario_path)[0])
                common_sha = non_width_config_sha256(
                    scenario, planner=planner, planner_config_sha256=config_sha
                )
                receipt_hook = session.hook(
                    planner=planner,
                    seed=seed,
                    map_sha256=asset["map_sha256"],
                    non_width_config_sha256=common_sha,
                )
                row = _run_map_episode(
                    scenario,
                    seed,
                    horizon=1,
                    dt=0.1,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=planner,
                    scenario_path=scenario_path,
                    algo_config=sf_config if planner == "social_force" else None,
                    algo_config_path=sf_path.as_posix() if planner == "social_force" else None,
                    pair_reset_hook=receipt_hook,
                )
                receipt = row.get("algorithm_metadata", {}).get("doorway_pair_receipt")
                if receipt != session.receipts[(planner, seed)][-1]:
                    raise ValueError(
                        f"episode row omitted or altered doorway receipt: {planner}/{seed}"
                    )
                rows.append(
                    {
                        "planner": planner,
                        "seed": seed,
                        "variant_id": asset["variant_id"],
                        "scenario_sha256": asset["scenario_sha256"],
                        "map_sha256": asset["map_sha256"],
                        "receipt": receipt,
                        "smoke_horizon_steps": 1,
                        "evidence_status": "diagnostic_only",
                    }
                )
    pair_manifest = build_pair_manifest(
        assets, tuple(manifest["_resolved"]["planner_seeds"]), _sha256(manifest_path)
    )
    paired = session.fill_pair_manifest(pair_manifest)
    if len(rows) != 18:
        raise ValueError("expected 18 doorway reset smoke rows")
    write_json(
        args.out_json,
        {
            "schema_version": "issue_9348_paired_reset_smoke.v1",
            "claim_boundary": "H1 diagnostic custody only; no width-comparison outcomes",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
            ).strip(),
            "manifest_sha256": _sha256(manifest_path),
            "preflight_checks": preflight["checks"],
            "preflight_variants": [
                {
                    "variant_id": record["variant_id"],
                    "map_sha256": record["assets"]["map_sha256"],
                    "scenario_sha256": record["assets"]["scenario_sha256"],
                    "oracle": record["oracle"],
                }
                for record in preflight["variants"]
            ],
            "pairs": paired,
            "rows": rows,
        },
    )
    print(f"issue #9348 paired reset smoke: cells={len(rows)} pairs=6 report={args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
