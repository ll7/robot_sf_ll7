"""Prove HYBDIAG regression witnesses against exact pre-Round-3 source bytes.

Load the old selector/evaluator/sensor modules in a fresh interpreter while using
current regression tests. No alternate checkout or production test seam is used.
All fixtures use synthetic geometry and perform no seeded environment reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODULES = (
    "robot_sf.planner.socnav_occupancy",
    "robot_sf.planner.grid_route",
    "robot_sf.planner.hybrid_rule_local_planner",
    "robot_sf.sensor.socnav_observation",
    "robot_sf.benchmark.map_runner.map_runner_observations",
)
TESTS = (
    "test_platform_speed_preference_does_not_saturate_at_comfort_cap",
    "test_platform_injected_speeds_preserve_nearest_pedestrian_braking_bound",
    "test_platform_checks_wall_stopping_distance_beyond_rollout_horizon",
    "test_successor_validity_is_independent_and_preserves_legitimate_origin",
    "test_sensor_emits_explicit_validity_only_when_opted_in",
    "test_debug_unknown_rejection_reason_is_observable_without_crashing",
    "test_physical_sweep_conservatively_covers_grazing_turn_arc",
    "test_grid_route_successor_validity_survives_config_builder",
    "test_physical_flag_keeps_explicitly_valid_origin_waypoint",
    "test_goal_validity_survives_real_observation_flattening",
    "test_map_observation_bridge_preserves_optional_successor_validity",
)
REASONS = (
    "comfort-cap normalization saturates added speeds",
    "injected speed exceeds physical braking cap",
    "finite horizon misses wall inside full stopping distance",
    "absent successor selected",
    "opt-in validity bit missing at sensor",
    "Unmapped evaluator constraint: future_physical_rule",
    "midpoint chord misses grazing arc contact",
    "route guide selects absent successor",
    "world-origin guard discards legitimate successor",
    "flattened observation loses successor validity",
    "map observation bridge drops opt-in successor validity",
)
TEST_PATH = "tests/planner/test_hybrid_feasibility_diagnostics.py"
CHILD = """import importlib.util,json,sys,pytest
names=json.loads(sys.argv[1])
for name,path in zip(names,sys.argv[2:2+len(names)]):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module
 spec.loader.exec_module(module)
raise SystemExit(pytest.main(sys.argv[2+len(names):]))
"""


def main() -> None:
    """Preserve module/test hashes, pytest failure log and intended-reason checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="14c1adf46436fc7b9e051b44981900acf525b9b2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=("all", "flattened"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", args.base_ref], cwd=root, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    selected = TESTS if args.group == "all" else TESTS[-2:]
    expected_count = 13 if args.group == "all" else 3
    selected_reasons = REASONS if args.group == "all" else REASONS[-2:]
    tests = [f"{TEST_PATH}::{name}" for name in selected] + ["-n", "0", "-q"]
    log = args.output / "counterproof.log"
    with tempfile.TemporaryDirectory(prefix="old-hybdiag-", dir=args.output) as directory:
        paths = []
        for module in MODULES:
            relative = module.replace(".", "/") + ".py"
            data = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=root)
            path = Path(directory) / (module.rsplit(".", 1)[1] + ".py")
            path.write_bytes(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
            paths.append(str(path.resolve()))
        with log.open("w") as stream:
            result = subprocess.run(
                [sys.executable, "-c", CHILD, json.dumps(MODULES), *paths, *tests],
                cwd=root,
                env=dict(os.environ, PYTHONPATH=str(root)),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
    text = log.read_text()
    missing = [reason for reason in selected_reasons if reason not in text]
    passed = result.returncode == 1 and f"{expected_count} failed" in text and not missing
    proof = {
        "head": head,
        "modules": hashes,
        "test_sha256": hashlib.sha256((root / TEST_PATH).read_bytes()).hexdigest(),
        "exit": result.returncode,
        "argv": tests,
        "intended_reasons_verified": passed,
        "missing_reasons": missing,
    }
    (args.output / "counterproof.json").write_text(json.dumps(proof, indent=2) + "\n")
    if not passed:
        raise SystemExit(f"Counterproof failed for unintended reason; inspect {log}")
    print(f"Verified {expected_count} intended failures at " + head)


if __name__ == "__main__":
    main()
