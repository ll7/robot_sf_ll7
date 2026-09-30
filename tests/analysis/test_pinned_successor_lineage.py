"""Actual frozen release configurations must resolve without rewriting their bytes."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.analysis.compare_release_0_0_7_to_0_0_8 import V4_SLOT_REPLACEMENTS

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"


def test_real_frozen_0_0_8_template_resolves_all_four_v4_lineages():
    """Exercise production pinned resolution with the real release matrix and frozen v4 files."""
    slots = [
        (key, "differential_drive", "classic_bottleneck_low", 111, "")
        for key in V4_SLOT_REPLACEMENTS.values()
    ]  # Seed 111 is a static join key; this resolver never steps an environment or planner.
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/analysis/_pinned_successor_runtime.py")],
        cwd=ROOT,
        input=json.dumps(
            {
                "config_path": TEMPLATE,
                "versioned_keys": sorted(V4_SLOT_REPLACEMENTS.values()),
                "rows": [{"slot": slot, "scenario_params": {}} for slot in slots],
            }
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert len(resolved["expected_slots"]) == 20160  # 14 arms * 48 scenarios * 30 seeds.
    assert {tuple(row["slot"]) for row in resolved["rows"]} == set(slots)
    for row in resolved["rows"]:
        assert row["algo"] == "hybrid_rule_local_planner"
        assert row["config"]["planner_variant"] == "hybrid_rule_v4_clearance_braking"
        assert row["path"].endswith("_release_0_0_8_frozen.yaml")
