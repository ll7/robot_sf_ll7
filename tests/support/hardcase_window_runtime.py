"""Fresh, source-CPU-pinned runtime for the retained seed-1001 regression.

Invoked only by test_generated_window_continuation.py. This helper executes the
production producer/replayer and writes temporary pytest outputs; the calling
test retains all independent, exact trace and event assertions.
"""

# evidence-writer-exempt: temporary pytest source/checkpoint/replay outputs only;
# this helper never writes to the committed evidence tree.

from __future__ import annotations

import gzip
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import build_map_policy, run_map_batch
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
from robot_sf.benchmark.scenario_generation.continuation import capture_window_checkpoint
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/hardcase_positive_control"
CONFIG = "configs/algos/social_force_release_v0_0_8.yaml"


def capture(output: Path) -> None:
    """Capture the real dev source boundary using the production serializer."""
    reference = json.loads(gzip.decompress((EVIDENCE / "source.jsonl.gz").read_bytes()))
    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    source = deepcopy(
        next(s for s in load_scenarios(matrix) if s["name"] == "classic_cross_trap_high")
    )
    source["seeds"] = [1001]

    def observe(**context) -> None:
        if context["step_idx"] == 370:
            capture_window_checkpoint(
                output / "checkpoint.json",
                env=context["env"],
                state=context["state"],
                loop_config=context["loop_config"],
                source_episode_id=reference["episode_id"],
                seed=1001,
                source_map=ROOT
                / "maps/successor_svg_maps/issue_9762_classic_crossing_goal_zone_entry_v2.svg",
                window_steps=20,
            )

    row = run_map_episode(
        source,
        1001,
        horizon=600,
        dt=0.1,
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        algo="social_force",
        scenario_path=matrix,
        algo_config=yaml.safe_load((ROOT / CONFIG).read_text()),
        algo_config_path=CONFIG,
        record_simulation_step_trace=True,
        before_step_hook=observe,
        policy_builder=build_map_policy,
    )
    (output / "captured.json").write_text(json.dumps(row))


def replay(output: Path, scenario: Path) -> None:
    """Replay one fresh environment with the same CPU/compiler identity."""
    summary = run_map_batch(
        scenario,
        output,
        ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        horizon=600,
        dt=0.1,
        record_forces=True,
        algo="social_force",
        algo_config_path=CONFIG,
        benchmark_profile="experimental",
        workers=1,
        resume=False,
        record_simulation_step_trace=True,
    )
    output.with_suffix(".summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "capture":
            capture(Path(sys.argv[2]))
        elif sys.argv[1] == "replay":
            replay(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            raise ValueError("expected capture or replay")
    except pytest.skip.Exception as exc:
        # Preserve an installed pytest audit interlock across the process boundary.
        print(str(exc), file=sys.stderr)
        sys.exit(77)
