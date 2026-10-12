"""Real dev-seed regression for the #10330 extracted near-miss window."""

# evidence-writer-exempt: temporary pytest checkpoint and malformed-companion
# fixtures are not durable generated evidence; exact native traces are preserved.

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.benchmark.scenario_generation.replay_adapter import (
    dump_generated_scenario_yaml,
    materialize_generated_scenario,
)
from robot_sf.benchmark.step_trace_comparator import (
    canonical_step_trace_digest,
    find_first_trace_difference,
)
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/hardcase_positive_control"


def _run_source_pinned_runtime(mode: str, output: Path, scenario: Path | None = None) -> None:
    """Use the producer's LLVM CPU target before any Numba kernel is imported.

    The retained oracle was produced on AMD Zen 2. Native fastmath dispatch on
    another CPU changes the exact trace even with identical checkout/dependency
    bytes. A fresh interpreter also avoids reusing a worker's compiled kernels.
    """
    environment = os.environ.copy()
    environment["NUMBA_CPU_NAME"] = "znver2"
    environment.pop("NUMBA_CPU_FEATURES", None)
    command = [
        sys.executable,
        str(ROOT / "tests/support/hardcase_window_runtime.py"),
        mode,
        str(output),
    ]
    if scenario is not None:
        command.append(str(scenario))
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode == 77:
        pytest.skip(result.stderr.strip().splitlines()[-1])
    assert result.returncode == 0, result.stderr[-4000:]


def test_seed1001_window_does_not_complete_at_its_boundary(tmp_path: Path) -> None:
    """The actual 0.8366 m window must reach its twenty-transition budget."""
    entry = json.loads((EVIDENCE / "entry.json").read_text())
    assert entry["source_episode"]["source_seed"] == 1001
    result = materialize_generated_scenario(entry, max_episode_steps=20)
    assert result.scenario_document is not None
    result.scenario_document["scenarios"][0]["map_file"] = str(
        ROOT / entry["source_episode"]["source_map"]
    )
    scenario = tmp_path / "window.yaml"
    scenario.write_text(dump_generated_scenario_yaml(result))
    output = tmp_path / "window.jsonl"
    summary = run_map_batch(
        scenario,
        output,
        ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        horizon=20,
        dt=0.1,
        algo="social_force",
        algo_config_path="configs/algos/social_force_release_v0_0_8.yaml",
        benchmark_profile="experimental",
        record_forces=True,
        record_simulation_step_trace=True,
        workers=1,
        resume=False,
    )
    assert summary["successful_jobs"] == 1 and summary["failed_jobs"] == 0
    row = json.loads(output.read_text())
    steps = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
    assert len(steps) == 20
    assert row["outcome"]["route_complete"] is False
    assert not any(step["rl"]["terminated"] for step in steps[:-1])


@pytest.fixture(scope="module")
def captured_seed1001(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Capture the real boundary once for serializer and fresh-replay checks.

    Returns:
        Source, independent reference, and checkpoint bindings.
    """
    tmp_path = tmp_path_factory.mktemp("seed1001-checkpoint")
    reference = json.loads(gzip.decompress((EVIDENCE / "source.jsonl.gz").read_bytes()))
    expected = reference["algorithm_metadata"]["simulation_step_trace"]["steps"][370:390]
    matrix = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    source = deepcopy(
        next(s for s in load_scenarios(matrix) if s["name"] == "classic_cross_trap_high")
    )
    source["seeds"] = [1001]
    checkpoint = tmp_path / "checkpoint.json"
    config_path = "configs/algos/social_force_release_v0_0_8.yaml"

    _run_source_pinned_runtime("capture", tmp_path)
    captured = json.loads((tmp_path / "captured.json").read_text())
    captured_steps = captured["algorithm_metadata"]["simulation_step_trace"]["steps"]
    assert canonical_step_trace_digest(captured_steps) == (
        "7a73ee858ba8248969015c4f31273c014d15fd328bd97671876f9520c2bf2b34"
    ), find_first_trace_difference(
        captured_steps, reference["algorithm_metadata"]["simulation_step_trace"]["steps"]
    )
    entry = json.loads((EVIDENCE / "entry.json").read_text())
    return {
        "reference": reference,
        "expected": expected,
        "source": source,
        "checkpoint": checkpoint,
        "entry": entry,
        "config_path": config_path,
    }


@pytest.mark.new_feature_no_base(reason="base lacks the typed environment continuation path")
def test_seed1001_checkpoint_replays_exact_near_miss_window_twice(
    tmp_path: Path, captured_seed1001: dict
) -> None:
    """Restore fresh environments and match the independently retained real trace."""
    reference = captured_seed1001["reference"]
    expected = captured_seed1001["expected"]
    source = captured_seed1001["source"]
    checkpoint = captured_seed1001["checkpoint"]
    entry = captured_seed1001["entry"]
    result = materialize_generated_scenario(
        entry,
        max_episode_steps=20,
        source_scenario=source,
        continuation_checkpoint=checkpoint,
    )
    assert result.scenario_document is not None
    scenario = result.scenario_document["scenarios"][0]
    scenario["map_file"] = str(ROOT / entry["source_episode"]["source_map"])
    assert scenario["simulation_config"]["max_episode_steps"] == 600
    assert scenario["simulation_config"]["goal_completion_policy"] == "goal_zone_entry_v1"
    replay_path = tmp_path / "window.yaml"
    replay_path.write_text(dump_generated_scenario_yaml(result))
    for index in (1, 2):
        output = tmp_path / f"replay-{index}.jsonl"
        _run_source_pinned_runtime("replay", output, replay_path)
        summary = json.loads(output.with_suffix(".summary.json").read_text())
        assert summary["successful_jobs"] == 1 and summary["failed_jobs"] == 0
        row = json.loads(output.read_text())
        actual = row["algorithm_metadata"]["simulation_step_trace"]["steps"]
        assert len(actual) == 20
        assert actual == expected
        assert row["metrics"]["near_misses"] == 11
        assert row["metrics"]["min_clearance"] == 0.47727323760374385
        assert actual[4]["step"] == 374 and actual[4]["time_s"] == 37.5
        assert row["outcome"] == reference["outcome"]


@pytest.mark.new_feature_no_base(reason="base lacks continuation companion validation")
def test_seed1001_checkpoint_rejects_tampered_companion(
    tmp_path: Path, captured_seed1001: dict
) -> None:
    """A modified numeric payload must fail its binding before restoration."""
    from robot_sf.benchmark.scenario_generation.continuation import read_window_checkpoint

    checkpoint = captured_seed1001["checkpoint"]
    copied = tmp_path / "copy"
    shutil.copytree(checkpoint.parent, copied)
    path = copied / checkpoint.name
    record = json.loads(path.read_text())
    (copied / record["aux_file"]).write_bytes(b"tampered numeric payload")
    with pytest.raises(ValueError, match="companion digest mismatch"):
        read_window_checkpoint(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.new_feature_no_base(reason="base lacks continuation companion validation")
def test_seed1001_checkpoint_rejects_parent_companion_path(
    tmp_path: Path, captured_seed1001: dict
) -> None:
    """Even a rehashed descriptor cannot read companions outside its packet."""
    from robot_sf.benchmark.scenario_generation.continuation import read_window_checkpoint

    checkpoint = captured_seed1001["checkpoint"]
    copied = tmp_path / "copy"
    shutil.copytree(checkpoint.parent, copied)
    path = copied / checkpoint.name
    record = json.loads(path.read_text())
    record["aux_file"] = "../outside.npz"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="companion must be a basename"):
        read_window_checkpoint(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
