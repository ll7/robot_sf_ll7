"""Tests for the issue #3574 pre-specified rank-reversal test CLI runner.

Exercises ``scripts/benchmark/run_rank_reversal_test_issue_3574.py`` end-to-end on a fixture
manifest plus constructed episode records so the CLI is proven before real campaign records
exist. The CLI fails closed on integration readiness and then renders the preregistered test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.heterogeneous_population_ablation import (
    build_mean_matched_harness_manifest,
)
from robot_sf.benchmark.heterogeneous_rank_sensitivity import RANK_REVERSAL_TEST_SCHEMA
from robot_sf.benchmark.pedestrian_control_trace import PEDESTRIAN_CONTROL_TRACE_LABELS_KEY

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_rank_reversal_test_issue_3574.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_rank_reversal_test_issue_3574", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_config() -> dict[str, Any]:
    """Two planners, two seeds, one scenario -> 8 paired manifest rows."""
    return {
        "trace_metric_keys": ["clearance_m"],
        "planners": [{"key": "goal"}, {"key": "social_force"}],
        "seeds": [101, 102],
        "scenarios": [
            {
                "id": "classic_density_002",
                "density": 0.02,
                "population_size": 4,
                "archetype_seed": 3574,
                "composition": {"cautious": 0.25, "standard": 0.5, "hurried": 0.25},
                "archetypes": {
                    "cautious": {"desired_speed_factor": 0.7, "radius_m": 0.35},
                    "standard": {"desired_speed_factor": 1.0, "radius_m": 0.3},
                    "hurried": {"desired_speed_factor": 1.4, "radius_m": 0.25},
                },
            }
        ],
    }


def _records_for_manifest(
    manifest: dict[str, Any],
    *,
    mean_clearance_by_key: dict[tuple[str, str, int, str], float],
) -> list[dict[str, Any]]:
    """Build ready episode records, varying ``mean_clearance`` per campaign key."""
    records: list[dict[str, Any]] = []
    for row in manifest["manifest_rows"]:
        key = (row["scenario_id"], row["planner"], row["seed"], row["population_arm"])
        labels = row["arm_population"][PEDESTRIAN_CONTROL_TRACE_LABELS_KEY]
        pedestrians = [
            {
                **label,
                "steps": [
                    {"step": 0, "clearance_m": 0.8},
                    {"step": 1, "clearance_m": 1.2},
                ],
            }
            for label in labels
        ]
        record = {
            "scenario_id": row["scenario_id"],
            "planner": row["planner"],
            "seed": row["seed"],
            "population_arm": row["population_arm"],
            "metrics": {"mean_clearance": mean_clearance_by_key[key]},
            "algorithm_metadata": {
                "pedestrian_control_trace": {
                    "schema_version": "pedestrian-control-trace.v1",
                    "near_field_clearance_threshold_m": 1.0,
                    "pedestrian_count": len(pedestrians),
                    "pedestrians": pedestrians,
                }
            },
        }
        if "response_law_fraction" in row:
            record["response_law_fraction"] = row["response_law_fraction"]
        records.append(record)
    return records


def _write_inputs(tmp_path: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    manifest = build_mean_matched_harness_manifest(_manifest_config())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path = tmp_path / "episode_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return manifest_path, records_path


def _run_cli(module: ModuleType, argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = argv
    try:
        return int(module.main())
    finally:
        sys.argv = old_argv


def test_cli_renders_reversal_on_constructed_records(tmp_path: Path) -> None:
    """A constructed reversal (goal ahead heterogeneous, social_force ahead mean-matched)."""
    manifest = build_mean_matched_harness_manifest(_manifest_config())
    clearance: dict[tuple[str, str, int, str], float] = {}
    for row in manifest["manifest_rows"]:
        key = (row["scenario_id"], row["planner"], row["seed"], row["population_arm"])
        arm = row["population_arm"]
        planner = row["planner"]
        seed = row["seed"]
        if arm == "heterogeneous":
            clearance[key] = 2.0 if planner == "goal" else 0.5
        else:
            clearance[key] = 0.5 if planner == "goal" else 2.0
        # Vary by seed slightly so the bootstrap has resampling structure.
        clearance[key] += 0.01 * (seed - 101)
    records = _records_for_manifest(manifest, mean_clearance_by_key=clearance)
    manifest_path, records_path = _write_inputs(tmp_path, records)
    output_dir = tmp_path / "out"

    module = _load_cli()
    code = _run_cli(
        module,
        [
            "run_rank_reversal_test_issue_3574.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--num-bootstrap",
            "200",
            "--seed",
            "42",
        ],
    )

    assert code == 0
    result = json.loads((output_dir / "rank_reversal_test.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == RANK_REVERSAL_TEST_SCHEMA
    assert result["status"] == "ready"
    assert result["decision"] == "reject_null_rank_stability"
    assert result["reversal_count"] == 1
    readiness = json.loads(
        (output_dir / "rank_reversal_test_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["ready"] is True
    markdown = (output_dir / "rank_reversal_test.md").read_text(encoding="utf-8")
    assert "reject_null_rank_stability" in markdown
    assert "Analysis tooling only" in markdown


def test_cli_selects_one_response_law_fraction(tmp_path: Path) -> None:
    """The preregistered test selects, rather than mixes, sweep fractions."""

    config = _manifest_config()
    config["response_law_fractions"] = [0.5]
    manifest = build_mean_matched_harness_manifest(config)
    clearance: dict[tuple[str, str, int, str], float] = {}
    for row in manifest["manifest_rows"]:
        key = (row["scenario_id"], row["planner"], row["seed"], row["population_arm"])
        clearance[key] = 2.0 if row["planner"] == "goal" else 0.5
    records = _records_for_manifest(manifest, mean_clearance_by_key=clearance)
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "episode_records.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    code = _run_cli(
        _load_cli(),
        [
            "run_rank_reversal_test_issue_3574.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--response-law-fraction",
            "0.5",
            "--num-bootstrap",
            "20",
        ],
    )

    assert code == 0
    result = json.loads((tmp_path / "out" / "rank_reversal_test.json").read_text(encoding="utf-8"))
    assert result["status"] == "ready"
    assert result["arms"] == [
        "heterogeneous/response_law_fraction_0.5",
        "mean_matched_homogeneous/response_law_fraction_0.5",
    ]


def test_cli_fails_closed_on_missing_records_file(tmp_path: Path) -> None:
    """A missing records file exits non-zero with a clear message."""
    manifest = build_mean_matched_harness_manifest(_manifest_config())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    module = _load_cli()
    code = _run_cli(
        module,
        [
            "run_rank_reversal_test_issue_3574.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(tmp_path / "does_not_exist.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert code == 1


def test_cli_blocks_on_unready_records(tmp_path: Path) -> None:
    """Records that fail the readiness check exit code 2 and write no test artifact."""
    manifest = build_mean_matched_harness_manifest(_manifest_config())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # One record short of the manifest -> readiness blocked.
    records = _records_for_manifest(
        manifest,
        mean_clearance_by_key={
            (row["scenario_id"], row["planner"], row["seed"], row["population_arm"]): 1.0
            for row in manifest["manifest_rows"]
        },
    )[:-1]
    records_path = tmp_path / "episode_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    output_dir = tmp_path / "out"

    module = _load_cli()
    code = _run_cli(
        module,
        [
            "run_rank_reversal_test_issue_3574.py",
            "--manifest",
            str(manifest_path),
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert code == 2
    assert (output_dir / "rank_reversal_test_readiness.json").exists()
    # Blocked readiness must not produce a test artifact.
    assert not (output_dir / "rank_reversal_test.json").exists()


def test_cli_default_metric_matches_readiness_contract() -> None:
    """The default ``--metric-key`` is the rank metric enforced by the readiness check."""
    module = _load_cli()
    from robot_sf.benchmark.heterogeneous_population_ablation import RANK_METRIC_KEY

    assert module.RANK_METRIC_KEY == RANK_METRIC_KEY == "mean_clearance"
