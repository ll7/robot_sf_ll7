"""Tests for benchmark JSONL to Parquet analytics export."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow.parquet as pq
import pytest

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.parquet_export import export_episodes_jsonl_to_parquet

if TYPE_CHECKING:
    from pathlib import Path


def _episode(
    *,
    episode_id: str,
    algo: str,
    scenario_family: str,
    seed: int,
    min_ttc: float,
    clearance: float,
    success: bool,
    termination_reason: str,
) -> dict[str, Any]:
    """Build a schema-like benchmark episode fixture."""
    return {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": f"{scenario_family}_dense_004",
        "seed": seed,
        "algo": algo,
        "scenario_params": {
            "algo": algo,
            "scenario_family": scenario_family,
            "density": 8,
            "spawn": {"pedestrian_count": 10},
        },
        "algorithm_metadata": {
            "algorithm": algo,
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
        },
        "metrics": {
            "min_ttc": min_ttc,
            "clearance": clearance,
            "success": success,
            "collisions": 0.0 if success else 1.0,
            "pedestrian_impact": {
                "canonical_reductions": {"accel_delta_mean": 0.1 if success else 0.4}
            },
        },
        "termination_reason": termination_reason,
        "outcome": {"collision_event": not success},
        "integrity": {"seed": seed},
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write benchmark records as JSONL."""
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_export_parquet_writes_stable_tables_and_metadata(tmp_path: Path) -> None:
    """Exporter should normalize benchmark JSONL into fixed analytics tables."""
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            _episode(
                episode_id="ep-a",
                algo="planner_a",
                scenario_family="crossing",
                seed=11,
                min_ttc=1.25,
                clearance=0.42,
                success=True,
                termination_reason="goal_reached",
            ),
            _episode(
                episode_id="ep-b",
                algo="planner_b",
                scenario_family="crossing",
                seed=12,
                min_ttc=0.2,
                clearance=0.05,
                success=False,
                termination_reason="collision",
            ),
        ],
    )

    result = export_episodes_jsonl_to_parquet(episodes_path, tmp_path / "analytics")

    assert result.record_count == 2
    assert result.output_dir == tmp_path / "analytics"
    assert (result.output_dir / "episodes.parquet").is_file()
    assert (result.output_dir / "metrics.parquet").is_file()
    assert (result.output_dir / "scenario_params.parquet").is_file()
    assert (result.output_dir / "algorithm_metadata.parquet").is_file()
    assert (result.output_dir / "duckdb_examples.sql").is_file()

    episodes = pq.read_table(result.output_dir / "episodes.parquet").to_pylist()
    assert [row["episode_id"] for row in episodes] == ["ep-a", "ep-b"]
    assert episodes[0]["algo"] == "planner_a"
    assert episodes[0]["scenario_family"] == "crossing"

    metrics = pq.read_table(result.output_dir / "metrics.parquet").to_pylist()
    metric_paths = {(row["episode_id"], row["metric_path"]) for row in metrics}
    assert ("ep-a", "min_ttc") in metric_paths
    assert ("ep-a", "pedestrian_impact.canonical_reductions.accel_delta_mean") in metric_paths

    params = pq.read_table(result.output_dir / "scenario_params.parquet").to_pylist()
    param_paths = {(row["episode_id"], row["param_path"]) for row in params}
    assert ("ep-a", "spawn.pedestrian_count") in param_paths

    metadata = json.loads((result.output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "benchmark_parquet_export.v1"
    assert metadata["jsonl_is_source_of_truth"] is True
    assert metadata["record_count"] == 2
    assert metadata["tables"]["episodes"]["rows"] == 2
    assert metadata["source_files"][0]["sha256"]


def test_export_parquet_refuses_to_overwrite_existing_outputs(tmp_path: Path) -> None:
    """Existing exports should require explicit overwrite to avoid silent replacement."""
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            _episode(
                episode_id="ep-a",
                algo="planner_a",
                scenario_family="crossing",
                seed=11,
                min_ttc=1.25,
                clearance=0.42,
                success=True,
                termination_reason="goal_reached",
            )
        ],
    )

    export_episodes_jsonl_to_parquet(episodes_path, tmp_path / "analytics")

    with pytest.raises(FileExistsError, match="already exists"):
        export_episodes_jsonl_to_parquet(episodes_path, tmp_path / "analytics")


def test_cli_export_parquet_supports_duckdb_fixture_query(tmp_path: Path) -> None:
    """CLI export should produce Parquet files DuckDB can query deterministically."""
    episodes_path = tmp_path / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            _episode(
                episode_id="ep-a",
                algo="planner_a",
                scenario_family="crossing",
                seed=11,
                min_ttc=1.25,
                clearance=0.42,
                success=True,
                termination_reason="goal_reached",
            ),
            _episode(
                episode_id="ep-b",
                algo="planner_a",
                scenario_family="crossing",
                seed=12,
                min_ttc=0.75,
                clearance=0.38,
                success=True,
                termination_reason="goal_reached",
            ),
        ],
    )
    out_dir = tmp_path / "analytics"

    assert cli_main(["export-parquet", "--in", str(episodes_path), "--out-dir", str(out_dir)]) == 0

    rows = duckdb.sql(
        f"""
        SELECT e.algo, e.scenario_family, AVG(m.value_number) AS avg_min_ttc
        FROM read_parquet('{out_dir / "episodes.parquet"}') AS e
        JOIN read_parquet('{out_dir / "metrics.parquet"}') AS m USING (episode_id)
        WHERE m.metric_path = 'min_ttc'
        GROUP BY e.algo, e.scenario_family
        """
    ).fetchall()
    assert rows == [("planner_a", "crossing", 1.0)]
