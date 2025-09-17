from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.cli import cli_main

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 3) -> None:
    scenarios = [
        {
            "id": "agg-smoke",
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
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def test_cli_aggregate_without_ci(tmp_path: Path, capsys):
    # Prepare episodes via run
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=2)
    episodes = tmp_path / "episodes.jsonl"

    rc_run = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(episodes),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "6",
            "--dt",
            "0.1",
        ]
    )
    cap = capsys.readouterr()
    assert rc_run == 0, f"run failed: {cap.err}"

    # Aggregate without CI
    out_json = tmp_path / "summary.json"
    rc = cli_main(["aggregate", "--in", str(episodes), "--out", str(out_json)])
    cap2 = capsys.readouterr()
    assert rc == 0, f"aggregate failed: {cap2.err}"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    # Default grouping is scenario_params.algo; when missing it falls back to scenario_id
    # Our demo matrix has id 'agg-smoke', so expect that key
    assert "agg-smoke" in data
    # Check that known metrics exist at least for mean/median/p95
    any_metric = next(iter(data["agg-smoke"].keys()))
    stats = data["agg-smoke"][any_metric]
    assert all(k in stats for k in ("mean", "median", "p95"))
    # CI keys should not be present when bootstrap_samples=0 (default)
    assert not any(k.endswith("_ci") for k in stats.keys())


def test_cli_aggregate_with_ci_and_seed(tmp_path: Path, capsys):
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=3)
    episodes = tmp_path / "episodes.jsonl"

    rc_run = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(episodes),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "6",
            "--dt",
            "0.1",
        ]
    )
    cap = capsys.readouterr()
    assert rc_run == 0, f"run failed: {cap.err}"

    out1 = tmp_path / "summary_ci1.json"
    out2 = tmp_path / "summary_ci2.json"

    args = [
        "aggregate",
        "--in",
        str(episodes),
        "--out",
        str(out1),
        "--bootstrap-samples",
        "200",
        "--bootstrap-confidence",
        "0.9",
        "--bootstrap-seed",
        "123",
    ]
    rc1 = cli_main(args)
    cap1 = capsys.readouterr()
    assert rc1 == 0, f"aggregate ci failed: {cap1.err}"

    # Deterministic with same seed
    # Run again with same seed but a different output path to verify determinism
    args2 = args.copy()
    out_idx = args2.index("--out") + 1
    args2[out_idx] = str(out2)
    rc2 = cli_main(args2)
    cap2 = capsys.readouterr()
    assert rc2 == 0, f"aggregate ci repeat failed: {cap2.err}"

    d1 = json.loads(out1.read_text(encoding="utf-8"))
    d2 = json.loads(out2.read_text(encoding="utf-8"))

    assert "agg-smoke" in d1
    # Pick a metric and verify CI keys present and shape
    _, stats = next(iter(d1["agg-smoke"].items()))
    assert "mean_ci" in stats and "median_ci" in stats and "p95_ci" in stats
    assert len(stats["mean_ci"]) == 2
    assert isinstance(stats["mean_ci"][0], float)
    assert stats["mean_ci"][0] <= stats["mean_ci"][1]

    # Determinism check: summaries should be identical with same seed
    assert d1 == d2
