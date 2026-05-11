"""TODO docstring. Document this module."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.aggregate import (
    compute_aggregates,
    compute_aggregates_with_ci,
    flatten_metrics,
    read_jsonl,
    write_episode_csv,
)
from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _make_sample_jsonl(tmp_path: Path) -> Path:
    # Use run_batch to generate 3 episodes across 2 algos (via scenario_params.algo)
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.

    Returns:
        TODO docstring.
    """
    scenarios = [
        {
            "id": "agg-uni-low-open-A",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 2,
            "algo": "A",
        },
        {
            "id": "agg-uni-low-open-B",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
            "algo": "B",
        },
    ]
    out_file = tmp_path / "episodes.jsonl"
    # run without forces for speed
    run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=10,
        horizon=8,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    return out_file


def test_read_and_flatten_and_write_csv(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    assert len(recs) == 3
    rows = [flatten_metrics(r) for r in recs]
    assert all("episode_id" in r for r in rows)
    # Write CSV
    csv_path = tmp_path / "episodes.csv"
    out = write_episode_csv(recs, csv_path)
    assert Path(out).exists()
    text = Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("episode_id,scenario_id,seed")


def test_compute_aggregates_group_by_algo(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    # We stored algo at the top-level of scenario params; group path is scenario_params.algo
    summary = compute_aggregates(recs, group_by="scenario_params.algo")
    # Should contain two groups A and B plus metadata
    algorithm_groups = {k for k in summary.keys() if k != "_meta"}
    assert algorithm_groups == {"A", "B"}
    # Should have _meta section
    assert "_meta" in summary
    # Each group should have numeric aggregates present for some core metric
    for group_name, metrics in summary.items():
        if group_name == "_meta":
            continue  # Skip metadata section
        assert "time_to_goal_norm" in metrics
        assert set(metrics["time_to_goal_norm"].keys()) == {"mean", "median", "p95"}


def test_compute_aggregates_with_ci_shape_and_determinism(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    # Compute with CIs
    summary_ci = compute_aggregates_with_ci(
        recs,
        group_by="scenario_params.algo",
        bootstrap_samples=200,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )
    # Basic shape: groups and keys
    algorithm_groups = {k for k in summary_ci.keys() if k != "_meta"}
    assert algorithm_groups == {"A", "B"}
    # Should have _meta section
    assert "_meta" in summary_ci
    any_group = next(iter(g for k, g in summary_ci.items() if k != "_meta"))
    # Ensure a known metric exists
    assert "time_to_goal_norm" in any_group
    m = any_group["time_to_goal_norm"]
    # Base stats still present
    assert {"mean", "median", "p95"}.issubset(set(m.keys()))
    # CI keys present and are [low, high]
    for key in ("mean_ci", "median_ci", "p95_ci"):
        assert key in m
        ci = m[key]
        assert isinstance(ci, list) and len(ci) == 2
        assert all(isinstance(v, float) for v in ci)
        assert ci[0] <= ci[1]

    # Determinism with same seed
    summary_ci_2 = compute_aggregates_with_ci(
        recs,
        group_by="scenario_params.algo",
        bootstrap_samples=200,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )
    assert summary_ci == summary_ci_2


def _paired_contrast_records() -> list[dict]:
    """Build synthetic records with a planted paired effect for group B over group A."""
    values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    deltas = [1.0, 2.0, 3.0, 1.0, 2.0]
    records = []
    for seed, (value_a, delta) in enumerate(zip(values_a, deltas, strict=True), start=1):
        for algo, score in (("A", value_a), ("B", value_a + delta)):
            records.append(
                {
                    "episode_id": f"{algo}-{seed}",
                    "scenario_id": "paired-scenario",
                    "seed": seed,
                    "scenario_params": {"algo": algo},
                    "metrics": {"score": score},
                }
            )
    return records


def test_compute_aggregates_with_ci_emits_pairwise_contrasts_for_planted_effect() -> None:
    """Pairwise contrast block should recover a planted paired group difference."""
    summary_ci = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=500,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )

    contrasts = summary_ci["pairwise_contrasts"]
    assert contrasts["_meta"]["pairing_keys"] == ["scenario_id", "seed_or_seed_index"]
    assert contrasts["_meta"]["p_value_correction"] == "holm"
    score = contrasts["A__vs__B"]["metrics"]["score"]
    assert score["n_pairs"] == 5
    assert score["delta_mean"] == 1.8
    assert score["delta_ci"][0] > 0.0
    assert score["p_value_holm"] >= score["p_value"]
    assert score["effect_size"]["type"] == "paired_cohens_dz"
    assert score["effect_size"]["value"] > 0.0


def test_compute_aggregates_with_ci_pairwise_contrasts_are_seed_deterministic() -> None:
    """Pairwise bootstrap output should be reproducible for identical inputs and seeds."""
    first = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=250,
        bootstrap_confidence=0.90,
        bootstrap_seed=456,
    )
    second = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=250,
        bootstrap_confidence=0.90,
        bootstrap_seed=456,
    )
    assert first["pairwise_contrasts"] == second["pairwise_contrasts"]
