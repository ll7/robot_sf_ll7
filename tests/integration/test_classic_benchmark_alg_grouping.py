"""Integration smoke test for benchmark aggregation algorithm grouping."""

from __future__ import annotations

from loguru import logger

from robot_sf.benchmark.aggregate import compute_aggregates_with_ci
from robot_sf.benchmark.full_classic import orchestrator


def _make_base_record(algo: str, episode_suffix: int) -> dict[str, object]:
    """Make base record.

    Args:
        algo: Auto-generated placeholder description.
        episode_suffix: Auto-generated placeholder description.

    Returns:
        dict[str, object]: Auto-generated placeholder description.
    """
    record: dict[str, object] = {
        "episode_id": f"{algo}-{episode_suffix}",
        "scenario_id": f"scenario-{episode_suffix}",
        "scenario_params": {"density": "medium"},
        "metrics": {"success_rate": 0.5 + episode_suffix * 0.1},
    }
    orchestrator._ensure_algo_metadata(record, algo=algo, episode_id=record["episode_id"])
    return record


def test_smoke_aggregation_workflow():
    """Aggregated output should track effective group keys and missing algorithms."""

    records = [
        _make_base_record("sf", 1),
        _make_base_record("random", 2),
    ]

    captured: list = []
    handle = logger.add(lambda message: captured.append(message), level="WARNING")
    try:
        result = compute_aggregates_with_ci(
            records,
            expected_algorithms={"sf", "ppo", "random"},
            return_ci=False,
        )
    finally:
        logger.remove(handle)

    assert set(result.keys()) >= {"sf", "random", "_meta"}
    assert result["_meta"]["group_by"] == "scenario_params.algo"
    assert result["_meta"]["effective_group_key"] == "scenario_params.algo | algo | scenario_id"
    assert result["_meta"]["missing_algorithms"] == ["ppo"]
    assert result["_meta"]["warnings"]
    assert any(
        msg.record["extra"].get("event") == "aggregation_missing_algorithms" for msg in captured
    )
