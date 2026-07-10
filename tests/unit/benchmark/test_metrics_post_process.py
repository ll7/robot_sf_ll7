"""Tests for post-processing helpers in benchmark metrics."""

from loguru import logger

from robot_sf.benchmark.metrics import post_process_metrics


def test_post_process_metrics_promotes_force_quantiles_and_success():
    """Verify success normalization and force quantile grouping for schema consistency."""
    metrics_raw = {
        "success": 1.0,
        "force_q50": 0.2,
        "force_q90": 0.4,
        "force_q95": 0.6,
        "collisions": 1.0,
    }

    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)

    assert metrics["success"] is True
    assert metrics["collisions"] == 1
    assert "force_quantiles" in metrics
    assert "force_q50" not in metrics
    assert metrics["force_quantiles"]["q50"] == 0.2


def test_post_process_metrics_drops_non_finite_values():
    """Verify NaN values are removed to keep JSONL outputs valid."""
    metrics_raw = {
        "success": 0.0,
        "mean_distance": float("nan"),
        "near_misses": 0.0,
    }

    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)

    assert "mean_distance" not in metrics
    assert metrics["near_misses"] == 0


def test_post_process_metrics_logs_and_drops_invalid_count_and_flag_values():
    """Invalid count/flag payloads should not survive into serialized metric output."""
    captured: list = []
    handle = logger.add(captured.append, level="WARNING")
    try:
        metrics = post_process_metrics(
            {
                "success": 0.0,
                "collisions": "not-a-count",
                "near_misses": 2.0,
                "social_proxemic_available": "not-a-flag",
            },
            snqi_weights=None,
            snqi_baseline=None,
        )
    finally:
        logger.remove(handle)

    assert "collisions" not in metrics
    assert "social_proxemic_available" not in metrics
    assert metrics["near_misses"] == 2
    logged_keys = {msg.record["extra"].get("metric_key") for msg in captured}
    assert {"collisions", "social_proxemic_available"} <= logged_keys
    count_warning = next(
        msg.record["message"]
        for msg in captured
        if msg.record["extra"].get("metric_key") == "collisions"
    )
    assert "collisions" in count_warning
    assert "'not-a-count'" in count_warning
