"""Contract tests for the issue #9609 actor/observation diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analysis.narrow_doorway_actor_response_issue_9609 import (
    OFFSETS,
    SEEDS,
    _nearest_forward_obstacle,
    _without_static_geometry,
    classify,
)


def _model_obs() -> dict[str, np.ndarray]:
    grid = np.zeros((3, 10, 10), dtype=np.float32)
    grid[0, 5, 7] = 1.0
    grid[1, 4, 6] = 1.0
    grid[2] = np.maximum(grid[0], grid[1])
    return {
        "occupancy_grid": grid,
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, -1, 2], dtype=np.int32),
        "occupancy_grid_meta_origin": np.asarray([-1.0, -1.0], dtype=np.float32),
        "occupancy_grid_meta_resolution": np.asarray([0.2], dtype=np.float32),
        "goal_current": np.asarray([2.0, 2.0], dtype=np.float32),
    }


def test_static_geometry_ablation_preserves_non_geometry_fields() -> None:
    source = _model_obs()
    ablated = _without_static_geometry(source)

    assert ablated["occupancy_grid"][0].sum() == 0.0
    np.testing.assert_array_equal(ablated["occupancy_grid"][1], source["occupancy_grid"][1])
    np.testing.assert_array_equal(ablated["occupancy_grid"][2], source["occupancy_grid"][1])
    np.testing.assert_array_equal(ablated["goal_current"], source["goal_current"])
    assert source["occupancy_grid"][0].sum() == 1.0


def test_nearest_forward_obstacle_uses_ego_grid_metadata() -> None:
    distance, count = _nearest_forward_obstacle(_model_obs())

    assert count == 1
    np.testing.assert_allclose(distance, np.hypot(0.5, 0.1), atol=1e-6)


def _supported_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for offset in OFFSETS:
            rows.append(
                {
                    "seed": seed,
                    "contact_step": 60,
                    "offset_before_contact": offset,
                    "static_geometry_present": True,
                    "nearest_forward_obstacle_m": float(offset) / 5.0,
                    "canonical_model_predict_v": 2.0,
                    "canonical_critic_value": -20.0 + float(offset) / 2.0,
                    "adapter_command_v": 2.0,
                    "executed_linear_speed_after": 2.0,
                    "adapter_forward_intent_preserved": True,
                    "fallback_or_degraded": False,
                }
            )
    return rows


def test_classify_supports_only_complete_matched_signal() -> None:
    summary = classify(_supported_rows())

    assert summary["row_count"] == 12
    assert summary["result_classification"] == "actor_value_response_mismatch_supported"
    assert summary["checks"]["fallback_or_degraded_rows"] == 0


def test_classify_fails_closed_when_geometry_is_missing() -> None:
    rows = _supported_rows()
    rows[0]["static_geometry_present"] = False

    assert classify(rows)["result_classification"] == "not_identifiable"


def test_tracked_summary_has_diagnostic_claim_boundary() -> None:
    evidence = (
        Path(__file__).resolve().parents[2]
        / "docs/context/evidence/issue_9609_narrow_doorway_actor_response/mechanism_summary.json"
    )
    if not evidence.exists():
        return
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["result_classification"] in {
        "actor_value_response_mismatch_supported",
        "not_identifiable",
    }
    assert payload["evidence_tier"] == "diagnostic-only"
    assert "not benchmark" in payload["claim_boundary"]
