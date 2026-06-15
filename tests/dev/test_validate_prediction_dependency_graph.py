"""Tests for the prediction-lane dependency graph validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.dev.validate_prediction_dependency_graph import validate

if TYPE_CHECKING:
    from pathlib import Path


def _write_graph(tmp_path: Path, graph: dict) -> Path:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path


def _minimal_graph() -> dict:
    return {
        "schema_version": "prediction_lane_dependency_graph.v1",
        "lane": "learned_prediction",
        "required_issue_set": [1, 2],
        "execution_order": [1, 2],
        "nodes": [
            {
                "issue": 1,
                "title": "parent",
                "issue_state": "open",
                "execution_state": "parent",
                "node_type": "epic",
                "depends_on": [],
                "blocked_by": [],
                "evidence_gates": [],
            },
            {
                "issue": 2,
                "title": "child",
                "issue_state": "closed",
                "execution_state": "completed",
                "node_type": "scaffold",
                "depends_on": [1],
                "blocked_by": [],
                "evidence_gates": [
                    {
                        "id": "ready",
                        "status": "passed",
                        "depends_on": [1],
                        "description": "validated",
                    }
                ],
            },
        ],
    }


def test_validate_prediction_dependency_graph_accepts_minimal_valid_graph(
    tmp_path: Path,
) -> None:
    """A well-formed graph should validate successfully."""
    assert validate(_write_graph(tmp_path, _minimal_graph())) == 0


def test_validate_prediction_dependency_graph_rejects_unknown_issue_reference(
    tmp_path: Path,
) -> None:
    """Unknown dependency references should fail validation."""
    graph = _minimal_graph()
    graph["nodes"][1]["depends_on"] = [99]

    assert validate(_write_graph(tmp_path, graph)) == 1


def test_validate_prediction_dependency_graph_rejects_unknown_status(
    tmp_path: Path,
) -> None:
    """Unknown execution states should fail validation."""
    graph = _minimal_graph()
    graph["nodes"][1]["execution_state"] = "mystery"

    assert validate(_write_graph(tmp_path, graph)) == 1


def test_validate_prediction_dependency_graph_rejects_cycles(tmp_path: Path) -> None:
    """Dependency cycles should fail validation without raising."""
    graph = _minimal_graph()
    graph["nodes"][0]["depends_on"] = [2]

    assert validate(_write_graph(tmp_path, graph)) == 1
