"""Tests for calibrate_scenario_priors_from_traces_issue_2726.py."""

from __future__ import annotations

import json
import pathlib
import tempfile

import yaml

from scripts.analysis.calibrate_scenario_priors_from_traces_issue_2726 import (
    assign_to_cluster,
    extract_features,
    generate_prior_cards,
    main,
)


def test_extract_features_minimal():
    """Verify feature extraction with a minimal mock trace data."""
    trace_data = {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": "test_trace_1",
        "source": {
            "scenario_id": "classic_bottleneck_medium",
            "seed": 111,
            "planner_id": "hybrid_rule_v0_minimal",
            "episode_id": "test_episode_1",
        },
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0], "velocity": [1.0, 0.0]},
                "pedestrians": [
                    {
                        "id": 1,
                        "position": [1.0, 0.0],
                        "signal_state": {"available": True, "label": "red"},
                    }
                ],
                "planner": {"active_hypothesis": "hyp_1"},
            },
            {
                "step": 1,
                "time_s": 0.1,
                "robot": {"position": [0.1, 0.0], "velocity": [0.0, 0.0]},
                "pedestrians": [
                    {
                        "id": 1,
                        "position": [1.0, 0.0],
                        "signal_state": {"available": True, "label": "green"},
                    }
                ],
                "planner": {"active_hypothesis": "hyp_2"},
            },
        ],
    }

    file_path = pathlib.Path("/tmp/mock_trace.json")
    result = extract_features(trace_data, file_path)

    assert result["trace_id"] == "test_trace_1"
    assert result["scenario_id"] == "classic_bottleneck_medium"
    assert result["episode_id"] == "test_episode_1"
    assert result["planner_id"] == "hybrid_rule_v0_minimal"
    assert result["seed"] == 111

    features = result["features"]
    assert features["bottleneck_width"] == 1.5
    assert features["pedestrian_count"] == 1
    assert features["pedestrian_density"] == "medium"
    assert features["has_signal"] is True
    assert features["signal_green_fraction"] == 0.5
    assert features["robot_displacement_m"] == 0.1
    assert features["robot_total_distance_m"] == 0.1
    # First frame speed = 1.0 >= 0.05, second speed = 0.0 < 0.05. stop fraction = 0.5
    assert features["stop_fraction"] == 0.5
    # topology switches: active_hypothesis transitioned hyp_1 -> hyp_2
    assert features["topology_switches"] == 1
    # min distance robot-ped is at step 1: dist([0.1, 0.0], [1.0, 0.0]) = 0.9.
    assert features["min_distance_m"] == 0.9
    assert features["outcome"] == "nominal"


def test_assign_to_cluster():
    """Verify cluster assignment returns deterministic names."""
    features = {
        "pedestrian_count": 3,
        "has_signal": True,
        "outcome": "near-miss",
    }
    cluster = assign_to_cluster(features, "issue_2868_signalized_crossing")
    assert cluster == "crossing_dense_signalized_near-miss"


def test_generate_prior_cards():
    """Verify prior registry cards structure matches sdd_scenario_distribution_candidate formatting."""
    clusters = {
        "crossing_sparse_signalized_nominal": [
            {
                "file_path": "tests/fixtures/mock.json",
                "trace_id": "t1",
                "episode_id": "ep1",
                "features": {
                    "min_distance_m": 0.9,
                    "stop_fraction": 0.0,
                },
            },
            {
                "file_path": "tests/fixtures/mock_2.json",
                "trace_id": "t2",
                "episode_id": "ep2",
                "features": {
                    "min_distance_m": 0.7,
                    "stop_fraction": 0.5,
                },
            },
        ]
    }
    registry = generate_prior_cards(clusters)

    assert registry["schema_version"] == "scenario-prior-card-registry.v1"
    assert len(registry["cards"]) == 1
    card = registry["cards"][0]
    assert card["card_id"] == "trace_cluster_crossing_sparse_signalized_nominal"
    assert card["classification"] == "repository_trace_derived"
    assert card["source_traces"] == [
        "tests/fixtures/mock.json (trace_id: t1, episode_id: ep1)",
        "tests/fixtures/mock_2.json (trace_id: t2, episode_id: ep2)",
    ]
    assert "repository-trace-grounded" in card["odd_conditions"][0]


def test_main_cli_execution():
    """Verify main function parses arguments, processes inputs, and writes reports correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        # Write mock trace file
        trace_file = trace_dir / "mock_trace.json"
        mock_data = {
            "schema_version": "simulation_trace_export.v1",
            "trace_id": "mock_1",
            "source": {
                "scenario_id": "classic_bottleneck_medium",
                "seed": 111,
                "planner_id": "hybrid_rule_v0_minimal",
                "episode_id": "ep_1",
            },
            "frames": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                    "pedestrians": [],
                    "planner": {},
                }
            ],
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(mock_data, f)

        output_dir = tmp_path / "output"

        exit_code = main(
            [
                "--trace-dir",
                str(trace_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert exit_code == 0
        assert (output_dir / "report.md").exists()
        assert (output_dir / "report.json").exists()
        assert (output_dir / "scenario_prior_cards_issue_2726.yaml").exists()
        assert (output_dir / "README.md").exists()

        # Load and verify content of report.json
        with open(output_dir / "report.json", encoding="utf-8") as f:
            report_data = json.load(f)
            assert report_data["total_processed"] == 1
            assert report_data["trace_count"] == 1
            assert report_data["cluster_count"] == 1
            assert "bottleneck_sparse_unsignalized_nominal" in report_data["clusters"]
            assert report_data["limitations"]
            assert report_data["follow_up_data_requirements"]
            assert report_data["scenario_prior_candidates"][0]["source_traces"]

        # Load and verify content of cards yaml
        with open(output_dir / "scenario_prior_cards_issue_2726.yaml", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            assert yaml_data["issue"] == 2726
            assert len(yaml_data["cards"]) == 1
