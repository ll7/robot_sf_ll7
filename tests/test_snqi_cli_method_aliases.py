"""Tests for the SNQI CLI method-name and dispatch surface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from robot_sf.benchmark.snqi import cli as snqi_cli
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES
from robot_sf.benchmark.snqi.types import SNQIWeights

if TYPE_CHECKING:
    from pathlib import Path


def _baseline_stats() -> dict[str, dict[str, float]]:
    """Return compact baseline stats for SNQI CLI tests."""
    return {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 1.0, "p95": 3.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "jerk_mean": {"med": 0.1, "p95": 0.4},
    }


def _episode_records() -> list[dict[str, object]]:
    """Return small episode records with all SNQI inputs represented."""
    return [
        {
            "scenario_id": "s1",
            "metrics": {
                "success": 1.0,
                "time_to_goal_norm": 0.4,
                "collisions": 0,
                "near_misses": 1,
                "comfort_exposure": 0.2,
                "force_exceed_events": 0,
                "jerk_mean": 0.1,
            },
        },
        {
            "scenario_id": "s2",
            "metrics": {
                "success": 0.0,
                "time_to_goal_norm": 0.9,
                "collisions": 1,
                "near_misses": 3,
                "comfort_exposure": 0.5,
                "force_exceed_events": 1,
                "jerk_mean": 0.3,
            },
        },
    ]


def _write_episodes_jsonl(path: Path) -> None:
    """Write compact episode records to a JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for record in _episode_records():
            handle.write(json.dumps(record) + "\n")


def test_snqi_recompute_help_lists_only_canonical_methods(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI help should advertise only the supported canonical method names."""
    parser = snqi_cli.create_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["recompute", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "canonical" in help_text
    assert "balanced" in help_text
    assert "optimized" in help_text
    assert "pareto_optimization" not in help_text
    assert "equal_weights" not in help_text
    assert "safety_focused" not in help_text
    assert "deprecated" not in help_text.lower()


@pytest.mark.parametrize(
    "deprecated_method", ["pareto_optimization", "equal_weights", "safety_focused"]
)
def test_snqi_recompute_rejects_deprecated_method_aliases(
    deprecated_method: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deprecated method aliases should fail at argument parsing with a clear choice error."""
    parser = snqi_cli.create_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "recompute",
                "--baseline-stats",
                "baseline.json",
                "--out",
                "weights.json",
                "--method",
                deprecated_method,
            ],
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr
    assert deprecated_method in stderr
    assert "canonical" in stderr
    assert "balanced" in stderr
    assert "optimized" in stderr


def test_cmd_recompute_weights_writes_canonical_method_output(tmp_path: Path) -> None:
    """Direct recompute command should write weights for canonical methods."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "weights.json"
    baseline_path.write_text(json.dumps(_baseline_stats()), encoding="utf-8")

    exit_code = snqi_cli.cmd_recompute_weights(
        argparse.Namespace(
            baseline_stats=str(baseline_path),
            out=str(output_path),
            method="balanced",
            seed=7,
        ),
    )

    assert exit_code == 0
    weights = SNQIWeights.load(output_path)
    assert weights.bootstrap_params["method"] == "balanced"
    assert weights.bootstrap_params["seed"] == 7
    assert set(weights.weights) == set(WEIGHT_NAMES)
    assert all(value == 1.0 for value in weights.weights.values())


def test_cmd_recompute_weights_rejects_missing_or_unknown_inputs(tmp_path: Path) -> None:
    """Direct recompute command should fail closed for missing stats and unknown methods."""
    missing = snqi_cli.cmd_recompute_weights(
        argparse.Namespace(
            baseline_stats=str(tmp_path / "missing.json"),
            out=str(tmp_path / "weights.json"),
            method="balanced",
            seed=7,
        ),
    )
    assert missing == 1

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline_stats()), encoding="utf-8")
    unknown_method = snqi_cli.cmd_recompute_weights(
        argparse.Namespace(
            baseline_stats=str(baseline_path),
            out=str(tmp_path / "weights.json"),
            method="old_alias",
            seed=7,
        ),
    )
    assert unknown_method == 2


def test_cmd_recompute_weights_logs_failing_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected SNQI recompute failures should log the stage before returning non-zero."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "weights.json"
    baseline_path.write_text(json.dumps(_baseline_stats()), encoding="utf-8")

    def _fail_recompute(**kwargs: object) -> SNQIWeights:
        raise ValueError("cannot recompute")

    monkeypatch.setattr(snqi_cli, "recompute_snqi_weights", _fail_recompute)
    captured: list = []
    handle = logger.add(captured.append, level="ERROR")
    try:
        exit_code = snqi_cli.cmd_recompute_weights(
            argparse.Namespace(
                baseline_stats=str(baseline_path),
                out=str(output_path),
                method="balanced",
                seed=7,
            ),
        )
    finally:
        logger.remove(handle)

    assert exit_code == 1
    assert any(
        msg.record["extra"].get("event") == "snqi_cli_failed"
        and msg.record["extra"].get("stage") == "recompute_weights"
        for msg in captured
    )


def test_episode_loading_and_baseline_stats_helpers(tmp_path: Path) -> None:
    """JSONL and stats helpers should skip malformed rows and compute neutral fallbacks."""
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
        "\n".join(
            [
                json.dumps({"metrics": {"collisions": 0, "near_misses": 1, "jerk_mean": "bad"}}),
                "{malformed",
                json.dumps({"metrics": {"collisions": 2, "near_misses": 5, "jerk_mean": 0.2}}),
                "",
            ],
        ),
        encoding="utf-8",
    )

    episodes = snqi_cli._load_episodes_jsonl(episodes_path)
    assert len(episodes) == 2
    assert snqi_cli._extract_metric_values(episodes, "collisions") == [0.0, 2.0]
    assert snqi_cli._extract_metric_values(episodes, "missing") == []
    stats = snqi_cli._compute_baseline_stats(episodes)
    assert stats["collisions"] == {"med": 1.0, "p95": 2.0}
    assert stats["force_exceed_events"] == {"med": 0.0, "p95": 1.0}
    assert stats["jerk_mean"] == {"med": 0.2, "p95": 1.2}


def test_cmd_ablation_analysis_writes_summary_with_default_weights(tmp_path: Path) -> None:
    """Ablation command should derive stats and write a summary without a weights file."""
    episodes_path = tmp_path / "episodes.jsonl"
    output_path = tmp_path / "ablation.json"
    _write_episodes_jsonl(episodes_path)

    exit_code = snqi_cli.cmd_ablation_analysis(
        argparse.Namespace(
            episodes=str(episodes_path),
            summary_out=str(output_path),
            weights=None,
            components=["w_success", "w_collisions"],
            seed=13,
        ),
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["components"] == ["w_success", "w_collisions"]
    assert payload["episode_count"] == 2
    assert payload["seed"] == 13
    assert set(payload["impacts"]) == {"w_success", "w_collisions"}
    assert (
        payload["baseline_stats"]["collisions"]["p95"]
        >= payload["baseline_stats"]["collisions"]["med"]
    )


def test_cmd_ablation_analysis_supports_weights_file_and_failure_modes(tmp_path: Path) -> None:
    """Ablation command should load explicit weights and fail closed on missing inputs."""
    episodes_path = tmp_path / "episodes.jsonl"
    output_path = tmp_path / "ablation.json"
    weights_path = tmp_path / "weights.json"
    _write_episodes_jsonl(episodes_path)
    SNQIWeights(
        weights_version="test",
        created_at="2026-01-01T00:00:00",
        git_sha="test",
        baseline_stats_path="baseline.json",
        baseline_stats_hash="hash",
        normalization_strategy="median_p95_clamp",
        components=list(WEIGHT_NAMES),
        weights=dict.fromkeys(WEIGHT_NAMES, 1.0),
    ).save(weights_path)

    exit_code = snqi_cli.cmd_ablation_analysis(
        argparse.Namespace(
            episodes=str(episodes_path),
            summary_out=str(output_path),
            weights=str(weights_path),
            components=None,
            seed=21,
        ),
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["components"] == WEIGHT_NAMES
    assert payload["weights_used"] == dict.fromkeys(WEIGHT_NAMES, 1.0)

    assert (
        snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(tmp_path / "missing.jsonl"),
                summary_out=str(output_path),
                weights=None,
                components=None,
                seed=21,
            ),
        )
        == 1
    )

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    assert (
        snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(empty_path),
                summary_out=str(output_path),
                weights=None,
                components=None,
                seed=21,
            ),
        )
        == 1
    )
    assert (
        snqi_cli.cmd_ablation_analysis(
            argparse.Namespace(
                episodes=str(episodes_path),
                summary_out=str(output_path),
                weights=str(tmp_path / "missing_weights.json"),
                components=None,
                seed=21,
            ),
        )
        == 1
    )


def test_main_dispatches_commands_and_requires_subcommand(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Main should dispatch parsed subcommands and show help when no subcommand is provided."""
    monkeypatch.setattr(sys, "argv", ["robot_sf_snqi"])
    assert snqi_cli.main() == 1
    assert "SNQI weight management" in capsys.readouterr().out

    monkeypatch.setattr(
        sys, "argv", ["robot_sf_snqi", "recompute", "--baseline-stats", "b", "--out", "o"]
    )
    monkeypatch.setattr(snqi_cli, "cmd_recompute_weights", lambda args: 42)
    assert snqi_cli.main() == 42

    monkeypatch.setattr(
        sys, "argv", ["robot_sf_snqi", "ablation", "--episodes", "e", "--summary-out", "s"]
    )
    monkeypatch.setattr(snqi_cli, "cmd_ablation_analysis", lambda args: 43)
    assert snqi_cli.main() == 43
