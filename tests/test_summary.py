"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.errors import EpisodeRecordInputError
from robot_sf.benchmark.summary import (
    aggregate_training_metrics_with_bootstrap,
    bootstrap_metric_confidence,
    collect_values,
    load_episode_records,
    summarize_to_plots,
)


def _write_sample_jsonl(path: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    records = [
        {
            "episode_id": "e1",
            "scenario_id": "s1",
            "seed": 0,
            "metrics": {"min_distance": 0.42, "avg_speed": 0.8},
        },
        {
            "episode_id": "e2",
            "scenario_id": "s1",
            "seed": 1,
            "metrics": {"min_distance": 0.31, "avg_speed": 1.2},
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_summary_creates_pngs(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    src = tmp_path / "episodes.jsonl"
    _write_sample_jsonl(src)
    out_dir = tmp_path / "figs"
    outs = summarize_to_plots(src, out_dir)
    # We expect two images (min_distance + avg_speed)
    assert len(outs) == 2
    for p in outs:
        assert Path(p).exists()


def test_load_episode_records_fails_closed_on_malformed_jsonl(tmp_path: Path) -> None:
    """Malformed JSONL should fail with path and line context by default."""
    src = tmp_path / "episodes_bad.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"metrics": {"min_distance": 0.4}}),
                "{bad json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeRecordInputError) as excinfo:
        load_episode_records(src)

    message = str(excinfo.value)
    assert "malformed_lines=1" in message
    assert f"{src}:2" in message


def test_load_episode_records_fails_closed_on_missing_path(tmp_path: Path) -> None:
    """Missing benchmark input paths should not be silently ignored."""
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(EpisodeRecordInputError) as excinfo:
        load_episode_records(missing)

    assert "missing_paths=1" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_load_episode_records_best_effort_is_explicit(tmp_path: Path) -> None:
    """Exploratory callers may opt into best-effort parsing explicitly."""
    src = tmp_path / "episodes_bad.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "ok", "metrics": {"min_distance": 0.4}}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    records = load_episode_records(src, strict=False)

    assert [record["episode_id"] for record in records] == ["ok"]


def test_load_episode_records_handles_multiple_paths_and_blank_lines(tmp_path: Path) -> None:
    """Strict loading should accept multiple clean files and ignore empty lines."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"episode_id": "a"}) + "\n\n", encoding="utf-8")
    second.write_text(json.dumps({"episode_id": "b"}) + "\n", encoding="utf-8")

    records = load_episode_records([first, second])

    assert [record["episode_id"] for record in records] == ["a", "b"]


def test_summarize_to_plots_best_effort_is_explicit_end_to_end(tmp_path: Path) -> None:
    """Best-effort plotting should be opt-in and still process valid records."""
    src = tmp_path / "episodes_bad.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_id": "ok",
                        "metrics": {"min_distance": 0.4, "avg_speed": 0.8},
                    }
                ),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    outputs = summarize_to_plots(src, tmp_path / "figs", strict=False)

    assert len(outputs) == 2
    assert all(Path(path).exists() for path in outputs)


def test_collect_values_reports_malformed_trajectory_fallback() -> None:
    """Invalid fallback velocity payloads should be visible in strict mode."""
    records = [
        {
            "episode_id": "bad-speed",
            "metrics": {"min_distance": 0.2},
            "trajectory": {"robot_vel": [["not-a-number", 0.0]]},
        }
    ]

    with pytest.raises(EpisodeRecordInputError) as excinfo:
        collect_values(records)

    assert "fallback_derivation_errors=1" in str(excinfo.value)
    assert "bad-speed" in str(excinfo.value)


def test_collect_values_derives_speed_from_valid_trajectory() -> None:
    """Valid robot velocity fallback data should contribute average speed."""
    mins, speeds = collect_values(
        [
            {
                "metrics": {"min_distance": 0.3, "avg_speed": float("nan")},
                "trajectory": {"robot_vel": [[3.0, 4.0], [0.0, 0.0]]},
            }
        ]
    )

    assert mins == [0.3]
    assert speeds == [2.5]


def test_training_metric_bootstrap_helpers_cover_empty_and_populated_inputs() -> None:
    """Training summary helpers should provide deterministic bootstrap summaries."""
    empty = bootstrap_metric_confidence([], n_samples=5, seed=1)
    assert empty == {"mean": 0.0, "median": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    populated = bootstrap_metric_confidence([1.0, 2.0, 3.0], n_samples=20, seed=1)
    assert populated["mean"] == pytest.approx(2.0)
    assert populated["median"] == pytest.approx(2.0)
    assert populated["ci_low"] <= populated["ci_high"]

    aggregate = aggregate_training_metrics_with_bootstrap(
        [
            {"metrics": {"success_rate": 1.0}},
            {"metrics": {"success_rate": 0.0}},
            {"metrics": {"success_rate": "not-numeric"}},
        ],
        ["metrics.success_rate", "metrics.missing"],
        n_samples=10,
        seed=2,
    )
    assert aggregate["metrics.success_rate"]["mean"] == pytest.approx(0.5)
    assert aggregate["metrics.missing"] == empty


def test_summary_cli_reports_malformed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark-facing summary command should fail closed with diagnostics."""
    from robot_sf.benchmark import cli

    src = tmp_path / "episodes_bad.jsonl"
    src.write_text('{"metrics":{"min_distance":0.4}}\n{bad json\n', encoding="utf-8")
    errors: list[str] = []

    def _record_error(message: str, *args: object, **_kwargs: object) -> None:
        errors.append(message % args)

    monkeypatch.setattr(cli.logging, "error", _record_error)

    exit_code = cli.cli_main(["summary", "--in", str(src), "--out-dir", str(tmp_path / "figs")])

    assert exit_code == 2
    assert errors
    assert "malformed_lines=1" in errors[0]
    assert f"{src}:2" in errors[0]
