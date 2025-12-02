from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.common.metrics_utils import metric_samples
from robot_sf.training.imitation_analysis import _generate_figures


def test_generate_figures_creates_all_expected_files(tmp_path: Path):
    baseline_metrics = {
        "timesteps_to_convergence": 1000.0,
        "timesteps_samples": [900.0, 1000.0, 1100.0],
        "success_rate": 0.6,
        "success_rate_samples": [0.55, 0.6, 0.65],
        "collision_rate": 0.2,
        "collision_rate_samples": [0.25, 0.2, 0.15],
        "snqi": 0.5,
        "snqi_samples": [0.45, 0.5, 0.55],
    }
    pretrained_metrics = {
        "timesteps_to_convergence": 600.0,
        "timesteps_samples": [580.0, 600.0, 620.0],
        "success_rate": 0.8,
        "success_rate_samples": [0.75, 0.8, 0.85],
        "collision_rate": 0.1,
        "collision_rate_samples": [0.12, 0.1, 0.08],
        "snqi": 0.7,
        "snqi_samples": [0.68, 0.7, 0.72],
    }

    paths = _generate_figures(
        baseline_metrics=baseline_metrics,
        pretrained_metrics=pretrained_metrics,
        output_dir=tmp_path,
    )

    expected = {
        "fig-sample-efficiency",
        "fig-success_rate-distribution",
        "fig-collision_rate-distribution",
        "fig-snqi-distribution",
        "fig-improvement-summary",
    }
    assert set(paths.keys()) == expected
    for name in expected:
        assert (tmp_path / f"{name}.png").exists()


def test_metric_samples_handles_suffix_and_filtering():
    payload = {
        "metrics": {
            "foo_samples": [1, 2.0, "bad", None],
            "bar": [3, "oops"],
        }
    }
    assert metric_samples(payload, "foo") == [1.0, 2.0]
    # falls back to base key when *_samples missing
    assert metric_samples(payload, "bar") == [3.0]
    # missing metrics yields empty list
    assert metric_samples({}, "baz") == []
