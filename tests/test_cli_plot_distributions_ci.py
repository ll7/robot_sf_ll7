"""TODO docstring. Document this module."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _write_matrix(path: Path, repeats: int = 4) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        repeats: TODO docstring.
    """
    scenarios = [
        {
            "id": "dist-ci",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        },
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def _write_distribution_records(path: Path) -> None:
    """Write enough varying values for the CI rendering path to execute."""
    records = [
        {
            "scenario_id": "dist-invalid-controls",
            "scenario_params": {"algo": "probe"},
            "metrics": {"collisions": float(value)},
        }
        for value in range(5)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


essential_markers = [
    # PNGs only; we just check existence of image files after CLI completes
]


@pytest.mark.parametrize(
    ("control_args", "error_fragment"),
    [
        (["--bins", "0"], "bins must be >= 1"),
        (["--ci", "--ci-samples", "0"], "ci_samples must be >= 1"),
        (["--ci", "--ci-confidence", "0"], "ci_confidence must be finite and in (0, 1)"),
        (["--ci", "--ci-confidence", "1"], "ci_confidence must be finite and in (0, 1)"),
        (["--ci", "--ci-confidence", "nan"], "ci_confidence must be finite and in (0, 1)"),
        (["--ci", "--ci-confidence", "inf"], "ci_confidence must be finite and in (0, 1)"),
    ],
    ids=[
        "bins-zero",
        "ci-samples-zero",
        "confidence-zero",
        "confidence-one",
        "confidence-nan",
        "confidence-inf",
    ],
)
def test_cli_plot_distributions_rejects_invalid_controls(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    control_args: list[str],
    error_fragment: str,
) -> None:
    """Invalid histogram/CI controls fail with exit 2 and a bounded error."""
    episodes = tmp_path / "episodes.jsonl"
    _write_distribution_records(episodes)

    caplog.set_level(logging.WARNING)
    rc = cli_main(
        [
            "plot-distributions",
            "--in",
            str(episodes),
            "--out-dir",
            str(tmp_path / "plots"),
            "--metrics",
            "collisions",
            *control_args,
        ],
    )

    assert rc == 2
    assert error_fragment in caplog.text


def test_cli_plot_distributions_ci(tmp_path: Path, capsys):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=4)
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
            "8",
            "--dt",
            "0.1",
        ],
    )
    capsys.readouterr()
    assert rc_run == 0

    out_dir = tmp_path / "plots"
    rc_plot = cli_main(
        [
            "plot-distributions",
            "--in",
            str(episodes),
            "--out-dir",
            str(out_dir),
            "--metrics",
            "collisions,comfort_exposure",
            "--bins",
            "10",
            "--kde",
            "--ci",
            "--ci-samples",
            "200",
            "--ci-confidence",
            "0.90",
            "--ci-seed",
            "123",
            "--out-pdf",
        ],
    )
    cap = capsys.readouterr()
    assert rc_plot == 0, f"plot-distributions --ci failed: {cap.err}"
    pngs = list(out_dir.glob("dist_*.png"))
    pdfs = list(out_dir.glob("dist_*.pdf"))
    assert pngs, "No PNGs generated with CI overlay"
    assert pdfs, "No PDFs generated with CI overlay"
