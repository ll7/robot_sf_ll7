"""Protect converted benchmark guards under Python optimized mode."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INTERNAL_ASSERT_COUNTS = {
    "robot_sf/benchmark/campaign_arm_admission.py": 1,
    "robot_sf/benchmark/campaign_atlas.py": 1,
    "robot_sf/benchmark/candidate_trace_resolution.py": 4,
    "robot_sf/benchmark/false_positive_injection_readiness.py": 1,
    "robot_sf/benchmark/figure_qa.py": 0,
    "robot_sf/benchmark/full_classic/encode.py": 1,
    "robot_sf/benchmark/long_horizon_route.py": 1,
    "robot_sf/benchmark/manifest_lineage_graph.py": 1,
    "robot_sf/benchmark/map_runner_native_command.py": 5,
    "robot_sf/benchmark/metrics.py": 2,
    "robot_sf/benchmark/predictive_checkpoint_schema_audit.py": 1,
    "robot_sf/benchmark/runner.py": 4,
    "robot_sf/benchmark/schemas/episode_schema.py": 1,
    "robot_sf/benchmark/schemas/forecast_batch_schema.py": 1,
    "robot_sf/benchmark/snqi/weights_inventory.py": 1,
    "robot_sf/benchmark/thresholds.py": 1,
}


@pytest.mark.parametrize(
    ("relative_path", "expected_count"),
    EXPECTED_INTERNAL_ASSERT_COUNTS.items(),
)
def test_benchmark_assert_triage_matches_classification(
    relative_path: str,
    expected_count: int,
) -> None:
    """Keep the reviewed split between internal assertions and explicit guards."""
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative_path)
    assert sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) == expected_count


def test_converted_guards_survive_python_optimized_mode(tmp_path: Path) -> None:
    """Exercise both converted guards with real ``-O``."""
    probe = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np

        from robot_sf.benchmark import figure_qa
        from robot_sf.benchmark.runner import _stack_or_zero


        def expect(exception_type, message, callback):
            try:
                callback()
            except exception_type as exc:
                if message not in str(exc):
                    raise AssertionError(f"missing {message!r} in {exc!r}") from exc
            else:
                raise AssertionError(f"{exception_type.__name__} was not raised")


        expect(
            ValueError,
            "empty_shape should have zero",
            lambda: _stack_or_zero([], stack_fn=np.stack, empty_shape=(1, 2)),
        )

        defect = figure_qa.FigureDefect("overlap", "error", "seeded defect")
        figure_qa.lint_figure = lambda *_args, **_kwargs: [defect]
        expect(
            AssertionError,
            "Figure has 1 defect(s)",
            lambda: figure_qa.assert_clean(object()),
        )

        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["DISPLAY"] = ""
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    result = subprocess.run(
        [sys.executable, "-O", "-c", probe, str(tmp_path / "optimized-mode.mp4")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
