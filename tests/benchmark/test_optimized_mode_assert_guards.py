"""Protect benchmark invariants from disappearing under Python optimized mode."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARDENED_MODULES = (
    "robot_sf/benchmark/campaign_arm_admission.py",
    "robot_sf/benchmark/campaign_atlas.py",
    "robot_sf/benchmark/candidate_trace_resolution.py",
    "robot_sf/benchmark/false_positive_injection_readiness.py",
    "robot_sf/benchmark/figure_qa.py",
    "robot_sf/benchmark/full_classic/encode.py",
    "robot_sf/benchmark/long_horizon_route.py",
    "robot_sf/benchmark/manifest_lineage_graph.py",
    "robot_sf/benchmark/map_runner_native_command.py",
    "robot_sf/benchmark/metrics.py",
    "robot_sf/benchmark/predictive_checkpoint_schema_audit.py",
    "robot_sf/benchmark/runner.py",
    "robot_sf/benchmark/schemas/episode_schema.py",
    "robot_sf/benchmark/schemas/forecast_batch_schema.py",
    "robot_sf/benchmark/snqi/weights_inventory.py",
    "robot_sf/benchmark/thresholds.py",
)


@pytest.mark.parametrize("relative_path", HARDENED_MODULES)
def test_hardened_benchmark_module_has_no_runtime_asserts(relative_path: str) -> None:
    """Keep scoped runtime guards explicit so ``python -O`` cannot remove them."""
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative_path)
    assert not [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def test_representative_guards_survive_python_optimized_mode(tmp_path: Path) -> None:
    """Exercise public, internal, metric, and type-narrowing guards with real ``-O``."""
    probe = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np

        from robot_sf.benchmark import figure_qa, metrics
        from robot_sf.benchmark.full_classic import encode
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

        episode = metrics.EpisodeData(
            robot_pos=np.empty((0, 2)),
            robot_vel=np.empty((0, 2)),
            robot_acc=np.empty((0, 2)),
            peds_pos=np.empty((0, 0, 2)),
            ped_forces=np.empty((0, 0, 2)),
            goal=np.zeros(2),
            dt=0.1,
            reached_goal_step=None,
        )
        metrics.success_rate = lambda *_args, **_kwargs: 1.0
        for metric_fn in (
            metrics.time_to_goal_norm,
            metrics.time_to_goal_norm_success_only,
        ):
            expect(
                RuntimeError,
                "reached_goal_step must not be None",
                lambda metric_fn=metric_fn: metric_fn(episode, horizon=10),
            )

        encode.moviepy_ready = lambda: True
        encode.ImageSequenceClip = object
        encode._iter_first = lambda _frames: (None, iter(()))
        encode._validate_first = lambda _first: (True, None)
        encode._start_memory_sampler = lambda *_args: (lambda: None, [])
        expect(
            RuntimeError,
            "first frame must not be None",
            lambda: encode.encode_frames([], Path(sys.argv[1]), sample_memory=False),
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
