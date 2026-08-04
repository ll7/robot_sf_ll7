"""Protect converted benchmark guards under Python optimized mode."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark import metrics as metrics_mod
from robot_sf.benchmark.full_classic import encode as encode_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ASSERT_COUNTS = {
    "robot_sf/benchmark/campaign_arm_admission.py": 0,
    "robot_sf/benchmark/campaign_atlas.py": 0,
    "robot_sf/benchmark/candidate_trace_resolution.py": 0,
    "robot_sf/benchmark/false_positive_injection_readiness.py": 0,
    "robot_sf/benchmark/figure_qa.py": 0,
    "robot_sf/benchmark/full_classic/encode.py": 0,
    "robot_sf/benchmark/long_horizon_route.py": 0,
    "robot_sf/benchmark/manifest_lineage_graph.py": 0,
    "robot_sf/benchmark/map_runner_native_command.py": 0,
    "robot_sf/benchmark/metrics.py": 0,
    "robot_sf/benchmark/predictive_checkpoint_schema_audit.py": 0,
    "robot_sf/benchmark/runner.py": 0,
    "robot_sf/benchmark/schemas/episode_schema.py": 0,
    "robot_sf/benchmark/schemas/forecast_batch_schema.py": 0,
    "robot_sf/benchmark/snqi/weights_inventory.py": 0,
    "robot_sf/benchmark/thresholds.py": 0,
}


@pytest.mark.parametrize(
    ("relative_path", "expected_count"),
    EXPECTED_ASSERT_COUNTS.items(),
)
def test_benchmark_runtime_asserts_are_eliminated(
    relative_path: str,
    expected_count: int,
) -> None:
    """Keep targeted benchmark runtime guards active under Python optimized mode."""
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative_path)
    assert sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) == expected_count


def test_converted_guards_survive_python_optimized_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise both converted guards with real ``-O``."""
    probe = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np

        from robot_sf.benchmark import figure_qa
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
            ValueError,
            "Figure has 1 defect(s)",
            lambda: figure_qa.assert_clean(object()),
        )

        encode.moviepy_ready = lambda: True
        encode.ImageSequenceClip = object
        encode._iter_first = lambda _frames: (None, iter(()))
        encode._validate_first = lambda _first: (True, None)
        expect(
            TypeError,
            "successful frame validation produced no first frame",
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

    # Exercise the encode guard in-process as well, so changed-line coverage sees
    # the same failure path that the optimized subprocess proves remains active.
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", object())
    monkeypatch.setattr(encode_mod, "_iter_first", lambda _frames: (None, iter(())))
    monkeypatch.setattr(encode_mod, "_validate_first", lambda _first: (True, None))
    with pytest.raises(TypeError, match="successful frame validation produced no first frame"):
        encode_mod.encode_frames([], tmp_path / "optimized-mode-inprocess.mp4", sample_memory=False)

    # Exercise both metric guards in-process as well, so changed-line coverage uses
    # the same test selected for the optimized-mode regression contract.
    data = metrics_mod.EpisodeData(
        robot_pos=np.zeros((2, 2)),
        robot_vel=np.zeros((2, 2)),
        robot_acc=np.zeros((2, 2)),
        peds_pos=np.zeros((2, 0, 2)),
        ped_forces=np.zeros((2, 0, 2)),
        goal=np.array([1.0, 0.0]),
        dt=0.1,
        reached_goal_step=None,
    )
    monkeypatch.setattr(metrics_mod, "success_rate", lambda *_args, **_kwargs: 1.0)

    for time_metric in (metrics_mod.time_to_goal_norm, metrics_mod.time_to_goal_norm_success_only):
        with pytest.raises(RuntimeError, match="successful episode has no recorded goal step"):
            time_metric(data, horizon=10)


def test_encode_frames_reports_empty_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the user-visible failure result for an empty frame iterable."""
    monkeypatch.setattr(encode_mod, "moviepy_ready", lambda: True)
    monkeypatch.setattr(encode_mod, "ImageSequenceClip", object())

    result = encode_mod.encode_frames([], tmp_path / "empty.mp4", sample_memory=False)

    assert result.status == "failed"
    assert result.note == "no-frames"
