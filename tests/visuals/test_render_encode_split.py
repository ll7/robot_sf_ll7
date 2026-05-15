"""Test for render vs encode timing split (T040A).

This test ensures that when a successful SimulationView encode occurs (or synthetic
fallback), the performance manifest includes the new keys with expected types.

We don't guarantee non-null render time unless SimulationView path succeeded;
so the test focuses on schema presence and numeric consistency when available.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from tests.perf_utils.minimal_matrix import write_minimal_matrix


class _Cfg:  # minimal inline config mirroring other tests
    """Minimal benchmark config for render/encode timing tests."""

    def __init__(self, tmp_path: Path, renderer: str = "synthetic"):
        """Create a one-episode benchmark config rooted at ``tmp_path``.

        Args:
            tmp_path: Directory used for the benchmark matrix and output artifacts.
            renderer: Video renderer mode to request from the benchmark pipeline.
        """
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.output_root = str(tmp_path)
        self.scenario_matrix_path = str(write_minimal_matrix(tmp_path))
        self.initial_episodes = 1
        self.max_episodes = 1
        self.batch_size = 1
        self.algo = "ppo"
        self.workers = 1
        self.master_seed = 123
        self.smoke = False
        self.disable_videos = False
        self.max_videos = 1
        self.capture_replay = True
        self.video_renderer = renderer
        # required targets
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read_json(path: Path):
    """Read a UTF-8 JSON file.

    Args:
        path: JSON file path.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def test_render_encode_split_tests_do_not_use_unconditional_skip_markers():
    """Guard this file against permanent placeholder skips in the default suite."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    skipped_tests: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "skip"
                and isinstance(decorator.func.value, ast.Attribute)
                and decorator.func.value.attr == "mark"
                and isinstance(decorator.func.value.value, ast.Name)
                and decorator.func.value.value.id == "pytest"
            ):
                skipped_tests.append(node.name)
    assert skipped_tests == []


def test_performance_manifest_has_render_encode_keys(tmp_path):
    """Synthetic rendering writes render/encode timing keys in the performance manifest.

    Args:
        tmp_path: Pytest temporary directory for generated benchmark artifacts.
    """
    cfg = _Cfg(tmp_path)
    run_full_benchmark(cfg)
    perf_path = Path(cfg.output_root) / "reports" / "performance_visuals.json"
    assert perf_path.exists(), "performance_visuals.json missing"
    data = _read_json(perf_path)
    # Presence
    assert "first_video_render_time_s" in data, data.keys()
    assert "first_video_encode_time_s" in data, data.keys()
    # Types (null allowed)
    rtime = data["first_video_render_time_s"]
    etime = data["first_video_encode_time_s"]
    assert (rtime is None) or isinstance(rtime, int | float), rtime
    assert (etime is None) or isinstance(etime, int | float), etime
    # If render time present, encode time must also be present and non-null
    if rtime is not None:
        assert etime is not None and etime >= 0, (rtime, etime)
