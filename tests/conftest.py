"""Minimal pytest configuration: headless graphics + slow test timing.

Rewritten to purge legacy pytest_sessionfinish hook with invalid signature.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from robot_sf.common.artifact_paths import ensure_canonical_tree

try:
    from tests.perf_utils.policy import PerformanceBudgetPolicy
    from tests.perf_utils.reporting import SlowTestSample, format_report, generate_report
except Exception:  # pragma: no cover - perf utils optional in some contexts
    PerformanceBudgetPolicy = None  # type: ignore[assignment]
    SlowTestSample = None  # type: ignore[assignment]
    format_report = None  # type: ignore[assignment]
    generate_report = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Generator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def _import_torch_optional():
    """TODO docstring. Document this function."""
    try:
        return importlib.import_module("torch")  # type: ignore
    except Exception:  # pragma: no cover - torch optional in some envs
        return None


def _snapshot_torch_determinism(torch_module):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
    """
    state: dict[str, object | None] = {
        "algos": None,
        "cudnn_backend": None,
        "cudnn_det": None,
        "cudnn_bench": None,
    }
    if hasattr(torch_module, "are_deterministic_algorithms_enabled"):
        try:
            state["algos"] = bool(torch_module.are_deterministic_algorithms_enabled())
        except Exception:  # pragma: no cover - defensive capture
            state["algos"] = None
    cudnn_backend = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    state["cudnn_backend"] = cudnn_backend
    if cudnn_backend is not None:
        state["cudnn_det"] = getattr(cudnn_backend, "deterministic", None)
        state["cudnn_bench"] = getattr(cudnn_backend, "benchmark", None)
    return state


def _apply_nondeterministic(torch_module, cudnn_backend):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
        cudnn_backend: TODO docstring.
    """
    try:
        if hasattr(torch_module, "use_deterministic_algorithms"):
            torch_module.use_deterministic_algorithms(False)
        if cudnn_backend is not None:
            cudnn_backend.deterministic = False  # type: ignore[attr-defined]
            cudnn_backend.benchmark = True  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort guard
        pass


def _restore_torch_determinism(torch_module, state):
    """TODO docstring. Document this function.

    Args:
        torch_module: TODO docstring.
        state: TODO docstring.
    """
    try:
        prev_algos = state.get("algos")
        cudnn_backend = state.get("cudnn_backend")
        if prev_algos is not None and hasattr(torch_module, "use_deterministic_algorithms"):
            torch_module.use_deterministic_algorithms(prev_algos)
        if cudnn_backend is not None:
            prev_det = state.get("cudnn_det")
            prev_bench = state.get("cudnn_bench")
            if prev_det is not None:
                cudnn_backend.deterministic = prev_det  # type: ignore[attr-defined]
            if prev_bench is not None:
                cudnn_backend.benchmark = prev_bench  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best effort restore
        pass


@pytest.fixture(scope="session", autouse=True)
def headless_pygame_environment() -> Generator[None, None, None]:
    """Force pygame/matplotlib to run headlessly for the duration of the test session."""
    originals: dict[str, str | None] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT"),
    }
    os.environ.update(
        {
            "DISPLAY": "",
            "SDL_VIDEODRIVER": "dummy",
            "MPLBACKEND": "Agg",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        },
    )
    yield
    for k, v in originals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session", autouse=True)
def reroute_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Override ROBOT_SF_ARTIFACT_ROOT so tests keep the repo tree clean.

    Args:
        tmp_path_factory: Pytest factory used to create a persistent temp directory.
    """
    original = os.environ.get("ROBOT_SF_ARTIFACT_ROOT")
    override_dir = tmp_path_factory.mktemp("robot_sf_artifacts")
    os.environ["ROBOT_SF_ARTIFACT_ROOT"] = str(override_dir)
    ensure_canonical_tree(override_dir)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("ROBOT_SF_ARTIFACT_ROOT", None)
        else:
            os.environ["ROBOT_SF_ARTIFACT_ROOT"] = original


@pytest.fixture(scope="session", autouse=True)
def torch_nondeterministic_guard():  # type: ignore[missing-return-type-doc]
    """Ensure torch deterministic algorithms aren't forced across the suite."""

    torch_module = _import_torch_optional()
    if torch_module is None:
        yield
        return

    state = _snapshot_torch_determinism(torch_module)
    _apply_nondeterministic(torch_module, state.get("cudnn_backend"))

    try:
        yield
    finally:
        _restore_torch_determinism(torch_module, state)


@pytest.fixture(scope="session")
def perf_policy():  # type: ignore[missing-return-type-doc]
    """TODO docstring. Document this function."""
    if PerformanceBudgetPolicy is not None:
        try:
            return PerformanceBudgetPolicy()
        except Exception:  # pragma: no cover
            pass

    class _Fallback:  # pragma: no cover - only used when perf utils missing
        """TODO docstring. Document this class."""

        soft_threshold_seconds = 20.0
        hard_timeout_seconds = 60.0
        report_count = 10
        relax_env_var = "ROBOT_SF_PERF_RELAX"

        def classify(self, duration_seconds: float):
            """TODO docstring. Document this function.

            Args:
                duration_seconds: TODO docstring.
            """
            if duration_seconds >= self.hard_timeout_seconds:
                return "hard"
            if duration_seconds >= self.soft_threshold_seconds:
                return "soft"
            return "none"

    return _Fallback()


_SLOW_SAMPLES: list[tuple[str, float]] = []


_FAST_PATH_FRAGMENTS = (
    "tests/common/",
    "tests/contract/",
    "tests/factories/",
    "tests/guard/",
    "tests/sensor/",
    "tests/sim/",
    "tests/unit/",
)
_FAST_FILE_PREFIXES = (
    "test_action_adapters",
    "test_config_validation",
    "test_environment_factory_signatures",
    "test_error_policy",
    "test_planner",
    "test_range_sensor",
    "test_seed_utils",
    "test_types",
)
_FAST_FILES = {
    "map_test.py",
    "navigation_test.py",
    "ped_grouping_test.py",
    "sim_config_test.py",
    "unicycle_drive_test.py",
    "zone_sampling_test.py",
}
_SLOW_FILE_OVERRIDES = {
    "test_edge_cases_recording.py",
    "test_runner_video.py",
}


def _should_auto_mark_slow(path_str: str) -> bool:
    """Return True when a test path should be auto-marked as slow."""
    normalized = path_str.replace("\\", "/")
    filename = Path(normalized).name
    if filename in _SLOW_FILE_OVERRIDES:
        return True
    if filename in _FAST_FILES:
        return False
    if any(fragment in normalized for fragment in _FAST_PATH_FRAGMENTS):
        return False
    if any(filename.startswith(prefix) for prefix in _FAST_FILE_PREFIXES):
        return False
    return True


def pytest_collection_modifyitems(config, items):  # type: ignore[missing-type-doc]
    """Auto-mark non-core tests as slow to keep fast unit runs small."""
    del config
    for item in items:
        path_str = str(item.fspath)
        if _should_auto_mark_slow(path_str):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[missing-type-doc]
    """TODO docstring. Document this function.

    Args:
        item: TODO docstring.
    """
    start = time.perf_counter()
    try:
        outcome = yield
    finally:
        # Always record duration, even if the test raised/failed (ensures slow report completeness).
        _SLOW_SAMPLES.append((item.nodeid, time.perf_counter() - start))
    return outcome


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[missing-type-doc]
    """Emit slow test report and optionally enforce soft breaches.

    Enforcement logic (feature 124):
      * If ROBOT_SF_PERF_ENFORCE=1 and any soft breach occurs, convert to failure.
      * Hard breaches are expected to be handled by individual test timeouts / assertions.
    """
    del exitstatus, config
    if not _SLOW_SAMPLES:
        return
    if PerformanceBudgetPolicy is None or SlowTestSample is None:
        return
    policy = PerformanceBudgetPolicy()
    # Optional environment overrides (not part of public API; used for enforcement test)
    try:
        soft_override = os.environ.get("ROBOT_SF_PERF_SOFT")
        hard_override = os.environ.get("ROBOT_SF_PERF_HARD")
        if soft_override:
            policy.soft_threshold_seconds = float(soft_override)  # type: ignore[misc]
        if hard_override:
            policy.hard_timeout_seconds = float(hard_override)  # type: ignore[misc]
    except Exception:  # pragma: no cover
        pass
    samples = [SlowTestSample(test_identifier=n, duration_seconds=d) for n, d in _SLOW_SAMPLES]
    records = generate_report(samples, policy)
    if not records:
        return
    relax = os.environ.get(policy.relax_env_var) == "1"
    enforce = os.environ.get(getattr(policy, "enforce_env_var", "ROBOT_SF_PERF_ENFORCE")) == "1"
    terminalreporter.write_line(
        "\n"
        + format_report(records, policy)
        + ("\n(relax mode active)" if relax else "")
        + ("\n(enforce mode)" if enforce else "")
        + "\n",
    )
    if enforce and not relax:
        # Treat any soft or hard breach as failure. Hard breaches ideally already handled
        # by per-test timeout markers, but we enforce here for determinism in minimal runs.
        if any(r.breach_type in {"soft", "hard"} for r in records):
            msg = "Performance breach (soft/hard) detected under enforce mode"
            terminalreporter.write_line(msg)
            # Use pytest's exit API to set return code cleanly (avoids raw SystemExit trace noise)
            pytest.exit(msg, returncode=pytest.ExitCode.TESTS_FAILED)


# Coverage-specific fixtures for test isolation


@pytest.fixture
def sample_coverage_data():
    """
    Provide sample coverage data for testing coverage tools.

    Returns a minimal but complete coverage.json structure
    for unit testing formatters, analyzers, and comparators.
    """
    return {
        "meta": {
            "version": "7.0.0",
            "timestamp": "2025-10-23T12:00:00",
            "branch_coverage": False,
            "show_contexts": False,
        },
        "files": {
            "robot_sf/gym_env/environment.py": {
                "executed_lines": [1, 2, 3, 10, 11, 12, 20],
                "summary": {
                    "covered_lines": 7,
                    "num_statements": 10,
                    "percent_covered": 70.0,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [4, 5, 6],
            },
            "robot_sf/sim/simulator.py": {
                "executed_lines": [1, 2, 3, 4, 5],
                "summary": {
                    "covered_lines": 5,
                    "num_statements": 8,
                    "percent_covered": 62.5,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [10, 11, 12],
            },
        },
        "totals": {
            "covered_lines": 12,
            "num_statements": 18,
            "percent_covered": 66.67,
            "missing_lines": 6,
            "excluded_lines": 0,
        },
    }


@pytest.fixture
def sample_gap_data():
    """Provide sample gap analysis data for testing."""
    return {
        "gaps": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "coverage_percent": 70.0,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add integration tests for reset() method",
            },
            {
                "file": "robot_sf/sim/simulator.py",
                "coverage_percent": 62.5,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add unit tests for step() method",
            },
        ],
        "summary": {
            "total_gaps": 2,
            "total_uncovered_lines": 6,
            "average_coverage": 66.25,
        },
    }


@pytest.fixture
def sample_trend_data():
    """Provide sample trend analysis data for testing."""
    return {
        "direction": "improving",
        "rate_per_week": 0.5,
        "current_coverage": 66.67,
        "oldest_coverage": 60.0,
        "snapshot_count": 10,
        "date_range": {
            "start": "2025-10-01",
            "end": "2025-10-23",
        },
    }


@pytest.fixture
def sample_baseline_data():
    """Provide sample baseline comparison data for testing."""
    return {
        "current_coverage": 66.67,
        "baseline_coverage": 70.0,
        "delta": -3.33,
        "threshold": 1.0,
        "changed_files": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "current": 70.0,
                "baseline": 75.0,
                "delta": -5.0,
            },
        ],
    }


# ============================================================================
# Occupancy Grid Fixtures (from conftest_occupancy.py)
# ============================================================================


@pytest.fixture
def simple_grid_config():
    """Basic 10x10m grid with 0.1m resolution (100x100 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def large_grid_config():
    """Larger 20x20m grid with 0.1m resolution (200x200 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=20.0,
        height=20.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.ROBOT],
    )


@pytest.fixture
def coarse_grid_config():
    """Coarse 10x10m grid with 0.5m resolution (20x20 cells)."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.5,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
    )


@pytest.fixture
def single_channel_config():
    """Grid with only obstacles channel."""
    from robot_sf.nav.occupancy_grid import GridChannel, GridConfig

    return GridConfig(
        resolution=0.1,
        width=10.0,
        height=10.0,
        channels=[GridChannel.OBSTACLES],
    )


@pytest.fixture
def occupancy_grid(simple_grid_config):
    """Instantiated OccupancyGrid with simple config."""
    from robot_sf.nav.occupancy_grid import OccupancyGrid

    return OccupancyGrid(config=simple_grid_config)


@pytest.fixture
def robot_pose_center():
    """Robot at center of a 10x10m grid (world frame origin)."""
    return ((5.0, 5.0), 0.0)


@pytest.fixture
def robot_pose_corner():
    """Robot at corner of grid."""
    return ((1.0, 1.0), 0.0)


@pytest.fixture
def robot_pose_rotated():
    """Robot at center with 45° rotation."""
    import numpy as np

    return ((5.0, 5.0), np.pi / 4)


@pytest.fixture
def simple_obstacles():
    """Simple obstacle layout: two horizontal walls."""
    return [
        ((1.0, 3.0), (9.0, 3.0)),  # Horizontal wall at Y=3
        ((1.0, 7.0), (9.0, 7.0)),  # Horizontal wall at Y=7
    ]


@pytest.fixture
def complex_obstacles():
    """More complex obstacle layout: rectangular room with interior walls."""
    return [
        # Outer walls
        ((0.5, 0.5), (9.5, 0.5)),  # Bottom
        ((0.5, 9.5), (9.5, 9.5)),  # Top
        ((0.5, 0.5), (0.5, 9.5)),  # Left
        ((9.5, 0.5), (9.5, 9.5)),  # Right
        # Interior walls
        ((3.0, 2.0), (3.0, 8.0)),  # Vertical divider
        ((7.0, 2.0), (7.0, 8.0)),  # Vertical divider
    ]


@pytest.fixture
def simple_pedestrians():
    """Simple pedestrian layout: two pedestrians."""
    return [
        ((3.0, 5.0), 0.3),  # Pedestrian at (3, 5)
        ((7.0, 5.0), 0.3),  # Pedestrian at (7, 5)
    ]


@pytest.fixture
def crowded_pedestrians():
    """Crowded layout: 5 pedestrians in middle of grid."""
    return [
        ((4.5, 4.5), 0.3),
        ((5.5, 4.5), 0.3),
        ((5.0, 5.5), 0.3),
        ((4.5, 5.5), 0.3),
        ((5.5, 5.5), 0.3),
    ]


@pytest.fixture
def empty_pedestrians():
    """Empty pedestrian list."""
    return []


@pytest.fixture
def pre_generated_grid(occupancy_grid, simple_obstacles, simple_pedestrians, robot_pose_center):
    """Pre-generated grid with simple layout."""
    grid = occupancy_grid
    grid.generate(
        obstacles=simple_obstacles,
        pedestrians=simple_pedestrians,
        robot_pose=robot_pose_center,
        ego_frame=False,
    )
    return grid
