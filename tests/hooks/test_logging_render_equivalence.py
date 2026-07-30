"""Behavioral test: migrated loguru calls render identical text before/after.

Issue #6468 migrated f-string ``logger.*()`` calls in the hot-path modules
(``robot_sf/sim/simulator.py``, ``robot_sf/gym_env/base_env.py``,
``robot_sf/gym_env/pedestrian_env.py``) to structured ``{key}`` + kwargs style.
For each migrated site this test proves the rendered log message text is
identical to the original f-string output, by capturing the actual
``record["message"]`` produced by a real loguru sink.

The task contract named "7 calls" using a single-line grep; AST inspection
found 2 additional multi-line concatenated f-string calls in pedestrian_env.py
(same allowed file), which were also migrated (9 sites total). One additional
site in robot_env.py is out of this PR's allowed-paths contract and was
grandfathered instead.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_PATH = _REPO_ROOT / "hooks" / "no_fstring_logger.py"
_SPEC = importlib.util.spec_from_file_location("no_fstring_logger", _HOOK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HOOK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HOOK)
find_violations = _HOOK.find_fstring_logger_violations


@dataclass(frozen=True)
class MigratedSite:
    """One migrated logger call: its new format string, sample values, and f-string ground truth."""

    id: str
    file: str
    level: str
    fmt: str
    values: dict[str, Any]
    before: Callable[..., str]


# Ground-truth "before" text is built with a real Python f-string (CPython
# rendering engine). The "after" text goes through loguru's str.format rendering
# via a capturing sink. Equality proves the two engines agree for these sites.
MIGRATED_SITES = [
    MigratedSite(
        id="sim-proximity-point",
        file="robot_sf/sim/simulator.py",
        level="warning",
        fmt="Could not find a valid proximity point: {point}.",
        values={"point": (1.5, 2.5)},
        before=lambda point: f"Could not find a valid proximity point: {point}.",
    ),
    MigratedSite(
        id="ped-reward-function",
        file="robot_sf/gym_env/pedestrian_env.py",
        level="info",
        fmt="Using reward function: {name}",
        values={"name": "simple_ped_reward"},
        before=lambda name: f"Using reward function: {name}",
    ),
    MigratedSite(
        id="ped-set-action-space",
        file="robot_sf/gym_env/pedestrian_env.py",
        level="warning",
        fmt="Failed to set robot model action space: {exc}",
        values={"exc": ValueError("boom")},
        before=lambda exc: f"Failed to set robot model action space: {exc}",
    ),
    MigratedSite(
        id="ped-predict-failed",
        file="robot_sf/gym_env/pedestrian_env.py",
        level="warning",
        fmt="Robot model predict failed ({exc}); using null action.",
        values={"exc": RuntimeError("predict failed")},
        before=lambda exc: f"Robot model predict failed ({exc}); using null action.",
    ),
    MigratedSite(
        id="ped-recording-saved",
        file="robot_sf/gym_env/pedestrian_env.py",
        level="info",
        fmt="Recording saved to {target_path}",
        values={"target_path": Path("/tmp/episode.pkl")},
        before=lambda target_path: f"Recording saved to {target_path}",
    ),
    MigratedSite(
        # The two action-space-shape sites (lines ~256 and ~275) share this message;
        # sample tuple shapes prove the implicitly-concatenated f-string renders
        # identically to the single structured format string.
        id="ped-action-space-shape",
        file="robot_sf/gym_env/pedestrian_env.py",
        level="warning",
        fmt=(
            "Robot model action space shape {model_shape} does not match env shape "
            "{env_shape}. Falling back to null actions."
        ),
        values={"model_shape": (2,), "env_shape": (3,)},
        before=lambda model_shape, env_shape: (
            "Robot model action space shape "
            f"{model_shape} does not match env shape "
            f"{env_shape}. Falling back to null actions."
        ),
    ),
    MigratedSite(
        id="base-video-fps",
        file="robot_sf/gym_env/base_env.py",
        level="debug",
        fmt="Video FPS not provided, setting to {video_fps}",
        values={"video_fps": 1.0 / 0.1},
        before=lambda video_fps: f"Video FPS not provided, setting to {video_fps}",
    ),
    MigratedSite(
        id="base-recording-saved",
        file="robot_sf/gym_env/base_env.py",
        level="info",
        fmt="Recording saved to {target_path}",
        values={"target_path": Path("output/rec.pkl")},
        before=lambda target_path: f"Recording saved to {target_path}",
    ),
]


@pytest.fixture
def captured_messages():
    """Capture rendered loguru messages into a list for the duration of the test."""
    captured: list[str] = []

    def sink(message) -> None:
        captured.append(message.record["message"])

    sink_id = logger.add(sink, level="TRACE", format="{message}")
    try:
        yield captured
    finally:
        logger.remove(sink_id)
        captured.clear()


@pytest.mark.parametrize("site", MIGRATED_SITES, ids=lambda s: s.id)
def test_structured_call_renders_identical_to_fstring(
    site: MigratedSite, captured_messages: list[str]
) -> None:
    """The migrated structured call renders the exact text the f-string did."""
    expected = site.before(**site.values)
    captured_messages.clear()
    getattr(logger, site.level)(site.fmt, **site.values)
    assert captured_messages == [expected], (
        f"loguru structured render diverged from f-string for {site.file}\n"
        f"  expected: {expected!r}\n  got:      {captured_messages!r}"
    )


@pytest.mark.parametrize("site", MIGRATED_SITES, ids=lambda s: s.id)
def test_migrated_placeholder_present_in_source(site: MigratedSite) -> None:
    """Each new {key} placeholder is present in the migrated source file (anchors the table)."""
    source = (_REPO_ROOT / site.file).read_text(encoding="utf-8")
    for key in site.values:
        assert "{" + key + "}" in source, f"{site.file}: placeholder {{{key}}} not found in source"


def test_hot_path_files_have_no_fstring_logger_calls() -> None:
    """The actual committed hot-path files contain zero f-string logger calls."""
    hot_paths = [
        "robot_sf/sim/simulator.py",
        "robot_sf/gym_env/base_env.py",
        "robot_sf/gym_env/pedestrian_env.py",
    ]
    for rel in hot_paths:
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        violations = find_violations(source, rel)
        assert violations == [], f"{rel} still has f-string logger calls: {violations}"
