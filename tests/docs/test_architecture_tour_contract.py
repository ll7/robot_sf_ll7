"""Contract tests for the newcomer architecture tour (issue #8730)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
TOUR = REPO_ROOT / "docs" / "architecture_tour.md"
ENTRY_POINTS = (
    REPO_ROOT / "docs" / "index.rst",
    REPO_ROOT / "docs" / "user-guide.md",
    REPO_ROOT / "CONTRIBUTING.md",
)

REQUIRED_SOURCES = (
    "robot_sf/api.py",
    "robot_sf/gym_env/environment_factory.py",
    "robot_sf/gym_env/robot_env.py",
    "robot_sf/gym_env/base_env.py",
    "robot_sf/sim/simulator.py",
    "robot_sf/baselines/interface.py",
    "robot_sf/planner/protocol.py",
    "robot_sf/benchmark/types.py",
    "robot_sf/benchmark/runner.py",
    "robot_sf/evidence/writers.py",
    "robot_sf/render/",
    "robot_sf/training/",
    "robot_sf_carla_bridge/",
    "docs/external_data_setup.md",
    "context/issue_928_carla_t0_t1_replay_contract.md",
    "configs/scenarios",
    "maps",
    "fast-pysf",
)


def _links(text: str) -> list[str]:
    """Return relative Markdown link destinations."""

    return re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)


def test_tour_has_stable_sections_and_source_boundaries() -> None:
    """The tour names the complete flow and keeps its claim boundary explicit."""

    text = TOUR.read_text(encoding="utf-8")
    for heading in (
        "# Robot SF Architecture Tour",
        "## One episode, end to end",
        "## Package map",
        "## Where common changes belong",
    ):
        assert heading in text
    for source in REQUIRED_SOURCES:
        assert source in text, source
    for phrase in (
        "not a benchmark claim",
        "planner quality or benchmark evidence",
        "Missing assets must remain unavailable",
        "robot_sf.api.run_episode(env, planner=...)",
        "benchmark.runner.run_episode(scenario_params, seed)",
        "Visualization and rendering",
        "CARLA integration",
        "not a CARLA fallback",
    ):
        assert phrase in text

    assert text.index("robot_sf.api.run_episode(env, planner=...") < text.index(
        "env.reset(seed=..."
    )
    assert text.index("env.reset(seed=...") < text.index("env.step(action)")
    assert text.index("benchmark.runner.run_episode(scenario_params, seed)") < text.index(
        "benchmark.runner.validate_and_write(...)"
    )


def test_tour_links_resolve_to_repository_paths() -> None:
    """Every local tour link resolves without escaping the repository."""

    text = TOUR.read_text(encoding="utf-8")
    for destination in _links(text):
        parsed = urlsplit(unquote(destination.strip().strip("<>")))
        if parsed.scheme or parsed.netloc:
            continue
        target = (TOUR.parent / parsed.path).resolve() if parsed.path else TOUR
        assert REPO_ROOT in target.parents or target == REPO_ROOT, destination
        assert target.exists(), destination


def test_entry_points_link_to_tour() -> None:
    """The requested newcomer entry points expose the architecture tour."""

    for path in ENTRY_POINTS:
        text = path.read_text(encoding="utf-8")
        expected = "architecture_tour" if path.suffix == ".rst" else "architecture_tour.md"
        assert expected in text, path
