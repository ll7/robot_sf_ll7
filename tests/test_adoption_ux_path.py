"""Keep the user-facing adoption path aligned with shipped UX surfaces."""

from pathlib import Path

import pytest

from robot_sf.cli import _build_parser

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADOPTION_PATH = REPOSITORY_ROOT / "docs" / "adoption_path.md"


@pytest.fixture(scope="module")
def adoption_text() -> str:
    """Read the canonical adoption path once for the contract tests."""
    return ADOPTION_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "command",
    (
        "uv run robot-sf doctor",
        "uv run robot-sf demo",
        "uv run robot-sf examples list",
        "uv run robot-sf recipe list",
        "uv run robot-sf gallery build",
    ),
)
def test_adoption_path_names_each_product_layer_command(adoption_text: str, command: str) -> None:
    """The documented install-to-gallery path must not silently lose a layer."""
    assert command in adoption_text


def test_adoption_path_claim_boundary_is_explicit(adoption_text: str) -> None:
    """Local UX output must not be presented as benchmark evidence."""
    assert "not benchmark evidence" in adoption_text
    assert "not a training result" in adoption_text
    assert "not measured per-scenario capabilities" in adoption_text


def test_adoption_path_artifact_locations_are_documented(adoption_text: str) -> None:
    """The visible demo and gallery outputs have stable, documented locations."""
    for artifact in (
        "output/demo/latest",
        "episode.jsonl",
        "summary.json",
        "viewer/index.html",
        "thumbnail.png",
        "output/gallery/index.html",
    ):
        assert artifact in adoption_text


def test_adoption_path_commands_are_registered_by_the_umbrella_cli() -> None:
    """The route names in the guide must remain available in the top-level parser."""
    parser = _build_parser()
    subparser_action = next(action for action in parser._actions if action.dest == "cmd")
    assert {
        "doctor",
        "demo",
        "examples",
        "recipe",
        "gallery",
    } <= set(subparser_action.choices)
