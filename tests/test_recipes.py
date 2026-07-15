"""Tests for the curated recipe catalog (``robot-sf recipe``; issue #5795).

These tests enforce the issue's acceptance criteria:

- ``recipe list`` / ``recipe run`` / ``recipe explain`` all work.
- Every shipped recipe id resolves to a real, loadable config.
- The 10 named blessed workflows from issue #5795 are present.
"""

from __future__ import annotations

import pytest

from robot_sf.cli import main
from robot_sf.recipes import discover_recipes, load_recipe
from robot_sf.recipes.catalog import recipes_dir
from robot_sf.recipes.cli import build_subparser, handle
from robot_sf.recipes.cli import main as recipes_main
from robot_sf.recipes.recipe import CATEGORIES, RecipeError, validate_configs_loadable

# The ten blessed workflows named in issue #5795.
EXPECTED_RECIPE_IDS: frozenset[str] = frozenset(
    {
        "first-demo",
        "custom-svg-map",
        "orca-smoke",
        "ppo-smoke",
        "planner-comparison",
        "benchmark-mini-run",
        "telemetry-headless-demo",
        "trace-viewer-demo",
        "map-validation",
        "scenario-thumbnail-generation",
    }
)


def test_catalog_directory_exists() -> None:
    """The recipe catalog lives under configs/recipes/."""
    catalog = recipes_dir()
    assert catalog.is_dir(), f"recipe catalog missing: {catalog}"
    assert catalog.name == "recipes"


def test_every_named_recipe_is_present() -> None:
    """All ten workflows named in issue #5795 ship in the catalog."""
    recipes = discover_recipes()
    ids = {recipe.id for recipe in recipes}
    assert EXPECTED_RECIPE_IDS <= ids, f"missing recipes: {EXPECTED_RECIPE_IDS - ids}"


def test_recipe_ids_match_filenames() -> None:
    """Each recipe id equals its filename stem so `run <id>` is unambiguous."""
    for manifest in sorted(recipes_dir().glob("*.yaml")):
        recipe = load_recipe(manifest.stem)
        assert recipe.id == manifest.stem


def test_every_recipe_resolves_to_real_loadable_config() -> None:
    """Acceptance criterion: every recipe id resolves to a real, loadable config.

    For recipes with no YAML config (e.g. quickstart examples) the referenced
    script/asset file must at least exist.
    """
    for recipe in discover_recipes():
        errors = validate_configs_loadable(recipe.config_paths)
        assert not errors, f"recipe {recipe.id!r} has unloadable configs: {errors}"
        assert recipe.command.strip(), f"recipe {recipe.id!r} has an empty command"


def test_every_recipe_category_is_known() -> None:
    """Every recipe must use a category from the canonical CATEGORIES tuple."""
    for recipe in discover_recipes():
        assert recipe.category in CATEGORIES, (
            f"recipe {recipe.id!r} has unknown category {recipe.category!r}"
        )


def test_recipe_list_outputs_all_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`robot-sf recipe list` prints every recipe id and a runtime estimate."""
    rc = main(["recipe", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Robot SF recipe catalog" in out
    for recipe_id in EXPECTED_RECIPE_IDS:
        assert recipe_id in out, f"`recipe list` missing id {recipe_id!r}"
    assert "RUNTIME" in out


def test_recipe_explain_prints_purpose_config_runtime_outputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`recipe explain <id>` shows purpose, config, runtime, and outputs."""
    rc = main(["recipe", "explain", "ppo-smoke"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# ppo-smoke" in out
    assert "purpose:" in out
    assert "runtime:" in out
    assert "command:" in out
    assert "configs (loadability contract):" in out
    # The config this recipe maps to must be named in the explanation.
    assert "issue_4017_unconstrained_smoke.yaml" in out


def test_recipe_explain_unknown_id_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown recipe id exits non-zero with a helpful message."""
    rc = main(["recipe", "explain", "does-not-exist"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "does-not-exist" in err


def test_recipe_run_dry_run_prints_command_without_executing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`recipe run <id> --dry-run` prints the command but does not execute it."""
    rc = main(["recipe", "run", "first-demo", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "first-demo" in out
    assert "examples/quickstart/01_basic_robot.py" in out
    assert "dry-run: command not executed" in out


def test_standalone_recipes_cli_list(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The recipe module is also usable on its own via its own entry point."""
    rc = recipes_main(["list"])
    assert rc == 0
    assert "Robot SF recipe catalog" in capsys.readouterr().out


def test_load_recipe_unknown_raises() -> None:
    """Loading an unknown id raises a clear RecipeError."""
    with pytest.raises(RecipeError, match="unknown recipe id"):
        load_recipe("not-a-recipe")


def test_build_subparser_registers_list_run_explain() -> None:
    """The recipe subparser exposes the three required verbs."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_subparser(sub)
    # Parsing each verb must succeed and record the recipe_cmd.
    for verb, extra in [("list", []), ("explain", ["ppo-smoke"]), ("run", ["ppo-smoke"])]:
        ns = parser.parse_args(["recipe", verb, *extra])
        assert ns.cmd == "recipe"
        assert ns.recipe_cmd == verb


def test_handle_dispatches_subcommands() -> None:
    """The handle() dispatcher routes to list/explain/run correctly."""
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_subparser(sub)
    ns = parser.parse_args(["recipe", "run", "ppo-smoke", "--dry-run"])
    # Dry-run must return 0 and not touch the filesystem.
    assert handle(ns) == 0


def test_recipe_with_no_configs_uses_bundled_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recipe may declare no config refs (bundled-default workflow)."""
    # first-demo references only the example script; explain still works.
    rc = main(["recipe", "explain", "first-demo"])
    assert rc == 0
    assert "01_basic_robot.py" in capsys.readouterr().out


def test_recipes_count_is_at_least_ten() -> None:
    """Issue #5795 asks for 8-12 blessed workflows; we ship at least 10."""
    recipes = discover_recipes()
    assert len(recipes) >= 10
