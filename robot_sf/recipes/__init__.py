"""Curated recipe catalog for the ``robot-sf recipe`` UX layer.

A *recipe* is a thin, blessed workflow that points at EXISTING configs and
scripts and hides path complexity for new users. Recipes never add new
simulation or training logic; they only orchestrate commands that already live
in the repository.

Public surface:

- :class:`Recipe` -- parsed and validated recipe manifest.
- :func:`load_recipe` -- load one recipe by id.
- :func:`discover_recipes` -- enumerate the catalog under ``configs/recipes``.
- :func:`resolve_recipe_config_paths` -- expand every referenced config file.

See :mod:`robot_sf.recipes.catalog` for discovery and
:mod:`robot_sf.recipes.recipe` for the data model.
"""

from __future__ import annotations

from robot_sf.recipes.catalog import discover_recipes, load_recipe, recipes_dir
from robot_sf.recipes.recipe import Recipe, RecipeError, load_recipe_file

__all__ = [
    "Recipe",
    "RecipeError",
    "discover_recipes",
    "load_recipe",
    "load_recipe_file",
    "recipes_dir",
]
