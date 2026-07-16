"""Recipe catalog discovery.

Recipes live as individual YAML manifests under ``configs/recipes/``. This
module discovers them, deduplicates by id, and provides lookup helpers.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.recipes.recipe import Recipe, RecipeError, load_recipe_file

#: Directory holding the recipe manifests, relative to the repository root.
RECIPES_SUBDIR = Path("configs/recipes")


def recipes_dir() -> Path:
    """Return the absolute path to the recipe catalog directory."""
    return get_repository_root() / RECIPES_SUBDIR


def discover_recipes() -> list[Recipe]:
    """Discover and validate every recipe manifest in the catalog.

    Returns:
        list[Recipe]: Recipes sorted by category order then id. The list is
        never empty for a healthy catalog; an empty catalog raises
        :class:`RecipeError` because it signals a broken install.

    Raises:
        RecipeError: If the catalog directory is missing, a manifest is
            invalid, or two recipes share an id.
    """
    catalog = recipes_dir()
    if not catalog.is_dir():
        raise RecipeError(f"recipe catalog directory not found: {catalog}")
    manifests = sorted(catalog.glob("*.yaml"))
    if not manifests:
        raise RecipeError(f"recipe catalog is empty: {catalog}")

    by_id: dict[str, Recipe] = {}
    for manifest in manifests:
        recipe = load_recipe_file(manifest)
        if recipe.id in by_id:
            raise RecipeError(
                f"duplicate recipe id {recipe.id!r}: {manifest} and "
                f"{by_id[recipe.id].id}",  # pragma: no cover - defensive msg
            )
        by_id[recipe.id] = recipe

    from robot_sf.recipes.recipe import CATEGORIES  # noqa: PLC0415 - avoid cycle at import

    category_index = {name: i for i, name in enumerate(CATEGORIES)}

    def _sort_key(recipe: Recipe) -> tuple[int, str]:
        return (category_index.get(recipe.category, len(CATEGORIES)), recipe.id)

    return sorted(by_id.values(), key=_sort_key)


def load_recipe(recipe_id: str) -> Recipe:
    """Load a single recipe by id.

    Args:
        recipe_id: The recipe id (must match a ``configs/recipes/<id>.yaml``).

    Returns:
        Recipe: The validated recipe.

    Raises:
        RecipeError: If the recipe id is unknown.
    """
    matches = [recipe for recipe in discover_recipes() if recipe.id == recipe_id]
    if not matches:
        raise RecipeError(
            f"unknown recipe id {recipe_id!r}; run `robot-sf recipe list` to see options",
        )
    return matches[0]
