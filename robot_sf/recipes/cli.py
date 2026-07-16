"""``robot-sf recipe`` subcommand: list / run / explain.

This is the user-facing UX layer described in issue #5795. It discovers the
recipe catalog under ``configs/recipes/`` and dispatches the three verbs:

- ``recipe list`` -- table of every blessed workflow, grouped by category.
- ``recipe explain <id>`` -- purpose, mapped config, runtime, and outputs.
- ``recipe run <id>`` -- execute the recipe's declared command.

The ``run`` verb shells out to the command stored in the manifest. It performs
no simulation itself: every recipe delegates to an EXISTING config/script.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.recipes.catalog import discover_recipes, load_recipe
from robot_sf.recipes.recipe import CATEGORIES, Recipe, RecipeError

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence
    from pathlib import Path


def _list_recipes() -> int:
    """Print every recipe grouped by category.

    Returns:
        int: Process-style exit code (0 on success, 1 on catalog error).
    """
    try:
        recipes = discover_recipes()
    except RecipeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    grouped: dict[str, list[Recipe]] = {name: [] for name in CATEGORIES}
    for recipe in recipes:
        grouped.setdefault(recipe.category, []).append(recipe)

    sys.stdout.write("Robot SF recipe catalog\n\n")
    sys.stdout.write(
        "Run a recipe with: uv run robot-sf recipe run <id>\n"
        "Explain a recipe with: uv run robot-sf recipe explain <id>\n\n",
    )

    header = f"{'ID':<22} {'TITLE':<40} {'RUNTIME':<16}"
    for category in CATEGORIES:
        rows = grouped.get(category, [])
        if not rows:
            continue
        sys.stdout.write(f"## {category}\n\n")
        sys.stdout.write(header + "\n")
        sys.stdout.write("-" * len(header) + "\n")
        for recipe in rows:
            title = recipe.title if len(recipe.title) <= 40 else recipe.title[:37] + "..."
            sys.stdout.write(f"{recipe.id:<22} {title:<40} {recipe.runtime:<16}\n")
        sys.stdout.write("\n")

    # Defensive: surface any recipe whose category slipped past validation.
    unknown = [r for r in recipes if r.category not in category_index]
    if unknown:  # pragma: no cover - load_recipe_file rejects these already
        sys.stdout.write("## other\n\n")
        for recipe in unknown:
            sys.stdout.write(f"{recipe.id:<22} {recipe.title}\n")
    return 0


def _explain_recipe(recipe_id: str) -> int:
    """Print the detailed explanation for one recipe.

    Returns:
        int: Process-style exit code (0 on success, 1 if the id is unknown).
    """
    try:
        recipe = load_recipe(recipe_id)
    except RecipeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(recipe.explain_text())
    return 0


def _run_recipe(recipe_id: str, *, dry_run: bool, cwd: Path | None = None) -> int:
    """Execute the recipe's command via the shell.

    Args:
        recipe_id: Recipe to run.
        dry_run: If True, print the command without executing it.
        cwd: Working directory for execution (defaults to repo root).

    Returns:
        int: The executed command's exit code, or 1 if the id is unknown / dry-run.
    """
    try:
        recipe = load_recipe(recipe_id)
    except RecipeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    command = recipe.command.strip()
    sys.stdout.write(f"recipe: {recipe.id} -- {recipe.title}\n")
    sys.stdout.write(f"runtime: {recipe.runtime}\n")
    sys.stdout.write(f"command: {command}\n")
    if dry_run:
        sys.stdout.write("(dry-run: command not executed)\n")
        return 0

    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.flush()
    argv = shlex.split(command)
    if not argv:
        sys.stderr.write(f"error: recipe {recipe.id!r} has an empty command\n")
        return 1
    completed = subprocess.run(
        argv,
        cwd=str(cwd or get_repository_root()),
        check=False,
    )
    return completed.returncode


def _add_recipe_verbs(subparsers) -> None:  # type: ignore[no-untyped-def]
    """Register the list/explain/run verbs on a subparsers object.

    Shared by :func:`build_subparser` (nested under ``robot-sf recipe``) and the
    standalone :func:`main` so the verb surface stays identical.
    """

    subparsers.add_parser("list", help="List every recipe grouped by category.")

    explain = subparsers.add_parser("explain", help="Show purpose, config, runtime, outputs.")
    explain.add_argument("id", help="Recipe id (see `robot-sf recipe list`).")

    run = subparsers.add_parser("run", help="Run a recipe's declared command.")
    run.add_argument("id", help="Recipe id (see `robot-sf recipe list`).")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would run without executing it.",
    )


def build_subparser(subparsers):  # type: ignore[no-untyped-def]
    """Register the ``recipe`` subcommand on an argparse subparsers object.

    Returns:
        argparse.ArgumentParser: The created ``recipe`` subparser.
    """
    import argparse  # noqa: PLC0415 - stdlib, avoids import-time cost

    recipe_parser = subparsers.add_parser(
        "recipe",
        help="Curated recipe catalog (uv run robot-sf recipe list|run|explain)",
        description=(
            "Browse and run blessed Robot SF workflows. Recipes point at existing "
            "configs and hide path complexity. No new simulation logic."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    recipe_sub = recipe_parser.add_subparsers(dest="recipe_cmd", required=True)
    _add_recipe_verbs(recipe_sub)
    return recipe_parser


def handle(args) -> int:  # type: ignore[no-untyped-def]
    """Dispatch a parsed ``recipe`` namespace to list/run/explain.

    Returns:
        int: Process-style exit code from the selected verb.
    """
    if args.recipe_cmd == "list":
        return _list_recipes()
    if args.recipe_cmd == "explain":
        return _explain_recipe(args.id)
    if args.recipe_cmd == "run":
        return _run_recipe(args.id, dry_run=bool(args.dry_run))
    sys.stderr.write(f"error: unknown recipe subcommand: {args.recipe_cmd}\n")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point for direct invocation (``python -m`` style).

    The primary path is the top-level ``robot-sf recipe`` subcommand wired in
    :mod:`robot_sf.cli`; this entry point mirrors that dispatch so the recipe
    module is also usable on its own, e.g. ``python -m robot_sf.recipes.cli list``.

    Returns:
        int: Process-style exit code from the selected verb.
    """
    import argparse  # noqa: PLC0415 - stdlib

    parser = argparse.ArgumentParser(
        prog="robot-sf recipe",
        description="Curated Robot SF recipe catalog.",
    )
    sub = parser.add_subparsers(dest="recipe_cmd", required=True)
    _add_recipe_verbs(sub)
    args = parser.parse_args(argv)
    return handle(args)
