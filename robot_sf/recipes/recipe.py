"""Recipe data model, loader, and validation.

A recipe manifest is a small YAML file living under ``configs/recipes/``. It
describes a blessed, user-facing workflow that delegates to EXISTING configs
and scripts. The recipe layer intentionally adds no simulation or training
logic: it only resolves paths, validates that the referenced inputs exist and
are loadable, and dispatches ``run`` to the declared command.

Manifest schema (all fields required unless noted):

```yaml
id: ppo-smoke                  # unique, kebab-case, matches filename
title: PPO baseline smoke test  # one-line human label
purpose: >                      # 1-3 sentence plain-language summary
  Exercise the PPO training pipeline on CPU with a tiny smoke config so a
  new contributor can confirm the training stack is wired up correctly.
runtime: "< 2 min CPU"          # honest wall-clock estimate
category: training              # grouping for `recipe list` (see CATEGORIES)
command: >                      # shell command executed by `recipe run`.
  uv run python scripts/training/train_ppo.py
  --config configs/training/ppo/issue_4017_unconstrained_smoke.yaml --dry-run
configs:                        # repo-relative files the recipe depends on
  - configs/training/ppo/issue_4017_unconstrained_smoke.yaml
outputs:                        # artifacts produced (informational; may not exist)
  - output/recipes/ppo-smoke/
docs: docs/recipes/README.md    # optional, repo-relative doc anchor
```

The ``configs`` list is the loadability contract: :func:`load_recipe` resolves
each entry against the repository root and :func:`Recipe.config_paths` returns
the absolute paths. The catalog test asserts every referenced config exists and
parses as YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence

#: Recipe categories used to group ``recipe list`` output. Order is presentation order.
CATEGORIES: tuple[str, ...] = (
    "getting-started",
    "maps",
    "planners",
    "training",
    "benchmark",
    "telemetry",
    "visualization",
)


class RecipeError(ValueError):
    """Raised when a recipe manifest is missing fields or fails validation."""


def _require_str(value: object, field_name: str, recipe_id: str) -> str:
    """Return a non-empty stripped string field or raise a RecipeError."""
    if not isinstance(value, str):
        raise RecipeError(
            f"recipe {recipe_id!r}: field {field_name!r} must be a non-empty string",
        )
    text = value.strip()
    if not text:
        raise RecipeError(
            f"recipe {recipe_id!r}: field {field_name!r} must be a non-empty string",
        )
    return text


@dataclass(frozen=True)
class Recipe:
    """A validated recipe manifest.

    Attributes mirror the YAML schema documented at the top of this module.
    ``config_paths`` and ``docs_path`` are resolved absolute paths (or ``None``
    for an unset ``docs`` field).
    """

    id: str
    title: str
    purpose: str
    runtime: str
    category: str
    command: str
    config_paths: tuple[Path, ...]
    outputs: tuple[str, ...] = ()
    docs_path: Path | None = None
    #: Repo-relative config strings exactly as written in the manifest.
    config_refs: tuple[str, ...] = field(default=(), repr=False)
    #: Repo-relative docs string exactly as written in the manifest.
    docs_ref: str | None = None

    def explain_text(self) -> str:
        """Return the multi-line ``recipe explain`` text for this recipe."""
        lines = [
            f"# {self.id}",
            f"title:    {self.title}",
            f"category: {self.category}",
            f"runtime:  {self.runtime}",
            "",
            "purpose:",
            _indent(self.purpose),
            "",
            "command:",
            _indent(self.command),
            "",
            "configs (loadability contract):",
        ]
        if self.config_refs:
            lines.extend(f"  - {ref}" for ref in self.config_refs)
        else:
            lines.append("  (none -- this recipe uses bundled defaults)")
        if self.outputs:
            lines.append("")
            lines.append("outputs:")
            lines.extend(f"  - {item}" for item in self.outputs)
        if self.docs_ref:
            lines.append("")
            lines.append(f"docs: {self.docs_ref}")
        return "\n".join(lines) + "\n"


def _indent(text: str, prefix: str = "  ") -> str:
    """Indent every line of ``text`` with ``prefix``.

    Returns:
        str: The indented text (empty lines left blank).
    """
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _resolve_repo_relative(ref: str, recipe_id: str, field_name: str) -> Path:
    """Resolve a repo-relative path reference against the repository root.

    Leading ``./`` is tolerated. The path is returned absolute but its
    existence is NOT checked here; callers (validators/tests) decide whether
    existence is required.

    Returns:
        Path: Absolute resolved path.
    """
    cleaned = ref.strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise RecipeError(
            f"recipe {recipe_id!r}: field {field_name!r} contains an empty path",
        )
    if Path(cleaned).is_absolute():
        raise RecipeError(f"recipe {recipe_id!r}: field {field_name!r} must be repository-relative")
    root = get_repository_root().resolve()
    resolved = (root / cleaned).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecipeError(
            f"recipe {recipe_id!r}: field {field_name!r} escapes repository root: {ref!r}"
        ) from exc
    return resolved


def _require_string_list(
    data: dict[str, object],
    field_name: str,
    recipe_id: str,
) -> list[str]:
    """Validate and return a list field of non-empty strings.

    Args:
        data: The parsed manifest mapping.
        field_name: The manifest key holding the list.
        recipe_id: Recipe id, used in error messages.

    Returns:
        list[str]: Stripped, non-empty string entries.

    Raises:
        RecipeError: If the field is missing, not a list, or contains bad entries.
    """
    raw = data.get(field_name, [])
    if not isinstance(raw, list):
        raise RecipeError(f"recipe {recipe_id!r}: field {field_name!r} must be a list")
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise RecipeError(
                f"recipe {recipe_id!r}: every {field_name!r} entry must be a non-empty string",
            )
        out.append(entry.strip())
    return out


def _validate_id(recipe_id: str, path: Path) -> None:
    """Ensure the recipe id matches its filename and is kebab-case.

    Args:
        recipe_id: The id parsed from the manifest.
        path: The manifest file path (``path.stem`` is the expected id).

    Raises:
        RecipeError: If the id is malformed or mismatches the filename.
    """
    if recipe_id != path.stem:
        raise RecipeError(
            f"recipe file {path.name}: id {recipe_id!r} must match filename {path.stem!r}",
        )
    if not all(ch.isalnum() or ch in "-_" for ch in recipe_id):
        raise RecipeError(
            f"recipe {recipe_id!r}: id must be kebab-case (alnum, '-', '_')",
        )


def _parse_yaml_manifest(path: Path) -> dict[str, object]:
    """Read and parse a recipe manifest, returning the top-level mapping.

    Args:
        path: Absolute path to a recipe YAML file.

    Returns:
        dict[str, object]: The parsed manifest.

    Raises:
        RecipeError: If the file is missing, invalid YAML, or not a mapping.
    """
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"recipe file {path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RecipeError(f"recipe file {path}: top level must be a mapping")
    return data


def load_recipe_file(path: Path) -> Recipe:
    """Load and validate a single recipe manifest from ``path``.

    Args:
        path: Absolute path to a recipe YAML file.

    Returns:
        Recipe: The validated recipe.

    Raises:
        RecipeError: If the manifest is malformed or fails schema validation.
    """
    path = Path(path).resolve()
    data = _parse_yaml_manifest(path)

    recipe_id = _require_str(data.get("id"), "id", path.stem)
    _validate_id(recipe_id, path)

    title = _require_str(data.get("title"), "title", recipe_id)
    purpose = _require_str(data.get("purpose"), "purpose", recipe_id)
    runtime = _require_str(data.get("runtime"), "runtime", recipe_id)
    category = _require_str(data.get("category"), "category", recipe_id)
    if category not in CATEGORIES:
        raise RecipeError(
            f"recipe {recipe_id!r}: category {category!r} not in {CATEGORIES}",
        )
    command = _require_str(data.get("command"), "command", recipe_id)

    config_refs = _require_string_list(data, "configs", recipe_id)
    config_paths = tuple(_resolve_repo_relative(ref, recipe_id, "configs") for ref in config_refs)

    outputs = _require_string_list(data, "outputs", recipe_id)

    docs_ref_raw = data.get("docs")
    docs_ref: str | None = None
    docs_path: Path | None = None
    if docs_ref_raw is not None:
        docs_ref = _require_str(docs_ref_raw, "docs", recipe_id)
        docs_path = _resolve_repo_relative(docs_ref, recipe_id, "docs")

    recipe = Recipe(
        id=recipe_id,
        title=title,
        purpose=purpose,
        runtime=runtime,
        category=category,
        command=command,
        config_paths=config_paths,
        outputs=tuple(outputs),
        docs_path=docs_path,
        config_refs=tuple(config_refs),
        docs_ref=docs_ref,
    )
    config_errors = validate_configs_loadable(recipe.config_paths)
    if config_errors:
        raise RecipeError(f"recipe {recipe_id!r}: " + "; ".join(config_errors))
    if recipe.docs_path is not None and not recipe.docs_path.is_file():
        raise RecipeError(f"recipe {recipe_id!r}: missing docs: {recipe.docs_path}")
    return recipe


def validate_configs_loadable(config_paths: Sequence[Path]) -> list[str]:
    """Return a list of human-readable errors for configs that fail to load.

    A "config" is considered loadable if it exists and parses as YAML. Binary
    or non-YAML referenced files (e.g. SVG maps) are checked for existence only;
    YAML files (``.yaml``/``.yml``) are parsed to catch corrupt configs early.

    Args:
        config_paths: Absolute paths to referenced recipe inputs.

    Returns:
        list[str]: Empty if all loadable; otherwise one message per failure.
    """
    errors: list[str] = []
    for path in config_paths:
        path = Path(path)
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                with path.open(encoding="utf-8") as handle:
                    yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                errors.append(f"unloadable YAML {path}: {exc}")
    return errors
