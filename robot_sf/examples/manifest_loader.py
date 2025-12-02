"""Load and validate the `examples/examples_manifest.yaml` configuration.

The examples manifest is the single source of truth for organizing example
scripts into tiered categories, powering both documentation generation and CI
smoke tests. This module provides a small typed layer for parsing the YAML file
into rich Python objects and performing core validation checks that other tools
build upon.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

__all__ = [
    "ExampleCategory",
    "ExampleManifest",
    "ExampleScript",
    "ManifestValidationError",
    "load_manifest",
]


class ManifestValidationError(ValueError):
    """Raised when the manifest data fails structural or semantic validation."""


@dataclass(frozen=True, slots=True)
class ExampleCategory:
    """Represents a tier of examples such as quickstart or advanced."""

    slug: str
    title: str
    description: str
    order: int
    ci_default: bool = True

    def __post_init__(self) -> None:
        """Post init.

        Returns:
            None: Auto-generated placeholder description.
        """
        if not self.slug:
            raise ManifestValidationError("Category slug cannot be empty.")
        if "/" in self.slug or "\\" in self.slug:
            raise ManifestValidationError(
                f"Category slug '{self.slug}' must not contain path separators."
            )
        if self.slug.strip() != self.slug:
            raise ManifestValidationError(
                f"Category slug '{self.slug}' must not contain leading or trailing whitespace."
            )
        if not isinstance(self.order, int):
            raise ManifestValidationError(
                f"Category '{self.slug}' must define an integer order index."
            )
        if not isinstance(self.ci_default, bool):
            raise ManifestValidationError(
                f"Category '{self.slug}' must declare ci_default as a boolean."
            )


@dataclass(frozen=True, slots=True)
class ExampleScript:
    """Describes a single example Python entry point."""

    path: PurePosixPath
    name: str
    summary: str
    category_slug: str
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    ci_enabled: bool = True
    ci_reason: str | None = None
    doc_reference: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Post init.

        Returns:
            None: Auto-generated placeholder description.
        """
        normalized_path = PurePosixPath(str(self.path))
        if normalized_path.is_absolute():
            raise ManifestValidationError(
                f"Example path '{normalized_path}' must be relative to the examples directory."
            )
        if any(part == ".." for part in normalized_path.parts):
            raise ManifestValidationError(
                f"Example path '{normalized_path}' cannot traverse out of the examples directory."
            )
        if normalized_path.suffix != ".py":
            raise ManifestValidationError(
                f"Example path '{normalized_path}' must point to a Python file."
            )
        object.__setattr__(self, "path", normalized_path)

        if not self.name:
            raise ManifestValidationError("Example name cannot be empty.")
        if not self.summary:
            raise ManifestValidationError(
                f"Example '{normalized_path}' must include a non-empty summary."
            )
        if not self.category_slug:
            raise ManifestValidationError(
                f"Example '{normalized_path}' must reference a category slug."
            )

        prerequisites = _validate_string_sequence(
            self.prerequisites, "prerequisites", normalized_path
        )
        object.__setattr__(self, "prerequisites", prerequisites)

        if not isinstance(self.ci_enabled, bool):
            raise ManifestValidationError(
                f"Example '{normalized_path}' must define ci_enabled as a boolean."
            )
        if not self.ci_enabled and not self.ci_reason:
            raise ManifestValidationError(
                f"Example '{normalized_path}' is disabled for CI but missing ci_reason."
            )
        tags = _validate_string_sequence(self.tags, "tags", normalized_path)
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True, slots=True)
class ExampleManifest:
    """Container object holding parsed manifest data."""

    version: str
    categories: tuple[ExampleCategory, ...]
    examples: tuple[ExampleScript, ...]
    manifest_path: Path
    examples_root: Path
    _categories_by_slug: dict[str, ExampleCategory] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post init.

        Returns:
            None: Auto-generated placeholder description.
        """
        if not self.version:
            raise ManifestValidationError("Manifest version string cannot be empty.")
        if not self.manifest_path.is_file():
            raise ManifestValidationError(
                f"Manifest path '{self.manifest_path}' does not point to a file."
            )
        if not self.examples_root.is_dir():
            raise ManifestValidationError(
                f"Examples root '{self.examples_root}' is not a directory."
            )

        categories_by_slug: dict[str, ExampleCategory] = {}
        for category in self.categories:
            if category.slug in categories_by_slug:
                raise ManifestValidationError(
                    f"Duplicate category slug '{category.slug}' detected in manifest."
                )
            categories_by_slug[category.slug] = category
        object.__setattr__(
            self,
            "categories",
            tuple(sorted(self.categories, key=lambda category: category.order)),
        )
        object.__setattr__(self, "_categories_by_slug", categories_by_slug)

        seen_paths: set[PurePosixPath] = set()
        normalized_examples: list[ExampleScript] = []
        for example in self.examples:
            if example.category_slug not in categories_by_slug:
                raise ManifestValidationError(
                    f"Example '{example.path}' references unknown category '{example.category_slug}'."
                )
            if example.path in seen_paths:
                raise ManifestValidationError(
                    f"Duplicate example path '{example.path}' detected in manifest."
                )
            seen_paths.add(example.path)
            normalized_examples.append(example)
        object.__setattr__(self, "examples", tuple(normalized_examples))

    @property
    def categories_by_slug(self) -> Mapping[str, ExampleCategory]:
        """Read-only mapping of category slug to category definition."""

        return self._categories_by_slug

    def examples_for_category(self, slug: str) -> tuple[ExampleScript, ...]:
        """Return all examples that belong to the provided category slug."""

        return tuple(example for example in self.examples if example.category_slug == slug)

    def iter_ci_enabled_examples(self) -> Iterator[ExampleScript]:
        """Yield each example that should execute in CI."""

        for example in self.examples:
            if example.ci_enabled:
                yield example

    def resolve_example_path(self, example: ExampleScript) -> Path:
        """Return the absolute filesystem path for the given example."""

        return (self.examples_root / Path(example.path)).resolve(strict=False)

    def validate_paths(self) -> None:
        """Ensure that every example path exists relative to the manifest directory."""

        missing: list[str] = []
        for example in self.examples:
            candidate = self.examples_root / Path(example.path)
            if not candidate.is_file():
                missing.append(example.path.as_posix())
        if missing:
            joined = ", ".join(sorted(missing))
            raise ManifestValidationError(f"Examples missing from filesystem: {joined}.")


def load_manifest(
    manifest_path: str | Path | None = None,
    *,
    validate_paths: bool = False,
) -> ExampleManifest:
    """Parse the examples manifest into typed data classes.

    Parameters
    ----------
    manifest_path:
        Optional path to the manifest file. When omitted, the loader resolves the
        canonical repository location (``examples/examples_manifest.yaml``).
    validate_paths:
        When ``True``, enforce that every example path exists relative to the
        manifest directory. Validation errors raise :class:`ManifestValidationError`.

    Returns
    -------
    ExampleManifest
        Parsed manifest data ready for downstream tooling.
    """

    resolved_manifest = _resolve_manifest_path(manifest_path)
    raw_data = _read_manifest_yaml(resolved_manifest)
    categories = _parse_categories(raw_data.get("categories", []))
    examples = _parse_examples(raw_data.get("examples", []), categories)

    manifest = ExampleManifest(
        version=_expect_string(raw_data.get("version"), "version"),
        categories=tuple(categories.values()),
        examples=tuple(examples),
        manifest_path=resolved_manifest,
        examples_root=resolved_manifest.parent,
    )

    if validate_paths:
        manifest.validate_paths()

    return manifest


def _resolve_manifest_path(manifest_path: str | Path | None) -> Path:
    """Resolve manifest path.

    Args:
        manifest_path: Auto-generated placeholder description.

    Returns:
        Path: Auto-generated placeholder description.
    """
    if manifest_path is not None:
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise ManifestValidationError(f"Manifest file not found at '{path}'.")
        return path

    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "examples" / "examples_manifest.yaml"
    if not default_path.is_file():
        raise ManifestValidationError(f"Default manifest file not found at '{default_path}'.")
    return default_path


def _read_manifest_yaml(path: Path) -> Mapping[str, Any]:
    """Read manifest yaml.

    Args:
        path: Auto-generated placeholder description.

    Returns:
        Mapping[str, Any]: Auto-generated placeholder description.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ManifestValidationError("Manifest root must be a mapping of keys to values.")
    return data


def _parse_categories(raw_categories: Any) -> dict[str, ExampleCategory]:
    """Parse categories.

    Args:
        raw_categories: Auto-generated placeholder description.

    Returns:
        dict[str, ExampleCategory]: Auto-generated placeholder description.
    """
    if not isinstance(raw_categories, Sequence):
        raise ManifestValidationError("Manifest 'categories' must be a list.")

    categories: dict[str, ExampleCategory] = {}
    for index, item in enumerate(raw_categories):
        if not isinstance(item, Mapping):
            raise ManifestValidationError(
                f"Category entry at index {index} must be a mapping, got {type(item)!r}."
            )
        category = ExampleCategory(
            slug=_expect_string(item.get("slug"), "categories.slug"),
            title=_expect_string(item.get("title"), f"categories[{index}].title"),
            description=_expect_string(item.get("description"), f"categories[{index}].description"),
            order=_expect_int(item.get("order"), f"categories[{index}].order"),
            ci_default=_expect_bool(
                item.get("ci_default", True), f"categories[{index}].ci_default"
            ),
        )
        if category.slug in categories:
            raise ManifestValidationError(
                f"Duplicate category slug '{category.slug}' detected in manifest."
            )
        categories[category.slug] = category
    return categories


def _parse_examples(
    raw_examples: Any,
    categories: Mapping[str, ExampleCategory],
) -> tuple[ExampleScript, ...]:
    """Parse examples.

    Args:
        raw_examples: Auto-generated placeholder description.
        categories: Auto-generated placeholder description.

    Returns:
        tuple[ExampleScript, ...]: Auto-generated placeholder description.
    """
    if not isinstance(raw_examples, Sequence):
        raise ManifestValidationError("Manifest 'examples' must be a list.")

    parsed: list[ExampleScript] = []
    for index, item in enumerate(raw_examples):
        if not isinstance(item, Mapping):
            raise ManifestValidationError(
                f"Example entry at index {index} must be a mapping, got {type(item)!r}."
            )
        category_slug = _expect_string(
            item.get("category_slug"), f"examples[{index}].category_slug"
        )
        category = categories.get(category_slug)
        if category is None:
            raise ManifestValidationError(
                f"Example at index {index} references unknown category '{category_slug}'."
            )

        ci_enabled = item.get("ci_enabled", category.ci_default)
        if not isinstance(ci_enabled, bool):
            raise ManifestValidationError(
                f"Example '{item.get('path')}' must declare ci_enabled as boolean if provided."
            )

        example = ExampleScript(
            path=PurePosixPath(_expect_string(item.get("path"), f"examples[{index}].path")),
            name=_expect_string(item.get("name"), f"examples[{index}].name"),
            summary=_expect_string(item.get("summary"), f"examples[{index}].summary"),
            category_slug=category_slug,
            prerequisites=_optional_string_sequence(
                item.get("prerequisites"),
                f"examples[{index}].prerequisites",
            ),
            ci_enabled=ci_enabled,
            ci_reason=_optional_string(item.get("ci_reason")),
            doc_reference=_optional_string(item.get("doc_reference")),
            tags=_optional_string_sequence(item.get("tags"), f"examples[{index}].tags"),
        )
        parsed.append(example)
    return tuple(parsed)


def _expect_string(value: Any, field_name: str) -> str:
    """Expect string.

    Args:
        value: Auto-generated placeholder description.
        field_name: Auto-generated placeholder description.

    Returns:
        str: Auto-generated placeholder description.
    """
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"Field '{field_name}' must be a non-empty string.")
    return value


def _expect_int(value: Any, field_name: str) -> int:
    """Expect int.

    Args:
        value: Auto-generated placeholder description.
        field_name: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    if not isinstance(value, int):
        raise ManifestValidationError(f"Field '{field_name}' must be an integer.")
    return value


def _expect_bool(value: Any, field_name: str) -> bool:
    """Expect bool.

    Args:
        value: Auto-generated placeholder description.
        field_name: Auto-generated placeholder description.

    Returns:
        bool: Auto-generated placeholder description.
    """
    if not isinstance(value, bool):
        raise ManifestValidationError(f"Field '{field_name}' must be a boolean.")
    return value


def _optional_string(value: Any) -> str | None:
    """Optional string.

    Args:
        value: Auto-generated placeholder description.

    Returns:
        str | None: Auto-generated placeholder description.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestValidationError("Optional string fields must be a string when provided.")
    stripped = value.strip()
    return stripped or None


def _optional_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    """Optional string sequence.

    Args:
        value: Auto-generated placeholder description.
        field_name: Auto-generated placeholder description.

    Returns:
        tuple[str, ...]: Auto-generated placeholder description.
    """
    if value is None:
        return ()
    return _validate_string_sequence(value, field_name)


def _validate_string_sequence(
    value: Any,
    field_name: str,
    context: PurePosixPath | None = None,
) -> tuple[str, ...]:
    """Validate string sequence.

    Args:
        value: Auto-generated placeholder description.
        field_name: Auto-generated placeholder description.
        context: Auto-generated placeholder description.

    Returns:
        tuple[str, ...]: Auto-generated placeholder description.
    """
    if isinstance(value, str):
        raise ManifestValidationError(
            f"Field '{field_name}' must be a list of strings, not a single string."
        )
    if value is None:
        return ()
    if not isinstance(value, Sequence):
        raise ManifestValidationError(
            f"Field '{field_name}' must be a sequence of strings, got {type(value)!r}."
        )

    items: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            location = f" for example '{context}'" if context else ""
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must contain only non-empty strings."
            )
        items.append(raw.strip())
    return tuple(items)
