"""Load and validate the `examples/examples_manifest.yaml` configuration.

The examples manifest is the single source of truth for organizing example
scripts into tiered categories, powering both documentation generation and CI
smoke tests. This module provides a small typed layer for parsing the YAML file
into rich Python objects and performing core validation checks that other tools
build upon.

Canonical schema (``schema_version: 1``, backward-compatible)
-------------------------------------------------------------
Top-level keys: ``version`` (content version string, required), ``categories``
(required), ``examples`` (required), plus optional ``schema_version`` (integer,
defaults to 1 when absent) and ``exclusions`` (explicit versioned exclusions
with rationale for files on disk that are not runnable examples, such as
compatibility mirrors/shims and shared helper modules).

Field contract per example: ``path`` (relative ``.py``), ``name`` (unique),
``summary`` (non-empty, matches module docstring first line), ``category_slug``
(known category), ``prerequisites`` (normalized list; empty list ``[]`` when
none), ``tags`` (normalized lowercase ``[a-z0-9][a-z0-9_-]*``), ``expected_runtime``
(runtime class surface: ``None`` or normalized non-empty estimate, never a
sentinel), ``ci_enabled``/``ci_reason`` (consistent: disabled requires a reason,
enabled forbids one), ``doc_reference`` (``None`` or ``path[#anchor]``).

Prerequisite sentinels such as ``None``/``"None"``/``"null"``/``"n/a"``/``"-"``
(case-insensitive) normalize to ``[]``; malformed sentinels (a bare string
instead of a list, non-string entries, empty-string entries) are rejected.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from robot_sf.errors import RobotSfError

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PREREQUISITE_SENTINELS",
    "ExampleCategory",
    "ExampleExclusion",
    "ExampleManifest",
    "ExampleScript",
    "ManifestValidationError",
    "load_manifest",
]

#: Current canonical manifest schema version. Manifests without an explicit
#: ``schema_version`` key are treated as version 1 for backward compatibility.
MANIFEST_SCHEMA_VERSION = 1

#: Case-insensitive prerequisite sentinel spellings that normalize to ``[]``.
#: YAML ``null`` entries (``None``) inside a prerequisites list are also treated
#: as sentinels. A bare string (e.g. ``prerequisites: "None"``), non-string
#: entries, or empty-string entries are malformed and rejected.
PREREQUISITE_SENTINELS = frozenset({"none", "null", "nil", "n/a", "na", "-"})

#: Normalized tag surface: lowercase alphanumeric with ``_``/``-`` separators.
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")

#: Allowed exclusion kinds for versioned non-example files on disk.
EXCLUSION_KINDS = frozenset({"archive", "mirror", "helper"})

#: Top-level fields required by every manifest version supported by this loader.
REQUIRED_MANIFEST_FIELDS = frozenset({"version", "categories", "examples"})


class ManifestValidationError(RobotSfError, ValueError):
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
        """Validate the example category configuration.

        Ensures all category fields meet structural requirements:
        - Slug is non-empty and has no path separators
        - Slug has no leading/trailing whitespace
        - Order is an integer
        - ci_default is a boolean
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
    """Describes a single example Python entry point.

    ``expected_runtime`` is an optional, human-readable estimate of how long the
    example takes to run (e.g. ``"~10s"``, ``"interactive"``). It is surfaced by
    discovery tooling (``robot-sf examples list``) and is left unset when no
    trustworthy estimate is available rather than being fabricated.
    """

    path: PurePosixPath
    name: str
    summary: str
    category_slug: str
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    ci_enabled: bool = True
    ci_reason: str | None = None
    doc_reference: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    expected_runtime: str | None = None

    def __post_init__(self) -> None:
        """Validate the example category configuration.

        Ensures all category fields meet structural requirements:
        - Slug is non-empty and has no path separators
        - Slug has no leading/trailing whitespace
        - Order is an integer
        - ci_default is a boolean
        """
        normalized_path = _normalize_example_path(self.path)
        object.__setattr__(self, "path", normalized_path)
        _check_example_identity(self.name, self.summary, self.category_slug, normalized_path)

        prerequisites = _normalize_prerequisites(
            self.prerequisites, "prerequisites", normalized_path
        )
        object.__setattr__(self, "prerequisites", prerequisites)

        _check_ci_fields(self.ci_enabled, self.ci_reason, normalized_path)
        tags = _validate_tags(self.tags, "tags", normalized_path)
        object.__setattr__(self, "tags", tags)

        _validate_runtime_class(self.expected_runtime, normalized_path)


@dataclass(frozen=True, slots=True)
class ExampleExclusion:
    """Explicit versioned exclusion for a non-example Python file on disk.

    ``kind`` is one of ``archive`` (deprecated reference copy), ``mirror``
    (compatibility shim re-exporting a registered example), or ``helper``
    (shared non-runnable module imported by examples). ``reason`` records why
    the file is not registered as its own example.
    """

    path: PurePosixPath
    kind: str
    reason: str

    def __post_init__(self) -> None:
        """Validate the exclusion entry."""
        normalized_path = PurePosixPath(str(self.path))
        if normalized_path.is_absolute():
            raise ManifestValidationError(
                f"Exclusion path '{normalized_path}' must be relative to the examples directory."
            )
        if any(part == ".." for part in normalized_path.parts):
            raise ManifestValidationError(
                f"Exclusion path '{normalized_path}' cannot traverse out of the examples directory."
            )
        if normalized_path.suffix != ".py":
            raise ManifestValidationError(
                f"Exclusion path '{normalized_path}' must point to a Python file."
            )
        object.__setattr__(self, "path", normalized_path)
        if self.kind not in EXCLUSION_KINDS:
            allowed = ", ".join(sorted(EXCLUSION_KINDS))
            raise ManifestValidationError(
                f"Exclusion '{normalized_path}' has unknown kind '{self.kind}' "
                f"(allowed: {allowed})."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ManifestValidationError(
                f"Exclusion '{normalized_path}' must declare a non-empty reason."
            )


def _normalize_example_path(raw_path: Any) -> PurePosixPath:
    """Validate that an example path is a relative ``.py`` path.

    Args:
        raw_path: The raw path value from the manifest or dataclass field.

    Returns:
        The normalized relative path.

    Raises:
        ManifestValidationError: If the path is absolute, escapes, or not Python.
    """

    normalized_path = PurePosixPath(str(raw_path))
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
    return normalized_path


def _check_example_identity(
    name: Any, summary: Any, category_slug: Any, path: PurePosixPath
) -> None:
    """Validate example name, summary, and category slug presence.

    Args:
        name: The example display name.
        summary: The example summary string.
        category_slug: The referenced category slug.
        path: The normalized example path used in error messages.

    Raises:
        ManifestValidationError: If any identity field is missing or un-normalized.
    """

    if not name:
        raise ManifestValidationError("Example name cannot be empty.")
    if isinstance(name, str) and name.strip() != name:
        raise ManifestValidationError(
            f"Example '{path}' has a name with leading/trailing whitespace."
        )
    if not summary:
        raise ManifestValidationError(f"Example '{path}' must include a non-empty summary.")
    if not category_slug:
        raise ManifestValidationError(f"Example '{path}' must reference a category slug.")


def _check_ci_fields(ci_enabled: Any, ci_reason: Any, path: PurePosixPath) -> None:
    """Validate ``ci_enabled``/``ci_reason`` consistency for one example.

    Args:
        ci_enabled: The CI eligibility flag.
        ci_reason: The optional CI rationale string.
        path: The normalized example path used in error messages.

    Raises:
        ManifestValidationError: If the flag is not boolean or the reason is
            missing on disabled entries or present on enabled entries.
    """

    if not isinstance(ci_enabled, bool):
        raise ManifestValidationError(f"Example '{path}' must define ci_enabled as a boolean.")
    if not ci_enabled and not ci_reason:
        raise ManifestValidationError(f"Example '{path}' is disabled for CI but missing ci_reason.")
    if ci_enabled and ci_reason:
        raise ManifestValidationError(
            f"Example '{path}' is CI-enabled but declares ci_reason (E_CI_REASON_CONTRADICTION)."
        )


def _check_unique_example(
    example: ExampleScript,
    seen_paths: set[PurePosixPath],
    seen_names: set[str],
) -> None:
    """Reject duplicate example paths and display names.

    Args:
        example: The example under inspection.
        seen_paths: Paths already accepted from this manifest.
        seen_names: Display names already accepted from this manifest.

    Raises:
        ManifestValidationError: On duplicate path (``E_DUPLICATE_PATH``) or
            duplicate name (``E_DUPLICATE_NAME``).
    """

    if example.path in seen_paths:
        raise ManifestValidationError(
            f"Duplicate example path '{example.path}' detected in manifest (E_DUPLICATE_PATH)."
        )
    if example.name in seen_names:
        raise ManifestValidationError(
            f"Duplicate example name '{example.name}' detected in manifest (E_DUPLICATE_NAME)."
        )


def _check_exclusions(
    exclusions: tuple[ExampleExclusion, ...],
    seen_paths: set[PurePosixPath],
) -> None:
    """Reject duplicate or example-overlapping exclusion paths.

    Args:
        exclusions: The parsed exclusion entries.
        seen_paths: Accepted example paths that exclusions must not repeat.

    Raises:
        ManifestValidationError: On duplicate exclusion paths or overlap.
    """

    seen_exclusions: set[PurePosixPath] = set()
    for exclusion in exclusions:
        if exclusion.path in seen_exclusions:
            raise ManifestValidationError(
                f"Duplicate exclusion path '{exclusion.path}' detected in manifest."
            )
        seen_exclusions.add(exclusion.path)
        if exclusion.path in seen_paths:
            raise ManifestValidationError(
                f"Exclusion '{exclusion.path}' overlaps a registered example path."
            )


def _check_unique_category_orders(categories: Sequence[ExampleCategory]) -> None:
    """Reject category-order collisions that would make rendering ambiguous."""

    seen_orders: dict[int, str] = {}
    for category in categories:
        if category.order in seen_orders:
            raise ManifestValidationError(
                f"Duplicate category order '{category.order}' for categories "
                f"'{seen_orders[category.order]}' and '{category.slug}' "
                "(E_DUPLICATE_CATEGORY_ORDER)."
            )
        seen_orders[category.order] = category.slug


@dataclass(frozen=True, slots=True)
class ExampleManifest:
    """Container object holding parsed manifest data."""

    version: str
    categories: tuple[ExampleCategory, ...]
    examples: tuple[ExampleScript, ...]
    manifest_path: Path
    examples_root: Path
    schema_version: int = MANIFEST_SCHEMA_VERSION
    exclusions: tuple[ExampleExclusion, ...] = ()
    _categories_by_slug: dict[str, ExampleCategory] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the example category configuration.

        Ensures all category fields meet structural requirements:
        - Slug is non-empty and has no path separators
        - Slug has no leading/trailing whitespace
        - Order is an integer
        - ci_default is a boolean
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
        _check_unique_category_orders(self.categories)
        object.__setattr__(
            self,
            "categories",
            tuple(sorted(self.categories, key=lambda category: category.order)),
        )
        object.__setattr__(self, "_categories_by_slug", categories_by_slug)

        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Unsupported manifest schema_version '{self.schema_version}' "
                f"(supported: {MANIFEST_SCHEMA_VERSION})."
            )

        seen_paths: set[PurePosixPath] = set()
        seen_names: set[str] = set()
        normalized_examples: list[ExampleScript] = []
        for example in self.examples:
            if example.category_slug not in categories_by_slug:
                raise ManifestValidationError(
                    f"Example '{example.path}' references unknown category '{example.category_slug}'."
                )
            _check_unique_example(example, seen_paths, seen_names)
            seen_paths.add(example.path)
            seen_names.add(example.name)
            normalized_examples.append(example)
        object.__setattr__(self, "examples", tuple(normalized_examples))

        _check_exclusions(self.exclusions, seen_paths)

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
        """Ensure that every example and exclusion path exists on disk."""

        missing: list[str] = []
        for example in self.examples:
            candidate = self.examples_root / Path(example.path)
            if not candidate.is_file():
                missing.append(example.path.as_posix())
        for exclusion in self.exclusions:
            candidate = self.examples_root / Path(exclusion.path)
            if not candidate.is_file():
                missing.append(exclusion.path.as_posix())
        if missing:
            joined = ", ".join(sorted(missing))
            raise ManifestValidationError(f"Examples missing from filesystem: {joined}.")

    def excluded_paths(self) -> frozenset[str]:
        """Return exclusion paths as a set of posix strings."""

        return frozenset(exclusion.path.as_posix() for exclusion in self.exclusions)


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
    missing_fields = sorted(REQUIRED_MANIFEST_FIELDS.difference(raw_data))
    if missing_fields:
        fields = ", ".join(missing_fields)
        raise ManifestValidationError(f"Manifest is missing required field(s): {fields}.")
    categories = _parse_categories(raw_data.get("categories", []))
    examples = _parse_examples(raw_data.get("examples", []), categories)

    manifest = ExampleManifest(
        version=_expect_string(raw_data.get("version"), "version"),
        categories=tuple(categories.values()),
        examples=tuple(examples),
        manifest_path=resolved_manifest,
        examples_root=resolved_manifest.parent,
        schema_version=_parse_schema_version(raw_data.get("schema_version")),
        exclusions=_parse_exclusions(raw_data.get("exclusions")),
    )

    if validate_paths:
        manifest.validate_paths()

    return manifest


def _resolve_manifest_path(manifest_path: str | Path | None) -> Path:
    """Resolve the manifest path from user input or default location.

    Args:
        manifest_path: Optional path string or Path object.

    Returns:
        Resolved Path object to the manifest file.

    Raises:
        ManifestValidationError: If manifest file is not found.
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
    """Read the manifest YAML file into a dictionary.

    Args:
        path: Path to the manifest YAML file.

    Returns:
        Parsed YAML data as a dictionary.

    Raises:
        ManifestValidationError: If the YAML is not a mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ManifestValidationError("Manifest root must be a mapping of keys to values.")
    return data


def _parse_categories(raw_categories: Any) -> dict[str, ExampleCategory]:
    """Parse raw category data into a dict of ExampleCategory objects.

    Args:
        raw_categories: Raw category data from YAML.

    Returns:
        Dict mapping category slugs to ExampleCategory objects.

    Raises:
        ManifestValidationError: If any category fails validation.
    """
    if isinstance(raw_categories, str) or not isinstance(raw_categories, Sequence):
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
    """Parse raw example data into ExampleScript objects.

    Args:
        raw_examples: Raw example data from YAML.
        categories: Mapping of category objects for reference.

    Returns:
        Tuple of ExampleScript objects.

    Raises:
        ManifestValidationError: If any example fails validation.
    """
    if isinstance(raw_examples, str) or not isinstance(raw_examples, Sequence):
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
            prerequisites=_normalize_prerequisites(
                item.get("prerequisites"),
                f"examples[{index}].prerequisites",
            ),
            ci_enabled=ci_enabled,
            ci_reason=_optional_string(item.get("ci_reason")),
            doc_reference=_optional_string(item.get("doc_reference")),
            tags=_normalize_tags(item.get("tags"), f"examples[{index}].tags"),
            expected_runtime=_optional_string(item.get("expected_runtime")),
        )
        parsed.append(example)
    return tuple(parsed)


def _parse_schema_version(value: Any) -> int:
    """Parse the optional top-level ``schema_version``.

    Args:
        value: The raw ``schema_version`` value (``None`` when absent).

    Returns:
        The validated schema version (``MANIFEST_SCHEMA_VERSION``).

    Raises:
        ManifestValidationError: If the value is not the supported version.
    """

    if value is None:
        return MANIFEST_SCHEMA_VERSION
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError("Field 'schema_version' must be an integer.")
    if value != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Unsupported manifest schema_version '{value}' (supported: {MANIFEST_SCHEMA_VERSION})."
        )
    return value


def _parse_exclusions(value: Any) -> tuple[ExampleExclusion, ...]:
    """Parse the optional top-level ``exclusions`` list.

    Args:
        value: The raw ``exclusions`` value (``None`` when absent).

    Returns:
        The parsed exclusion entries, empty when the key is absent.

    Raises:
        ManifestValidationError: If any exclusion entry fails validation.
    """

    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ManifestValidationError("Manifest 'exclusions' must be a list.")
    parsed: list[ExampleExclusion] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ManifestValidationError(
                f"Exclusion entry at index {index} must be a mapping, got {type(item)!r}."
            )
        exclusion = ExampleExclusion(
            path=PurePosixPath(_expect_string(item.get("path"), f"exclusions[{index}].path")),
            kind=_expect_string(item.get("kind"), f"exclusions[{index}].kind"),
            reason=_expect_string(item.get("reason"), f"exclusions[{index}].reason"),
        )
        parsed.append(exclusion)
    return tuple(parsed)


def _expect_string(value: Any, field_name: str) -> str:
    """Validate that a field is a non-empty string.

    Args:
        value: The field value to check.
        field_name: Name of the field for error messages.

    Returns:
        The string value if valid.

    Raises:
        ManifestValidationError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"Field '{field_name}' must be a non-empty string.")
    return value


def _expect_int(value: Any, field_name: str) -> int:
    """Validate that a field is an integer.

    Args:
        value: The field value to check.
        field_name: Name of the field for error messages.

    Returns:
        The integer value if valid.

    Raises:
        ManifestValidationError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"Field '{field_name}' must be an integer.")
    return value


def _expect_bool(value: Any, field_name: str) -> bool:
    """Validate that a field is a boolean.

    Args:
        value: The field value to check.
        field_name: Name of the field for error messages.

    Returns:
        The boolean value if valid.

    Raises:
        ManifestValidationError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ManifestValidationError(f"Field '{field_name}' must be a boolean.")
    return value


def _optional_string(value: Any) -> str | None:
    """Validate that a field is an optional string.

    Args:
        value: The field value to check.

    Returns:
        The string value if provided, None otherwise.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestValidationError("Optional string fields must be a string when provided.")
    stripped = value.strip()
    return stripped or None


def _optional_string_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate that a field is an optional string sequence.

    Args:
        value: The field value to check.
        field_name: Name of the field for error messages.

    Returns:
        Tuple of strings if provided, empty tuple otherwise.
    """
    if value is None:
        return ()
    return _validate_string_sequence(value, field_name)


def _normalize_prerequisites(
    value: Any,
    field_name: str,
    context: PurePosixPath | None = None,
) -> tuple[str, ...]:
    """Normalize a prerequisites list, filtering sentinel no-op entries.

    Sentinel spellings (``PREREQUISITE_SENTINELS`` plus YAML ``null``) mean "no
    prerequisite" and are dropped; an all-sentinel list normalizes to ``()``.
    Malformed sentinels — a bare string instead of a list, non-string entries
    (other than ``None``), or empty-string entries — raise
    :class:`ManifestValidationError` with an ``E_MALFORMED_PREREQUISITE`` marker.

    Args:
        value: The raw prerequisites value.
        field_name: Field name used in error messages.
        context: Optional example path used in error messages.

    Returns:
        The normalized prerequisite tuple with sentinels removed.

    Raises:
        ManifestValidationError: If the value is a malformed sentinel.
    """

    if value is None:
        return ()
    if isinstance(value, str):
        raise ManifestValidationError(
            f"Field '{field_name}' must be a list of strings, not a single string "
            "(E_MALFORMED_PREREQUISITE)."
        )
    if not isinstance(value, Sequence):
        raise ManifestValidationError(
            f"Field '{field_name}' must be a sequence of strings, got {type(value)!r} "
            "(E_MALFORMED_PREREQUISITE)."
        )

    items: list[str] = []
    for raw in value:
        if raw is None:
            continue
        if not isinstance(raw, str):
            location = f" for example '{context}'" if context else ""
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must contain only non-empty strings "
                f"(E_MALFORMED_PREREQUISITE, got {type(raw)!r})."
            )
        stripped = raw.strip()
        if not stripped:
            location = f" for example '{context}'" if context else ""
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must contain only non-empty strings "
                "(E_MALFORMED_PREREQUISITE)."
            )
        if stripped.lower() in PREREQUISITE_SENTINELS:
            continue
        items.append(stripped)
    return tuple(items)


def _normalize_tags(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate tags from raw manifest data.

    Args:
        value: The raw tags value (``None`` when absent).
        field_name: Field name used in error messages.

    Returns:
        The validated normalized tag tuple, empty when absent.

    Raises:
        ManifestValidationError: If any tag is not in normalized form.
    """

    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ManifestValidationError(
            f"Field '{field_name}' must be a sequence of strings, got {type(value)!r}."
        )
    for raw in value:
        if not isinstance(raw, str) or raw.strip() != raw or not raw:
            raise ManifestValidationError(
                f"Field '{field_name}' must use normalized tags "
                f"(E_TAG_NOT_NORMALIZED, got {raw!r})."
            )
    tags = _validate_string_sequence(value, field_name)
    return _validate_tags(tags, field_name)


def _validate_tags(
    value: Any,
    field_name: str,
    context: PurePosixPath | None = None,
) -> tuple[str, ...]:
    """Validate that tags are normalized (stripped, lowercase, patterned).

    Args:
        value: The tag sequence to check.
        field_name: Field name used in error messages.
        context: Optional example path used in error messages.

    Returns:
        The validated tag tuple.

    Raises:
        ManifestValidationError: If any tag is not normalized.
    """

    tags = _validate_string_sequence(value, field_name, context)
    seen: set[str] = set()
    for tag in tags:
        location = f" for example '{context}'" if context else ""
        if tag.strip() != tag or tag.lower() != tag:
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must use normalized tags "
                f"(E_TAG_NOT_NORMALIZED, got {tag!r})."
            )
        if not _TAG_PATTERN.match(tag):
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must use normalized tags "
                f"(E_TAG_NOT_NORMALIZED, got {tag!r})."
            )
        if tag in seen:
            raise ManifestValidationError(
                f"Field '{field_name}'{location} must not contain duplicate tags "
                f"(E_TAG_NOT_NORMALIZED, got {tag!r})."
            )
        seen.add(tag)
    return tags


def _validate_runtime_class(value: Any, context: PurePosixPath | None = None) -> None:
    """Validate the runtime-class surface (``expected_runtime``).

    Must be ``None`` or an already-stripped non-empty string that is not a
    prerequisite-style sentinel. Sentinel spellings are rejected so estimates
    stay explicit (``E_RUNTIME_CLASS_INVALID``).

    Args:
        value: The raw ``expected_runtime`` value.
        context: Optional example path used in error messages.

    Raises:
        ManifestValidationError: If the value is not a valid runtime class.
    """

    if value is None:
        return
    if not isinstance(value, str) or not value:
        location = f" for example '{context}'" if context else ""
        raise ManifestValidationError(
            f"Field 'expected_runtime'{location} must be a non-empty string when provided "
            "(E_RUNTIME_CLASS_INVALID)."
        )
    if value.strip() != value:
        location = f" for example '{context}'" if context else ""
        raise ManifestValidationError(
            f"Field 'expected_runtime'{location} must be normalized (E_RUNTIME_CLASS_INVALID)."
        )
    if value.lower() in PREREQUISITE_SENTINELS:
        location = f" for example '{context}'" if context else ""
        raise ManifestValidationError(
            f"Field 'expected_runtime'{location} must not be a sentinel "
            f"(E_RUNTIME_CLASS_INVALID, got {value!r})."
        )


def _validate_string_sequence(
    value: Any,
    field_name: str,
    context: PurePosixPath | None = None,
) -> tuple[str, ...]:
    """Validate that a field is a sequence of strings.

    Args:
        value: The field value to check.
        field_name: Name of the field for error messages.
        context: Optional path context for error messages.

    Returns:
        Tuple of strings if valid.

    Raises:
        ManifestValidationError: If the value is not a string sequence.
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
