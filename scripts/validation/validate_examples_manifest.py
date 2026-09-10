"""Validate that the examples manifest matches the repository state.

This script performs a series of structural checks against
``examples/examples_manifest.yaml`` to guarantee that documentation tooling and
CI smoke tests have an accurate view of the available example scripts. It makes
sure that every maintained example on disk is represented exactly once (or
matches an explicit versioned exclusion with rationale), that category slugs
align with directory layout, that tags/runtime class are normalized, that
``ci_enabled``/``ci_reason`` are consistent, that documentation references
exist, and that module docstrings match the metadata summaries stored in the
manifest.

Every error is deterministic and reported as ``<path>: [<REASON_CODE>] message``
(or ``manifest: [<REASON_CODE>]`` for file-level issues), sorted by
``(path, code)`` so repeated runs are byte-identical. Stable reason codes:

- ``E_UNREGISTERED_FILE`` — ``.py`` on disk (minus ``__init__.py``) that is
  neither a registered example nor a versioned exclusion.
- ``E_MISSING_PATH`` — manifest example/exclusion path absent from disk.
- ``E_EXCLUSION_OVERLAP`` — exclusion path that duplicates a registered example.
- ``E_CATEGORY_MISMATCH`` — leading directory does not match ``category_slug``.
- ``E_DOCSTRING_MISSING`` / ``E_DOCSTRING_MISMATCH`` — docstring contract.
- ``E_DOCSTRING_PARSE`` — registered example has invalid Python syntax.
- ``E_DOC_REFERENCE_MISSING`` — ``doc_reference`` file target or anchor absent.
- ``E_TAG_NOT_NORMALIZED`` — tag not in normalized lowercase form.
- ``E_RUNTIME_CLASS_INVALID`` — ``expected_runtime`` sentinel/un-normalized.
- ``E_CI_REASON_MISSING`` / ``E_CI_REASON_CONTRADICTION`` — CI consistency.
- ``E_DUPLICATE_NAME`` — duplicate example display name.

Example::

    uv run python scripts/validation/validate_examples_manifest.py

"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from robot_sf.examples import ExampleManifest, ManifestValidationError, load_manifest

_MARKDOWN_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_HTML_ANCHOR = re.compile(r'<a\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validator."""

    parser = argparse.ArgumentParser(
        description="Validate the examples manifest and related documentation contracts."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the manifest file (defaults to examples/examples_manifest.yaml)",
    )
    parser.add_argument(
        "--skip-docstring-checks",
        action="store_true",
        help="Skip validating that module docstrings match manifest summaries.",
    )
    parser.add_argument(
        "--allow-missing-docstrings",
        action="store_true",
        help=(
            "Treat missing module docstrings as warnings instead of hard errors. "
            "Only applies when docstring checks are enabled."
        ),
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=None,
        help="Override the examples directory root (defaults to manifest parent directory).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the validation script."""

    args = parse_args()
    manifest_path: Path | None = args.manifest

    try:
        manifest = load_manifest(manifest_path, validate_paths=False)
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    examples_root = args.examples_root or manifest.examples_root

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_check_manifest_coverage(manifest, examples_root))
    errors.extend(_check_category_directory_alignment(manifest))
    errors.extend(_check_unique_names(manifest))
    errors.extend(_check_ci_consistency(manifest))
    errors.extend(_check_tags_normalized(manifest))
    errors.extend(_check_runtime_class(manifest))
    errors.extend(_check_doc_references(manifest))

    if not args.skip_docstring_checks:
        doc_errors, doc_warnings = _check_docstrings(
            manifest,
            allow_missing=args.allow_missing_docstrings,
        )
        errors.extend(doc_errors)
        warnings.extend(doc_warnings)

    errors = sorted(errors)
    warnings = sorted(warnings)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        for issue in errors:
            print(f"ERROR: {issue}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("Examples manifest validation passed.")
    return 0


def _coded(path: str, code: str, message: str) -> str:
    """Format a deterministic ``path: [CODE] message`` error line."""

    return f"{path}: [{code}] {message}"


def _check_manifest_coverage(manifest: ExampleManifest, examples_root: Path) -> list[str]:
    """Ensure every maintained script is registered once or explicitly excluded."""

    root = examples_root.resolve()
    if not root.is_dir():
        return [_coded(str(root), "E_MISSING_PATH", f"Examples root '{root}' is not a directory.")]

    manifest_paths = {example.path.as_posix() for example in manifest.examples}
    exclusion_paths = {exclusion.path.as_posix() for exclusion in manifest.exclusions}
    registered = manifest_paths | exclusion_paths

    overlap = sorted(manifest_paths & exclusion_paths)
    errors: list[str] = [
        _coded(path, "E_EXCLUSION_OVERLAP", "exclusion overlaps a registered example path.")
        for path in overlap
    ]

    discovered: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if _should_ignore_file(path):
            continue
        relative = path.relative_to(root).as_posix()
        discovered.add(relative)

    for missing in sorted(discovered - registered):
        errors.append(
            _coded(missing, "E_UNREGISTERED_FILE", "script on disk is not registered in manifest.")
        )

    # Manifest entries that do not exist on disk will already have triggered
    # ManifestValidationError during load when validate_paths=True, but we
    # defensively highlight any anomalies discovered here as well.
    for undefined in sorted(manifest_paths - discovered):
        errors.append(
            _coded(
                undefined,
                "E_MISSING_PATH",
                "manifest references a script that was not found on disk.",
            )
        )
    for undefined in sorted(exclusion_paths - discovered):
        errors.append(
            _coded(
                undefined,
                "E_MISSING_PATH",
                "manifest exclusion references a script that was not found on disk.",
            )
        )

    return errors


def _should_ignore_file(path: Path) -> bool:
    """Return True when the given path should be ignored for coverage checks."""

    if path.name == "__init__.py":
        return True
    return False


def _check_category_directory_alignment(manifest: ExampleManifest) -> list[str]:
    """Ensure each entry resides in the directory that matches its category slug."""

    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        parts = example.path.parts
        slug = example.category_slug

        if slug == "uncategorized":
            # Root-level examples remain acceptable during migration.
            if len(parts) > 1:
                errors.append(
                    _coded(
                        example.path.as_posix(),
                        "E_CATEGORY_MISMATCH",
                        "expected to be at repository root for 'uncategorized'.",
                    )
                )
            continue

        if not parts:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_CATEGORY_MISMATCH",
                    "example path is empty in manifest.",
                )
            )
            continue

        if parts[0] != slug:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_CATEGORY_MISMATCH",
                    f"leading directory '{parts[0]}' does not match category slug '{slug}'.",
                )
            )

    return errors


def _check_unique_names(manifest: ExampleManifest) -> list[str]:
    """Defensively re-check display-name uniqueness with a stable code."""

    seen: dict[str, str] = {}
    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        if example.name in seen:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DUPLICATE_NAME",
                    f"duplicate example name '{example.name}' "
                    f"(also used by '{seen[example.name]}').",
                )
            )
        else:
            seen[example.name] = example.path.as_posix()
    return errors


def _check_ci_consistency(manifest: ExampleManifest) -> list[str]:
    """Check ``ci_enabled``/``ci_reason`` consistency with stable codes."""

    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        path = example.path.as_posix()
        if not example.ci_enabled and not example.ci_reason:
            errors.append(
                _coded(
                    path,
                    "E_CI_REASON_MISSING",
                    "example is disabled for CI but missing ci_reason.",
                )
            )
        if example.ci_enabled and example.ci_reason:
            errors.append(
                _coded(
                    path,
                    "E_CI_REASON_CONTRADICTION",
                    "example is CI-enabled but declares ci_reason.",
                )
            )
    return errors


def _check_tags_normalized(manifest: ExampleManifest) -> list[str]:
    """Defensively re-check tag normalization with a stable code."""

    import re as _re

    pattern = _re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        seen: set[str] = set()
        for tag in example.tags:
            if tag.strip() != tag or tag.lower() != tag or not pattern.match(tag) or tag in seen:
                errors.append(
                    _coded(
                        example.path.as_posix(),
                        "E_TAG_NOT_NORMALIZED",
                        f"tag {tag!r} is not normalized.",
                    )
                )
                break
            seen.add(tag)
    return errors


def _check_runtime_class(manifest: ExampleManifest) -> list[str]:
    """Check the runtime-class surface (``expected_runtime``) with a stable code."""

    from robot_sf.examples.manifest_loader import PREREQUISITE_SENTINELS

    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        runtime = example.expected_runtime
        if runtime is None:
            continue
        if not runtime or runtime.strip() != runtime or runtime.lower() in PREREQUISITE_SENTINELS:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_RUNTIME_CLASS_INVALID",
                    f"expected_runtime {runtime!r} is not a valid runtime class.",
                )
            )
    return errors


def _check_doc_references(manifest: ExampleManifest) -> list[str]:
    """Check that every ``doc_reference`` target and optional anchor exists."""

    repo_root = manifest.manifest_path.parents[1]
    errors: list[str] = []
    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        if not example.doc_reference:
            continue
        target = example.doc_reference.split("#", 1)[0].strip()
        if not target:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DOC_REFERENCE_MISSING",
                    f"doc_reference '{example.doc_reference}' has no file target.",
                )
            )
            continue
        candidate = (repo_root / target).resolve(strict=False)
        try:
            inside = candidate.is_relative_to(repo_root.resolve(strict=False))
        except ValueError:
            inside = False
        if not inside or not candidate.is_file():
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DOC_REFERENCE_MISSING",
                    f"doc_reference target '{target}' was not found.",
                )
            )
            continue

        if "#" not in example.doc_reference:
            continue
        anchor = example.doc_reference.split("#", 1)[1].strip()
        if not anchor or anchor not in _read_markdown_anchors(candidate):
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DOC_REFERENCE_MISSING",
                    f"doc_reference anchor '{anchor}' was not found in '{target}'.",
                )
            )
    return errors


def _read_markdown_anchors(path: Path) -> set[str]:
    """Return explicit and heading-derived anchors from a readable Markdown file."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()

    anchors = {match.group(1) for match in _HTML_ANCHOR.finditer(text)}
    occurrences: dict[str, int] = {}
    for match in _MARKDOWN_HEADING.finditer(text):
        heading = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        heading = re.sub(r"[ \t]+#+[ \t]*$", "", heading).strip().lower()
        slug = re.sub(r"[^\w\s-]", "", heading)
        slug = re.sub(r"[\s-]+", "-", slug).strip("-")
        if not slug:
            continue
        occurrence = occurrences.get(slug, 0)
        anchors.add(slug if occurrence == 0 else f"{slug}-{occurrence}")
        occurrences[slug] = occurrence + 1
    return anchors


def _check_docstrings(
    manifest: ExampleManifest,
    *,
    allow_missing: bool,
) -> tuple[list[str], list[str]]:
    """Confirm module docstrings exist and match manifest summaries."""

    errors: list[str] = []
    warnings: list[str] = []

    for example in sorted(manifest.examples, key=lambda item: item.path.as_posix()):
        module_path = manifest.resolve_example_path(example)
        try:
            docstring = _read_module_docstring(module_path)
        except ManifestValidationError as exc:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DOCSTRING_PARSE",
                    str(exc),
                )
            )
            continue

        if docstring is None:
            message = _coded(
                example.path.as_posix(), "E_DOCSTRING_MISSING", "missing module docstring."
            )
            if allow_missing:
                warnings.append(message)
            else:
                errors.append(message)
            continue

        first_line = docstring.splitlines()[0].strip() if docstring.splitlines() else ""
        summary = example.summary.strip()
        if first_line != summary:
            errors.append(
                _coded(
                    example.path.as_posix(),
                    "E_DOCSTRING_MISMATCH",
                    "docstring first line does not match manifest summary.",
                )
            )

    return errors, warnings


def _read_module_docstring(module_path: Path) -> str | None:
    """Read and return the cleaned module docstring for a Python script."""

    try:
        source = module_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    try:
        module = ast.parse(source, filename=str(module_path))
    except SyntaxError as exc:
        raise ManifestValidationError(f"Failed to parse {module_path.as_posix()}: {exc}") from exc

    return ast.get_docstring(module, clean=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
