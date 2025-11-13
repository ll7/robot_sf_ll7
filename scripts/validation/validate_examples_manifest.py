"""Validate that the examples manifest matches the repository state.

This script performs a series of structural checks against
``examples/examples_manifest.yaml`` to guarantee that documentation tooling and
CI smoke tests have an accurate view of the available example scripts. It makes
sure that every example on disk is represented in the manifest, that category
slugs align with directory layout, and that module docstrings match the metadata
summaries stored in the manifest.

Example::

    uv run python scripts/validation/validate_examples_manifest.py

"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from robot_sf.examples import ExampleManifest, ManifestValidationError, load_manifest


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
        manifest = load_manifest(manifest_path, validate_paths=True)
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    examples_root = args.examples_root or manifest.examples_root

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_check_manifest_coverage(manifest, examples_root))
    errors.extend(_check_category_directory_alignment(manifest))

    if not args.skip_docstring_checks:
        doc_errors, doc_warnings = _check_docstrings(
            manifest,
            allow_missing=args.allow_missing_docstrings,
        )
        errors.extend(doc_errors)
        warnings.extend(doc_warnings)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        for issue in errors:
            print(f"ERROR: {issue}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("Examples manifest validation passed.")
    return 0


def _check_manifest_coverage(manifest: ExampleManifest, examples_root: Path) -> list[str]:
    """Ensure every script on disk appears in the manifest."""

    root = examples_root.resolve()
    if not root.is_dir():
        return [f"Examples root '{root}' is not a directory."]

    manifest_paths = {example.path.as_posix() for example in manifest.examples}

    discovered: set[str] = set()
    for path in root.rglob("*.py"):
        if _should_ignore_file(path):
            continue
        relative = path.relative_to(root).as_posix()
        discovered.add(relative)

    missing = sorted(discovered - manifest_paths)
    errors: list[str] = []
    if missing:
        errors.append("Scripts missing from manifest: " + ", ".join(missing))

    # Manifest entries that do not exist on disk will already have triggered
    # ManifestValidationError during load when validate_paths=True, but we
    # defensively highlight any anomalies discovered here as well.
    undefined = sorted(manifest_paths - discovered)
    if undefined:
        errors.append(
            "Manifest references scripts that were not found on disk: " + ", ".join(undefined)
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
    for example in manifest.examples:
        parts = example.path.parts
        slug = example.category_slug

        if slug == "uncategorized":
            # Root-level examples remain acceptable during migration.
            if len(parts) > 1:
                errors.append(
                    f"{example.path.as_posix()}: expected to be at repository root for 'uncategorized'."
                )
            continue

        if not parts:
            errors.append(f"{example.path.as_posix()}: example path is empty in manifest.")
            continue

        if parts[0] != slug:
            errors.append(
                f"{example.path.as_posix()}: leading directory '{parts[0]}' does not match category slug '{slug}'."
            )

    return errors


def _check_docstrings(
    manifest: ExampleManifest,
    *,
    allow_missing: bool,
) -> tuple[list[str], list[str]]:
    """Confirm module docstrings exist and match manifest summaries."""

    errors: list[str] = []
    warnings: list[str] = []

    for example in manifest.examples:
        module_path = manifest.resolve_example_path(example)
        try:
            docstring = _read_module_docstring(module_path)
        except ManifestValidationError as exc:
            errors.append(str(exc))
            continue

        if docstring is None:
            message = f"{example.path.as_posix()}: missing module docstring"
            if allow_missing:
                warnings.append(message)
            else:
                errors.append(message)
            continue

        first_line = docstring.splitlines()[0].strip() if docstring.splitlines() else ""
        summary = example.summary.strip()
        if first_line != summary:
            errors.append(
                f"{example.path.as_posix()}: docstring first line does not match manifest summary."
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
