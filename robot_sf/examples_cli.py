"""Discovery CLI for the examples catalog.

Backs the ``robot-sf examples list/run`` commands described in issue #5794.
Everything is driven from ``examples/examples_manifest.yaml`` (the single source
of truth) via :func:`robot_sf.examples.load_manifest`; this module never
duplicates example metadata.

Stable identifiers
------------------
Each example's *id* is its manifest ``path`` with the ``.py`` suffix removed,
for example ``quickstart/01_basic_robot``. The full manifest path
(e.g. ``quickstart/01_basic_robot.py``) and the bare filename stem
(e.g. ``01_basic_robot``) are accepted as aliases when unambiguous.

Fast mode
---------
``run --fast`` activates the repository's established reduced-step convention by
setting ``ROBOT_SF_FAST_DEMO=1`` and capping ``ROBOT_SF_EXAMPLES_MAX_STEPS``.
Examples that honour these env vars (most quickstart/benchmark examples do)
then execute a short smoke-style rollout instead of their default length.
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from robot_sf.examples.manifest_loader import (
    ExampleManifest,
    ExampleScript,
    ManifestValidationError,
    load_manifest,
)

__all__ = [
    "ExampleIdentityError",
    "ExamplesCliError",
    "build_examples_parser",
    "example_id",
    "examples_cli_main",
    "find_example",
    "format_examples_table",
    "run_example",
]

# Reduced-step budget applied in fast mode. Mirrors the CI smoke harness value
# so ``run --fast`` behaves like a local smoke check.
_FAST_MAX_STEPS = "64"


def example_id(example: ExampleScript) -> str:
    """Return the stable discovery id for an example.

    The id is the manifest ``path`` with the trailing ``.py`` removed, so
    ``quickstart/01_basic_robot.py`` becomes ``quickstart/01_basic_robot``.

    Args:
        example: The example to compute an id for.

    Returns:
        The example's discovery id.
    """

    return example.path.with_suffix("").as_posix()


@dataclass(frozen=True, slots=True)
class _ExampleEntry:
    """Internal lookup record bundling an example with its discovery id."""

    example: ExampleScript
    identifier: str
    aliases: tuple[str, ...]


class ExamplesCliError(Exception):
    """Base class for examples CLI failures (reported as user-facing errors)."""


class ExampleIdentityError(ExamplesCliError):
    """Raised when an example id cannot be resolved to a single entry."""


def _build_lookup(
    manifest: ExampleManifest,
) -> tuple[list[_ExampleEntry], dict[str, _ExampleEntry]]:
    """Index every example by id, full path, and (where unique) filename stem.

    Args:
        manifest: The loaded examples manifest.

    Returns:
        A ``(entries, index)`` pair where ``entries`` preserves the manifest
        order and ``index`` maps every accepted alias (lower-cased) to its entry.
        When a filename stem collides across examples it is omitted from the
        index so resolution stays unambiguous.
    """

    entries: list[_ExampleEntry] = []
    stem_counts: dict[str, int] = {}
    for example in manifest.examples:
        identifier = example_id(example)
        full_path = example.path.as_posix()
        stem = example.path.stem
        entries.append(
            _ExampleEntry(
                example=example,
                identifier=identifier,
                aliases=(identifier, full_path, stem),
            )
        )
        stem_counts[stem] = stem_counts.get(stem, 0) + 1

    index: dict[str, _ExampleEntry] = {}
    for entry in entries:
        stem = entry.example.path.stem
        # The bare filename stem is only a valid alias when it is unique.
        unique_aliases = tuple(
            alias for alias in entry.aliases if alias != stem or stem_counts[stem] == 1
        )
        for alias in unique_aliases:
            key = alias.lower()
            if key in index and index[key] is not entry:
                # An alias that maps to two different examples is ambiguous and
                # removed entirely so resolution never silently picks one.
                index[key] = None  # type: ignore[assignment]
            else:
                index[key] = entry
    # Drop any alias that collided across entries.
    index = {key: value for key, value in index.items() if value is not None}
    return entries, index


def find_example(manifest: ExampleManifest, query: str) -> ExampleScript:
    """Resolve a user-provided id to a single example.

    Matching is case-insensitive and accepts the discovery id, the full manifest
    path, or the bare filename stem (when unique). On failure, raises
    :class:`ExampleIdentityError` listing the closest matches.

    Args:
        manifest: The loaded examples manifest.
        query: The user-provided id/path/stem.

    Returns:
        The resolved example.

    Raises:
        ExampleIdentityError: If no example matches (the exception message lists
            the closest matches).
    """

    _, index = _build_lookup(manifest)
    entry = index.get(query.strip().lower())
    if entry is not None:
        return entry.example
    raise ExampleIdentityError(_format_unknown_id(manifest, query))


def _format_unknown_id(manifest: ExampleManifest, query: str) -> str:
    """Build a helpful error message for an unresolved example id.

    Args:
        manifest: The loaded examples manifest.
        query: The unresolved user-provided id.

    Returns:
        A multi-line error message including the closest known ids.
    """

    entries, _ = _build_lookup(manifest)
    candidates = [entry.identifier for entry in entries]
    closest = difflib.get_close_matches(query.strip().lower(), [c.lower() for c in candidates], n=5)
    # Map back to the original-cased identifiers for display.
    lower_to_id = {c.lower(): c for c in candidates}
    suggestions = [lower_to_id[match] for match in closest if match in lower_to_id]
    lines = [f"Unknown example id: {query!r}"]
    if suggestions:
        lines.append("Closest matches:")
        for suggestion in suggestions:
            lines.append(f"  - {suggestion}")
    else:
        lines.append("Run `robot-sf examples list` to see available ids.")
    return "\n".join(lines)


def format_examples_table(
    manifest: ExampleManifest,
    *,
    tag: str | None = None,
    category: str | None = None,
) -> str:
    """Render the catalog as a human-readable table.

    Args:
        manifest: The loaded examples manifest.
        tag: Optional tag filter (case-insensitive, exact match on a tag).
        category: Optional category slug filter (case-insensitive).

    Returns:
        The formatted table as a string.
    """

    entries, _ = _build_lookup(manifest)
    rows: list[_ExampleEntry] = []
    tag_filter = tag.lower() if tag else None
    category_filter = category.lower() if category else None
    for entry in entries:
        if tag_filter is not None and tag_filter not in {t.lower() for t in entry.example.tags}:
            continue
        if category_filter is not None and entry.example.category_slug.lower() != category_filter:
            continue
        rows.append(entry)

    if not rows:
        qualifier_parts: list[str] = []
        if tag_filter is not None:
            qualifier_parts.append(f"tag={tag}")
        if category_filter is not None:
            qualifier_parts.append(f"category={category}")
        qualifier = " ".join(qualifier_parts) or "no examples"
        return f"No examples match {qualifier}."

    headers = ("ID", "TITLE", "CATEGORY", "TAGS", "RUNTIME", "CI")
    table_rows: list[tuple[str, ...]] = [headers]
    for entry in rows:
        example = entry.example
        tags = ", ".join(example.tags) if example.tags else "-"
        runtime = example.expected_runtime or "-"
        ci = "yes" if example.ci_enabled else "no"
        table_rows.append(
            (
                entry.identifier,
                example.name,
                example.category_slug,
                tags,
                runtime,
                ci,
            )
        )

    widths = [max(len(str(row[i])) for row in table_rows) for i in range(len(headers))]
    lines: list[str] = []
    for row_index, row in enumerate(table_rows):
        rendered = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        lines.append(rendered)
        if row_index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    lines.append(f"\n{len(rows)} example(s) listed.")
    return "\n".join(lines)


def _headless_env(base: dict[str, str], *, fast: bool) -> dict[str, str]:
    """Return an environment suitable for running an example.

    In fast mode the reduced-step env vars are enabled and headless rendering
    backends are forced so smoke-style runs succeed without a display. In normal
    mode the caller's environment is passed through unchanged (apart from a
    repo-root ``PYTHONPATH``) so interactive examples can still open a window.

    Args:
        base: The base environment to copy (usually ``os.environ``).
        fast: Whether to enable fast/headless mode.

    Returns:
        A new environment dict.
    """

    env = dict(base)
    repo_root = _repo_root()
    env["PYTHONPATH"] = _merge_pythonpath(repo_root, env.get("PYTHONPATH"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if fast:
        env["DISPLAY"] = ""
        env["MPLBACKEND"] = "Agg"
        env["SDL_VIDEODRIVER"] = "dummy"
        env["ROBOT_SF_FAST_DEMO"] = "1"
        env["ROBOT_SF_EXAMPLES_MAX_STEPS"] = _FAST_MAX_STEPS
    return env


def _repo_root() -> Path:
    """Return the repository root inferred from this file's location."""

    return Path(__file__).resolve().parents[1]


def _merge_pythonpath(root: Path, existing: str | None) -> str:
    """Prepend ``root`` to an existing ``PYTHONPATH`` without duplicates.

    Args:
        root: The repository root to prepend.
        existing: The current ``PYTHONPATH`` value, if any.

    Returns:
        A de-duplicated ``PYTHONPATH`` with ``root`` first.
    """

    parts: list[str] = [str(root)]
    if existing:
        parts.extend(element for element in existing.split(os.pathsep) if element)
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return os.pathsep.join(ordered)


def run_example(
    manifest: ExampleManifest,
    query: str,
    *,
    fast: bool = False,
    extra_args: Sequence[str] = (),
    runner=None,
    timeout: float | None = None,
) -> int:
    """Execute an example resolved by id from the manifest.

    The example script is launched as a subprocess using the current Python
    interpreter so it behaves exactly like
    ``uv run python examples/<path>.py``.

    Args:
        manifest: The loaded examples manifest.
        query: The example id/path/stem to run.
        fast: When True, enable reduced-step/fast mode via env vars.
        extra_args: Additional arguments forwarded to the example script.
        runner: Optional callable replacing :func:`subprocess.run` (for tests).
            It must accept ``(command, env, cwd, timeout)`` keyword arguments
            and return an object with ``returncode``.
        timeout: Optional subprocess timeout in seconds.

    Returns:
        The example process exit code.

    Raises:
        ExampleIdentityError: If the id cannot be resolved.
    """

    example = find_example(manifest, query)
    script_path = manifest.resolve_example_path(example)
    if not script_path.is_file():
        raise ExamplesCliError(f"Example script not found on disk: {script_path}")
    command = [sys.executable, str(script_path), *extra_args]
    env = _headless_env(os.environ.copy(), fast=fast)
    cwd = _repo_root()
    if fast:
        sys.stderr.write(
            "Running in fast mode (ROBOT_SF_FAST_DEMO=1, "
            f"ROBOT_SF_EXAMPLES_MAX_STEPS={_FAST_MAX_STEPS}).\n"
        )
    if runner is not None:
        result = runner(command=command, env=env, cwd=cwd, timeout=timeout)
        return int(getattr(result, "returncode", 1))
    completed = subprocess.run(command, env=env, cwd=cwd, timeout=timeout, check=False)
    return int(completed.returncode)


def iter_example_ids(manifest: ExampleManifest) -> Iterator[str]:
    """Yield the discovery id of every example in manifest order.

    Args:
        manifest: The loaded examples manifest.

    Yields:
        Each example's discovery id.
    """

    entries, _ = _build_lookup(manifest)
    for entry in entries:
        yield entry.identifier


def build_examples_parser() -> argparse.ArgumentParser:
    """Build and return the ``argparse`` parser for ``robot-sf examples``.

    Returns:
        The configured argument parser for the examples subcommand.
    """

    parser = argparse.ArgumentParser(
        prog="robot-sf examples",
        description="Discover and run Robot SF examples from examples_manifest.yaml.",
    )
    sub = parser.add_subparsers(dest="examples_command", required=True)

    list_parser = sub.add_parser("list", help="List examples from the manifest.")
    list_parser.add_argument(
        "--tag",
        default=None,
        help="Only show examples carrying this tag (case-insensitive).",
    )
    list_parser.add_argument(
        "--category",
        default=None,
        help="Only show examples in this category slug (case-insensitive).",
    )
    list_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the manifest file (defaults to examples/examples_manifest.yaml).",
    )

    run_parser = sub.add_parser("run", help="Run an example by id.")
    run_parser.add_argument("id", help="Example id, path, or unique filename stem.")
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in reduced-step/fast mode (sets ROBOT_SF_FAST_DEMO=1).",
    )
    run_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the manifest file (defaults to examples/examples_manifest.yaml).",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional subprocess timeout in seconds.",
    )
    run_parser.add_argument(
        "extra",
        nargs="...",
        help="Additional arguments forwarded to the example script.",
    )
    return parser


def examples_cli_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``robot-sf examples`` subcommand.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A process exit code.
    """

    parser = build_examples_parser()
    args = parser.parse_args(argv)
    command = args.examples_command

    try:
        manifest = load_manifest(args.manifest, validate_paths=True)
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if command == "list":
        table = format_examples_table(manifest, tag=args.tag, category=args.category)
        print(table)
        return 0

    if command == "run":
        try:
            return run_example(
                manifest,
                args.id,
                fast=args.fast,
                extra_args=args.extra,
                timeout=args.timeout,
            )
        except ExampleIdentityError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except ExamplesCliError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # argparse enforces a valid subcommand, but guard defensively.
    parser.error(f"Unknown examples command: {command!r}")  # pragma: no cover
    return 2  # pragma: no cover
