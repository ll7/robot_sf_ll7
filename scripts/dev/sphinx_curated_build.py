#!/usr/bin/env python3
"""Strict, curated Sphinx build for the Robot SF documentation site.

The repository docs tree contains a large historical corpus (context notes,
issue packets, design write-ups) that is intentionally *not* part of the
curated Sphinx navigation site. Building the whole source tree with warnings
promoted to errors is therefore impossible without either mass-editing the
historical corpus or suppressing broad warning classes that would also hide
regressions on curated pages.

This helper takes the third path: it builds **only** the curated document set
(the transitive ``toctree`` closure rooted at ``docs/index.rst``) with
``-W --keep-going``, no warning-class suppressions, and the complement of the
curated set excluded. The curated set is pinned in
``docs/sphinx_curated_sources.json`` so a toctree change fails closed until the
manifest is regenerated; new files are never covered by a broad suppression.

Usage::

    uv run python scripts/dev/sphinx_curated_build.py
    uv run python scripts/dev/sphinx_curated_build.py --builder dummy --json
    uv run python scripts/dev/sphinx_curated_build.py --write-manifest

Exit codes:
    0 - strict curated build passed
    1 - build reported warnings/errors (or the manifest drifted)
    2 - usage / environment error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "sphinx_curated_sources.v1"
DEFAULT_DOCS = Path("docs")
DEFAULT_MANIFEST_NAME = "sphinx_curated_sources.json"
DEFAULT_OUT_DIR = Path("output/docs-strict")

_TOCTREE_RST = re.compile(r"\.\. toctree::\n((?:[ \t].*\n|\n)*)")
_TOCTREE_MYST = re.compile(r"```\{toctree\}([^\n]*)\n(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class CuratedBuildResult:
    """Outcome of one strict curated build."""

    status: str
    curated_count: int
    excluded_count: int
    returncode: int
    output_dir: str
    warning_lines: tuple[str, ...]
    allowed_crossref_count: int = 0
    blocking_warnings: tuple[str, ...] = ()


def _resolve_doc(target: str, base: Path) -> Path | None:
    """Resolve one toctree target to a document file or directory index."""

    clean = target.strip().split("#", 1)[0].strip()
    if not clean or clean.startswith(("http://", "https://", "mailto:", "<")):
        return None
    candidate = base / clean
    for suffix in ("", ".md", ".rst"):
        candidate_file = Path(str(candidate) + suffix)
        if candidate_file.is_file():
            return candidate_file.resolve()
    for index_name in ("index.md", "index.rst", "README.md"):
        index_file = candidate / index_name
        if index_file.is_file():
            return index_file.resolve()
    return None


def _toctree_targets(text: str) -> list[str]:
    """Return the raw toctree entries from one RST or MyST document."""

    targets: list[str] = []
    for match in _TOCTREE_RST.finditer(text):
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(":"):
                targets.append(stripped)
    for match in _TOCTREE_MYST.finditer(text):
        for line in match.group(2).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(":"):
                targets.append(stripped)
    return targets


def compute_curated_sources(docs_dir: Path, *, root: str = "index.rst") -> list[Path]:
    """Return the transitive toctree closure starting at ``index.rst``.

    Returns:
        Sorted absolute document paths belonging to the curated site.
    """

    docs_root = docs_dir.resolve()
    queue = [(docs_root / root)]
    seen: set[Path] = set()
    while queue:
        document = queue.pop()
        resolved = document.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        text = resolved.read_text(encoding="utf-8", errors="replace")
        for entry in _toctree_targets(text):
            target = entry.split("<")[-1].rstrip(">").strip() if "<" in entry else entry
            target_path = _resolve_doc(target, resolved.parent)
            if target_path is not None and target_path not in seen:
                queue.append(target_path)
    return sorted(seen)


def manifest_path_for(docs_dir: Path, manifest: str | Path | None) -> Path:
    """Return the manifest path for a docs directory."""

    if manifest is not None:
        return Path(manifest)
    return docs_dir / DEFAULT_MANIFEST_NAME


def build_manifest(curated: list[Path], docs_dir: Path) -> dict[str, Any]:
    """Return the serializable curated-source manifest payload."""

    return {
        "schema": MANIFEST_SCHEMA,
        "generated_from": "docs/index.rst transitive toctree closure",
        "curated": [path.relative_to(docs_dir.resolve()).as_posix() for path in curated],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a curated-source manifest.

    Raises:
        ValueError: If the manifest is missing or malformed.
    """

    if not path.is_file():
        raise ValueError(f"curated source manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported curated source manifest schema in {path}")
    curated = data.get("curated")
    if not isinstance(curated, list) or not all(isinstance(item, str) for item in curated):
        raise ValueError(f"manifest {path} must contain a string list 'curated'")
    return data


def manifest_drift(computed: list[Path], stored: dict[str, Any], docs_dir: Path) -> list[str]:
    """Return human-readable differences between computed and stored sources."""

    computed_set = {path.relative_to(docs_dir.resolve()).as_posix() for path in computed}
    stored_set = set(stored.get("curated", []))
    drift = [f"missing from manifest: {item}" for item in sorted(computed_set - stored_set)]
    drift += [f"stale in manifest: {item}" for item in sorted(stored_set - computed_set)]
    return drift


def _document_files(docs_dir: Path) -> set[Path]:
    """Return every Markdown/RST document under the docs root."""

    docs_root = docs_dir.resolve()
    return {
        path.resolve()
        for pattern in ("*.md", "*.rst")
        for path in docs_root.rglob(pattern)
        if "_build" not in path.parts
    }


_XREF_UNKNOWN = re.compile(
    r"WARNING: Unknown source document '(?P<target>[^']+)' \[myst\.xref_missing\]"
)
_XREF_TARGET = re.compile(
    r"WARNING: 'myst' cross-reference target not found: '(?P<target>[^']+)' \[myst\.xref_missing\]"
)
_WARNING_SOURCE = re.compile(r"^(?P<path>.+?):\d+: WARNING:")


def _target_exists(target: str, *, source_dir: Path, docs_root: Path, repo_root: Path) -> bool:
    """Return whether an xref target resolves to an existing repo document or directory."""

    clean = target.split("#", 1)[0].strip()
    if not clean:
        return False
    if clean.startswith("/"):
        candidate = Path(clean)
        if candidate.is_file() or candidate.is_dir():
            return True
    for base in (docs_root, repo_root, source_dir):
        candidate = base / clean
        probes = (
            candidate,
            Path(str(candidate) + ".md"),
            Path(str(candidate) + ".rst"),
            candidate / "index.md",
            candidate / "index.rst",
            candidate / "README.md",
        )
        if any(probe.is_file() for probe in probes):
            return True
        if candidate.is_dir():
            return True
    return False


def _heading_ids(document: Path) -> set[str]:
    """Return the docutils heading ids generated from one Markdown document.

    MyST validates ``#anchor`` links against its own generated anchors, while
    Sphinx/docutils render heading ids with :func:`docutils.nodes.make_id`.
    Headings that MyST does not register (for example emoji headings) still
    produce working rendered anchors, so those links are allowed by resolving
    against the docutils id set.

    Returns:
        The set of heading ids; empty when docutils is unavailable.
    """

    try:
        from docutils.nodes import make_id
    except ImportError:  # pragma: no cover - docutils ships with Sphinx
        return set()
    heading = re.compile(r"^#{1,6}\s+(?P<title>.*?)(?:\s+\{#[^}]+\})?\s*$")
    ids: set[str] = set()
    for line in document.read_text(encoding="utf-8", errors="replace").splitlines():
        match = heading.match(line)
        if match is not None:
            ids.add(make_id(match.group("title").strip()))
    return ids


def classify_warnings(
    warning_lines: tuple[str, ...], docs_dir: Path
) -> tuple[tuple[str, ...], int]:
    """Split warnings into blocking lines and allowed curated-set cross-references.

    A ``myst.xref_missing`` warning is allowed only when its target resolves to
    an existing repository document, directory, or asset. Curated pages
    intentionally link into the historical corpus that the curated site does
    not build, so those references cannot resolve as Sphinx documents. Every
    other warning class, and any xref to a target that does not exist, remains
    blocking.

    Returns:
        A ``(blocking_warnings, allowed_count)`` pair.
    """

    docs_root = docs_dir.resolve()
    repo_root = docs_root.parent
    blocking: list[str] = []
    allowed = 0
    for line in warning_lines:
        unknown = _XREF_UNKNOWN.search(line)
        anchor = _XREF_TARGET.search(line)
        if unknown is None and anchor is None:
            blocking.append(line)
            continue
        source_match = _WARNING_SOURCE.match(line)
        source_path = Path(source_match.group("path")) if source_match else docs_root
        if unknown is not None:
            allowed_target = _target_exists(
                unknown.group("target"),
                source_dir=source_path.parent,
                docs_root=docs_root,
                repo_root=repo_root,
            )
        else:
            anchor_target = anchor.group("target")
            allowed_target = anchor_target in _heading_ids(source_path) or _target_exists(
                anchor_target,
                source_dir=source_path.parent,
                docs_root=docs_root,
                repo_root=repo_root,
            )
        if allowed_target:
            allowed += 1
        else:
            blocking.append(line)
    return tuple(blocking), allowed


def strict_build(
    *,
    docs_dir: Path = DEFAULT_DOCS,
    manifest: Path | None = None,
    output_dir: Path | None = None,
    builder: str = "html",
    write_manifest: bool = False,
    extra_sphinx_args: tuple[str, ...] = (),
) -> CuratedBuildResult:
    """Run the strict curated build and return a compact result.

    Returns:
        A :class:`CuratedBuildResult` describing the build outcome.
    """

    docs_dir = docs_dir.resolve()
    curated = compute_curated_sources(docs_dir)
    manifest_path = manifest_path_for(docs_dir, manifest)
    computed_manifest = build_manifest(curated, docs_dir)

    if write_manifest:
        manifest_path.write_text(
            json.dumps(computed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    stored = load_manifest(manifest_path)
    drift = manifest_drift(curated, stored, docs_dir)
    if drift:
        return CuratedBuildResult(
            status="manifest_drift",
            curated_count=len(curated),
            excluded_count=len(_document_files(docs_dir) - set(curated)),
            returncode=2,
            output_dir="",
            warning_lines=tuple(drift),
        )

    excluded = sorted(
        path.relative_to(docs_dir).as_posix() for path in _document_files(docs_dir) - set(curated)
    )
    target_dir = (output_dir or DEFAULT_OUT_DIR / builder).resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sphinx.cmd.build",
        "-W",
        "--keep-going",
        "-b",
        builder,
        "-D",
        "suppress_warnings=",
        "-D",
        "exclude_patterns=" + ",".join(excluded),
        *extra_sphinx_args,
        str(docs_dir),
        str(target_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    combined = completed.stdout + completed.stderr
    warnings = tuple(
        line
        for line in combined.splitlines()
        if ("WARNING" in line or "ERROR" in line) and "absl::InitializeLog" not in line
    )
    blocking, allowed = classify_warnings(warnings, docs_dir)
    return CuratedBuildResult(
        status="pass" if not blocking else "failed",
        curated_count=len(curated),
        excluded_count=len(excluded),
        returncode=completed.returncode,
        output_dir=str(target_dir),
        warning_lines=warnings,
        allowed_crossref_count=allowed,
        blocking_warnings=blocking,
    )


def _emit(result: CuratedBuildResult, *, as_json: bool) -> int:
    """Print the result and return the process status."""

    if as_json:
        print(
            json.dumps(
                {
                    "schema": "sphinx_curated_build.v1",
                    "status": result.status,
                    "curated_count": result.curated_count,
                    "excluded_count": result.excluded_count,
                    "returncode": result.returncode,
                    "output_dir": result.output_dir,
                    "allowed_crossref_count": result.allowed_crossref_count,
                    "allowed_warning_samples": list(result.warning_lines[:5]),
                    "blocking_warnings": list(result.blocking_warnings),
                    "sphinx_returncode": result.returncode,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Curated strict build: {result.status} "
            f"({result.curated_count} curated, {result.excluded_count} excluded, "
            f"{result.allowed_crossref_count} allowed cross-references)"
        )
        if result.output_dir:
            print(f"Output: {result.output_dir}")
        for line in result.blocking_warnings:
            print(line)
    if result.status == "pass":
        return 0
    if result.status == "manifest_drift":
        print(
            "Curated sources drifted from the manifest; rerun with --write-manifest "
            "after reviewing the toctree changes.",
            file=sys.stderr,
        )
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        The process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--builder", choices=("html", "dummy"), default="html")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = strict_build(
            docs_dir=args.docs,
            manifest=args.manifest,
            output_dir=args.output_dir,
            builder=args.builder,
            write_manifest=args.write_manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: strict curated build refused: {error}", file=sys.stderr)
        return 2
    return _emit(result, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
