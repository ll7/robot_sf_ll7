#!/usr/bin/env python3
"""Resolve generated-file merge conflicts in ``docs/context/catalog.yaml`` and ``INDEX.md``.

Why this exists
---------------
``docs/context/catalog.yaml`` and ``docs/context/INDEX.md`` are append-maintained
generated indexes: every evidence/context PR adds a distinct entry, so two PRs that
land near each other produce a content merge conflict on the same file. The correct
resolution is mechanical (keep both sides' entries), but doing it by hand is
error-prone -- an entry can lose its ``status``/``freshness`` fields or be duplicated,
and only ``check_docs_evidence_integrity`` catches it downstream (issue #6781).

This helper parses both sides of a git conflict and emits the de-duplicated union:

- ``catalog.yaml``: union of the ``entries:`` list, de-duplicated by ``path``, each
  entry kept verbatim (field-complete), sorted by path; the header is preserved.
- ``INDEX.md``: union of the body paragraphs, de-duplicated by content; the header
  and original paragraph order are preserved (HEAD order first, then THEIRS-only
  additions).

Usage
-----
After a merge leaves a conflict::

    uv run python scripts/dev/resolve_context_catalog_conflict.py docs/context/catalog.yaml
    uv run python scripts/dev/resolve_context_catalog_conflict.py docs/context/INDEX.md

The file is rewritten in place with the resolved union and the conflict markers
removed. Re-run ``check_docs_evidence_integrity`` and ``git add`` afterwards. The
helper refuses to overwrite a file that has no conflict markers (so it cannot
clobber a clean file) and exits nonzero on a file with unbalanced markers.

This is a deterministic text resolver, not a policy change: it never invents
entries, fields, or provenance -- it only keeps what either side already declared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEAD_MARK = "<<<<<<<"
SEP_MARK = "======="
THEIRS_MARK = ">>>>>>>"


def _split_conflict_sides(text: str) -> tuple[str, str]:
    """Reconstruct the HEAD-only and THEIRS-only documents from a conflicted file.

    Common (non-conflict) lines are included in both sides; HEAD-side lines only in
    the HEAD document; THEIRS-side lines only in the THEIRS document.
    """
    head_lines: list[str] = []
    theirs_lines: list[str] = []
    state = "common"
    seen_head = seen_sep = seen_theirs = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(HEAD_MARK):
            state = "head"
            seen_head = True
            continue
        if stripped.startswith(THEIRS_MARK):
            state = "common"
            seen_theirs = True
            continue
        if stripped.startswith(SEP_MARK):
            if state != "head":
                raise ValueError("conflict separator '=======' without a preceding HEAD marker")
            state = "theirs"
            seen_sep = True
            continue
        if state == "common":
            head_lines.append(line)
            theirs_lines.append(line)
        elif state == "head":
            head_lines.append(line)
        else:  # theirs
            theirs_lines.append(line)
    if seen_head and not seen_sep:
        raise ValueError("unbalanced conflict markers: HEAD marker without a separator")
    if seen_sep and not seen_theirs:
        raise ValueError("unbalanced conflict markers: separator without a THEIRS marker")
    return "".join(head_lines), "".join(theirs_lines)


def _entry_path(block: str) -> str | None:
    """Return the ``path:`` value from a single catalog entry block, or None."""
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- path:") or stripped.startswith("path:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return None


def _parse_catalog_entries(doc: str) -> tuple[str, list[str], str]:
    """Split a catalog.yaml doc into (pre, entry_blocks, post) around the entries list.

    ``pre`` ends at the ``entries:`` line (inclusive); ``post`` is anything after the
    final entry block (normally empty). Each entry block keeps its trailing newline.
    """
    lines = doc.splitlines(keepends=True)
    try:
        entries_idx = next(i for i, line in enumerate(lines) if line.strip() == "entries:")
    except StopIteration:
        return doc, [], ""
    pre = "".join(lines[: entries_idx + 1])
    blocks, post_start = _collect_entry_blocks(lines, entries_idx + 1)
    post = "".join(lines[post_start:]) if post_start is not None else ""
    return pre, blocks, post


def _collect_entry_blocks(lines: list[str], start: int) -> tuple[list[str], int | None]:
    """Collect ``- path:`` entry blocks starting at ``start``; return (blocks, post_start).

    A top-level key (column 0, not ``- ``) after the entries ends the list; its index is
    returned as ``post_start`` (or None if entries run to end of document).
    """
    blocks: list[str] = []
    current: list[str] = []
    post_start: int | None = None
    for offset in range(start, len(lines)):
        line = lines[offset]
        if line.startswith("- "):
            if current:
                blocks.append("".join(current))
            current = [line]
        elif current and (line[:1] in (" ", "\t") or line.strip() == ""):
            current.append(line)
        elif current:
            blocks.append("".join(current))
            current = []
            post_start = offset
            break
    if current:
        blocks.append("".join(current))
    return blocks, post_start


def resolve_catalog(head_doc: str, theirs_doc: str) -> str:
    """Return a catalog.yaml doc with the union of both sides' entries."""
    head_pre, head_blocks, head_post = _parse_catalog_entries(head_doc)
    _, theirs_blocks, theirs_post = _parse_catalog_entries(theirs_doc)
    merged: dict[str, str] = {}
    for block in head_blocks + theirs_blocks:
        path = _entry_path(block)
        key = path or block  # fall back to the raw block if no path (keeps malformed entries)
        if key not in merged:
            merged[key] = block
    ordered = [merged[k] for k in sorted(merged)]
    return head_pre + "".join(ordered) + (head_post or theirs_post)


def _split_paragraphs(doc: str) -> tuple[str, list[str]]:
    """Split an INDEX.md doc into (header, body_paragraphs).

    ``header`` is the leading ``# Context Retrieval Index`` title line plus the first
    blank line. Body paragraphs are non-empty line groups separated by blank lines,
    each kept verbatim (including its trailing separator).
    """
    lines = doc.splitlines(keepends=True)
    header_end = 0
    for idx, line in enumerate(lines):
        if line.strip() == "":
            header_end = idx + 1
            break
    else:
        header_end = len(lines)
    header = "".join(lines[:header_end])
    body = "".join(lines[header_end:])
    paragraphs = [p for p in (body.split("\n\n")) if p.strip()]
    # Re-attach the double-newline separators lost by split.
    normalized = [(p if p.endswith("\n") else p + "\n") for p in paragraphs]
    return header, normalized


def resolve_index(head_doc: str, theirs_doc: str) -> str:
    """Return an INDEX.md doc with the union of both sides' body paragraphs."""
    header, head_paras = _split_paragraphs(head_doc)
    _, theirs_paras = _split_paragraphs(theirs_doc)
    seen: set[str] = set()
    ordered: list[str] = []
    for para in head_paras + theirs_paras:
        key = para.strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(para)
    return header + "\n\n".join(p.rstrip("\n") for p in ordered) + "\n"


def resolve_conflict_file(path: Path) -> str:
    """Resolve a conflicted file in place; return the resolved text (also written)."""
    text = path.read_text(encoding="utf-8")
    if HEAD_MARK not in text:
        raise ValueError(f"{path}: no conflict markers found; refusing to overwrite a clean file")
    head_doc, theirs_doc = _split_conflict_sides(text)
    name = path.name
    if name == "catalog.yaml":
        resolved = resolve_catalog(head_doc, theirs_doc)
    elif name == "INDEX.md":
        resolved = resolve_index(head_doc, theirs_doc)
    else:
        raise ValueError(f"{path}: unsupported file (expected catalog.yaml or INDEX.md)")
    path.write_text(resolved, encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """CLI entry: resolve a conflicted catalog.yaml/INDEX.md file in place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "file", type=Path, help="conflicted file (docs/context/catalog.yaml or INDEX.md)"
    )
    args = parser.parse_args(argv)
    try:
        resolve_conflict_file(args.file)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"resolved: {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
