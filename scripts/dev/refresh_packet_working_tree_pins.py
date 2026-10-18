#!/usr/bin/env python3
"""Refresh only mutable current-tree identities in explicitly selected source packets.

Immutable source hashes and base commits are preserved, including YAML formatting.
A dry check exits one when any existing current-tree pin needs a refresh.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROTECTED_PARTS = {
    "release",
    "releases",
    "released",
    "frozen",
    "evidence",
    "released_evidence",
    "0.0.2",
    "0.0.7",
    "0.0.8",
    "release_0_0_2",
    "release_0_0_7",
    "release_0_0_8",
}


def repository_path(root: Path, path: Path) -> Path:
    """Reject paths outside the source tree, including symlink escapes."""
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes the repository")
    return resolved


def _mapping_nodes(node, seen=None):
    """Traverse mappings once, including aliases, without following cycles."""
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, MappingNode):
        fields = {}
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.value in fields:
                raise ValueError("packet mapping keys must be unique scalars")
            fields[key.value] = value
        yield fields
        children = fields.values()
    elif isinstance(node, SequenceNode):
        children = node.value
    else:
        return
    for child in children:
        yield from _mapping_nodes(child, seen)


def _pin_replacement(root, fields, source):
    pin = fields["working_tree_sha256"]
    path = fields.get("path")
    immutable = fields.get("sha256")
    if not (
        isinstance(pin, ScalarNode)
        and _SHA256.fullmatch(pin.value)
        and isinstance(immutable, ScalarNode)
        and _SHA256.fullmatch(immutable.value)
        and isinstance(path, ScalarNode)
        and path.tag == "tag:yaml.org,2002:str"
    ):
        raise ValueError("current-tree pins require a path and two SHA256 scalars")
    spelling = source[pin.start_mark.index : pin.end_mark.index]
    # Anchored or tagged scalars could also alter an immutable alias elsewhere.
    if spelling not in (pin.value, f"'{pin.value}'", f'"{pin.value}"'):
        raise ValueError("anchored or tagged current-tree pins are refused")
    actual = hashlib.sha256(repository_path(root, Path(path.value)).read_bytes()).hexdigest()
    if actual == pin.value:
        return None
    return pin.start_mark.index, pin.end_mark.index, spelling.replace(pin.value, actual), path.value


def refreshed_bytes(root: Path, packet: Path) -> tuple[bytes, list[str]]:
    """Plan scalar replacements without serializing or changing other packet fields."""
    packet = repository_path(root, packet)
    relative = packet.relative_to(root)
    protected = any(part.lower() in _PROTECTED_PARTS for part in packet.parts)
    if protected or packet.name.endswith(("_frozen.yaml", "_frozen.yml", "_frozen.json")):
        raise ValueError(f"protected packet: {relative.as_posix()}")
    source = packet.read_bytes().decode("utf-8")
    document = yaml.compose(source, Loader=yaml.SafeLoader)
    fields = [node for node in _mapping_nodes(document) if "working_tree_sha256" in node]
    if not fields:
        raise ValueError(f"no working_tree_sha256 fields: {relative.as_posix()}")
    changes = [change for node in fields if (change := _pin_replacement(root, node, source))]
    for start, end, replacement, _ in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source.encode("utf-8"), [path for _, _, _, path in changes]


def main(argv=None) -> int:
    """Check or refresh selected packets after validating the entire request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true", help="report stale pins without writing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plans = []
    try:
        # Validate the entire request before writing any packet.
        for packet in sorted(set(args.packets)):
            content, inputs = refreshed_bytes(root, packet)
            plans.append((repository_path(root, packet), content, inputs))
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"packet pin refresh refused: {exc}\n")
    changed = False
    for packet, content, inputs in plans:
        if not inputs:
            continue
        changed = True
        print(f"{packet.relative_to(root).as_posix()}: {', '.join(inputs)}")
        if not args.check:
            packet.write_bytes(content)
    return int(args.check and changed)


if __name__ == "__main__":
    raise SystemExit(main())
