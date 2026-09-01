#!/usr/bin/env python3
"""Resolver-aware inventory of YAML config families (issue #7901).

Scans declared YAML roots under ``configs/``, resolves ``base_config``
inheritance through the canonical resolver
(``scripts.training.train_ppo._load_expert_training_config_mapping``), and
groups near-duplicate families from stable naming plus resolved-mapping
equivalence.  Candidate bases must be byte-identical across every member's
resolved paths.  No production YAML is rewritten; no training, simulation,
benchmark, external-data, or scheduler work is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.training.train_ppo import (
    _load_expert_training_config_mapping,
)

SCHEMA = "config_family_inventory.v1"
DEFAULT_ROOTS = ("configs/algos", "configs/training", "configs/adversarial")
MIN_FAMILY_SIZE = 3
MIN_LINE_REDUCTION = 0.20

#: Config categories that use the canonical base_config resolver.
SUPPORTED_CATEGORIES = ("ppo", "adversarial", "training")
UNSUPPORTED_CATEGORY_HINTS = ("carla", "benchmark_data_release", "scenario_set")


def _digest_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _digest_mapping(mapping: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonicalized mapping."""
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_raw_yaml(path: Path) -> dict[str, Any] | None:
    """Load a YAML file as a mapping, returning ``None`` for non-mappings."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        return None
    return payload


def _inheritance_chain(path: Path, raw: dict[str, Any]) -> list[str]:
    """Return the base_config inheritance chain as relative path strings."""
    chain: list[str] = []
    current: Path | None = path
    seen: set[Path] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"base_config cycle detected at {current}")
        seen.add(current)
        chain.append(str(current))
        raw_current = _load_raw_yaml(current)
        if raw_current is None:
            break
        base_raw = raw_current.get("base_config")
        if base_raw is None:
            break
        base_path = Path(str(base_raw))
        if not base_path.is_absolute():
            base_path = current.parent / base_path
        current = base_path.resolve()
    return chain


def _category_of(raw: dict[str, Any]) -> str:
    """Classify the config category from schema/category markers."""
    for hint in ("schema_version", "config_type", "category"):
        value = raw.get(hint)
        if isinstance(value, str) and value:
            if any(unsupported in value.lower() for unsupported in UNSUPPORTED_CATEGORY_HINTS):
                return f"unsupported:{value}"
            return value
    text = json.dumps(raw, default=str)[:2000].lower()
    for hint in UNSUPPORTED_CATEGORY_HINTS:
        if hint in text:
            return f"unsupported:{hint}"
    return "ppo"


def scan_config(path: Path) -> dict[str, Any]:
    """Scan one config file and return its resolved record.

    Raises:
        ValueError: On resolver errors, cycles, or missing bases.
    """
    raw = _load_raw_yaml(path)
    if raw is None:
        return {
            "path": str(path),
            "category": "unsupported:not_a_mapping",
            "resolved": None,
            "resolved_digest": None,
            "inheritance_chain": [str(path)],
            "key_count": 0,
            "line_count": 0,
            "error": "not_a_mapping",
        }
    category = _category_of(raw)
    if category.startswith("unsupported"):
        return {
            "path": str(path),
            "category": category,
            "resolved": None,
            "resolved_digest": None,
            "inheritance_chain": [str(path)],
            "key_count": len(raw),
            "line_count": len(path.read_text(encoding="utf-8").splitlines()),
            "error": f"unsupported category {category}",
        }
    try:
        resolved = _load_expert_training_config_mapping(path)
        chain = _inheritance_chain(path, raw)
    except ValueError as exc:
        return {
            "path": str(path),
            "category": category,
            "resolved": None,
            "resolved_digest": None,
            "inheritance_chain": [str(path)],
            "key_count": len(raw),
            "line_count": len(path.read_text(encoding="utf-8").splitlines()),
            "error": str(exc),
        }
    return {
        "path": str(path),
        "category": category,
        "resolved": resolved,
        "resolved_digest": _digest_mapping(resolved),
        "inheritance_chain": chain,
        "key_count": len(resolved),
        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
        "error": None,
    }


def _family_key(path: str) -> str:
    """Derive a stable family key from the file name (strip seeds/suffixes)."""
    name = Path(path).stem
    for token in ("_seed", "_v", "_camera_ready", "_cpu", "_gpu", "_smoke"):
        if token in name:
            name = name.split(token)[0]
    return name


def _family_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records by stable family key, keeping only resolved members."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("resolved") is None:
            continue
        key = _family_key(record["path"])
        groups.setdefault(key, []).append(record)
    return groups


def _common_resolved_paths(members: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    """Return resolved key paths shared byte-identically across all members."""
    if not members:
        return set()
    common: set[tuple[str, ...]] | None = None
    for member in members:
        paths = _resolved_paths(member["resolved"])
        if common is None:
            common = paths
        else:
            common &= paths
    return common or set()


def _resolved_paths(mapping: dict[str, Any], prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Flatten a resolved mapping into key-path tuples with leaf values."""
    paths: set[tuple[str, ...]] = set()
    for key, value in mapping.items():
        key_text = str(key)
        path = (*prefix, key_text)
        if isinstance(value, dict):
            paths |= _resolved_paths(value, path)
        else:
            paths.add((*path, repr(value)))
    return paths


def _split_key_values(
    k: str,
    mappings: list[dict[str, Any]],
    base: dict[str, Any],
    leaf_overrides: list[dict[str, Any]],
) -> None:
    """Split a single key's values across member mappings into base and leaf overrides."""
    if not all(k in m for m in mappings):
        for idx, m in enumerate(mappings):
            if k in m:
                leaf_overrides[idx][k] = m[k]
        return

    values = [m[k] for m in mappings]
    if all(isinstance(v, dict) for v in values):
        sub_base, sub_leaves = _extract_common_and_overrides(values)
        if sub_base:
            base[k] = sub_base
        for idx, sub_leaf in enumerate(sub_leaves):
            if sub_leaf:
                leaf_overrides[idx][k] = sub_leaf
    elif all(v == values[0] and type(v) is type(values[0]) for v in values):
        base[k] = values[0]
    else:
        for idx, m in enumerate(mappings):
            leaf_overrides[idx][k] = m[k]


def _extract_common_and_overrides(
    mappings: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recursively split member mappings into a shared base and leaf overrides."""
    if not mappings:
        return {}, []

    base: dict[str, Any] = {}
    leaf_overrides: list[dict[str, Any]] = [{} for _ in mappings]

    all_keys = sorted({k for m in mappings for k in m.keys()})
    for k in all_keys:
        _split_key_values(k, mappings, base, leaf_overrides)

    return base, leaf_overrides


def _yaml_line_count(payload: dict[str, Any]) -> int:
    """Return the line count of a deterministically serialized YAML mapping."""
    if not payload:
        return 0
    dumped = yaml.dump(payload, sort_keys=True, default_flow_style=False)
    return len(dumped.strip().splitlines())


def run_inventory(roots: list[Path]) -> dict[str, Any]:
    """Scan config roots and return the versioned family inventory."""
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*.yaml")
        if path.is_file() and path.name != "base.yaml"
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in files:
        try:
            record = scan_config(path)
        except (OSError, ValueError) as exc:
            record = {
                "path": str(path),
                "category": "error",
                "resolved": None,
                "resolved_digest": None,
                "inheritance_chain": [str(path)],
                "key_count": 0,
                "line_count": 0,
                "error": str(exc),
            }
            errors.append(str(exc))
        records.append(record)

    groups = _family_groups(records)
    candidates: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        if len(members) < MIN_FAMILY_SIZE:
            continue
        # Skip already-migrated families: members that already declare a
        # base_config are covered by an existing base (issue #6484 family
        # records); recommending them again would duplicate ownership.
        if any(_load_raw_yaml(Path(member["path"])).get("base_config") for member in members):
            continue
        common = _common_resolved_paths(members)
        common_count = len(common)
        before_lines = sum(member["line_count"] for member in members)

        member_mappings = [m["resolved"] for m in members]
        base_mapping, leaf_overrides = _extract_common_and_overrides(member_mappings)
        estimated_base_lines = _yaml_line_count(base_mapping)
        leaf_mappings = [{"base_config": "base.yaml", **overrides} for overrides in leaf_overrides]
        estimated_leaf_lines = [_yaml_line_count(leaf) for leaf in leaf_mappings]
        estimated_after_lines = estimated_base_lines + sum(estimated_leaf_lines)
        reduction = 1.0 - (estimated_after_lines / max(before_lines, 1))
        candidates.append(
            {
                "family": key,
                "member_paths": [member["path"] for member in members],
                "member_count": len(members),
                "common_resolved_path_count": common_count,
                "before_lines": before_lines,
                "estimated_base_lines": estimated_base_lines,
                "estimated_leaf_lines": estimated_leaf_lines,
                "estimated_after_lines": estimated_after_lines,
                "estimated_reduction": round(reduction, 3),
                "risk_flags": [],
            }
        )
    ready = [
        candidate
        for candidate in candidates
        if candidate["estimated_reduction"] >= MIN_LINE_REDUCTION
        and candidate["member_count"] >= MIN_FAMILY_SIZE
    ]
    return {
        "schema": SCHEMA,
        "roots": [str(root) for root in roots],
        "scan": {
            "file_count": len(files),
            "resolved_count": sum(1 for r in records if r.get("resolved") is not None),
            "unsupported_count": sum(1 for r in records if r.get("error")),
            "resolver_error_count": len(errors),
        },
        "candidate_families": candidates,
        "ready_families": ready,
        "disposition": "one_family_ready_for_child" if ready else "no_safe_family",
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the config-family inventory CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=[Path(r) for r in DEFAULT_ROOTS])
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = run_inventory(args.roots)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown:
        print(_markdown(report))
    elif args.report is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown summary."""
    lines = [
        f"# Config-family inventory (`{report['schema']}`)",
        "",
        f"- Files: {report['scan']['file_count']} | resolved: "
        f"{report['scan']['resolved_count']} | unsupported/error: "
        f"{report['scan']['unsupported_count']}",
        f"- Candidate families (>= {MIN_FAMILY_SIZE} members): "
        f"{len(report['candidate_families'])} | ready: {len(report['ready_families'])}",
        "",
    ]
    for candidate in report["candidate_families"]:
        lines.append(
            f"- `{candidate['family']}`: {candidate['member_count']} members, "
            f"common={candidate['common_resolved_path_count']}, "
            f"before={candidate['before_lines']}, base={candidate['estimated_base_lines']}, "
            f"after={candidate['estimated_after_lines']}, "
            f"reduction={candidate['estimated_reduction']:.1%}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
