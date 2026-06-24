#!/usr/bin/env python3
"""Preflight a SLURM launch packet and report machine-readable run-readiness (#3549).

Plain-language summary: a "launch packet" is a YAML file that pins the exact configs, seed budget,
and command a SLURM campaign should run (e.g. the #3216 headline packet). This tool checks whether a
packet is *safe to submit right now* — every pinned config still exists and its sha256 still matches
(drift guard), the seed budget resolves, a runnable command is present, and a claim/evidence boundary
is declared — and emits a structured result the autonomous queue discovery can consume.

It is honest and fail-closed: launch packets use heterogeneous schemas, so anything this tool cannot
verify (drifted sha, missing config, no command, no claim gate) yields ``ready: false`` with an
explicit reason. It NEVER submits anything and makes no benchmark/paper-grade claim.

Generalizes the inline drift-guard + seed-resolve checks in
``scripts/benchmark/run_issue3216_headline_campaign.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "run-ready-launch-packet-preflight.v1"

# Scalar string values that look like in-repo asset paths we can existence-check.
_PATH_RE = re.compile(r"^(configs|scripts|SLURM)/[\w./-]+\.(ya?ml|sh|sl|py)$")
# Seed-set tokens like ``paper_eval_s20`` we can resolve to a seed count.
_SEED_SET_RE = re.compile(r"\b(paper_eval_s\d+|[\w]*_s\d+)\b")
_SEED_SETS_FILE = "configs/benchmarks/seed_sets_v1.yaml"

# Keys whose presence indicates a runnable command / payload.
_COMMAND_KEYS = (
    "campaign_command",
    "report_command",
    "validation_command",
    "slurm_command_shape",
)
# Keys whose presence indicates a declared claim / evidence boundary.
_CLAIM_GATE_KEYS = (
    "claim_gate",
    "claim_map_gate",
    "claim_boundary",
    "no_claim_until_run",
    "no_claim_statement",
    "evidence_status",
    "execution_boundary",
)


def sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk(node: Any) -> Iterable[Any]:
    """Yield every mapping and scalar reachable from ``node`` (depth-first)."""
    yield node
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk(value)


def find_sha_pairs(packet: Any) -> list[dict[str, str]]:
    """Find ``<key>`` / ``<key>_sha256`` sibling pairs anywhere in the packet.

    Returns a list of ``{"path": ..., "expected_sha256": ...}`` dicts.
    """
    pairs: list[dict[str, str]] = []
    for node in _walk(packet):
        if not isinstance(node, Mapping):
            continue
        for key, value in node.items():
            sha_key = f"{key}_sha256"
            if isinstance(value, str) and sha_key in node and isinstance(node[sha_key], str):
                pairs.append({"path": value, "expected_sha256": node[sha_key]})
    return pairs


def find_config_paths(packet: Any) -> list[str]:
    """Return the sorted unique set of in-repo asset paths referenced as scalars."""
    found: set[str] = set()
    for node in _walk(packet):
        if isinstance(node, str) and _PATH_RE.match(node):
            found.add(node)
    return sorted(found)


def _has_key(packet: Any, keys: Iterable[str]) -> list[str]:
    """Return which of ``keys`` appear anywhere in the packet mappings."""
    present: set[str] = set()
    keyset = set(keys)
    for node in _walk(packet):
        if isinstance(node, Mapping):
            present.update(keyset.intersection(node.keys()))
    return sorted(present)


def _issue_from_path(packet_path: Path) -> int | None:
    """Parse an issue number from a packet filename like ``..._issue_3216_...``."""
    m = re.search(r"issue[_-]?(\d+)", packet_path.name)
    return int(m.group(1)) if m else None


def _kind_from_path(packet_path: Path) -> str:
    """Infer a coarse packet kind from its location."""
    parts = packet_path.parts
    if "benchmarks" in parts or "benchmark" in parts:
        return "benchmark"
    if "training" in parts:
        return "training"
    return "unknown"


def _deep_find(node: Any, name: str) -> Any:
    """Return the first value stored under key ``name`` anywhere in ``node``."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == name:
                return value
            hit = _deep_find(value, name)
            if hit is not None:
                return hit
    elif isinstance(node, (list, tuple)):
        for value in node:
            hit = _deep_find(value, name)
            if hit is not None:
                return hit
    return None


def _resolve_seed_count(packet: Any, repo_root: Path) -> dict[str, Any]:
    """Best-effort resolve a declared seed-set token to a concrete seed count."""
    text = yaml.safe_dump(packet)
    seeds_file = repo_root / _SEED_SETS_FILE
    tokens = sorted({m.group(1) for m in _SEED_SET_RE.finditer(text)})
    if not tokens or not seeds_file.exists():
        return {"seed_set": tokens[0] if tokens else None, "seed_count": None}
    seed_sets = yaml.safe_load(seeds_file.read_text()) or {}

    for token in tokens:
        resolved = _deep_find(seed_sets, token)
        seeds = resolved.get("seeds") if isinstance(resolved, Mapping) else resolved
        if isinstance(seeds, list):
            return {"seed_set": token, "seed_count": len(seeds)}
    return {"seed_set": tokens[0], "seed_count": None}


def preflight_packet(packet_path: Path, repo_root: Path) -> dict[str, Any]:
    """Validate one launch packet and return a structured readiness report.

    Args:
        packet_path: Path to the launch-packet YAML.
        repo_root: Repository root used to resolve referenced asset paths.

    Returns:
        A dict with readiness fields; ``ready`` is True only when all pinned configs exist,
        all declared sha256 pairs match, a command and a claim gate are present, and at least
        one config path is referenced.
    """
    reasons: list[str] = []
    try:
        packet = yaml.safe_load(packet_path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        return {
            "issue": _issue_from_path(packet_path),
            "packet_path": str(packet_path.relative_to(repo_root))
            if packet_path.is_absolute()
            else str(packet_path),
            "ready": False,
            "reasons": [f"unparseable packet: {exc}"],
        }

    rel_packet = (
        str(packet_path.relative_to(repo_root)) if packet_path.is_absolute() else str(packet_path)
    )
    schema_version = packet.get("schema_version") if isinstance(packet, Mapping) else None

    # sha-pinned config pairs (drift guard).
    sha_pairs = find_sha_pairs(packet)
    checked_pairs: list[dict[str, Any]] = []
    for pair in sha_pairs:
        target = repo_root / pair["path"]
        if not target.exists():
            checked_pairs.append({**pair, "actual_sha256": None, "ok": False})
            reasons.append(f"sha-pinned config missing: {pair['path']}")
            continue
        actual = sha256_of(target)
        ok = actual == pair["expected_sha256"]
        checked_pairs.append({**pair, "actual_sha256": actual, "ok": ok})
        if not ok:
            reasons.append(f"sha256 drift: {pair['path']}")

    # All referenced in-repo asset paths exist.
    config_paths = find_config_paths(packet)
    checked_paths: list[dict[str, Any]] = []
    for rel in config_paths:
        exists = (repo_root / rel).exists()
        checked_paths.append({"path": rel, "exists": exists})
        if not exists:
            reasons.append(f"referenced path missing: {rel}")

    if not config_paths:
        reasons.append("no in-repo config paths referenced")

    command_keys = _has_key(packet, _COMMAND_KEYS)
    if not command_keys:
        reasons.append("no runnable command/payload key present")

    claim_gate_keys = _has_key(packet, _CLAIM_GATE_KEYS)
    if not claim_gate_keys:
        reasons.append("no claim/evidence boundary declared")

    seed_info = _resolve_seed_count(packet, repo_root)
    seed_budget_present = bool(
        _has_key(packet, ("seed_budget", "seed_policy", "seeds")) or seed_info["seed_set"]
    )

    drift_guarded = bool(checked_pairs) and all(p["ok"] for p in checked_pairs)
    ready = (
        all(p["ok"] for p in checked_pairs)
        and all(p["exists"] for p in checked_paths)
        and bool(config_paths)
        and bool(command_keys)
        and bool(claim_gate_keys)
    )

    return {
        "issue": _issue_from_path(packet_path),
        "packet_path": rel_packet,
        "schema_version": schema_version,
        "kind": _kind_from_path(packet_path),
        "ready": ready,
        "drift_guarded": drift_guarded,
        "reasons": reasons,
        "sha_pairs": checked_pairs,
        "config_paths": checked_paths,
        "seed_budget_present": seed_budget_present,
        "seed_set": seed_info["seed_set"],
        "seed_count": seed_info["seed_count"],
        "command_keys": command_keys,
        "claim_gate_keys": claim_gate_keys,
    }


def _repo_root(start: Path) -> Path:
    """Find the repository root by walking up to a directory containing ``configs``."""
    for candidate in [start, *start.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return start


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: preflight one packet and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", type=Path, required=True, help="Path to the launch-packet YAML."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Repository root (auto-detected)."
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the packet is not run-ready.",
    )
    args = parser.parse_args(argv)

    packet_path = args.packet.resolve()
    repo_root = (args.repo_root or _repo_root(packet_path.parent)).resolve()
    report = preflight_packet(packet_path, repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and not report["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
