#!/usr/bin/env python3
"""Build the machine-readable run-ready launch-packet manifest (#3549).

Plain-language summary: scan the repository's SLURM launch packets, preflight each one (drift guard +
seed resolve + config existence + claim-gate presence via ``preflight_launch_packet``), and emit a
single manifest the private autonomous-queue discovery can read to propose queue entries. This is the
public half of closing the throughput gap; the private companions are the scheduler and queue
auto-population (robot_sf_ll7-private-ops #2 / #3).

The manifest reports *preflight* readiness only. It asserts no benchmark/paper-grade result and never
submits anything; submission stays gated by the private engine's budget + fair-use rules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.dev.preflight_launch_packet import (
    SCHEMA_VERSION as PREFLIGHT_SCHEMA_VERSION,
)
from scripts.dev.preflight_launch_packet import (
    _repo_root,
    preflight_packet,
)

MANIFEST_SCHEMA_VERSION = "run-ready-launch-packet-manifest.v1"
DEFAULT_GLOB = "configs/**/*launch_packet*.yaml"
DEFAULT_OUTPUT = "experiments/run_ready_manifest.yaml"


def discover_packets(repo_root: Path, glob: str = DEFAULT_GLOB) -> list[Path]:
    """Return the sorted unique list of launch-packet files under ``repo_root``."""
    return sorted({p.resolve() for p in repo_root.glob(glob) if p.is_file()})


def _git_head(repo_root: Path) -> str | None:
    """Return the short git HEAD sha for provenance, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def build_manifest(repo_root: Path, glob: str = DEFAULT_GLOB) -> dict[str, Any]:
    """Preflight every discovered packet and assemble the manifest dict."""
    packets = discover_packets(repo_root, glob)
    entries = [preflight_packet(p, repo_root) for p in packets]
    entries.sort(key=lambda e: (e.get("issue") is None, e.get("issue") or 0, e["packet_path"]))
    ready = [e for e in entries if e["ready"]]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_from_commit": _git_head(repo_root),
        "packet_glob": glob,
        "summary": {
            "total": len(entries),
            "ready": len(ready),
            "drift_guarded": sum(1 for e in entries if e.get("drift_guarded")),
        },
        "packets": entries,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build the manifest and write it to disk (yaml or json)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Repository root (auto-detected)."
    )
    parser.add_argument(
        "--glob", type=str, default=DEFAULT_GLOB, help="Launch-packet glob pattern."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path (default: {DEFAULT_OUTPUT}). Use '-' for stdout.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of YAML.")
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or _repo_root(Path.cwd())).resolve()
    manifest = build_manifest(repo_root, args.glob)

    text = (
        json.dumps(manifest, indent=2, sort_keys=True)
        if args.json
        else yaml.safe_dump(manifest, sort_keys=False, width=100)
    )

    if args.output is not None and str(args.output) == "-":
        print(text)
    else:
        out_path = (args.output or (repo_root / DEFAULT_OUTPUT)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text if text.endswith("\n") else text + "\n")
        s = manifest["summary"]
        print(
            f"wrote {out_path.relative_to(repo_root)} — "
            f"{s['ready']}/{s['total']} ready ({s['drift_guarded']} drift-guarded)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
