#!/usr/bin/env python3
"""Forward route resolution to the canonical shared resolver without copying its policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CANONICAL_POINTER = (
    "https://github.com/ll7/codex-personal-skills/blob/main/scripts/resolve-route.py"
)


def _canonical_script() -> tuple[Path | None, str | None]:
    """Return the configured canonical resolver or a safe unavailable reason."""
    configured = os.environ.get("CODEX_ROUTING_REPO", "").strip()
    if not configured:
        return None, "CODEX_ROUTING_REPO is not set"
    root = Path(configured).expanduser()
    if not root.is_dir():
        return None, f"CODEX_ROUTING_REPO is not a directory: {root}"
    script = (root / "scripts" / "resolve-route.py").resolve()
    if not script.is_file():
        return None, f"canonical resolver is missing: {script}"
    if script == Path(__file__).resolve():
        return None, "CODEX_ROUTING_REPO points to this compatibility wrapper"
    return script, None


def _requested_output(arguments: list[str]) -> Path | None:
    """Find an optional shared-resolver ``--out`` destination."""
    for index, argument in enumerate(arguments):
        if argument == "--out" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith("--out="):
            return Path(argument.partition("=")[2])
    return None


def _unavailable_report(reason: str) -> dict[str, object]:
    return {
        "schema_version": "route_resolution.v1",
        "status": "unavailable",
        "route_evidence_only": True,
        "canonical_pointer": CANONICAL_POINTER,
        "reason": reason,
        "next_action": "configure CODEX_ROUTING_REPO or continue with conservative local routing",
    }


def main(argv: list[str] | None = None) -> int:
    """Resolve a route or emit a machine-readable unavailable state."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    script, reason = _canonical_script()
    if script is not None:
        completed = subprocess.run([sys.executable, str(script), *arguments], check=False)
        return completed.returncode
    if any(argument in {"-h", "--help"} for argument in arguments):
        print(f"Usage: {Path(__file__).name} [canonical-resolver-options]")
        print("Set CODEX_ROUTING_REPO to delegate to the shared route resolver.")
        return 0

    report = _unavailable_report(reason or "canonical resolver unavailable")
    output_path = _requested_output(arguments)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
