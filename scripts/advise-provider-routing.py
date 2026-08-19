#!/usr/bin/env python3
"""Forward provider advice to the canonical shared advisor without local route tables."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CANONICAL_POINTER = (
    "https://github.com/ll7/codex-personal-skills/blob/main/scripts/advise-provider-routing.py"
)


def _canonical_script() -> tuple[Path | None, str | None]:
    """Return the configured canonical advisor or a safe unavailable reason."""
    configured = os.environ.get("CODEX_ROUTING_REPO", "").strip()
    if not configured:
        return None, "CODEX_ROUTING_REPO is not set"
    root = Path(configured).expanduser()
    if not root.is_dir():
        return None, f"CODEX_ROUTING_REPO is not a directory: {root}"
    script = (root / "scripts" / "advise-provider-routing.py").resolve()
    if not script.is_file():
        return None, f"canonical provider advisor is missing: {script}"
    if script == Path(__file__).resolve():
        return None, "CODEX_ROUTING_REPO points to this compatibility wrapper"
    return script, None


def main(argv: list[str] | None = None) -> int:
    """Advise on routing or emit a machine-readable unavailable state."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    script, reason = _canonical_script()
    if script is not None:
        completed = subprocess.run([sys.executable, str(script), *arguments], check=False)
        return completed.returncode
    if any(argument in {"-h", "--help"} for argument in arguments):
        print(f"Usage: {Path(__file__).name} [canonical-advisor-options]")
        print("Set CODEX_ROUTING_REPO to delegate to the shared provider advisor.")
        return 0

    report = {
        "schema_version": "provider_routing_advice.v1",
        "status": "unavailable",
        "route_evidence_only": True,
        "canonical_pointer": CANONICAL_POINTER,
        "reason": reason or "canonical provider advisor unavailable",
        "next_action": "continue conservatively; do not invent a provider or model route",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
