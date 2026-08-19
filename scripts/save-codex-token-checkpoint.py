#!/usr/bin/env python3
"""Build a compact token-saving checkpoint without duplicating route policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

CANONICAL_ROUTE_POINTER = (
    "https://github.com/ll7/codex-personal-skills/blob/main/scripts/resolve-route.py"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-class", default="unknown", help="Canonical task class, if known.")
    parser.add_argument(
        "--prompt", default="", help="Short task description used only for the checkpoint."
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--json", action="store_true", help="Alias for --format json.")
    return parser


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _entrypoint_status(name: str) -> dict[str, object]:
    configured = os.environ.get("CODEX_ROUTING_REPO", "").strip()
    if not configured:
        return {"status": "unavailable", "reason": "CODEX_ROUTING_REPO is not set"}
    script = Path(configured).expanduser() / "scripts" / name
    if not script.is_file():
        return {"status": "unavailable", "reason": f"canonical entrypoint is missing: {script}"}
    return {"status": "available", "path": str(script.resolve())}


def _build_checkpoint(task_class: str, prompt: str) -> dict[str, object]:
    common_git_dir = _git_value("rev-parse", "--path-format=absolute", "--git-common-dir")
    ledger_dir = (
        str(Path(common_git_dir) / "codex-agent-runs" / "active") if common_git_dir else None
    )
    route = _entrypoint_status("resolve-route.py")
    advisor = _entrypoint_status("advise-provider-routing.py")
    route_status = "available" if route["status"] == "available" else "unavailable"
    return {
        "schema_version": "token_saving_checkpoint.v1",
        "task_class": task_class,
        "prompt_excerpt": prompt[:240],
        "prompt_length": len(prompt),
        "repository": {
            "head": _git_value("rev-parse", "HEAD"),
            "origin_main": _git_value("rev-parse", "origin/main"),
            "common_git_dir": common_git_dir,
        },
        "route": {
            "status": route_status,
            "resolver": route,
            "provider_advice": advisor,
            "canonical_pointer": CANONICAL_ROUTE_POINTER,
            "route_evidence_only": True,
        },
        "active_ledger": {
            "status": "available" if ledger_dir else "unavailable",
            "path": ledger_dir,
        },
        "recommended_commands": [
            "python3 scripts/advise-provider-routing.py --json",
            "python3 scripts/read-active-ledger.py --json --limit 1",
            "python3 scripts/resolve-route.py --help",
        ],
        "acceptance_gate": [
            "route selection is evidence only",
            "the controller reviews the exact diff and local validation",
            "do not invent a provider or model when the canonical route is unavailable",
        ],
        "next_action": (
            "use the canonical shared resolver"
            if route_status == "available"
            else "continue conservatively with compact local snapshots and record route-unavailable"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Emit the checkpoint in the requested compact format."""
    args = _parser().parse_args(argv)
    checkpoint = _build_checkpoint(args.task_class, args.prompt)
    if args.json or args.format == "json":
        print(json.dumps(checkpoint, indent=2, sort_keys=True))
        return 0
    print("token_saving_checkpoint.v1")
    print(f"task_class={checkpoint['task_class']}")
    print(f"route_status={checkpoint['route']['status']}")
    print(f"active_ledger={checkpoint['active_ledger']['path']}")
    print("recommended_commands:")
    for command in checkpoint["recommended_commands"]:
        print(f"- {command}")
    print(f"next_action={checkpoint['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
