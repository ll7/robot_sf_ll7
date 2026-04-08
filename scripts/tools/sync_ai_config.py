#!/usr/bin/env python3
"""Validate or repair repository AI assistant configuration links."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LinkSpec:
    """Expected compatibility symlink from a tool-specific path to `.agents`."""

    path: str
    target: str


LINK_SPECS = (
    LinkSpec(".codex/skills", "../.agents/skills"),
    LinkSpec(".opencode/skills", "../.agents/skills"),
    LinkSpec(".claude/skills", "../.agents/skills"),
    LinkSpec(".codex/prompts", "../.agents/prompts/codex"),
    LinkSpec(".github/prompts", "../.agents/prompts/github"),
    LinkSpec(".github/agents", "../.agents/agents/github"),
    LinkSpec(".gemini/commands", "../.agents/commands/gemini"),
)


def _repo_path(path: str) -> Path:
    """Resolve a repo-relative path and reject paths that escape the repository root."""
    candidate = REPO_ROOT / path
    if not candidate.parent.resolve(strict=False).is_relative_to(REPO_ROOT):
        raise ValueError(f"Path escapes repository root: {path}")
    return candidate


def _check_link(spec: LinkSpec, *, fix: bool) -> list[str]:
    """Check one symlink spec, optionally repairing missing or stale symlinks."""
    path = _repo_path(spec.path)
    target = Path(spec.target)
    expected_abs = (path.parent / target).resolve()
    if not expected_abs.is_relative_to(REPO_ROOT):
        raise ValueError(f"Symlink target escapes repository root: {spec.path} -> {spec.target}")
    errors: list[str] = []

    if path.is_symlink():
        if not expected_abs.exists():
            errors.append(
                f"{spec.path}: target does not exist: {expected_abs.relative_to(REPO_ROOT)}"
            )
            return errors
        if path.readlink() == target and path.resolve() == expected_abs:
            return errors
        if fix:
            path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
            return errors
        errors.append(f"{spec.path}: expected symlink to {spec.target}, got {path.readlink()}")
        return errors

    if path.exists():
        errors.append(f"{spec.path}: exists but is not a symlink")
        return errors

    if fix:
        if not expected_abs.exists():
            errors.append(
                f"{spec.path}: target does not exist: {expected_abs.relative_to(REPO_ROOT)}"
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
    else:
        errors.append(f"{spec.path}: missing symlink to {spec.target}")
    return errors


def _check_pointer_file(path: str, expected_text: str) -> list[str]:
    """Check that a small tool-specific pointer file references the canonical source."""
    pointer_path = _repo_path(path)
    if not pointer_path.exists():
        return [f"{path}: missing pointer file"]
    if expected_text not in pointer_path.read_text(encoding="utf-8"):
        return [f"{path}: expected to reference {expected_text!r}"]
    return []


def main() -> int:
    """Return non-zero when supported AI configuration surfaces drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="validate without changing files")
    group.add_argument("--fix", action="store_true", help="repair supported symlinks")
    args = parser.parse_args()

    fix = args.fix
    errors: list[str] = []
    for spec in LINK_SPECS:
        errors.extend(_check_link(spec, fix=fix))

    errors.extend(_check_pointer_file(".cursorrules", "AGENTS.md"))
    errors.extend(_check_pointer_file(".github/copilot-instructions.md", "AGENTS.md"))

    if errors:
        raise SystemExit("AI config drift detected:\n- " + "\n- ".join(errors))

    mode = "repaired" if fix else "validated"
    print(f"AI config links {mode}: {len(LINK_SPECS)} symlinks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
