#!/usr/bin/env python3
"""Validate repo-local skill metadata and index discoverability."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
README = SKILLS_ROOT / "README.md"
PATH_PATTERN = re.compile(
    r"`((?:\.agents|\.codex|\.opencode|docs|scripts|configs|tests|\.github)/[^`]+)`"
)


def _frontmatter_value(lines: list[str], key: str, path: Path) -> str:
    prefix = f"{key}: "
    for line in lines[:10]:
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().strip('"')
    raise AssertionError(f"{path}: missing frontmatter key {key!r}")


def _validate_skill_metadata(
    path: Path,
    readme_text: str,
    missing_from_readme: list[str],
    directory_mismatches: list[str],
) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise AssertionError(f"{path}: missing opening YAML frontmatter marker")
    name = _frontmatter_value(lines, "name", path)
    _frontmatter_value(lines, "description", path)
    if f"`{name}`" not in readme_text:
        missing_from_readme.append(name)
    if path.parent.name != name:
        directory_mismatches.append(f"{path.parent.name} != {name}")
    return path.read_text(encoding="utf-8")


def _reference_path(match: str) -> str:
    return match.split(maxsplit=1)[0].rstrip(".,:;)")


def _find_broken_paths(path: Path, text: str) -> list[str]:
    broken_paths: list[str] = []
    for match in PATH_PATTERN.findall(text):
        reference = _reference_path(match)
        if not (REPO_ROOT / reference).exists():
            broken_paths.append(f"{path.relative_to(REPO_ROOT)} -> {reference}")
    return broken_paths


def main() -> int:
    """Return non-zero when skill metadata or README coverage is stale."""
    readme_text = README.read_text(encoding="utf-8")
    skill_paths = sorted(SKILLS_ROOT.glob("*/SKILL.md"))
    if not skill_paths:
        raise AssertionError(f"No SKILL.md files found under {SKILLS_ROOT}")

    missing_from_readme: list[str] = []
    directory_mismatches: list[str] = []
    broken_paths = _find_broken_paths(README, readme_text)
    for path in skill_paths:
        text = _validate_skill_metadata(
            path, readme_text, missing_from_readme, directory_mismatches
        )
        broken_paths.extend(_find_broken_paths(path, text))

    if missing_from_readme:
        raise AssertionError(
            "Skills missing from .agents/skills/README.md: "
            + ", ".join(sorted(missing_from_readme))
        )
    if directory_mismatches:
        raise AssertionError(
            "Skill directory names should match frontmatter names: "
            + "; ".join(sorted(directory_mismatches))
        )
    if broken_paths:
        raise AssertionError(
            "Skill docs reference missing paths: " + "; ".join(sorted(broken_paths))
        )

    print(f"Validated {len(skill_paths)} skills and README coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
