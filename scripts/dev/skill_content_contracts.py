"""Declarative skill content contracts (issue #7661).

Skill-specific Markdown content policies live as versioned YAML fixtures under
``.agents/skills/tests/contracts/<skill>.content-contract.v1.yaml`` instead of
hard-coded phrase tuples in ``scripts/dev/check_skills.py``.

Schema (strict; unknown fields are rejected):

    version: 1
    skill: <skill name, must match the file stem>
    requirements:
      - id: <stable semantic identifier>
        description: <remediation hint>
        scope: raw | lowercase | normalized
        operator: all_of | any_of
        values: [<literal that must (or must not all) appear in the text>]

``scope`` controls text normalization before matching:

- ``raw``: the text exactly as read;
- ``lowercase``: lowercased text;
- ``normalized``: lowercased text with whitespace collapsed to single spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONTRACTS_DIR = Path(".agents/skills/tests/contracts")
CONTRACT_SUFFIX = ".content-contract.v1.yaml"
ALLOWED_SCOPES = ("raw", "lowercase", "normalized")
ALLOWED_OPERATORS = ("all_of", "any_of")
ALLOWED_TOP_LEVEL_FIELDS = {"version", "skill", "requirements"}
ALLOWED_REQUIREMENT_FIELDS = {"id", "description", "scope", "operator", "values"}


class ContractError(ValueError):
    """Raised when a content-contract fixture is missing or malformed."""


def _normalize(text: str, scope: str) -> str:
    if scope == "lowercase":
        return text.lower()
    if scope == "normalized":
        return " ".join(text.lower().split())
    return text


def _validate_requirement(index: int, entry: Any) -> dict[str, Any]:
    label = f"requirement[{index}]"
    if not isinstance(entry, dict):
        raise ContractError(f"{label}: must be a mapping")
    unknown = set(entry) - ALLOWED_REQUIREMENT_FIELDS
    if unknown:
        raise ContractError(f"{label}: unknown field(s): {sorted(unknown)}")
    req_id = entry.get("id")
    if not isinstance(req_id, str) or not req_id.strip():
        raise ContractError(f"{label}: 'id' must be a non-empty string")
    label = f"requirement {req_id!r}"
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ContractError(f"{label}: 'description' must be a non-empty remediation hint")
    scope = entry.get("scope")
    if scope not in ALLOWED_SCOPES:
        raise ContractError(
            f"{label}: unknown scope {scope!r} (allowed: {', '.join(ALLOWED_SCOPES)})"
        )
    operator = entry.get("operator")
    if operator not in ALLOWED_OPERATORS:
        raise ContractError(
            f"{label}: unknown operator {operator!r} (allowed: {', '.join(ALLOWED_OPERATORS)})"
        )
    values = entry.get("values")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, str) and v for v in values)
    ):
        raise ContractError(f"{label}: 'values' must be a non-empty list of strings")
    return {
        "id": req_id,
        "description": description,
        "scope": scope,
        "operator": operator,
        "values": values,
    }


def parse_contract(raw: Any, expected_stem: str) -> dict[str, Any]:
    """Parse untyped YAML data into a validated contract mapping."""
    if not isinstance(raw, dict):
        raise ContractError("contract top level must be a mapping")
    unknown = set(raw) - ALLOWED_TOP_LEVEL_FIELDS
    if unknown:
        raise ContractError(f"unknown top-level field(s): {sorted(unknown)}")
    if raw.get("version") != 1:
        raise ContractError(f"unsupported contract version: {raw.get('version')!r} (expected 1)")
    skill = raw.get("skill")
    if not isinstance(skill, str) or not skill.strip():
        raise ContractError("'skill' must be a non-empty string")
    if skill != expected_stem:
        raise ContractError(f"'skill' {skill!r} does not match file stem {expected_stem!r}")
    raw_requirements = raw.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ContractError("'requirements' must be a non-empty list")
    requirements = [_validate_requirement(i, e) for i, e in enumerate(raw_requirements)]
    ids = [req["id"] for req in requirements]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ContractError(f"duplicate requirement id(s): {duplicates}")
    return {"version": 1, "skill": skill, "requirements": requirements}


def load_contract(repo_root: Path, skill: str) -> dict[str, Any]:
    """Load and validate the content contract for one skill.

    Raises :class:`ContractError` when no fixture exists or the fixture is invalid.
    """
    path = repo_root / CONTRACTS_DIR / f"{skill}{CONTRACT_SUFFIX}"
    if not path.is_file():
        raise ContractError(f"content contract not found: {path.relative_to(repo_root)}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"{path.relative_to(repo_root)}: YAML parse error: {exc}") from exc
    return parse_contract(raw, path.name[: -len(CONTRACT_SUFFIX)])


def has_contract(repo_root: Path, skill: str) -> bool:
    """Return whether a content-contract fixture exists for the skill."""
    return (repo_root / CONTRACTS_DIR / f"{skill}{CONTRACT_SUFFIX}").is_file()


def evaluate_contract(contract: dict[str, Any], text: str) -> list[str]:
    """Evaluate one contract against skill text; return fail-closed error strings."""
    errors: list[str] = []
    skill = contract["skill"]
    for requirement in contract["requirements"]:
        haystack = _normalize(text, requirement["scope"])
        values = requirement["values"]
        if requirement["operator"] == "all_of":
            missing = [v for v in values if v not in haystack]
            for value in missing:
                errors.append(
                    f"[{skill}] contract '{requirement['id']}': missing {value!r} "
                    f"({requirement['description']})"
                )
        elif not any(v in haystack for v in values):
            errors.append(
                f"[{skill}] contract '{requirement['id']}': missing one of "
                f"{values!r} ({requirement['description']})"
            )
    return errors


def validate_skill_text(repo_root: Path, skill: str, rel_path: Path, text: str) -> list[str]:
    """Load the skill's contract (if any) and evaluate it against the text."""
    if not has_contract(repo_root, skill):
        return []
    try:
        contract = load_contract(repo_root, skill)
    except ContractError as exc:
        return [f"{rel_path}: invalid content contract: {exc}"]
    return [f"{rel_path}: {error}" for error in evaluate_contract(contract, text)]
