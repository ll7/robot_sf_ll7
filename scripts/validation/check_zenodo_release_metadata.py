"""Validate repository-controlled metadata used by Zenodo GitHub releases.

The root ``.zenodo.json`` file overrides metadata inferred from a GitHub release.
This check intentionally stays offline: it validates the documented Zenodo field
contract and binds the release metadata to ``CITATION.cff`` and ``pyproject.toml``.
It does not resolve a DOI, inspect Zenodo, or authorize publication.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZENODO = REPO_ROOT / ".zenodo.json"
DEFAULT_CITATION = REPO_ROOT / "CITATION.cff"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"

# These are the top-level properties from Zenodo's official legacy deposit
# schema. Keeping the allow-list local makes the release gate deterministic and
# avoids downloading a schema during CI.
ZENODO_PROPERTIES = frozenset(
    {
        "$schema",
        "access_conditions",
        "access_right",
        "communities",
        "conference_acronym",
        "conference_dates",
        "conference_place",
        "conference_session",
        "conference_session_part",
        "conference_title",
        "conference_url",
        "contributors",
        "creators",
        "description",
        "doi",
        "embargo_date",
        "grants",
        "image_type",
        "imprint_isbn",
        "imprint_place",
        "imprint_publisher",
        "journal_issue",
        "journal_pages",
        "journal_title",
        "journal_volume",
        "keywords",
        "license",
        "notes",
        "openaire_type",
        "partof_pages",
        "partof_title",
        "publication_date",
        "publication_type",
        "references",
        "related_identifiers",
        "subjects",
        "thesis_supervisors",
        "thesis_university",
        "title",
        "upload_type",
    }
)
ZENODO_REQUIRED = frozenset(
    {"access_right", "creators", "description", "license", "title", "upload_type"}
)
ZENODO_CREATOR_PROPERTIES = frozenset({"affiliation", "gnd", "name", "orcid"})
ZENODO_UPLOAD_TYPES = frozenset(
    {
        "dataset",
        "image",
        "lesson",
        "other",
        "physicalobject",
        "poster",
        "presentation",
        "publication",
        "software",
        "video",
        "workflow",
    }
)
ZENODO_ACCESS_RIGHTS = frozenset({"closed", "embargoed", "open", "restricted"})


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a JSON object and return deterministic validation errors."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{label} is unreadable or invalid JSON: {path} ({exc})"]
    if not isinstance(payload, Mapping):
        return None, [f"{label} must be a JSON object: {path}"]
    return dict(payload), []


def _load_yaml_object(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a YAML mapping and return deterministic validation errors."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"{label} is unreadable or invalid YAML: {path} ({exc})"]
    if not isinstance(payload, Mapping):
        return None, [f"{label} must be a YAML mapping: {path}"]
    return dict(payload), []


def _load_toml_object(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a TOML mapping and return deterministic validation errors."""

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, [f"{label} is unreadable or invalid TOML: {path} ({exc})"]
    if not isinstance(payload, Mapping):
        return None, [f"{label} must be a TOML mapping: {path}"]
    return dict(payload), []


def _text(value: Any, label: str, errors: list[str]) -> str | None:
    """Return a non-empty string or record a missing/malformed field."""

    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return None
    return value.strip()


def _name_signature(value: str) -> tuple[str, ...]:
    """Normalize a personal name while accepting Zenodo family-first format."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return tuple(sorted(tokens))


def _citation_author_names(citation: Mapping[str, Any], errors: list[str]) -> list[str]:
    """Read creator names from the citation metadata."""

    authors = citation.get("authors")
    if not isinstance(authors, list) or not authors:
        errors.append("CITATION.cff authors must be a non-empty list")
        return []

    names: list[str] = []
    for index, author in enumerate(authors):
        if not isinstance(author, Mapping):
            errors.append(f"CITATION.cff authors[{index}] must be a mapping")
            continue
        name = author.get("name")
        if not isinstance(name, str) or not name.strip():
            given = author.get("given-names")
            family = author.get("family-names")
            if isinstance(given, str) and isinstance(family, str):
                name = f"{given} {family}"
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        else:
            errors.append(f"CITATION.cff authors[{index}] lacks a usable name")
    return names


def _project_author_names(project: Mapping[str, Any], errors: list[str]) -> list[str]:
    """Read creator names from the Python project metadata."""

    authors = project.get("authors")
    if not isinstance(authors, list) or not authors:
        errors.append("pyproject.toml [project].authors must be a non-empty list")
        return []

    names: list[str] = []
    for index, author in enumerate(authors):
        if not isinstance(author, Mapping):
            errors.append(f"pyproject.toml [project].authors[{index}] must be a mapping")
            continue
        name = author.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        else:
            errors.append(f"pyproject.toml [project].authors[{index}] lacks a usable name")
    return names


def _source_author_names(
    citation: Mapping[str, Any], project: Mapping[str, Any], errors: list[str]
) -> list[str]:
    """Read creator names from both authoritative repository metadata surfaces."""

    return _citation_author_names(citation, errors) + _project_author_names(project, errors)


def _citation_text(citation: Mapping[str, Any], field: str, errors: list[str]) -> str | None:
    """Read a required text field from CFF metadata."""

    return _text(citation.get(field), f"CITATION.cff {field!r}", errors)


def _validate_zenodo_shape(zenodo: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the closed-world top-level shape and required fields."""

    unknown = sorted(set(zenodo) - ZENODO_PROPERTIES)
    if unknown:
        errors.append(f".zenodo.json contains unsupported fields: {', '.join(unknown)}")
    missing = sorted(ZENODO_REQUIRED - set(zenodo))
    if missing:
        errors.append(f".zenodo.json is missing required fields: {', '.join(missing)}")


def _release_values(
    zenodo: Mapping[str, Any], errors: list[str]
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Read and type-check the release fields used by this repository."""

    return (
        _text(zenodo.get("title"), ".zenodo.json title", errors),
        _text(zenodo.get("description"), ".zenodo.json description", errors),
        _text(zenodo.get("license"), ".zenodo.json license", errors),
        _text(zenodo.get("access_right"), ".zenodo.json access_right", errors),
        _text(zenodo.get("upload_type"), ".zenodo.json upload_type", errors),
    )


def _source_values(
    citation: Mapping[str, Any], project: Mapping[str, Any], errors: list[str]
) -> tuple[str | None, str | None, str | None, str | None, Mapping[str, Any]]:
    """Read the authoritative release fields from CFF and TOML."""

    citation_title = _citation_text(citation, "title", errors)
    citation_abstract = _citation_text(citation, "abstract", errors)
    citation_license = _citation_text(citation, "license", errors)
    project_section = project.get("project")
    if not isinstance(project_section, Mapping):
        errors.append("pyproject.toml [project] table is missing")
        project_section = {}
    project_license = _text(
        project_section.get("license"), "pyproject.toml project license", errors
    )
    _text(project_section.get("name"), "pyproject.toml project name", errors)
    return citation_title, citation_abstract, citation_license, project_license, project_section


def _expect_match(left: str | None, right: str | None, message: str, errors: list[str]) -> None:
    """Record a mismatch only when both source values are valid strings."""

    if left is not None and right is not None and left != right:
        errors.append(message)


def _validate_release_values(
    release_values: tuple[str | None, str | None, str | None, str | None, str | None],
    source_values: tuple[str | None, str | None, str | None, str | None, Mapping[str, Any]],
    errors: list[str],
) -> None:
    """Validate source alignment and official Zenodo enumerated values."""

    title, description, license_id, access_right, upload_type = release_values
    citation_title, citation_abstract, citation_license, project_license, _ = source_values
    _expect_match(
        title,
        citation_title,
        f".zenodo.json title {title!r} does not match CITATION.cff title {citation_title!r}",
        errors,
    )
    _expect_match(
        description,
        citation_abstract,
        ".zenodo.json description does not match CITATION.cff abstract",
        errors,
    )
    _expect_match(
        citation_license,
        project_license,
        "CITATION.cff and pyproject.toml licenses do not match",
        errors,
    )
    _expect_match(
        license_id, citation_license, ".zenodo.json license does not match CITATION.cff", errors
    )
    _expect_match(
        license_id,
        project_license,
        ".zenodo.json license does not match pyproject.toml",
        errors,
    )
    if access_right is not None and access_right not in ZENODO_ACCESS_RIGHTS:
        errors.append(
            f".zenodo.json access_right is not an official Zenodo value: {access_right!r}"
        )
    if access_right == "open" and license_id is None:
        errors.append(".zenodo.json license is required for open access")
    if upload_type is not None and upload_type not in ZENODO_UPLOAD_TYPES:
        errors.append(f".zenodo.json upload_type is not an official Zenodo value: {upload_type!r}")
    if upload_type is not None and upload_type != "software":
        errors.append(".zenodo.json upload_type must be 'software' for this repository")


def _creator_signatures(creators: Any, errors: list[str]) -> list[tuple[str, ...]]:
    """Read and type-check Zenodo creator names."""

    if not isinstance(creators, list) or not creators:
        errors.append(".zenodo.json creators must be a non-empty list")
        return []

    signatures: list[tuple[str, ...]] = []
    for index, creator in enumerate(creators):
        if not isinstance(creator, Mapping):
            errors.append(f".zenodo.json creators[{index}] must be an object")
            continue
        creator_unknown = sorted(set(creator) - ZENODO_CREATOR_PROPERTIES)
        if creator_unknown:
            errors.append(
                f".zenodo.json creators[{index}] contains unsupported fields: "
                f"{', '.join(creator_unknown)}"
            )
        name = _text(creator.get("name"), f".zenodo.json creators[{index}] name", errors)
        if name is not None:
            signatures.append(_name_signature(name))
    return signatures


def _validate_creators(creators: Any, expected_names: list[str], errors: list[str]) -> None:
    """Require Zenodo creators to equal the source-author union."""

    actual_signatures = _creator_signatures(creators, errors)
    if not actual_signatures:
        return
    if len(actual_signatures) != len(set(actual_signatures)):
        errors.append(".zenodo.json creators must not contain duplicate names")
    expected_signatures = [_name_signature(name) for name in expected_names]
    if len(expected_signatures) != len(set(expected_signatures)):
        errors.append("authoritative metadata contains duplicate creator names")
    if set(actual_signatures) != set(expected_signatures):
        missing_creators = sorted(set(expected_signatures) - set(actual_signatures))
        extra_creators = sorted(set(actual_signatures) - set(expected_signatures))
        if missing_creators:
            errors.append(f".zenodo.json creators omit authoritative names: {missing_creators}")
        if extra_creators:
            errors.append(
                f".zenodo.json creators contain non-authoritative names: {extra_creators}"
            )


def _validate_publication_boundary(zenodo: Mapping[str, Any], errors: list[str]) -> None:
    """Reject fields that would claim a DOI or related publication."""

    if "doi" in zenodo:
        errors.append(".zenodo.json must not assert a DOI; publication is verified separately")
    if "related_identifiers" in zenodo:
        errors.append(
            ".zenodo.json must not declare related identifiers in this release metadata contract"
        )


def validate_release_metadata(
    zenodo_path: Path = DEFAULT_ZENODO,
    citation_path: Path = DEFAULT_CITATION,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> list[str]:
    """Return validation errors for the offline Zenodo release metadata contract."""

    errors: list[str] = []
    zenodo, zenodo_errors = _load_json_object(zenodo_path, ".zenodo.json")
    citation, citation_errors = _load_yaml_object(citation_path, "CITATION.cff")
    project, project_errors = _load_toml_object(pyproject_path, "pyproject.toml")
    errors.extend(zenodo_errors)
    errors.extend(citation_errors)
    errors.extend(project_errors)
    if zenodo is None or citation is None or project is None:
        return errors

    _validate_zenodo_shape(zenodo, errors)
    release_values = _release_values(zenodo, errors)
    source_values = _source_values(citation, project, errors)
    _validate_release_values(
        release_values,
        source_values,
        errors,
    )
    expected_names = _source_author_names(citation, source_values[4], errors)
    _validate_creators(zenodo.get("creators"), expected_names, errors)
    _validate_publication_boundary(zenodo, errors)
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zenodo", type=Path, default=DEFAULT_ZENODO)
    parser.add_argument("--citation", type=Path, default=DEFAULT_CITATION)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline release metadata check."""

    args = _parser().parse_args(argv)
    errors = validate_release_metadata(args.zenodo, args.citation, args.pyproject)
    if errors:
        print("Zenodo release metadata: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Zenodo release metadata: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
