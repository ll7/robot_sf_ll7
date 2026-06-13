"""Validate a generated_scenario_candidate.v1 JSON file against the canonical schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "generated_scenario_candidate.v1.json"
)


def load_schema() -> dict:
    """Load the generated_scenario_candidate.v1 JSON Schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_candidate(candidate_path: str) -> list[str]:
    """Validate a candidate JSON file against the schema.

    Returns a list of error messages (empty if valid).
    """
    path = Path(candidate_path)
    if not path.exists():
        return [f"Candidate file not found: {path}"]

    schema = load_schema()
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON syntax in {path}: {exc}"]

    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for error in validator.iter_errors(candidate):
        errors.append(f"  {error.json_path}: {error.message}")

    return errors


def main() -> None:
    """CLI entry point: validate one or more candidate files."""
    if len(sys.argv) < 2:
        print(
            "Usage: validate_generated_scenario_candidate.py <candidate.json> [candidate2.json ...]"
        )
        sys.exit(1)

    all_ok = True
    for candidate_path in sys.argv[1:]:
        errors = validate_candidate(candidate_path)
        if errors:
            all_ok = False
            print(f"FAIL: {candidate_path}")
            for err in errors:
                print(err)
        else:
            print(f"PASS: {candidate_path}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
