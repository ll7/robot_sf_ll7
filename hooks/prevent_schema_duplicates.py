"""
Git hook to prevent schema duplication.

This hook scans staged files for schema files that would duplicate
existing canonical schemas, preventing commits that introduce duplication.
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def prevent_schema_duplicates(
    staged_files: List[str],
    schema_pattern: str = r".*\.schema\.v[0-9]+\.json$",
    canonical_dir: Path = Path("robot_sf/benchmark/schemas"),
) -> Dict:
    """
    Check staged files for schema duplicates against canonical schemas.

    Args:
        staged_files: List of file paths staged for commit
        schema_pattern: Regex pattern to identify schema files
        canonical_dir: Directory containing canonical schemas

    Returns:
        Dict with status, duplicates_found, and message
    """
    # Find schema files in staged files
    schema_files = [f for f in staged_files if re.match(schema_pattern, f)]

    if not schema_files:
        return {
            "status": "pass",
            "duplicates_found": [],
            "message": "No schema files found in staged changes - no duplicates detected",
        }

    # Get canonical schema directory
    if not canonical_dir.exists():
        return {
            "status": "fail",
            "duplicates_found": [],
            "message": f"Canonical schema directory not found: {canonical_dir}",
        }

    duplicates = []

    for staged_file in schema_files:
        staged_path = Path(staged_file)

        # Skip files that are already in the canonical directory
        if staged_path.parent == canonical_dir:
            continue

        # Check if this file would duplicate a canonical schema
        duplicate_info = _check_for_duplicate(staged_path, canonical_dir)
        if duplicate_info:
            duplicates.append(duplicate_info)

    if duplicates:
        return {
            "status": "fail",
            "duplicates_found": duplicates,
            "message": f"Found {len(duplicates)} duplicate schema file(s). "
            "Schema consolidation requires using canonical schemas.",
        }

    return {
        "status": "pass",
        "duplicates_found": [],
        "message": f"Validated {len(schema_files)} schema file(s) - no duplicates found",
    }


def _check_for_duplicate(staged_path: Path, canonical_dir: Path) -> Optional[Dict]:
    """
    Check if a staged schema file duplicates a canonical schema.

    Args:
        staged_path: Path to the staged schema file
        canonical_dir: Directory containing canonical schemas

    Returns:
        Dict with duplicate info if found, None otherwise
    """
    if not staged_path.exists():
        return None

    try:
        # Read staged file content
        with open(staged_path, "r", encoding="utf-8") as f:
            staged_content = f.read()

        # Calculate hash of staged content
        staged_hash = hashlib.sha256(staged_content.encode("utf-8")).hexdigest()

        # Check against all canonical schemas
        for canonical_file in canonical_dir.glob("*.schema.v*.json"):
            try:
                with open(canonical_file, "r", encoding="utf-8") as f:
                    canonical_content = f.read()

                canonical_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

                if staged_hash == canonical_hash:
                    return {
                        "file": str(staged_path),
                        "canonical_file": str(canonical_file),
                        "reason": "Content hash matches existing canonical schema",
                    }

            except (IOError, OSError) as e:
                # Log error but continue checking other files
                print(
                    f"Warning: Could not read canonical schema {canonical_file}: {e}",
                    file=sys.stderr,
                )

    except (IOError, OSError) as e:
        print(f"Warning: Could not read staged schema {staged_path}: {e}", file=sys.stderr)

    return None


def main():
    """Main entry point for the git hook."""
    import argparse

    parser = argparse.ArgumentParser(description="Prevent schema file duplication")
    parser.add_argument("staged_files", nargs="+", help="Files staged for commit")
    parser.add_argument(
        "--schema-pattern",
        default=r".*\.schema\.v[0-9]+\.json$",
        help="Regex pattern for schema files",
    )
    parser.add_argument(
        "--canonical-dir",
        default="robot_sf/benchmark/schemas",
        help="Directory containing canonical schemas",
    )

    args = parser.parse_args()

    canonical_dir = Path(args.canonical_dir)
    result = prevent_schema_duplicates(args.staged_files, args.schema_pattern, canonical_dir)

    print(result["message"])

    if result["duplicates_found"]:
        print("\nDuplicate schemas found:")
        for dup in result["duplicates_found"]:
            print(f"  {dup['file']} duplicates {dup['canonical_file']}")
            print(f"    Reason: {dup['reason']}")

    # Exit with appropriate code
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
