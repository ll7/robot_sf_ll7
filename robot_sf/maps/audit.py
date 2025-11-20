"""Audit tool for detecting stray map assets outside canonical hierarchy.

This script scans the repository for map-related files that exist outside
the canonical locations (maps/svg_maps/ and maps/metadata/) and generates
a report.

Usage
-----
python -m robot_sf.maps.audit [--output OUTPUT_FILE]

Exit Codes
----------
0: No stray files found (audit passed)
1: Stray files detected (audit failed)
2: Error during audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def get_canonical_paths() -> tuple[Path, Path, Path]:
    """Get canonical and repository root paths.
    
    Returns
    -------
    tuple[Path, Path, Path]
        Tuple of (repo_root, svg_dir, metadata_dir)
    """
    # Assume we're running from repo root
    repo_root = Path.cwd()
    svg_dir = repo_root / "maps" / "svg_maps"
    metadata_dir = repo_root / "maps" / "metadata"
    return repo_root, svg_dir, metadata_dir


def find_stray_files(repo_root: Path, svg_dir: Path, metadata_dir: Path) -> dict[str, list[Path]]:
    """Find map files outside canonical locations.
    
    Parameters
    ----------
    repo_root : Path
        Repository root directory
    svg_dir : Path
        Canonical SVG directory
    metadata_dir : Path
        Canonical metadata directory
    
    Returns
    -------
    dict[str, list[Path]]
        Dictionary with keys 'svg', 'json' containing lists of stray file paths
    """
    stray_files = {"svg": [], "json": []}
    
    # Directories to exclude from scan
    exclude_dirs = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "fast-pysf",  # Separate project
        ".specify",
        ".codex",
        ".gemini",
        "class_diagram",  # UML diagrams, not map assets
        "svg_conv",  # SVG conversion examples, not map assets
        "docs",  # Documentation diagrams
    }
    
    # Scan for SVG files
    for svg_file in repo_root.rglob("*.svg"):
        # Skip if in canonical directory or excluded directories
        if any(excluded in svg_file.parts for excluded in exclude_dirs):
            continue
        if svg_dir in svg_file.parents or svg_file.parent == svg_dir:
            continue
        
        stray_files["svg"].append(svg_file)
        logger.warning(f"Stray SVG file: {svg_file.relative_to(repo_root)}")
    
    # Scan for JSON files in map-related directories
    # Look for JSON files in directories named "maps" but not in canonical location
    for json_file in repo_root.rglob("maps/*.json"):
        # Skip if in canonical directory or excluded directories  
        if any(excluded in json_file.parts for excluded in exclude_dirs):
            continue
        if metadata_dir in json_file.parents or json_file.parent == metadata_dir:
            continue
        if svg_dir in json_file.parents or json_file.parent == svg_dir:
            continue  # README or other docs
        
        stray_files["json"].append(json_file)
        logger.warning(f"Stray JSON file: {json_file.relative_to(repo_root)}")
    
    return stray_files


def search_hard_coded_references(repo_root: Path) -> list[tuple[Path, int, str]]:
    """Search for hard-coded map references in Python files.
    
    Parameters
    ----------
    repo_root : Path
        Repository root directory
    
    Returns
    -------
    list[tuple[Path, int, str]]
        List of (file_path, line_number, line_content) tuples
    """
    patterns = [
        "robot_sf/maps",  # Old path references
        "uni_campus_big",  # Hard-coded map ID example
    ]
    
    matches = []
    exclude_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "fast-pysf"}
    
    for py_file in repo_root.rglob("*.py"):
        # Skip excluded directories and the audit script itself
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        if "audit.py" in py_file.name or "registry.py" in py_file.name:
            continue
        
        try:
            with open(py_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue
                    for pattern in patterns:
                        if pattern in line:
                            matches.append((py_file, line_num, line.strip()))
                            logger.warning(f"Hard-coded reference in {py_file.name}:{line_num}: {line.strip()[:80]}")
        except (UnicodeDecodeError, PermissionError):
            logger.debug(f"Skipped file: {py_file}")
    
    return matches


def main():
    """Main audit function."""
    parser = argparse.ArgumentParser(description="Audit map files for canonical hierarchy compliance")
    parser.add_argument("--output", "-o", help="Output JSON file path", default=None)
    args = parser.parse_args()
    
    try:
        repo_root, svg_dir, metadata_dir = get_canonical_paths()
        
        logger.info("Starting map audit...")
        logger.info(f"Repository root: {repo_root}")
        logger.info(f"Canonical SVG directory: {svg_dir}")
        logger.info(f"Canonical metadata directory: {metadata_dir}")
        
        # Find stray files
        stray_files = find_stray_files(repo_root, svg_dir, metadata_dir)
        
        # Search for hard-coded references
        hard_coded = search_hard_coded_references(repo_root)
        
        # Generate report
        report = {
            "audit_passed": len(stray_files["svg"]) == 0 and len(stray_files["json"]) == 0,
            "stray_svg_count": len(stray_files["svg"]),
            "stray_json_count": len(stray_files["json"]),
            "stray_svg_files": [str(p.relative_to(repo_root)) for p in stray_files["svg"]],
            "stray_json_files": [str(p.relative_to(repo_root)) for p in stray_files["json"]],
            "hard_coded_references_count": len(hard_coded),
            "hard_coded_references": [
                {"file": str(f.relative_to(repo_root)), "line": ln, "content": content}
                for f, ln, content in hard_coded
            ],
        }
        
        # Output report
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Audit report written to: {output_path}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Stray SVG files: {report['stray_svg_count']}")
        logger.info(f"Stray JSON files: {report['stray_json_count']}")
        logger.info(f"Hard-coded references: {report['hard_coded_references_count']}")
        
        if report["audit_passed"]:
            logger.info("✓ Audit PASSED: No stray files found")
            return 0
        else:
            logger.error("✗ Audit FAILED: Stray files detected")
            return 1
            
    except Exception as e:
        logger.error(f"Error during audit: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
