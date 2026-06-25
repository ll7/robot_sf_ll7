"""
Git hook to prevent accidental absolute home-dir paths in ``configs/**``.

Autonomously-generated configs occasionally hardcode absolute user-home paths
(``/home/<user>/...``), which are non-portable for other contributors and
automated runners. This hook fails when a tracked config file contains such a
path, UNLESS the line is explicitly annotated as intentional (e.g. private-ops
SLURM routing) with an ``allow-abs-path`` marker.

See issue #3605.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Absolute home-dir prefixes that should not appear in portable configs.
ABS_PATH_PATTERN = re.compile(r"(/home/|/Users/|/root/)")

# A line carrying this marker is an intentional, documented absolute path.
ALLOW_MARKER = "allow-abs-path"

CONFIG_ROOT = Path("configs")


def _iter_config_files(files: list[str]) -> list[Path]:
    """
    Return the subset of ``files`` that live under a ``configs/`` directory.

    Accepts both repo-relative paths (as pre-commit passes them, e.g.
    ``configs/training/x.yaml``) and absolute paths (as tests / direct calls may
    pass), so membership is keyed on a ``configs`` path component rather than a
    fixed relative prefix.
    """
    selected: list[Path] = []
    for f in files:
        path = Path(f)
        if CONFIG_ROOT.name in path.parts and path.is_file():
            selected.append(path)
    return selected


def find_abs_path_violations(files: list[str]) -> dict:
    """
    Scan config files for unannotated absolute home-dir paths.

    Args:
        files: Candidate file paths (only those under ``configs/`` are checked).

    Returns:
        Dict with ``status`` ("pass"/"fail"), ``violations`` (list of
        ``{file, line, text}``), and a human-readable ``message``.
    """
    config_files = _iter_config_files(files)

    if not config_files:
        return {
            "status": "pass",
            "violations": [],
            "message": "No config files in scope - nothing to check.",
        }

    violations: list[dict] = []
    for path in config_files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Non-text or unreadable config (e.g. binary asset) - skip.
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not ABS_PATH_PATTERN.search(line):
                continue
            if ALLOW_MARKER in line:
                # Explicitly annotated as an intentional absolute path.
                continue
            violations.append({"file": str(path), "line": lineno, "text": line.strip()})

    if violations:
        return {
            "status": "fail",
            "violations": violations,
            "message": (
                f"Found {len(violations)} unannotated absolute home-dir path(s) "
                f"in configs/. Use a repo-relative path, or annotate the line "
                f"with '# {ALLOW_MARKER}: <reason>' if the absolute path is "
                f"intentional (e.g. private-ops routing)."
            ),
        }

    return {
        "status": "pass",
        "violations": [],
        "message": f"Checked {len(config_files)} config file(s); no leaks found.",
    }


def main() -> None:
    """CLI entry point for the git hook."""
    parser = argparse.ArgumentParser(
        description="Prevent accidental absolute home-dir paths in configs/**"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (only those under configs/ are scanned).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every tracked file under configs/ instead of the given list.",
    )
    args = parser.parse_args()

    if args.all:
        files = [str(p) for p in CONFIG_ROOT.rglob("*") if p.is_file()]
    else:
        files = args.files

    result = find_abs_path_violations(files)

    for v in result["violations"]:
        logging.error("Absolute path in config: %s:%s\n  %s", v["file"], v["line"], v["text"])
    if result["violations"]:
        logging.error(result["message"])

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
