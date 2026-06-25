"""
Git hook to catch issue-provenance mismatches in cloned training configs.

Training configs are cloned from a "leader" recipe (the intended workflow), but
the leading header comment's issue reference is not always updated to match the
clone's identity fields. The result is a config whose top-of-file comment names
the wrong issue while its ``policy_id`` / tracking ``group`` / ``tags`` /
``campaign_id`` all target the correct one (see #3570, where the header read
"Issue-1024" while every identity field targeted issue-3068).

This hook keys off the in-file identity fields as the source of truth: if those
agree on a single canonical issue number and the leading comment block mentions
issue numbers but NOT that canonical one, the header is flagged as
misattributed. A header that mentions the canonical issue (even alongside a
source/leader issue, e.g. "clones issue-791 for issue-3068") passes.

Immutable consideration: configs under ``configs/training/ppo/ablations/`` map
to already-run SLURM jobs and are treated as historical records. Correcting a
*comment* does not change run semantics, so the check still applies, but a line
annotated with ``allow-provenance-mismatch: <reason>`` is exempt for the rare
case where editing the historical header is undesirable.

See issue #3606.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING)

TRAINING_ROOT = Path("configs/training")

# Matches "issue-3068", "issue_3068", "Issue 3068", "issue3068".
ISSUE_RE = re.compile(r"issue[\s_-]?(\d+)", re.IGNORECASE)

# A header carrying this marker intentionally references a different issue.
ALLOW_MARKER = "allow-provenance-mismatch"


def _issue_numbers(text: str) -> set[int]:
    """Return the set of issue numbers referenced in ``text``."""
    return {int(m) for m in ISSUE_RE.findall(text)}


def _leading_comment_block(text: str) -> str:
    """Return the contiguous comment lines at the very top of the file."""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines.append(stripped)
        elif stripped == "":
            # Allow blank lines interleaved within the leading block.
            if lines:
                lines.append("")
            continue
        else:
            break
    return "\n".join(lines)


def _identity_issue(data: object) -> set[int]:
    """Collect issue numbers referenced by a config's identity fields."""
    issues: set[int] = set()
    if not isinstance(data, dict):
        return issues
    for key in ("policy_id", "campaign_id"):
        value = data.get(key)
        if isinstance(value, str):
            issues |= _issue_numbers(value)
    tracking = data.get("tracking")
    if isinstance(tracking, dict):
        wandb = tracking.get("wandb")
        if isinstance(wandb, dict):
            group = wandb.get("group")
            if isinstance(group, str):
                issues |= _issue_numbers(group)
            tags = wandb.get("tags")
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        issues |= _issue_numbers(tag)
    return issues


def _iter_training_configs(files: list[str]) -> list[Path]:
    """Return the subset of ``files`` that are training YAML configs."""
    selected: list[Path] = []
    for f in files:
        path = Path(f)
        parts = path.parts
        in_training = "configs" in parts and "training" in parts
        if in_training and path.suffix in {".yaml", ".yml"} and path.is_file():
            selected.append(path)
    return selected


def find_provenance_mismatches(files: list[str]) -> dict:
    """
    Scan training configs for header/identity issue-provenance mismatches.

    Args:
        files: Candidate file paths (only training YAML configs are checked).

    Returns:
        Dict with ``status`` ("pass"/"fail"), ``violations`` (list of
        ``{file, header_issues, identity_issue}``), and a ``message``.
    """
    configs = _iter_training_configs(files)
    if not configs:
        return {
            "status": "pass",
            "violations": [],
            "message": "No training configs in scope - nothing to check.",
        }

    violations: list[dict] = []
    for path in configs:
        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
        except (UnicodeDecodeError, OSError, yaml.YAMLError):
            # Unreadable / unparseable config - not this hook's concern.
            continue

        identity = _identity_issue(data)
        # Only act when identity fields agree on a single canonical issue.
        if len(identity) != 1:
            continue
        canonical = next(iter(identity))

        header = _leading_comment_block(text)
        if ALLOW_MARKER in header:
            continue
        header_issues = _issue_numbers(header)
        if header_issues and canonical not in header_issues:
            violations.append(
                {
                    "file": str(path),
                    "header_issues": sorted(header_issues),
                    "identity_issue": canonical,
                }
            )

    if violations:
        return {
            "status": "fail",
            "violations": violations,
            "message": (
                f"Found {len(violations)} training config(s) whose header comment "
                f"names a different issue than their identity fields "
                f"(policy_id/group/tags/campaign_id). Update the header to match, "
                f"or annotate it with '# {ALLOW_MARKER}: <reason>' if intentional."
            ),
        }

    return {
        "status": "pass",
        "violations": [],
        "message": f"Checked {len(configs)} training config(s); provenance consistent.",
    }


def main() -> None:
    """CLI entry point for the git hook."""
    parser = argparse.ArgumentParser(
        description="Catch issue-provenance mismatches in cloned training configs"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (only training YAML configs are scanned).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every YAML file under configs/training/ instead of the list.",
    )
    args = parser.parse_args()

    if args.all:
        files = [
            str(p)
            for p in TRAINING_ROOT.rglob("*")
            if p.is_file() and p.suffix in {".yaml", ".yml"}
        ]
    else:
        files = args.files

    result = find_provenance_mismatches(files)

    for v in result["violations"]:
        logging.error(
            "Provenance mismatch: %s\n  header references issue(s) %s but identity "
            "fields target issue %s",
            v["file"],
            v["header_issues"],
            v["identity_issue"],
        )
    if result["violations"]:
        logging.error(result["message"])

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
