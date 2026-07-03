"""
Git hook to prevent absolute home-dir paths in ``configs/**`` and committed
evidence packets under ``docs/context/evidence/**``.

Autonomously-generated configs and evidence packets occasionally hardcode
absolute user-home paths (``/home/<user>/...``, including author-specific
worktree paths), which are non-portable for other contributors and automated
runners and violate the repository's reproducibility hard-rule. This hook fails
when a tracked config or evidence file contains such a path, UNLESS the line is
explicitly annotated as intentional (e.g. private-ops SLURM routing) with an
``allow-abs-path`` marker.

Configs coverage was added in issue #3605. Evidence-packet coverage was added in
issue #4324 after the same defect recurred in a committed provenance file
(``config.path`` / ``gate_spec.path`` baking in a local worktree path; cf. the
#4302/#4303 SNQI fix). To keep the guard cheap it scans only the files it is
handed (staged files at commit time), plus the whole tracked tree under ``--all``.
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Absolute home-dir prefixes that should not appear in portable configs/evidence.
ABS_PATH_PATTERN = re.compile(r"(/home/|/Users/|/root/)")

# A line carrying this marker is an intentional, documented absolute path.
ALLOW_MARKER = "allow-abs-path"

CONFIG_ROOT = Path("configs")
EVIDENCE_ROOT = Path("docs/context/evidence")

# Directory roots whose committed files must stay free of absolute home-dir paths.
SCANNED_ROOTS = (CONFIG_ROOT, EVIDENCE_ROOT)

# Evidence packets committed *before* the guard was extended to
# ``docs/context/evidence/**`` (issue #4324). These historical benchmark
# command/provenance records already bake in absolute worktree paths in durable,
# checksummed artifacts. They are grandfathered so the guard can fail closed for
# NEW packets without forcing a retroactive rewrite (and SHA256SUMS churn) of
# durable historical evidence. This is tracked pre-existing debt: do NOT add new
# entries here — fix the leak at generation time instead.
LEGACY_EVIDENCE_ALLOWLIST = frozenset(
    {
        "issue_1023_candidate_augmented_local_full_2026-05-06",
        "issue_1023_scenario_horizons_local_full_2026-05-06",
        "issue_1023_scenario_horizons_preflight_2026-05-06",
        "issue_1454_stage_a_fixed_h100_2026-05-22",
        "issue_1470_oracle_imitation_traces_12911_2026-06-17",
        "issue_1475_orca_residual_bc_smoke_12913_2026-06-17",
        "issue_2258_topology_primary_route_audit_2026-06-05",
        "issue_2282_topology_selection_instrumentation_2026-06-05",
        "issue_3266_ppo_snqi_smoke_2026-06-23",
    }
)


def _under_root(path: Path, root: Path) -> bool:
    """Return whether ``path`` lives under ``root``.

    Membership is keyed on ``root``'s components appearing as a contiguous
    subsequence of ``path``'s parts, so both repo-relative paths (as pre-commit
    passes them, e.g. ``configs/training/x.yaml``) and absolute paths (as tests
    or direct calls may pass, e.g. ``/home/u/repo/configs/training/x.yaml``) are
    recognised.
    """
    parts = path.parts
    root_parts = root.parts
    span = len(root_parts)
    return any(parts[i : i + span] == root_parts for i in range(len(parts) - span + 1))


def _is_grandfathered_evidence(path: Path) -> bool:
    """Return whether ``path`` belongs to a grandfathered legacy evidence packet.

    The packet directory is the component immediately after the
    ``docs/context/evidence`` root; if that name is in
    :data:`LEGACY_EVIDENCE_ALLOWLIST` the file is skipped (pre-existing debt).
    """
    parts = path.parts
    root_parts = EVIDENCE_ROOT.parts
    span = len(root_parts)
    for i in range(len(parts) - span + 1):
        if parts[i : i + span] == root_parts and i + span < len(parts):
            return parts[i + span] in LEGACY_EVIDENCE_ALLOWLIST
    return False


def _iter_scanned_files(files: list[str]) -> list[Path]:
    """
    Return the subset of ``files`` under a scanned root, minus grandfathered ones.

    A file is in scope when it lives under any of :data:`SCANNED_ROOTS`
    (``configs/`` or ``docs/context/evidence/``) and is not part of a
    grandfathered legacy evidence packet.
    """
    selected: list[Path] = []
    for f in files:
        path = Path(f)
        if not path.is_file():
            continue
        if not any(_under_root(path, root) for root in SCANNED_ROOTS):
            continue
        if _is_grandfathered_evidence(path):
            continue
        selected.append(path)
    return selected


def _iter_tracked_scanned_files() -> list[str]:
    """Return Git-tracked files under the scanned roots for ``--all`` checks.

    Uses ``git ls-files -z`` so paths containing spaces or other special
    characters survive verbatim (default output C-quotes such paths, which
    would no longer match a real file and could be silently skipped).
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *(str(root) for root in SCANNED_ROOTS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [name for name in result.stdout.split("\0") if name and Path(name).is_file()]


def find_abs_path_violations(files: list[str]) -> dict:
    """
    Scan config/evidence files for unannotated absolute home-dir paths.

    Args:
        files: Candidate file paths (only those under a scanned root and not
            grandfathered are checked).

    Returns:
        Dict with ``status`` ("pass"/"fail"), ``violations`` (list of
        ``{file, line, text}``), and a human-readable ``message``.
    """
    scanned_files = _iter_scanned_files(files)

    if not scanned_files:
        return {
            "status": "pass",
            "violations": [],
            "message": "No config or evidence files in scope - nothing to check.",
        }

    violations: list[dict] = []
    for path in scanned_files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Non-text or unreadable file (e.g. binary asset) - skip.
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
                f"in configs/ or docs/context/evidence/. Use a repo-relative path, "
                f"or annotate the line with '# {ALLOW_MARKER}: <reason>' if the "
                f"absolute path is intentional (e.g. private-ops routing)."
            ),
        }

    return {
        "status": "pass",
        "violations": [],
        "message": f"Checked {len(scanned_files)} file(s); no leaks found.",
    }


def main() -> None:
    """CLI entry point for the git hook."""
    parser = argparse.ArgumentParser(
        description=("Prevent absolute home-dir paths in configs/** and docs/context/evidence/**")
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (only those under a scanned root are scanned).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan every tracked file under the scanned roots instead of the given list.",
    )
    args = parser.parse_args()

    if args.all:
        files = _iter_tracked_scanned_files()
    else:
        files = args.files

    result = find_abs_path_violations(files)

    for v in result["violations"]:
        logging.error("Absolute path: %s:%s\n  %s", v["file"], v["line"], v["text"])
    if result["violations"]:
        logging.error(result["message"])

    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
