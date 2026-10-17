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
import hashlib
import logging
import os
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

# Verbatim evidence sources may retain absolute paths only when the exact producer bytes are
# separately bound and a portable normalized copy is retained. These exceptions are file- and
# digest-specific: any change to a pinned file restores the normal absolute-path scan.
PINNED_VERBATIM_EVIDENCE_SHA256 = {
    # TRAIN2: exact producer bytes, with portable copies and full checksum custody
    # in docs/context/evidence/issue_10007_train2_custody/path_custody.json.
    "docs/context/evidence/issue_10056_wall_order_2026-09/doorway_plausibility.json": "f6c70b91be024f5ea8023baa4c4aa8602853d42c011e4470dff14ccd523ab254",
    "docs/context/evidence/issue_10064_no_admissible_recovery/changed_outcomes.json": "caa70860aa3b9133a9f66b1653fb4ce2c5eaf977c4e3afcc11ac0e68e79f8794",
    "docs/context/evidence/issue_10064_no_admissible_recovery/empty_world.json": "c85204242a7ab45cfa594019e942ccfa84aa420d8ff959c200ae6f93cba07d03",
    "docs/context/evidence/issue_10064_no_admissible_recovery/inner_checkpoints.json": "83e1dd9cd0cc56b6c6c8c94b2c7897a1509187af96481d1f637d10dc24ddecd2",
    "docs/context/evidence/issue_10064_no_admissible_recovery/paired_analysis.json": "e260334d0e073cc59e41ebd379ec8609877781e5ec739b11996763611c9b4b5e",
    "docs/context/evidence/issue_10064_no_admissible_recovery/reproduce.txt": "32418fb811b79c23d375faba6866ea0a97ef3dcd63a43bed1968d121a43d2e51",
    "docs/context/evidence/issue_10064_no_admissible_recovery/trace_samples.json": "ed825c26c993548d8027d64e6206d0c60e4918e0647e58cc7afe76e32948ca06",
    "docs/context/evidence/issue_10064_no_admissible_recovery/validation.json": "3122a83a4304ac39f083be560150d0ea32d80652484ddb904011f121238c10c8",
    "docs/context/evidence/issue_9952_auditor/aud2/focused.txt": "d5a065f1c009951360857891e59477f3bfb733df4ba03a959c4c17640b14a971",
    "docs/context/evidence/issue_9952_auditor/aud2/green.txt": "af4ec8eb7ec387e0ab9bfd9fc406e5b44822ebefafc1c03e9b84f2a671be1a09",
    "docs/context/evidence/issue_9952_auditor/aud2/queue_validation.json": "0b9aa55142bbc5a3749f5f5c0507f629acb49b6ffbe4977c839362508ab59646",
    "docs/context/evidence/issue_9952_auditor/aud2/test_value.md": "2ed1fd10dd93c1946847e5cb298a77db70e0933d756c6e4ecec938dca0222987",
    "docs/context/evidence/issue_9952_auditor/focused-final.txt": "e8b7df75fc17ea4bc88532d828ac3ad260cc8abdcd9f7a028b57d326e837e9bd",
    "docs/context/evidence/issue_9952_auditor/parent-failures-final.txt": "3f56728356c531dd33397565fee182726fd9d2a0dd0d1d7f5fa26afc4a67103d",
    (
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "source_reports/13268/campaign_summary.json"
    ): "f29e6c5ee12679408b1d65add0149e4cfe07390f0c8828208114f39dd900c257",
    (
        "docs/context/evidence/issue_3810_h600_interpretation_2026-07/"
        "source_reports/13273/campaign_summary.json"
    ): "f456580bad70167e42d6e24c9570547042fd02ce39c35687ed152928f6a0698e",
    (
        "docs/context/evidence/issue_6474_social_compliance_nominal_campaign_manifest.json"
    ): "10c45f44ec5679144671c6247644a7e88b1444fdf9b25a7373b343bbf732e1bc",
    (
        "docs/context/evidence/issue_6102_robot_speed_tier_recovery/README.md"
    ): "14a131eceb2f767e70609d573d32a942ba15e703378d2bb921cb7da82e768179",
    (
        "docs/context/evidence/issue_6102_robot_speed_tier_recovery/recovery_manifest.json"
    ): "2e14b777170450825f7671418ea8ed7130576adbd6bd473bf0d63062d9ee49ae",
}


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


def _repo_relative_path(path: Path) -> str | None:
    """Normalize a lexical path relative to its repository root, or the test cwd.

    Git supplies repo-relative paths to ``--all``; tests and direct callers may supply
    absolute paths. Normalize ``.`` and ``..`` without resolving symlinks, so an alias to
    a pinned source remains a different path. Looking for the nearest ``.git`` marker
    keeps both forms tied to the actual repository/worktree root. Temporary test roots
    without Git metadata use their current working directory.
    """
    try:
        lexical_absolute = Path(os.path.abspath(path.expanduser()))
    except (OSError, RuntimeError, TypeError):
        return None

    for parent in (lexical_absolute.parent, *lexical_absolute.parent.parents):
        if (parent / ".git").exists():
            try:
                return lexical_absolute.relative_to(parent).as_posix()
            except ValueError:
                return None

    try:
        cwd = Path(os.path.abspath(Path.cwd()))
        return lexical_absolute.relative_to(cwd).as_posix()
    except (OSError, ValueError, RuntimeError):
        return None


def _is_pinned_verbatim_evidence(path: Path) -> bool:
    """Return whether ``path`` is an exact pinned recovery artifact.

    The lexical path must match the exact repository-relative key. The digest match makes the
    exception fail closed if a recovered artifact is edited or replaced.
    """

    normalized = _repo_relative_path(path)
    if normalized is None:
        return False
    for repo_path, expected_sha256 in PINNED_VERBATIM_EVIDENCE_SHA256.items():
        if normalized != repo_path:
            continue
        try:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 16), b""):
                    hasher.update(chunk)
            observed_sha256 = hasher.hexdigest()
        except OSError:
            return False
        return observed_sha256 == expected_sha256
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
        if _is_pinned_verbatim_evidence(path):
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
