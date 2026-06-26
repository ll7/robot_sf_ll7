#!/usr/bin/env python3
"""Propose ``docs/context/catalog.yaml`` entries for uncataloged evidence anchors.

The evidence-cataloging backlog (issue #3014) is fully regular: each tracked
evidence bundle under ``docs/context/evidence/`` needs one catalog row with a
``path``/``status``/``freshness``/``area`` shape.  Draining that backlog one
bundle at a time costs many mechanical PRs.  This tool turns N manual rows into
one reviewed diff by:

1. reusing the checker's anchor discovery
   (:func:`scripts.validation.check_docs_proof_consistency.uncovered_evidence_bundles`)
   so it can never diverge from the gate it is trying to satisfy,
2. inferring ``status``/``freshness``/``area`` from deterministic, documented
   heuristics,
3. routing anything it cannot infer with confidence to an explicit
   ``needs-human-review`` list rather than guessing, and
4. emitting an additive, stably-ordered preview (default) or applying it
   (``--apply``).

Safety contract
---------------
* Discovery is delegated, not reimplemented.
* Every auto-proposed entry is vetted to pass the checker's own catalog
  validation (canonical vocabulary, existing path, no ``output/`` or absolute
  local paths in the referenced evidence file).
* ``--apply`` is *additive only*: it appends rows to the end of the ``entries:``
  list and never edits, reorders, or deletes existing rows.
* The tool is idempotent: once a bundle is covered, discovery stops reporting it,
  so a second run proposes nothing.

Usage
-----
::

    uv run python scripts/tools/catalog_evidence.py                      # dry-run preview
    uv run python scripts/tools/catalog_evidence.py --json-output out.json
    uv run python scripts/tools/catalog_evidence.py --apply              # write entries
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.validation.check_docs_proof_consistency import (
    _ABSOLUTE_LOCAL_PATH_RE,
    _CATALOG_DEFAULT_FRESHNESS,
    _CATALOG_STATUSES,
    _CONTEXT_CATALOG,
    _EVIDENCE_DIR,
    _OUTPUT_PATH_RE,
    _TEXT_EVIDENCE_SUFFIXES,
    _is_within_dir,
    _repo_root,
    _strip_fenced_code_blocks,
    evidence_bundle_members,
    uncovered_evidence_bundles,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- Inference vocabulary -------------------------------------------------
#
# All anchors discovered by the checker live under ``docs/context/evidence/``,
# which the catalog vocabulary classifies as tracked proof: ``status: evidence``
# and ``freshness: evidence`` are therefore high-confidence defaults for that
# directory.  ``area`` is the only genuinely ambiguous field, so it gets an
# explicit, ordered keyword map and a conservative ``needs-human-review``
# fallback (never a silent guess).
_EVIDENCE_STATUS = "evidence"
_EVIDENCE_FRESHNESS = "evidence"

# Ordered (substring, area) rules; first match wins.  Substrings are matched
# against the lowercased bundle-key path so multi-word topic signals work.  The
# specific topic families come first so e.g. "adversarial" beats a generic
# "scenario".  Areas must already exist in the catalog's vocabulary.
_AREA_RULES: tuple[tuple[str, str], ...] = (
    # External-simulator / CARLA evidence.
    ("carla", "carla_external_sim"),
    # Adversarial scenario search/generation evidence.
    ("adversarial", "adversarial_search"),
    # Manual-control and real-environment trace evidence.
    ("manual_control", "manual_control_trace"),
    ("trace_viewer", "manual_control_trace"),
    ("real_env_trace", "manual_control_trace"),
    ("real_trace", "manual_control_trace"),
    # Learned-policy / imitation / shielded-PPO evidence.
    ("orca_residual", "learned_policy"),
    ("shielded_ppo", "learned_policy"),
    ("bc_smoke", "learned_policy"),
    ("bc_warm", "learned_policy"),
    ("warm_start", "learned_policy"),
    ("imitation", "learned_policy"),
    ("oracle", "learned_policy"),
    ("lidar_ppo", "learned_policy"),
    ("selector_orca", "learned_policy"),
    # Scenario belief/prior evidence is learned-policy uncertainty work; keep it
    # ahead of the generic ("scenario", "benchmark_evidence") fallback below.
    ("scenario_belief", "learned_policy"),
    ("scenario_prior", "learned_policy"),
    # Predictive / forecast planner evidence.
    ("predictive", "predictive_planner"),
    ("forecast", "predictive_planner"),
    ("same_seed", "predictive_planner"),
    ("coupling_gate", "predictive_planner"),
    # Infrastructure areas.
    ("root_layout", "root_layout"),
    ("local_artifact", "evidence_policy"),
    ("artifact_retirement", "evidence_policy"),
    ("release_evidence_gate", "evidence_policy"),
    ("slurm", "slurm"),
    ("policy_search", "policy_search"),
    # Benchmark-evidence family (broad scenario/seed/metric diagnostics).  Kept
    # last so the specific families above take precedence.
    ("perturbation", "benchmark_evidence"),
    ("topology", "benchmark_evidence"),
    ("seed", "benchmark_evidence"),
    ("amv", "benchmark_evidence"),
    ("actuation", "benchmark_evidence"),
    ("kinematic", "benchmark_evidence"),
    ("scenario", "benchmark_evidence"),
    ("recenter", "benchmark_evidence"),
    ("scaling", "benchmark_evidence"),
    ("sensitivity", "benchmark_evidence"),
    ("deadlock", "benchmark_evidence"),
    ("intersection", "benchmark_evidence"),
    ("crossing", "benchmark_evidence"),
    ("head_on", "benchmark_evidence"),
    ("signalized", "benchmark_evidence"),
    ("occluded", "benchmark_evidence"),
    ("emergence", "benchmark_evidence"),
    ("calibration", "benchmark_evidence"),
    ("disruption", "benchmark_evidence"),
    ("ablation", "benchmark_evidence"),
    ("horizon", "benchmark_evidence"),
    ("density", "benchmark_evidence"),
    ("dense_pedestrian", "benchmark_evidence"),
    ("solvability", "benchmark_evidence"),
    ("mechanism", "benchmark_evidence"),
    ("replay", "benchmark_evidence"),
    ("benchmark", "benchmark_evidence"),
    ("matrix", "benchmark_evidence"),
    ("sweep", "benchmark_evidence"),
    ("smoke", "benchmark_evidence"),
    ("pilot", "benchmark_evidence"),
    ("route_offset", "benchmark_evidence"),
    ("corridor_trace", "benchmark_evidence"),
    ("ped_timing", "benchmark_evidence"),
    ("leave_group_speed", "benchmark_evidence"),
    ("hazard_odd", "benchmark_evidence"),
    ("failure_pack", "benchmark_evidence"),
    ("ammv", "benchmark_evidence"),
    ("one_factor", "benchmark_evidence"),
    ("component_synthesis", "benchmark_evidence"),
    ("hot_path", "benchmark_evidence"),
    ("criticality", "benchmark_evidence"),
    ("failure_synthesis", "benchmark_evidence"),
    ("trace_case", "benchmark_evidence"),
    ("panel_candidate", "benchmark_evidence"),
    ("learned_risk", "benchmark_evidence"),
    ("reward_curriculum", "benchmark_evidence"),
    ("observation_noise", "benchmark_evidence"),
    ("counterfactual_pair", "benchmark_evidence"),
    ("robot_influence", "benchmark_evidence"),
    ("sensor_noise", "benchmark_evidence"),
    ("fast_pysf", "benchmark_evidence"),
    ("first_use", "adversarial_search"),
    ("external_prior", "benchmark_evidence"),
    ("pedestrian_archetype", "benchmark_evidence"),
    ("hardcase", "benchmark_evidence"),
    ("actor_injection", "benchmark_evidence"),
    ("belief_mode", "benchmark_evidence"),
    ("planner_obs", "benchmark_evidence"),
)

# Representative-file selection order for a bundle directory.  Exact basenames
# are tried first (canonical proof/manifest names), then by suffix group.
_PREFERRED_BASENAMES: tuple[str, ...] = (
    "summary.json",
    "report.json",
    "report.md",
    "readme.md",
    "register.json",
    "candidate.json",
    "manifest.json",
)
_SUFFIX_ORDER: tuple[str, ...] = (".json", ".yaml", ".yml", ".md", ".txt")


@dataclass(frozen=True)
class ProposedEntry:
    """One additive catalog row proposed for an uncovered evidence bundle."""

    bundle: str
    path: str
    status: str
    freshness: str
    area: str


@dataclass(frozen=True)
class ReviewItem:
    """An anchor the tool declined to auto-classify, with the blocking reason."""

    bundle: str
    reason: str


@dataclass(frozen=True)
class CatalogProposal:
    """The full proposal: confident additive entries plus the review queue."""

    proposed: list[ProposedEntry]
    needs_human_review: list[ReviewItem]


def infer_area(bundle_key: Path) -> str | None:
    """Return the catalog ``area`` for a bundle, or ``None`` when ambiguous.

    Matches the ordered :data:`_AREA_RULES` keyword map against the lowercased
    bundle-key path.  Returns ``None`` (route to ``needs-human-review``) when no
    rule matches, rather than guessing a default.
    """
    haystack = bundle_key.as_posix().lower()
    for needle, area in _AREA_RULES:
        if needle in haystack:
            return area
    return None


def infer_status(bundle_key: Path) -> str | None:
    """Return the catalog ``status`` for a bundle, or ``None`` when ambiguous.

    Anchors under ``docs/context/evidence/`` are tracked proof, so they map to
    ``evidence``.  Anything outside that root is not confidently classifiable.
    """
    return _EVIDENCE_STATUS if _is_within_dir(bundle_key, _EVIDENCE_DIR) else None


def infer_freshness(bundle_key: Path) -> str | None:
    """Return the catalog ``freshness`` for a bundle, or ``None`` when ambiguous."""
    return _EVIDENCE_FRESHNESS if _is_within_dir(bundle_key, _EVIDENCE_DIR) else None


def _evidence_text_is_clean(path: Path, repo_root: Path) -> bool:
    """Return whether a text evidence file would pass the checker's content scan.

    Mirrors ``check_docs_proof_consistency`` evidence validation: a referenced
    text file must not embed ``output/`` artifact pointers or absolute local
    filesystem paths.  Non-text suffixes are never content-scanned by the
    checker, so they are treated as clean.
    """
    if path.suffix not in _TEXT_EVIDENCE_SUFFIXES:
        return True
    try:
        text = (repo_root / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Unreadable as UTF-8: the checker skips content scanning, so it cannot
        # raise a content diagnostic for this file.
        return True
    scan_text = _strip_fenced_code_blocks(text) if path.suffix == ".md" else text
    return not (_OUTPUT_PATH_RE.search(scan_text) or _ABSOLUTE_LOCAL_PATH_RE.search(scan_text))


def _representative_member(members: Sequence[Path], repo_root: Path) -> Path | None:
    """Pick a deterministic, checker-clean evidence file to reference for a bundle.

    Tries canonical proof/manifest basenames first, then files by suffix group,
    each in sorted order, and returns the first candidate whose content passes
    the checker's evidence scan.  Returns ``None`` when no member is clean (the
    anchor is then routed to ``needs-human-review``).
    """
    ordered: list[Path] = []
    seen: set[Path] = set()

    by_basename = {member.name.lower(): member for member in members}
    for basename in _PREFERRED_BASENAMES:
        member = by_basename.get(basename)
        if member is not None and member not in seen:
            ordered.append(member)
            seen.add(member)

    for suffix in _SUFFIX_ORDER:
        for member in sorted(members):
            if member.suffix.lower() == suffix and member not in seen:
                ordered.append(member)
                seen.add(member)

    for member in sorted(members):
        if member not in seen:
            ordered.append(member)
            seen.add(member)

    for member in ordered:
        if _evidence_text_is_clean(member, repo_root):
            return member
    return None


def build_proposal(repo_root: Path, *, catalog_path: Path = _CONTEXT_CATALOG) -> CatalogProposal:
    """Compute additive catalog entries (and the review queue) for uncovered anchors.

    Reuses the checker's :func:`uncovered_evidence_bundles` discovery and member
    grouping, then applies the deterministic inference heuristics.  Proposed
    entries are sorted by their referenced path for stable, reviewable output.
    """
    members_by_bundle = evidence_bundle_members(repo_root)
    proposed: list[ProposedEntry] = []
    review: list[ReviewItem] = []

    for bundle_key in uncovered_evidence_bundles(repo_root, catalog_path=catalog_path):
        bundle = bundle_key.as_posix()
        status = infer_status(bundle_key)
        freshness = infer_freshness(bundle_key)
        area = infer_area(bundle_key)

        if status is None or freshness is None:
            review.append(
                ReviewItem(bundle=bundle, reason="anchor is outside docs/context/evidence/")
            )
            continue
        if area is None:
            review.append(
                ReviewItem(bundle=bundle, reason="no area keyword matched; classify area manually")
            )
            continue

        members = members_by_bundle.get(bundle_key, [])
        representative = _representative_member(members, repo_root)
        if representative is None:
            # Fail closed: a bundle with no checker-clean representative file is
            # left for human review rather than auto-cataloged at the directory
            # level.  Emitting the directory would bypass the file-only durable
            # evidence scan and silently admit output/local pointers; dirty
            # bundles must be acknowledged explicitly (legacy_dirty_evidence) by
            # a human instead.
            review.append(
                ReviewItem(
                    bundle=bundle,
                    reason="no checker-clean evidence file to reference (output/ or local paths)",
                )
            )
            continue

        proposed.append(
            ProposedEntry(
                bundle=bundle,
                path=representative.as_posix(),
                status=status,
                freshness=freshness,
                area=area,
            )
        )

    proposed.sort(key=lambda entry: entry.path)
    review.sort(key=lambda item: item.bundle)
    return CatalogProposal(proposed=proposed, needs_human_review=review)


def _validate_entry_vocabulary(entry: ProposedEntry) -> None:
    """Defensive guard: never emit an entry outside the canonical vocabulary."""
    if entry.status not in _CATALOG_STATUSES:
        raise ValueError(
            f"refusing to emit non-canonical status {entry.status!r} for {entry.bundle}"
        )
    if entry.freshness not in _CATALOG_DEFAULT_FRESHNESS:
        raise ValueError(
            f"refusing to emit non-canonical freshness {entry.freshness!r} for {entry.bundle}"
        )


def render_entry_block(entry: ProposedEntry) -> str:
    """Render one catalog row matching the existing 2/4-space YAML indentation."""
    _validate_entry_vocabulary(entry)
    return (
        f"  - path: {entry.path}\n"
        f"    status: {entry.status}\n"
        f"    freshness: {entry.freshness}\n"
        f"    area: {entry.area}\n"
    )


def render_dry_run(proposal: CatalogProposal, *, catalog_rel: Path = _CONTEXT_CATALOG) -> str:
    """Render a unified-diff-style preview of the additive ``entries:`` rows."""
    lines: list[str] = []
    if proposal.proposed:
        lines.append(f"--- {catalog_rel.as_posix()} (entries: additions)")
        lines.append(f"+++ {catalog_rel.as_posix()}")
        for entry in proposal.proposed:
            for block_line in render_entry_block(entry).splitlines():
                lines.append(f"+{block_line}")
    else:
        lines.append("No new catalog entries proposed.")

    if proposal.needs_human_review:
        lines.append("")
        lines.append("needs-human-review:")
        for item in proposal.needs_human_review:
            lines.append(f"  - {item.bundle}: {item.reason}")

    lines.append("")
    lines.append(
        f"Summary: proposed {len(proposal.proposed)},"
        f" needs-human-review {len(proposal.needs_human_review)}"
    )
    return "\n".join(lines)


def apply_proposal(proposal: CatalogProposal, catalog_file: Path) -> int:
    """Append proposed entries to the catalog file additively; return rows written.

    The new rows are appended to the end of the existing ``entries:`` list (the
    last top-level block in the catalog).  Existing bytes are preserved exactly,
    so ``git diff`` shows only additions.
    """
    if not proposal.proposed:
        return 0

    text = catalog_file.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    addition = "".join(render_entry_block(entry) for entry in proposal.proposed)
    catalog_file.write_text(text + addition, encoding="utf-8")
    return len(proposal.proposed)


def _proposal_to_json(proposal: CatalogProposal) -> dict[str, object]:
    """Serialize a proposal to a machine-readable mapping."""
    return {
        "proposed": [asdict(entry) for entry in proposal.proposed],
        "needs_human_review": [asdict(item) for item in proposal.needs_human_review],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the catalog-evidence proposer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append proposed entries to the catalog (additive only). Default is dry-run.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write the machine-readable proposal (proposed + needs-human-review) to this path.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Override the catalog file path (default: docs/context/catalog.yaml under repo root).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the repository root (default: git toplevel of the current checkout).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the catalog-evidence proposer CLI."""
    args = _parse_args(argv)
    repo_root = (args.repo_root or _repo_root()).resolve()
    catalog_rel = args.catalog or _CONTEXT_CATALOG
    catalog_file = catalog_rel if catalog_rel.is_absolute() else repo_root / catalog_rel

    proposal = build_proposal(repo_root, catalog_path=catalog_rel)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(_proposal_to_json(proposal), indent=2) + "\n", encoding="utf-8"
        )

    if args.apply:
        written = apply_proposal(proposal, catalog_file)
        print(
            f"Applied {written} catalog entr{'y' if written == 1 else 'ies'};"
            f" needs-human-review {len(proposal.needs_human_review)}."
        )
    else:
        print(render_dry_run(proposal, catalog_rel=catalog_rel))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
