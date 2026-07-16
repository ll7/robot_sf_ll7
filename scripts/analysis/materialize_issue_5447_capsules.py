"""Materialize the pinned Chapter 7 case-capsule artifact (issue #5447).

This script is the faithful, reproducible driver that turns the validated
``#5446`` seed-flip / planner-inversion candidate manifest into the frozen
``ch7_case_capsule_manifest.v1`` artifact, a pre-selection ledger (full
candidate pool vs. chosen / unavailable), and a ``SHA256SUMS`` sidecar so the
capsule set is pinned to its inputs and re-derivable in a clean worktree.

It does NOT fabricate capsules: it simply re-runs the issue-#5447 builder
(``robot_sf.benchmark.case_capsules``) over the real candidate manifest and
records exactly what came out, including the genuinely unavailable archetypes.

Run from the repository root:

    uv run python scripts/analysis/materialize_issue_5447_capsules.py

Outputs (under ``docs/context/evidence/issue_5447_ch7_case_capsules/``):
    ch7_case_capsule_manifest.v1.json  - the capsule manifest (schema-versioned)
    pre_selection_ledger.v1.json       - full pool audit + selection decision
    build_command.v1.txt               - the exact figure/export commands
    SHA256SUMS                          - checksums of every emitted file
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.case_capsules import (
    canonical_sha256,
    validate_ch7_case_capsule_manifest,
)
from robot_sf.evidence.writers import write_json, write_text

EVID_DIR = Path("docs/context/evidence/issue_5447_ch7_case_capsules")
CANDIDATE_REL = (
    "docs/context/evidence/issue_5446_release_0_0_3_candidates/"
    "seed_flip_inversion_candidates.v1.json.gz"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize() -> int:
    """Build and write the frozen capsule artifact set for issue #5447.

    Returns:
        The number of admitted capsules (informational; does not fail closed).
    """
    candidate_path = Path(CANDIDATE_REL)
    if not candidate_path.exists():
        raise SystemExit(f"missing validated candidate manifest at {candidate_path}")
    raw = candidate_path.read_bytes()
    # Decode a gzipped manifest transparently; detect gzip by magic bytes so a
    # plain .json copy of the same manifest also works (hermetic test path).
    if raw[:2] == b"\x1f\x8b":
        import gzip

        raw = gzip.decompress(raw)
    candidate_manifest: dict[str, Any] = json.loads(raw.decode("utf-8"))

    # Import the builder here so a missing candidate file fails fast above.
    from robot_sf.benchmark.case_capsules import build_ch7_case_capsule_manifest

    manifest = build_ch7_case_capsule_manifest(candidate_manifest)
    validation = validate_ch7_case_capsule_manifest(manifest)
    if not validation.ok:
        raise SystemExit(
            f"capsule manifest failed structural validation: {validation.structural_violations}"
        )

    EVID_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = EVID_DIR / "ch7_case_capsule_manifest.v1.json"
    write_json(manifest_path, manifest)

    # Pre-selection ledger: full candidate pool vs. chosen / unavailable.
    candidates = candidate_manifest["candidates"]
    chosen_ids = {
        c["source_candidate_id"]
        for c in manifest["capsules"]
        if c["status"] == "admitted" and c.get("source_candidate_id")
    }
    ledger = {
        "schema_version": "pre_selection_ledger.v1",
        "issue": "#5447",
        "candidate_manifest": {
            "rel_path": CANDIDATE_REL,
            "sha256_file": _sha256_file(candidate_path),
            "sha256_canonical_dict": canonical_sha256(candidate_manifest),
            "schema_version": candidate_manifest.get("schema_version"),
            "n_candidates": len(candidates),
            "archetypes": sorted({c.get("archetype") for c in candidates}),
        },
        "selection_result": {
            "status": manifest["status"],
            "n_admitted": manifest["summary"]["n_admitted"],
            "n_unavailable": manifest["summary"]["n_unavailable"],
            "admitted_archetypes": manifest["summary"]["admitted_archetypes"],
            "unavailable_archetypes": manifest["summary"]["unavailable_archetypes"],
            "meets_min_capsules": manifest["summary"]["meets_min_capsules"],
        },
        "capsules": [
            {
                "archetype": c["archetype"],
                "title": c["title"],
                "status": c["status"],
                "evidence_grade": c.get("evidence_grade"),
                "source_candidate_id": c.get("source_candidate_id"),
                "reason": c.get("reason"),
            }
            for c in manifest["capsules"]
        ],
        # Full pre-selection pool: every candidate, with whether it was chosen.
        "candidate_pool": [
            {
                "candidate_id": c.get("candidate_id"),
                "archetype": c.get("archetype"),
                "scenario_id": c.get("scenario_id"),
                "planner": c.get("planner"),
                "selected": bool(c.get("selected")),
                "triage_only": bool(c.get("triage_only")),
                "chosen": c.get("candidate_id") in chosen_ids,
            }
            for c in sorted(
                candidates, key=lambda x: (x.get("archetype") or "", x.get("candidate_id") or "")
            )
        ],
        "author_pending_fields": validation.author_pending,
    }
    ledger_path = EVID_DIR / "pre_selection_ledger.v1.json"
    write_json(ledger_path, ledger)

    # Exact dissertation / figure-export commands (issue #5447 remaining work).
    build_cmd_path = EVID_DIR / "build_command.v1.txt"
    write_text(
        build_cmd_path,
        (
            "# Issue #5447 Chapter 7 case-capsule build & export commands (pinned).\n"
            "# Reproduce the capsule manifest + ledger from the frozen #5446 candidate manifest.\n"
            "uv run python scripts/analysis/materialize_issue_5447_capsules.py\n"
            "\n"
            "# One-off CLI form (values pinned to the frozen-SHA candidate manifest):\n"
            "uv run python scripts/analysis/build_ch7_case_capsules_issue_5447.py \\\n"
            f"    --candidates {CANDIDATE_REL} \\\n"
            "    --json docs/context/evidence/issue_5447_ch7_case_capsules/ch7_case_capsule_manifest.v1.json \\\n"
            "    --validate\n"
            "\n"
            "# Downstream vector-figure rendering (reuses robot_sf.benchmark.figures);\n"
            "# requires the pinned episode trajectories resolved by #5615/#5446 trace resolution.\n"
            "# Author-pending narrative/figure fields must be completed and an\n"
            "# independent pinned-SHA visual/evidence review recorded before dissertation use.\n"
        ),
        issue_ref="robot_sf#5447 cheap-lane worker",
    )

    # Sidecar checksums.
    sums_path = EVID_DIR / "SHA256SUMS"
    lines = []
    for name in (
        "ch7_case_capsule_manifest.v1.json",
        "pre_selection_ledger.v1.json",
        "build_command.v1.txt",
    ):
        p = EVID_DIR / name
        if p.exists():
            lines.append(f"{_sha256_file(p)}  {name}")
    lines.append(f"{_sha256_file(candidate_path)}  {CANDIDATE_REL}")
    write_text(
        sums_path,
        "\n".join(lines) + "\n",
        issue_ref="robot_sf#5447 cheap-lane worker",
    )

    return manifest["summary"]["n_admitted"]


if __name__ == "__main__":
    n = materialize()
    print(f"materialized issue #5447 capsule artifact set; admitted={n}")
