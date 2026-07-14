#!/usr/bin/env python3
"""Build the (scenario, planner, seed, episode-id) trace re-export list for #5447.

Context
-------
The #5615 resolver (``resolve_candidate_trace_resolution``) joins each
``seed_flip_inversion_candidates.v1`` candidate to a campaign episode by
looking up ``candidate.get("seed")`` / ``candidate.get("episode_id")``. The
#5446 miner never populates those two fields on a candidate: a candidate is a
*(scenario, planner) cell* aggregate, and its per-seed outcomes live inside
``reproducibility.raw_seed_outcomes`` (``seed_flip`` archetype) or
``upset_outcome.raw_paired_outcomes`` (``planner_upset`` archetype). Running
the resolver as shipped therefore reports every real candidate as
``provenance-incomplete`` before it ever reaches trace search -- this is a
genuine contract gap between #5446's candidate granularity and #5615's
per-episode resolution model, not a bug in either tool (see the evidence
README for the full writeup).

This script does not change mining or resolution semantics. It performs the
one join #5447 actually needs: for the Pareto-*selected* candidates in a
mined manifest, expand each candidate's embedded raw per-seed outcomes into
concrete ``(scenario_id, planner, seed, episode_id)`` tuples by looking the
episode id up in the same flat mining-rows file the miner was run against
(the adapter's ``--mining-rows-out``, which already carries the bundle's real
``episode_id`` per row). Only selected candidates are expanded: they are the
non-dominated, "worth confirming" cases per the miner's own claim boundary;
un-selected/triage-only candidates are not asked to re-export by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load rows from a JSONL file.

    Returns:
        The parsed row list.
    """
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _episode_index(mining_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    """Index mining rows by ``(scenario_id, planner, seed)`` -> ``episode_id``.

    Returns:
        The lookup index built from every mining row's real episode id.
    """
    index: dict[tuple[str, str, str], str] = {}
    for row in mining_rows:
        key = (str(row.get("scenario_id")), str(row.get("algo")), str(row.get("seed")))
        index[key] = str(row.get("episode_id"))
    return index


def build_reexport_list(
    candidate_manifest: dict[str, Any], mining_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Expand every selected candidate's raw seed evidence into tuples needing a trace.

    Returns:
        A small manifest: ``source_manifest_hash`` echo, ``tuples`` (sorted,
        deduplicated), and per-candidate provenance for each tuple.
    """
    index = _episode_index(mining_rows)
    tuples: dict[tuple[str, str, str], dict[str, Any]] = {}

    for cand in candidate_manifest.get("candidates", []):
        if not cand.get("selected"):
            continue
        scenario_id = str(cand["scenario_id"])
        seed_planner_pairs: list[tuple[str, str]] = []
        if cand["archetype"] == "seed_flip":
            for seed in cand["reproducibility"]["raw_seed_outcomes"]:
                seed_planner_pairs.append((cand["planner"], seed))
        elif cand["archetype"] == "planner_upset":
            raw = cand["upset_outcome"]["raw_paired_outcomes"]
            for planner, seed_outcomes in raw.items():
                for seed in seed_outcomes:
                    seed_planner_pairs.append((planner, seed))
        else:
            continue

        for planner, seed in seed_planner_pairs:
            key = (scenario_id, planner, seed)
            episode_id = index.get(key)
            entry = tuples.setdefault(
                key,
                {
                    "scenario_id": scenario_id,
                    "planner": planner,
                    "seed": seed,
                    "episode_id": episode_id,
                    "episode_id_status": "found"
                    if episode_id is not None
                    else "not_found_in_mining_rows",
                    "requested_by_candidates": [],
                },
            )
            entry["requested_by_candidates"].append(cand["candidate_id"])

    ordered = sorted(tuples.values(), key=lambda e: (e["scenario_id"], e["planner"], e["seed"]))
    for entry in ordered:
        entry["requested_by_candidates"] = sorted(set(entry["requested_by_candidates"]))

    return {
        "schema_version": "issue_5446_trace_reexport_list.v1",
        "issue": "#5446 -> #5447",
        "claim_boundary": (
            "Targeted trace re-export request list only. Derived from the Pareto-selected "
            "candidates' embedded raw per-seed outcomes because the #5615 resolver, as shipped, "
            "cannot resolve per-seed episodes from a cell-level candidate (see README)."
        ),
        "source_candidate_manifest_hash": candidate_manifest.get("summary"),
        "n_tuples": len(ordered),
        "n_episode_id_found": sum(1 for e in ordered if e["episode_id_status"] == "found"),
        "tuples": ordered,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--mining-rows", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args(argv)

    candidate_manifest = json.loads(args.candidates.read_text(encoding="utf-8"))
    mining_rows = _load_jsonl(args.mining_rows)
    manifest = build_reexport_list(candidate_manifest, mining_rows)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest['n_tuples']} trace re-export tuples to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
