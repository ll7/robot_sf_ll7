#!/usr/bin/env python3
"""Thin, non-semantic adapter: publication release bundle -> issue #5446/#5615/#5616 inputs.

This script does **not** implement mining, resolution, or rendering. It only
reshapes rows from a pinned publication-facing release bundle's
``runs/<arm>/episodes.jsonl`` layout into the flat row contracts the three
merged tools already expect:

- ``robot_sf/benchmark/seed_flip_mining.py`` (issue #5446): needs top-level
  ``repo_commit`` and ``execution_mode`` fields that this bundle carries only
  as nested provenance (``result_provenance.repo_commit`` and
  ``algorithm_metadata.planner_kinematics.execution_mode``). No mining
  threshold, gate, or semantic is touched here; this only *projects* fields
  that are already present in the row under a different path.
- ``robot_sf/benchmark/campaign_atlas.py`` (issue #5616): needs a flat
  ``planner`` / ``scenario_family`` / ``outcome`` string per episode. This
  bundle carries the planner under ``algo``, a real per-scenario archetype
  under ``scenario_params.metadata.archetype`` (used verbatim as
  ``scenario_family``), and a structured ``outcome`` object + ``metrics``
  block that this adapter folds into one label using only literal boolean
  fields already on the row (``collision_event``, ``timeout_event``,
  ``metrics.success``). No outcome semantics are invented.
- ``scripts/tools/campaign_result_store.py`` (``campaign-result-store.v1``,
  consumed read-only by the #5615 resolver): optionally emits a minimal store
  so the resolver can be exercised against a real (if trace-less) episode
  index. The bundle has no ``simulation_trace_export.v1`` artifacts, so the
  emitted ``artifact_uri``/``artifact_sha256`` point at the *episode row*
  itself (a real, hashable, non-fabricated artifact) rather than a trace; the
  resolver is expected to report ``trace-missing`` once it reaches that stage
  -- that is the honest, intended outcome, not a bug in this adapter.

Streaming contract
-------------------
Each arm's ``episodes.jsonl`` is read one line at a time and each output row
is written immediately. Full per-episode payloads (event ledgers, safety
predicate blocks, tracking-precision arrays, ...) are never accumulated
across arms; only the handful of scalar fields each downstream contract
needs are kept in memory, and only for the optional campaign-store row list.

This file lives under ``scripts/analysis/`` per the issue #5446 allowed-paths
scope (a new adapter file is explicitly permitted there) and is reused
read-only by the #5615/#5616 CLIs, which independently list
``scripts/analysis/`` as an allowed path for their own entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: Difficulty suffixes stripped when an episode row lacks a scenario archetype
#: (defensive fallback only; every row observed in the release v0.0.3 bundle
#: carries ``scenario_params.metadata.archetype``).
_DIFFICULTY_SUFFIXES = ("low", "medium", "high")

#: campaign_result_store.py's ROW_STATUS_VALUES enum does not have a "mixed"
#: member; a mixed-execution row (partially native, partially adapted
#: commands) is recorded as "adapter" in the store with this fact stated here
#: rather than silently coerced.
_STORE_ROW_STATUS_MAP = {
    "native": "native",
    "adapter": "adapter",
    "mixed": "adapter",
}


def _get(d: dict[str, Any], *path: str) -> Any:
    """Read a nested dict path, returning ``None`` on any missing hop."""
    cur: Any = d
    for part in path:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _derive_scenario_family(scenario_id: str, archetype: str | None) -> str:
    """Real archetype when present; else difficulty-suffix-stripped scenario id."""
    if archetype:
        return str(archetype)
    parts = scenario_id.rsplit("_", 1)
    if len(parts) > 1 and parts[-1] in _DIFFICULTY_SUFFIXES:
        return parts[0]
    return scenario_id


def _derive_outcome_label(row: dict[str, Any]) -> str:
    """Fold the row's literal outcome/metrics booleans into one atlas outcome label.

    Uses only fields already present on the row (``outcome.collision_event``,
    ``outcome.timeout_event``, ``metrics.success``); no new outcome semantics
    are computed. Collision takes precedence over timeout, which takes
    precedence over the binary success metric; anything else is ``other``.
    """
    outcome = row.get("outcome") or {}
    metrics = row.get("metrics") or {}
    if bool(outcome.get("collision_event")):
        return "collision"
    if bool(outcome.get("timeout_event")):
        return "timeout"
    success = metrics.get("success")
    if isinstance(success, bool) and success:
        return "success"
    if isinstance(success, int | float) and float(success) >= 0.5:
        return "success"
    return "other"


def _mining_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one bundle episode row into the #5446 miner's flat row contract."""
    repo_commit = _get(row, "result_provenance", "repo_commit") or row.get("git_hash")
    execution_mode = _get(row, "algorithm_metadata", "planner_kinematics", "execution_mode")
    return {
        "episode_id": row.get("episode_id"),
        "scenario_id": row.get("scenario_id"),
        "seed": row.get("seed"),
        "config_hash": row.get("config_hash"),
        "algo": row.get("algo"),
        "repo_commit": repo_commit,
        "execution_mode": execution_mode,
        "metrics": row.get("metrics") or {},
    }


def _atlas_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one bundle episode row into the #5616 atlas inventory contract.

    No ``trajectory``/``event_anchors``/``predicate_timeline`` are emitted:
    this release bundle contains no per-step trace exports, so the ensemble
    context view is out of scope for this adapter's output by construction
    (the atlas builder defaults those fields to empty and treats the
    population-level cell summary independently of them).
    """
    scenario_id = str(row.get("scenario_id"))
    archetype = _get(row, "scenario_params", "metadata", "archetype")
    return {
        "episode_id": row.get("episode_id"),
        "planner": row.get("algo"),
        "scenario_id": scenario_id,
        "scenario_family": _derive_scenario_family(scenario_id, archetype),
        "seed": row.get("seed"),
        "outcome": _derive_outcome_label(row),
        "metrics": {
            k: v
            for k, v in (row.get("metrics") or {}).items()
            if k in ("success", "total_collision_count", "near_misses", "snqi")
        },
    }


def _store_row(
    row: dict[str, Any], *, run_id: str, line_no: int, source_path: Path
) -> dict[str, Any]:
    """Build one ``campaign-result-store.v1`` episode row from a bundle episode row.

    ``artifact_uri``/``artifact_sha256`` point at the episode row itself (a
    real, verifiable, non-fabricated artifact) because this bundle has no
    per-step trace export; the #5615 resolver is expected to reach
    ``trace-missing`` when it tries to validate that artifact against
    ``simulation_trace_export.v1``.
    """
    execution_mode = _get(row, "algorithm_metadata", "planner_kinematics", "execution_mode")
    row_status = _STORE_ROW_STATUS_MAP.get(str(execution_mode), "unavailable")
    scenario_id = str(row.get("scenario_id"))
    archetype = _get(row, "scenario_params", "metadata", "archetype")
    raw_line = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return {
        "run_id": run_id,
        "episode_id": row.get("episode_id"),
        "planner": row.get("algo"),
        "scenario_id": scenario_id,
        "scenario_family": _derive_scenario_family(scenario_id, archetype),
        "seed": row.get("seed"),
        "row_status": row_status,
        "artifact_uri": f"{source_path}#L{line_no}",
        "artifact_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
    }


def iter_arm_dirs(payload_dir: Path, *, arms: list[str] | None = None) -> list[Path]:
    """Return sorted ``runs/<arm>`` directories that contain an ``episodes.jsonl``."""
    runs_dir = payload_dir / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs/ directory under {payload_dir}")
    out = []
    for arm_dir in sorted(runs_dir.iterdir()):
        if arms and arm_dir.name not in arms:
            continue
        if (arm_dir / "episodes.jsonl").is_file():
            out.append(arm_dir)
    if not out:
        raise SystemExit(f"no runs/<arm>/episodes.jsonl found under {payload_dir}")
    return out


def _resolve_run_id(arm_dir: Path) -> str:
    """Return the provenance sidecar's real ``run.run_id``, or the arm dir name.

    Returns:
        The run id string.
    """
    provenance_path = arm_dir / "episodes.jsonl.provenance.json"
    if not provenance_path.is_file():
        return arm_dir.name
    try:
        prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return arm_dir.name
    return str(_get(prov, "run", "run_id") or arm_dir.name)


def _adapt_one_arm(
    arm_dir: Path,
    *,
    mining_fh: Any,
    atlas_fh: Any,
    store_rows: list[dict[str, Any]] | None,
    execution_mode_counts: dict[str, int],
) -> int:
    """Stream one arm's ``episodes.jsonl``, writing to the open output handles.

    Returns:
        The number of rows processed for this arm.
    """
    episodes_path = arm_dir / "episodes.jsonl"
    run_id = _resolve_run_id(arm_dir)
    arm_count = 0
    with episodes_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            arm_count += 1
            exec_mode = str(_get(row, "algorithm_metadata", "planner_kinematics", "execution_mode"))
            execution_mode_counts[exec_mode] = execution_mode_counts.get(exec_mode, 0) + 1
            if mining_fh is not None:
                mining_fh.write(json.dumps(_mining_row(row), sort_keys=True) + "\n")
            if atlas_fh is not None:
                atlas_fh.write(json.dumps(_atlas_row(row), sort_keys=True) + "\n")
            if store_rows is not None:
                store_rows.append(
                    _store_row(row, run_id=run_id, line_no=line_no, source_path=episodes_path)
                )
    return arm_count


def adapt(
    payload_dir: Path,
    *,
    mining_rows_out: Path | None,
    atlas_inventory_out: Path | None,
    store_rows_out: Path | None,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    """Stream every arm's ``episodes.jsonl`` once, writing the requested outputs.

    Returns:
        A small run summary (per-arm row counts, execution-mode tally) for the
        caller to fold into an evidence README. No episode payload is
        retained beyond the current line.
    """
    arm_dirs = iter_arm_dirs(payload_dir, arms=arms)

    mining_fh = mining_rows_out.open("w", encoding="utf-8") if mining_rows_out else None
    atlas_fh = atlas_inventory_out.open("w", encoding="utf-8") if atlas_inventory_out else None
    store_rows: list[dict[str, Any]] | None = [] if store_rows_out is not None else None

    per_arm_counts: dict[str, int] = {}
    execution_mode_counts: dict[str, int] = {}
    try:
        for arm_dir in arm_dirs:
            per_arm_counts[arm_dir.name] = _adapt_one_arm(
                arm_dir,
                mining_fh=mining_fh,
                atlas_fh=atlas_fh,
                store_rows=store_rows,
                execution_mode_counts=execution_mode_counts,
            )
    finally:
        if mining_fh is not None:
            mining_fh.close()
        if atlas_fh is not None:
            atlas_fh.close()

    total = sum(per_arm_counts.values())

    if store_rows_out is not None:
        from scripts.tools.campaign_result_store import write_result_store

        write_result_store(
            store_rows_out,
            store_rows,
            study_id="issue_5446_release_0_0_3_candidates",
            command="scripts/analysis/adapt_release_bundle_issue_5446_mining_run.py",
            source_commit=store_rows[0]["run_id"] if store_rows else None,
        )

    return {
        "n_arms": len(arm_dirs),
        "n_rows_total": total,
        "per_arm_row_counts": per_arm_counts,
        "execution_mode_counts": execution_mode_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the adapter CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Reshape a publication release bundle's runs/<arm>/episodes.jsonl rows into the "
            "flat contracts the #5446 miner, #5616 atlas, and (optionally) the #5615 resolver's "
            "campaign-result-store already expect. No mining/resolution/rendering semantics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--bundle-payload",
        required=True,
        type=Path,
        help="Path to the release bundle's payload/ directory (contains runs/<arm>/episodes.jsonl).",
    )
    p.add_argument(
        "--mining-rows-out", type=Path, default=None, help="Write miner input JSONL here."
    )
    p.add_argument(
        "--atlas-inventory-out", type=Path, default=None, help="Write atlas inventory JSONL here."
    )
    p.add_argument(
        "--store-rows-out",
        type=Path,
        default=None,
        help="Write a minimal campaign-result-store.v1 directory here (optional).",
    )
    p.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="Restrict to these runs/<arm> directory names (default: all arms found).",
    )
    p.add_argument(
        "--summary-json", type=Path, default=None, help="Write the run summary JSON here."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the adapter CLI; returns a process exit code."""
    args = build_parser().parse_args(argv)
    summary = adapt(
        args.bundle_payload,
        mining_rows_out=args.mining_rows_out,
        atlas_inventory_out=args.atlas_inventory_out,
        store_rows_out=args.store_rows_out,
        arms=args.arms,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
