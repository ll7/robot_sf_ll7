#!/usr/bin/env python3
"""Issue #5416 campaign runner: execute the full frozen SIPP four-geometry matrix.

Plain-language summary
----------------------
This runner is the campaign dispatcher the armed private-ops queue entry needs
for issue #6082. It iterates the exact 100 rows of the issue #5416 preregistered
SIPP four-geometry matrix (5 planners x 4 failure geometries x 5 paired seeds),
executes each row through the merged native-command entrypoint, and writes one
episode JSONL per row in the shape the paired-outcome analyzer consumes.

It is the campaign-execution plumbing only. It does **not** interpret benchmark
outcomes, promote planner/safety/liveness claims, or edit paper/dissertation
text. Every produced row is ``exploratory_synthetic_benchmark_only`` until the
native matrix, geometry gate, paired seeds, and full provenance pass the frozen
acceptance contract.

What it reuses (read-only, never redefined here)
------------------------------------------------
* ``check_issue_5416_sipp_four_geometry_packet.py``: the preregistration
  acceptance oracle. The runner enumerates exactly the rows that packet defines
  (its scenario/seed/planner dimensions) and fails closed when the packet's
  geometry gate is not ready.
* ``check_issue_5416_sipp_native_smoke.py``: its ``_FROZEN_NATIVE_CONFIGS``
  mapping (planner id -> tracked native-command config) and its
  ``_selected_scenario``/``REPO_ROOT``/``SCHEMA_PATH`` helpers.
* ``analyze_issue_5416_sipp_four_geometry.py``: its per-row parse logic is the
  single source of truth for "this episode is an eligible native evidence row",
  so a row skipped-on-resume is judged by the same contract that scores it.
* ``robot_sf.benchmark.map_runner.run_map_batch`` with ``algo="native_command"``:
  the same execution path the frozen native smoke validator drives.

Resume discipline (#5538/#5940)
-------------------------------
A row whose output already exists **and is checker-valid** (analyzer-eligible:
native execution, no fallback, correct planner/scenario/seed identity, deadlock
metric present, no integrity contradictions, complete provenance) is skipped.
A row is never re-executed once it is complete; a partial or invalid row is
re-run rather than silently trusted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner import run_map_batch

# Reuse the frozen packet checker (acceptance oracle) and the native smoke
# validator's frozen planner->config mapping and helpers. These are imported,
# never redefined, so the campaign cannot drift from the frozen contract.
from scripts.analysis.analyze_issue_5416_sipp_four_geometry import (
    _parse_row as analyzer_parse_row,
)
from scripts.validation.check_issue_5416_sipp_four_geometry_packet import (
    load_packet,
    validate_packet,
)
from scripts.validation.check_issue_5416_sipp_native_smoke import (
    _FROZEN_NATIVE_CONFIGS,
    REPO_ROOT,
    SCHEMA_PATH,
    _selected_scenario,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ISSUE = 5416
HORIZON = 500
DT_SECONDS = 0.1
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"


class CampaignError(ValueError):
    """Raised when the frozen campaign contract cannot be established."""


@dataclass(frozen=True)
class CampaignRow:
    """One frozen (planner, scenario, seed) cell of the 100-row matrix."""

    index: int
    planner_id: str
    scenario_id: str
    seed: int
    native_config_path: Path

    @property
    def key(self) -> tuple[str, str, int]:
        """The (scenario_id, seed, planner_id) identity used by the analyzer."""
        return (self.scenario_id, self.seed, self.planner_id)


def _native_config_for(planner_id: str) -> Path:
    """Return the tracked native-command config path for one roster planner."""
    mapping = dict(_FROZEN_NATIVE_CONFIGS)
    if planner_id not in mapping:
        raise CampaignError(f"planner {planner_id!r} has no frozen native-command config mapping")
    return mapping[planner_id]


def _packet_dimensions(
    packet: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """Return the frozen (planners, scenarios, seeds) tuples from the packet."""
    contract = packet["scenario_contract"]
    scenarios = tuple(str(row["scenario_id"]) for row in contract["selected_rows"])
    seeds = tuple(int(seed) for seed in contract["result_producing_seeds"])
    planners = tuple(str(row["planner_id"]) for row in packet["planner_roster"]["required"])
    return planners, scenarios, seeds


def enumerate_rows(packet: dict[str, Any]) -> list[CampaignRow]:
    """Enumerate the frozen 100-row matrix in a deterministic order.

    The planner order follows the frozen roster, scenarios follow the frozen
    ``selected_rows`` order, and seeds follow the frozen result-producing order.
    This keeps ``--dry-run`` output stable and ``--rows N-M`` reproducible.
    """
    planners, scenarios, seeds = _packet_dimensions(packet)
    rows: list[CampaignRow] = []
    index = 0
    for planner_id in planners:
        native_config_path = _native_config_for(planner_id)
        for scenario_id in scenarios:
            for seed in seeds:
                index += 1
                rows.append(
                    CampaignRow(
                        index=index,
                        planner_id=planner_id,
                        scenario_id=scenario_id,
                        seed=seed,
                        native_config_path=native_config_path,
                    )
                )
    return rows


def plan_output_path(output_root: Path, row: CampaignRow) -> Path:
    """Return the one-row episode JSONL path for a campaign cell."""
    return output_root / row.planner_id / row.scenario_id / f"seed_{row.seed}" / "episodes.jsonl"


def read_completed_episode(path: Path) -> dict[str, Any] | None:
    """Read exactly one episode record from a row's JSONL, or ``None``."""
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    if len(lines) != 1:
        return None
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def episode_is_checker_valid(
    episode: dict[str, Any], row: CampaignRow, *, planners: Sequence[str]
) -> tuple[bool, list[str]]:
    """Return whether one episode is an analyzer-eligible native evidence row.

    This is the resume-skip contract: it reuses the paired-outcome analyzer's
    per-row parse so a skipped row is judged by exactly the rules that score it.
    A row is valid only when it is native, non-fallback, correctly identified,
    carries the deadlock/stall metric, has complete provenance, and has no
    integrity contradictions.
    """
    measurement, reasons, _diagnostic_errors = analyzer_parse_row(
        episode, row.planner_id, row.scenario_id, row.seed, planners
    )
    if reasons or measurement is None:
        return False, sorted(set(reasons or ["row cannot be keyed"]))
    return True, []


def row_is_complete(
    row: CampaignRow, output_root: Path, *, planners: Sequence[str]
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    """Return whether a campaign cell has a checker-valid episode and its data."""
    episode = read_completed_episode(plan_output_path(output_root, row))
    if episode is None:
        return False, None, ["episode output is missing or not exactly one record"]
    valid, reasons = episode_is_checker_valid(episode, row, planners=planners)
    return valid, episode if valid else None, reasons


def execute_row(
    row: CampaignRow,
    *,
    output_root: Path,
    packet: dict[str, Any],
    workers: int = 1,
) -> dict[str, Any]:
    """Execute one frozen native row through the merged native-command path.

    Mirrors ``check_issue_5416_sipp_native_smoke._run_native_row``: it selects
    the scenario, pins its single seed, and runs ``run_map_batch`` with
    ``algo="native_command"`` so every planner flows through the same tracked
    geometry-aware native protocol and emits the per-row deadlock/stall metric.
    """
    scenario = _selected_scenario(packet, row.scenario_id, row.seed)
    episodes_path = plan_output_path(output_root, row)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    # ``resume=False`` so a single-cell rerun overwrites a stale/partial file;
    # campaign-level resume (skip already-valid rows) is handled by the caller.
    run_map_batch(
        [scenario],
        episodes_path,
        SCHEMA_PATH,
        scenario_path=REPO_ROOT / packet["scenario_contract"]["scenario_matrix"],
        horizon=HORIZON,
        dt=DT_SECONDS,
        algo="native_command",
        algo_config_path=str(row.native_config_path),
        workers=workers,
        resume=False,
        record_forces=False,
    )
    return {"row": row.index, "episodes_path": str(episodes_path)}


def _parse_rows_spec(spec: str | None, total: int) -> tuple[int, int]:
    """Parse a 1-based inclusive ``N`` or ``N-M`` slice against ``total`` rows."""
    if spec is None:
        return 1, total
    try:
        if "-" in spec:
            start_text, end_text = spec.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(spec)
    except ValueError as exc:
        raise CampaignError(f"--rows must be N or N-M (got {spec!r})") from exc
    if start < 1 or end < start or end > total:
        raise CampaignError(f"--rows {spec!r} is out of range for {total} planned rows")
    return start, end


def select_rows(rows: list[CampaignRow], spec: str | None) -> list[CampaignRow]:
    """Return the inclusive 1-based slice of planned rows selected by ``--rows``."""
    start, end = _parse_rows_spec(spec, len(rows))
    return [row for row in rows if start <= row.index <= end]


def run_campaign(  # noqa: C901
    *,
    config_path: Path,
    output_root: Path | None = None,
    rows_spec: str | None = None,
    dry_run: bool = False,
    workers: int = 1,
    packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan, (optionally) execute, and summarize the frozen campaign matrix.

    Args:
        config_path: Path to the issue #5416 preregistration packet YAML.
        output_root: Directory for per-row episode JSONL. Required unless
            ``dry_run`` is set.
        rows_spec: Optional 1-based inclusive ``N`` or ``N-M`` row slice.
        dry_run: When ``True``, list the planned rows without executing.
        workers: Worker count forwarded to ``run_map_batch``.
        packet: Pre-loaded packet (used by tests to avoid re-reading the file).

    Returns:
        A compact campaign summary with planned, selected, executed, skipped,
        and failed row counts plus per-row deadlock/stall evidence.
    """
    if packet is None:
        packet = load_packet(config_path)
    gate = validate_packet(packet, repo_root=REPO_ROOT)
    if gate.get("status") != "ready":
        raise CampaignError(
            "frozen packet geometry gate is not ready: "
            + ", ".join(str(row) for row in gate.get("blocked_rows", []))
        )
    rows = enumerate_rows(packet)
    planned_count = len(rows)
    if planned_count != len(_FROZEN_NATIVE_CONFIGS) * 4 * 5:
        # Stop condition from issue #6082: do not improvise rows.
        raise CampaignError(
            f"preregistration matrix enumerated {planned_count} rows, "
            "expected the frozen 5x4x5=100 grid"
        )
    selected = select_rows(rows, rows_spec)
    planners = _packet_dimensions(packet)[0]

    summary: dict[str, Any] = {
        "issue": ISSUE,
        "schema_version": "issue_5416_campaign_run.v1",
        "claim_boundary": (
            "campaign execution plumbing only; exploratory_synthetic_benchmark_only "
            "until the native matrix, geometry gate, paired seeds, and provenance pass"
        ),
        "config_path": str(config_path),
        "output_root": str(output_root) if output_root else None,
        "dry_run": dry_run,
        "rows_spec": rows_spec,
        "planned_rows": planned_count,
        "selected_rows": len(selected),
        "executed_rows": 0,
        "skipped_rows": 0,
        "failed_rows": 0,
        "rows": [],
    }

    for row in selected:
        entry: dict[str, Any] = {
            "index": row.index,
            "planner_id": row.planner_id,
            "scenario_id": row.scenario_id,
            "seed": row.seed,
            "key": list(row.key),
        }
        summary["rows"].append(entry)

    if dry_run:
        return summary

    if output_root is None:
        raise CampaignError("--output-root is required unless --dry-run is set")
    output_root = Path(output_root)

    for row, entry in zip(selected, summary["rows"], strict=True):
        complete, episode, reasons = row_is_complete(row, output_root, planners=planners)
        if complete:
            entry["status"] = "skipped"
            entry["reason"] = "already checker-valid"
            summary["skipped_rows"] += 1
            metrics = episode.get("metrics") if isinstance(episode, dict) else None
            if isinstance(metrics, dict) and isinstance(metrics.get("deadlock"), bool):
                entry["deadlock"] = metrics["deadlock"]
            continue
        try:
            execute_row(
                row,
                output_root=output_root,
                packet=packet,
                workers=workers,
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            entry["status"] = "failed"
            entry["reason"] = str(exc) or type(exc).__name__
            summary["failed_rows"] += 1
            continue
        # Re-validate the freshly written episode so the summary reflects the
        # same checker-valid contract used for resume-skip.
        complete, episode, reasons = row_is_complete(row, output_root, planners=planners)
        if not complete:
            entry["status"] = "failed"
            entry["reason"] = "post-execute row not checker-valid: " + "; ".join(reasons)
            summary["failed_rows"] += 1
            continue
        entry["status"] = "executed"
        entry["episodes_path"] = str(plan_output_path(output_root, row))
        metrics = episode.get("metrics") if isinstance(episode, dict) else None
        if isinstance(metrics, dict) and isinstance(metrics.get("deadlock"), bool):
            # The per-row deadlock/stall metric delivered by the native-command
            # arm (#5887); surfaced verbatim for the campaign ledger.
            entry["deadlock"] = metrics["deadlock"]
        summary["executed_rows"] += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    """Plan and (optionally) execute the frozen #5416 campaign matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--rows",
        default=None,
        help="1-based inclusive row slice (N or N-M) of the planned matrix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list planned rows without executing episodes",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        summary = run_campaign(
            config_path=args.config,
            output_root=args.output_root,
            rows_spec=args.rows,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    # Convert expected validation/contract failures into a non-zero exit with a
    # clear message. Fatal process conditions deliberately remain unhandled.
    except (CampaignError, OSError, ValueError, KeyError) as exc:
        if args.as_json:
            print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        else:
            print(f"blocked: {exc}")
        return 1
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        status = (
            "failed"
            if summary["failed_rows"]
            else ("executed" if summary["executed_rows"] else "planned")
        )
        print(
            f"{status}: issue #{ISSUE} campaign "
            f"({summary['planned_rows']} planned, {summary['selected_rows']} selected)"
        )
        for entry in summary["rows"]:
            print(
                f"  row {entry['index']:>3}: {entry['planner_id']}"
                f" / {entry['scenario_id']} / seed {entry['seed']}"
            )
    return 1 if summary["failed_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
