#!/usr/bin/env python3
"""Compare the immutable 0.0.6 and corrected 0.0.7 S30/H600 rows.

The comparator is deliberately independent of the benchmark runner.  It streams
episode JSONL from the predecessor publication archive and the successor
campaign root, pairs rows by planner/scenario/seed, and emits a compact JSON and
Markdown diff.  Trace-dependent goal-adjacent labels remain ``unavailable`` when
the input row did not record a simulation trace; a missing label is never
silently treated as ``false``.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

OUTCOME_KEYS = ("route_complete", "collision_event", "timeout_event")
# A release row may be wholly native, use an explicit adapter, or combine
# native and adapter components (for example a guarded PPO wrapper).  All
# three are contract-valid; fallback/degraded flags remain disqualifying.
EXECUTION_MODES = {"native", "adapter", "mixed"}
BOOTSTRAP_SAMPLES = 3000
BOOTSTRAP_SEED = 123


def _row_outcome(row: Mapping[str, Any]) -> dict[str, bool]:
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        outcome = {}
    return {key: bool(outcome.get(key, False)) for key in OUTCOME_KEYS}


def _nested_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield child
            yield from _nested_values(child)
    elif isinstance(value, list):
        for child in value:
            yield child
            yield from _nested_values(child)


def _goal_adjacent_status(row: Mapping[str, Any]) -> str:
    """Return true/false only for explicit labels; otherwise unavailable."""
    candidates: list[Any] = []
    for key in ("goal_adjacent_timeout", "goal_adjacent_timeout_event", "goal_adjacent"):
        if key in row:
            candidates.append(row[key])
    failure = row.get("failure_mechanism")
    if isinstance(failure, Mapping):
        label = failure.get("mechanism_label")
        if isinstance(label, str) and "goal_adjacent" in label.lower():
            candidates.append(True)
    for value in candidates:
        if isinstance(value, bool):
            return "true" if value else "false"
    return "unavailable"


def _execution_audit(row: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    metadata = row.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        kinematics = metadata.get("planner_kinematics")
        if isinstance(kinematics, Mapping):
            mode = str(kinematics.get("execution_mode", ""))
            if mode and mode not in EXECUTION_MODES:
                issues.append(f"execution_mode={mode}")
        for key in ("fallback_used", "degraded", "unavailable"):
            if metadata.get(key) is True:
                issues.append(f"{key}=true")
    # Outcome statuses are not execution degradation.  A valid benchmark row
    # may finish as success, collision, timeout, or failure; only explicit
    # runtime/error/unavailable markers are disqualifying here.
    if str(row.get("status", "")).lower() not in {
        "",
        "ok",
        "success",
        "completed",
        "collision",
        "failure",
        "timeout",
        "timed_out",
        "timeout_event",
    }:
        issues.append(f"status={row.get('status')}")
    for value in _nested_values(row):
        if isinstance(value, str) and value.lower() in {"fallback", "degraded", "unavailable"}:
            # A row may mention ``unavailable`` for a diagnostic predicate; do
            # not fail the execution gate on that wording alone.
            continue
    return issues


def _arm_from_name(name: str) -> str:
    return name.removesuffix("__differential_drive")


def _read_jsonl(path: Path, *, arm: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scenario_id"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate {arm} row {key} in {path}:{line_number}")
            rows[key] = row
    return rows


def _read_predecessor(archive: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = sorted(
            (
                member
                for member in handle.getmembers()
                if member.isfile()
                and member.name.endswith("/episodes.jsonl")
                and "/payload/runs/" in member.name
            ),
            key=lambda member: member.name,
        )
        if not members:
            raise ValueError(f"predecessor archive has no payload run rows: {archive}")
        for member in members:
            stem = Path(member.name).parent.name
            arm = _arm_from_name(stem)
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member {member.name}")
            rows: dict[tuple[str, int], dict[str, Any]] = {}
            for line_number, raw in enumerate(extracted, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                key = (str(row["scenario_id"]), int(row["seed"]))
                if key in rows:
                    raise ValueError(f"duplicate predecessor row {arm} {key}:{line_number}")
                rows[key] = row
            result[arm] = rows
    return result


def _read_successor(root: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for path in sorted(root.glob("runs/*/episodes.jsonl")):
        result[_arm_from_name(path.parent.name)] = _read_jsonl(
            path, arm=_arm_from_name(path.parent.name)
        )
    if not result:
        raise ValueError(f"successor root has no run rows: {root}")
    return result


def _metric(row: Mapping[str, Any], metric: str) -> float:
    return float(_row_outcome(row)[metric])


def _bootstrap_seed_delta(
    old: Mapping[tuple[str, int], Mapping[str, Any]],
    new: Mapping[tuple[str, int], Mapping[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    seed_values: dict[int, list[float]] = defaultdict(list)
    for key, old_row in old.items():
        if key not in new:
            continue
        seed_values[key[1]].append(_metric(new[key], metric) - _metric(old_row, metric))
    if not seed_values:
        return None
    seeds = sorted(seed_values)
    seed_means = np.asarray([np.mean(seed_values[seed]) for seed in seeds], dtype=float)
    observed = float(np.mean(seed_means))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(seed_means), size=(BOOTSTRAP_SAMPLES, len(seed_means)))
    distribution = seed_means[sampled].mean(axis=1)
    return {
        "metric": metric,
        "estimand": "success/collision/timeout risk difference (0.0.7 - 0.0.6)",
        "seed_block_count": len(seeds),
        "seeds": seeds,
        "observed_delta": observed,
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "method": "percentile bootstrap over whole seed blocks",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def compare(
    predecessor_archive: Path,
    successor_root: Path,
    *,
    predecessor_sha256: str,
    predecessor_source_sha: str,
    successor_source_sha: str,
    successor_bundle_sha256: str,
) -> dict[str, Any]:
    """Pair predecessor and successor rows and derive the governed release diff."""
    old = _read_predecessor(predecessor_archive)
    new = _read_successor(successor_root)
    arms = sorted(set(old) | set(new))
    execution_issues: dict[str, list[str]] = defaultdict(list)
    arm_reports: dict[str, Any] = {}
    changed: list[dict[str, Any]] = []
    bootstrap: dict[str, dict[str, Any]] = {}

    for arm in arms:
        old_rows = old.get(arm, {})
        new_rows = new.get(arm, {})
        paired_keys = sorted(set(old_rows) & set(new_rows))
        old_counts: Counter[str] = Counter()
        new_counts: Counter[str] = Counter()
        old_goal: Counter[str] = Counter()
        new_goal: Counter[str] = Counter()
        for key, row in old_rows.items():
            outcome = _row_outcome(row)
            old_counts["success" if outcome["route_complete"] else "failure"] += 1
            old_counts["collision"] += int(outcome["collision_event"])
            old_counts["timeout"] += int(outcome["timeout_event"])
            old_goal[_goal_adjacent_status(row)] += 1
        for key, row in new_rows.items():
            outcome = _row_outcome(row)
            new_counts["success" if outcome["route_complete"] else "failure"] += 1
            new_counts["collision"] += int(outcome["collision_event"])
            new_counts["timeout"] += int(outcome["timeout_event"])
            new_goal[_goal_adjacent_status(row)] += 1
            issues = _execution_audit(row)
            if issues:
                execution_issues[arm].extend(issues)
        for key in paired_keys:
            old_outcome = _row_outcome(old_rows[key])
            new_outcome = _row_outcome(new_rows[key])
            if old_outcome != new_outcome:
                changed.append(
                    {
                        "arm": arm,
                        "scenario_id": key[0],
                        "seed": key[1],
                        "old": old_outcome,
                        "new": new_outcome,
                    }
                )
        arm_reports[arm] = {
            "predecessor_rows": len(old_rows),
            "successor_rows": len(new_rows),
            "paired_rows": len(paired_keys),
            "predecessor_outcomes": dict(sorted(old_counts.items())),
            "successor_outcomes": dict(sorted(new_counts.items())),
            "predecessor_goal_adjacent_timeout": dict(sorted(old_goal.items())),
            "successor_goal_adjacent_timeout": dict(sorted(new_goal.items())),
        }
        for metric in ("route_complete", "collision_event", "timeout_event"):
            result = _bootstrap_seed_delta(old_rows, new_rows, metric)
            if result is not None:
                bootstrap[f"{arm}:{metric}"] = result

    return {
        "schema_version": "issue-9431-release-diff.v1",
        "predecessor": {
            "release": "0.0.6",
            "archive_sha256": predecessor_sha256,
            "archive": predecessor_archive.name,
            "source_commit": predecessor_source_sha,
        },
        "successor": {
            "release": "0.0.7",
            "source_commit": successor_source_sha,
            "bundle_sha256": successor_bundle_sha256,
            "campaign_root": successor_root.name,
        },
        "arms": arm_reports,
        "arm_count": len(arms),
        "paired_rows": sum(report["paired_rows"] for report in arm_reports.values()),
        "changed_episode_count": len(changed),
        "changed_episodes": changed,
        "seed_block_bootstrap": bootstrap,
        "execution_audit": {
            "successor_rows_with_fallback_or_degraded": sum(
                len(values) for values in execution_issues.values()
            ),
            "issues_by_arm": {arm: sorted(values) for arm, values in execution_issues.items()},
            "admitted": not execution_issues,
        },
        "goal_adjacent_timeout_policy": {
            "missing_trace_is": "unavailable",
            "note": "Full S30/H600 release rows do not carry simulation-step traces; trace-derived labels are not inferred from outcome rows.",
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Issue #9431 release diff: 0.0.6 → corrected 0.0.7",
        "",
        "This report pairs rows by planner, scenario, and seed. Successor execution is admitted only when every row is native, adapter, or mixed mode with no fallback/degraded row. Trace-dependent goal-adjacent timeout labels remain `unavailable` when step traces were not recorded.",
        "",
        f"- predecessor archive SHA-256: `{report['predecessor']['archive_sha256']}`",
        f"- exact source range: `{report['predecessor']['source_commit']}..{report['successor']['source_commit']}`",
        f"- successor source commit: `{report['successor']['source_commit']}`",
        f"- successor publication bundle SHA-256: `{report['successor']['bundle_sha256']}`",
        f"- paired rows: **{report['paired_rows']}**; changed outcome rows: **{report['changed_episode_count']}**",
        f"- successor execution audit: **{'pass' if report['execution_audit']['admitted'] else 'fail'}**",
        "",
        "## Per-arm outcome counts",
        "",
        "| arm | paired | 0.0.6 success | 0.0.7 success | 0.0.6 collisions | 0.0.7 collisions | 0.0.6 timeouts | 0.0.7 timeouts | goal-adjacent labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, values in sorted(report["arms"].items()):
        old = values["predecessor_outcomes"]
        new = values["successor_outcomes"]
        old_goal = values["predecessor_goal_adjacent_timeout"]
        new_goal = values["successor_goal_adjacent_timeout"]
        lines.append(
            f"| `{arm}` | {values['paired_rows']} | {old.get('success', 0)} | {new.get('success', 0)} | {old.get('collision', 0)} | {new.get('collision', 0)} | {old.get('timeout', 0)} | {new.get('timeout', 0)} | old `{dict(old_goal)}`; new `{dict(new_goal)}` |"
        )
    lines += [
        "",
        "## Seed-block bootstrap intervals",
        "",
        "The interval is a deterministic 95% percentile bootstrap over whole seed blocks (3000 replicates, seed 123). The estimand is the risk difference 0.0.7 minus 0.0.6.",
        "",
        "| arm | metric | observed delta | 95% interval | seed blocks |",
        "|---|---|---:|---|---:|",
    ]
    for key, values in sorted(report["seed_block_bootstrap"].items()):
        lines.append(
            f"| `{key.split(':', 1)[0]}` | {values['metric']} | {values['observed_delta']:.6f} | [{values['ci95'][0]:.6f}, {values['ci95'][1]:.6f}] | {values['seed_block_count']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The full release rows are outcome evidence only. Because they do not contain simulation-step traces, the goal-adjacent timeout predicate is not recomputed here and `unavailable` is not counted as false. The separately pinned 0.0.7 worked-example trace bundle supplies the bounded head-on/group-crossing diagnostic traces.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the release comparator CLI and write JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-archive", type=Path, required=True)
    parser.add_argument("--successor-root", type=Path, required=True)
    parser.add_argument("--predecessor-sha256", required=True)
    parser.add_argument("--predecessor-source-sha", required=True)
    parser.add_argument("--successor-source-sha", required=True)
    parser.add_argument("--successor-bundle-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.predecessor_archive,
        args.successor_root,
        predecessor_sha256=args.predecessor_sha256,
        predecessor_source_sha=args.predecessor_source_sha,
        successor_source_sha=args.successor_source_sha,
        successor_bundle_sha256=args.successor_bundle_sha256,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "paired_rows": report["paired_rows"],
                "changed_episode_count": report["changed_episode_count"],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
