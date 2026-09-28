#!/usr/bin/env python3
"""Classify issue #9631 social-force residual failures in the frozen 0.0.6 bundle.

The frozen 0.0.6 S30/H600 publication bundle does not retain simulation step
traces, so the trace-dependent goal-adjacent exclusion owned by issue #9429 is
unavailable for every timeout row.  This analyzer therefore reports the
goal-adjacent/residual split as not separable and instead classifies every
social-force episode by its retained termination signal (success, collision
with pedestrian/obstacle/wall subtype, timeout with terminated/max_steps
split) crossed with scenario family and seed.  No episode is stepped or
rerun, and no runtime or frozen artifact is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPORT_SCHEMA_VERSION = "issue_9631_sf_residual_report.v1"
SF_ARM = "social_force__differential_drive"
EXPECTED_EPISODES = 1440


@dataclass(frozen=True, slots=True)
class EpisodeClass:
    """Retained-signal termination class for one social-force episode."""

    scenario_id: str
    scenario_family: str
    seed: int
    status: str
    termination: str
    collision_subtype: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Malformed JSON on line {line_number} of {path}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_checksums(bundle_root: Path) -> tuple[int, int]:
    """Verify every payload checksum and return ``(files, bytes)``."""
    checksum_path = bundle_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing bundle checksum manifest: {checksum_path}")
    files = 0
    total_bytes = 0
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}: {raw_line!r}") from exc
        target = bundle_root / relative.lstrip("* ")
        if not target.is_file():
            raise FileNotFoundError(f"Checksum target is missing: {target}")
        observed = _sha256(target)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: expected {expected}, observed {observed}"
            )
        files += 1
        total_bytes += target.stat().st_size
    return files, total_bytes


def _scenario_family(row: Mapping[str, Any]) -> str:
    params = row.get("scenario_params")
    metadata = params.get("metadata") if isinstance(params, Mapping) else None
    family = metadata.get("archetype") if isinstance(metadata, Mapping) else None
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"Episode {row.get('episode_id')} has no scenario archetype")
    return family.strip()


def _collision_subtype(row: Mapping[str, Any]) -> str:
    """Split a collision episode by retained collision-count fields."""
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    ped = metrics.get("ped_collision_count") or 0
    obs = metrics.get("obstacle_collision_count") or 0
    wall = metrics.get("wall_collisions") or 0
    parts = []
    if ped:
        parts.append("pedestrian")
    if obs:
        parts.append("obstacle")
    if wall:
        parts.append("wall")
    return "+".join(parts) if parts else "untyped"


def classify_episode(row: Mapping[str, Any]) -> EpisodeClass:
    """Classify one episode by retained termination signals only."""
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    status = str(row.get("status", ""))
    termination = str(row.get("termination_reason", ""))
    subtype: str | None = None
    if outcome.get("route_complete") is True:
        status = "success"
    elif outcome.get("collision_event") is True:
        status = "collision"
        subtype = _collision_subtype(row)
    elif outcome.get("timeout_event") is True:
        status = "timeout"
    return EpisodeClass(
        scenario_id=str(row.get("scenario_id", "")),
        scenario_family=_scenario_family(row),
        seed=int(row.get("seed", -1)),
        status=status,
        termination=termination,
        collision_subtype=subtype,
    )


def is_noncollision_timeout(row: Mapping[str, Any]) -> bool:
    """Return whether a row is a timeout without a collision event."""
    outcome = row.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    return outcome.get("timeout_event") is True and outcome.get("collision_event") is not True


def summarize_timeouts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize retained progress signals over non-collision timeouts."""
    timeouts = [row for row in rows if is_noncollision_timeout(row)]
    deadlocks = 0
    deadlock_known = 0
    ftp_values: list[float] = []
    stalled_values: list[float] = []
    speed_values: list[float] = []
    for row in timeouts:
        metrics = row.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        deadlock = metrics.get("deadlock")
        if isinstance(deadlock, bool):
            deadlock_known += 1
            deadlocks += deadlock
        for key, sink in (
            ("failure_to_progress", ftp_values),
            ("stalled_time", stalled_values),
            ("avg_speed", speed_values),
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                sink.append(float(value))
    max_steps = sum(1 for row in timeouts if str(row.get("termination_reason")) == "max_steps")
    deadlock_true_all = sum(
        1
        for row in rows
        if isinstance((row.get("metrics") or {}).get("deadlock"), bool)
        and (row.get("metrics") or {}).get("deadlock")
    )
    deadlock_true_success = sum(
        1
        for row in rows
        if (row.get("metrics") or {}).get("deadlock") is True
        and classify_episode(row).status == "success"
    )

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "timeouts": len(timeouts),
        "terminated": len(timeouts) - max_steps,
        "max_steps": max_steps,
        "deadlock_true": deadlocks,
        "deadlock_known": deadlock_known,
        "deadlock_true_non_timeout": deadlock_true_all - deadlocks,
        "deadlock_true_success": deadlock_true_success,
        "failure_to_progress": _stats(ftp_values),
        "stalled_time_s": _stats(stalled_values),
        "avg_speed_m_s": _stats(speed_values),
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: Any) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_number(value) for value in row) + " |")
    return lines


def build_report(args: argparse.Namespace) -> str:
    """Build the complete Markdown report from validated frozen inputs."""
    bundle_root = args.bundle_root.resolve()
    publication_manifest = _read_json(bundle_root / "publication_manifest.json")
    payload_manifest = _read_json(bundle_root / "payload/manifest.json")
    checksum_files, checksum_bytes = verify_bundle_checksums(bundle_root)
    declared_totals = publication_manifest.get("totals", {})
    if checksum_files != int(declared_totals.get("file_count", -1)):
        raise ValueError("Checksum file count does not match publication_manifest.json")
    if checksum_bytes != int(declared_totals.get("total_bytes", -1)):
        raise ValueError("Checksum byte count does not match publication_manifest.json")

    episodes_path = bundle_root / "payload" / "runs" / SF_ARM / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing social-force episode file: {episodes_path}")
    rows = _load_jsonl(episodes_path)
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} social-force rows, found {len(rows)}")
    source_commit = str(payload_manifest["git_hash"])
    mismatched = [row["episode_id"] for row in rows if row.get("git_hash") != source_commit]
    if mismatched:
        raise ValueError(f"{len(mismatched)} rows do not record source commit {source_commit}")

    classes = [classify_episode(row) for row in rows]
    by_family: dict[str, list[EpisodeClass]] = {}
    for item in classes:
        by_family.setdefault(item.scenario_family, []).append(item)
    subtype_counter: Counter[str] = Counter(
        item.collision_subtype for item in classes if item.collision_subtype
    )
    timeout_summary = summarize_timeouts(rows)
    successes = sum(1 for item in classes if item.status == "success")
    collisions = sum(1 for item in classes if item.status == "collision")

    seed_rows = [
        (row, classify_episode(row))
        for row in rows
        if row.get("seed") == args.case_seed and "group_crossing" in str(row.get("scenario_id"))
    ]

    lines = [
        "# Social-force residual failures in the frozen 0.0.6 campaign",
        "",
        "Claim boundary: descriptive analysis of the frozen release bundle only. No episode was "
        "stepped or rerun, and no runtime or frozen artifact was changed. Evidence tier is "
        "analysis-only; no causal, planner-general, paper, or benchmark-success claim follows.",
        "",
        "Goal-adjacent exclusion status: not separable. Issue #9429 proved its trace-dependent "
        "`goal_adjacent_timeout.v1` predicate unavailable for all 2,108 classified/timeout rows "
        "because the bundle sets `record_simulation_step_trace=false`. `NA` is not zero, so no "
        "episode below can be assigned to, or excluded as, a goal-adjacent timeout. The residual "
        "subset is therefore the full non-success set described by retained termination signals, "
        "not a positively identified post-exclusion remainder.",
        "",
        "## Input identity",
        "",
        f"- Bundle: `{publication_manifest.get('bundle_name')}`",
        f"- Campaign: `{payload_manifest.get('campaign_name', 'benchmark_0_0_6_s30_h600_20260911')}`",
        f"- Source commit: `{source_commit}`",
        f"- Checksum verification: {checksum_files} files, {checksum_bytes} bytes",
        f"- Social-force rows read: {len(rows)}",
        f"- Row source-commit agreement: {len(rows) - len(mismatched)}/{len(rows)}",
        "",
        "## Termination summary (retained signals)",
        "",
        f"- Successes (`route_complete`): {successes}/{len(rows)} ({successes / len(rows):.4f})",
        "- Collisions (`collision_event`): "
        f"{collisions}/{len(rows)} ({collisions / len(rows):.4f})",
        "- Timeouts without collision: "
        f"{timeout_summary['timeouts']}/{len(rows)} ({timeout_summary['timeouts'] / len(rows):.4f}), "
        f"of which `terminated`: {timeout_summary['terminated']}, `max_steps`: "
        f"{timeout_summary['max_steps']}",
        "",
        "Collision subtypes (retained count fields; disjoint in this arm, so the "
        "counts below sum to the collision total):",
        "",
        *_markdown_table(
            ["collision subtype", "episodes"],
            [(subtype, count) for subtype, count in sorted(subtype_counter.items())],
        ),
        "",
        "## Outcomes by scenario family",
        "",
        *_markdown_table(
            ["scenario family", "episodes", "success", "collision", "timeout"],
            [
                (
                    family,
                    len(items),
                    sum(1 for item in items if item.status == "success"),
                    sum(1 for item in items if item.status == "collision"),
                    sum(1 for item in items if item.status == "timeout"),
                )
                for family, items in sorted(by_family.items())
            ],
        ),
        "",
        "## Retained progress signals over non-collision timeouts",
        "",
        f"- Deadlock detector true: {timeout_summary['deadlock_true']}/"
        f"{timeout_summary['deadlock_known']} timeouts with a known deadlock flag "
        f"({timeout_summary['deadlock_true_non_timeout']} true outside timeouts, "
        f"{timeout_summary['deadlock_true_success']} true on successes)",
        "- `failure_to_progress` steps: "
        + ", ".join(
            f"{key}={_format_number(value)}"
            for key, value in timeout_summary["failure_to_progress"].items()
        ),
        "- `stalled_time` seconds: "
        + ", ".join(
            f"{key}={_format_number(value)}"
            for key, value in timeout_summary["stalled_time_s"].items()
        ),
        "- `avg_speed` m/s: "
        + ", ".join(
            f"{key}={_format_number(value)}"
            for key, value in timeout_summary["avg_speed_m_s"].items()
        ),
        "",
        f"## Retained-trace case audit (group-crossing seed {args.case_seed})",
        "",
    ]
    if not seed_rows:
        lines += [
            f"No group-crossing rows for seed {args.case_seed} were retained; the dissertation "
            "seed-22 stall cannot be re-examined from this bundle.",
            "",
        ]
    else:
        lines += [
            "The dissertation records a group-crossing seed-22 stall with about 2.9 m net "
            "displacement while the nearest pedestrian remains more than about 6.5 m away, and "
            "states the episode 'does not identify what holds the robot back'. The retained "
            "rows below are the same seed index under `paper_eval_s30` seeds 111-140 "
            f"(seed {args.case_seed}); they are timeout cases with the recorded signals shown, "
            "not a mechanism identification:",
            "",
            *_markdown_table(
                [
                    "scenario",
                    "status",
                    "steps",
                    "avg_speed m/s",
                    "stalled_time s",
                    "failure_to_progress steps",
                    "deadlock",
                    "ped collisions",
                    "obstacle collisions",
                    "min clearance m",
                ],
                [
                    (
                        item.scenario_id,
                        item.status,
                        row.get("steps"),
                        (row.get("metrics") or {}).get("avg_speed"),
                        (row.get("metrics") or {}).get("stalled_time"),
                        (row.get("metrics") or {}).get("failure_to_progress"),
                        (row.get("metrics") or {}).get("deadlock"),
                        (row.get("metrics") or {}).get("ped_collision_count"),
                        (row.get("metrics") or {}).get("obstacle_collision_count"),
                        (row.get("metrics") or {}).get("clearing_distance_min"),
                    )
                    for row, item in seed_rows
                ],
            ),
            "",
        ]
    lines += [
        "## Table 7.1 cross-check",
        "",
        "The dissertation table behind issue #9631 reports 30 social-force successes out of "
        f"1,440 (2.08%). The retained bundle rows record {successes} `route_complete` episodes "
        f"out of {len(rows)} ({successes / len(rows):.2%}). The counting definitions differ "
        "(the dissertation cell is a typed aggregate; the bundle field is the runtime "
        "completion flag), so neither number is corrected here. The dissertation owner owns "
        "reconciling the two definitions; this report uses the retained row fields throughout.",
        "",
        "## Competing-explanation verdicts",
        "",
        "1. Residual failures are predominantly collision terminations in specific "
        "interaction-heavy families: supported as a descriptive concentration. Collisions are "
        f"{collisions}/{len(rows) - successes} of the non-success episodes, and the family "
        "table above shows where they concentrate. Whether any collision is itself a "
        "goal-adjacent limit-cycle variant is unidentifiable without step traces.",
        "2. Some failures are non-goal-adjacent low-progress timeouts: observed as timeout "
        "cases with retained stall signatures (deadlock-true timeouts and the seed-case "
        "audit), but 'non-goal-adjacent' cannot be asserted while the goal-adjacent "
        "predicate is unavailable.",
        "3. Failures are scenario-specific with no planner-wide mechanism: not excluded. "
        "Both collisions and timeouts concentrate in a subset of families, which is "
        "compatible with scenario-specific causes and with a planner-wide weakness that "
        "only some families trigger.",
        "4. Available release outputs do not preserve enough mechanism signals: confirmed "
        "for the goal-adjacent question (no step traces, no force time series) and for any "
        "claim requiring per-step command/yaw or force-component series.",
        "",
        "Overall: at least one recurring residual signature is connected to recorded "
        "signals across many episodes — collision termination concentrated by family, and "
        "stall-flagged timeouts — so the result is a descriptive classification, not an "
        "unresolved verdict. The causal attribution of either signature remains "
        "unestablished, and the goal-adjacent share remains unmeasured.",
        "",
        "## Stop-rule compliance",
        "",
        "Only the exact pinned bundle, the #9429 report, source/configuration identity, and "
        "retained traces were audited. No new trace was captured, no simulator was run, no "
        "parameter was swept, and no planner or campaign comparison was performed.",
        "",
        "## Reproduction",
        "",
        "Run the committed analyzer against the checksummed release bundle:",
        "",
        "```bash",
        "uv run python scripts/analysis/analyze_sf_residual_failures_issue_9631.py \\",
        "  --bundle-root /path/to/benchmark_0_0_6_s30_h600_20260911_publication_bundle \\",
        "  --case-seed 132 \\",
        "  --output docs/analysis/issue_9631_sf_residual_failures_0_0_6.md",
        "```",
        "",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the issue #9631 residual-failure analyzer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--case-seed", type=int, default=132)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args)
    args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
