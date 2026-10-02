"""Publish compact dev-only HYBDIAG evidence and regenerate its CSV/JSON tables.

Import complete native run folders once with --import-round; subsequently --check
verifies all tables using only committed per-episode summaries, never raw traces.
These experiments are exploratory evidence, not release evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

ARMS = ("off", "static", "platform", "both")
DEFAULT_OUTPUT = Path("docs/validation/hybdiag")


def wilson(count: int, total: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    z = 1.959963984540054
    proportion = count / total
    divisor = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / divisor
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / divisor
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(rows: list[dict]) -> dict:
    """Pool exposure-weighted rates, episode counts and executed separation."""
    total = sum(row["duration_s"] for row in rows)
    result = {"episodes": len(rows), "robot_seconds": total}
    for outcome in ("success", "collision", "timeout", "freezing"):
        count = sum(
            row["freezing"] if outcome == "freezing" else row["outcome"] == outcome for row in rows
        )
        low, high = wilson(count, len(rows))
        result.update(
            {outcome: count, f"{outcome}_wilson_low": low, f"{outcome}_wilson_high": high}
        )
    stopped = sum(row["stopped_time_fraction"] * row["duration_s"] for row in rows)
    no_moving = sum(row["no_feasible_moving_s"] for row in rows)
    events = sum(row["near_miss_events"] for row in rows)
    near_seconds = sum(row["near_miss_steps"] for row in rows) * 0.1
    separations = [row["minimum_pedestrian_separation_m"] for row in rows]
    result.update(
        freezing_rate=result["freezing"] / len(rows),
        stopped_time_fraction=stopped / total,
        no_feasible_moving_s=no_moving,
        mean_no_feasible_moving_s=no_moving / len(rows),
        no_feasible_moving_fraction=no_moving / total,
        near_miss_events=events,
        near_miss_events_per_1000_robot_seconds=events * 1000 / total,
        near_miss_time_fraction=near_seconds / total,
        minimum_pedestrian_separation_m=min(
            (v for v in separations if v is not None), default=None
        ),
    )
    return result


def validate(payload: dict) -> None:
    """Refuse partial grids, duplicate episodes and non-dev experimental seeds."""
    if any(row["world"] not in {"crowd", "empty"} for row in payload["episodes"]):
        raise ValueError("Unknown episode world")
    for world in ("crowd", "empty"):
        manifest = payload["manifests"][world]
        seeds = list(range(1001, 1031)) if world == "crowd" else [1001, 1002]
        scenario_count = 10 if world == "crowd" else 51
        if manifest["seeds"] != seeds or len(set(manifest["scenarios"])) != scenario_count:
            raise ValueError(f"Wrong scenario/seed domain: {world}")
        if manifest["arms"] != list(ARMS):
            raise ValueError(f"Incomplete arms: {world}")
        expected = set(itertools.product(manifest["scenarios"], seeds, ARMS))
        rows = [row for row in payload["episodes"] if row["world"] == world]
        keys = [(row["scenario"], row["seed"], row["arm"]) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"Incomplete or duplicate paired grid: {world}")
        if any(row["outcome"] not in {"success", "collision", "timeout"} for row in rows):
            raise ValueError(f"Invalid outcome: {world}")
        if any(row["duration_s"] <= 0 for row in rows):
            raise ValueError(f"Invalid exposure: {world}")


def import_runs(round_number: int, crowd: Path, empty: Path) -> dict:
    """Strip private runtime metadata and trajectories from complete native results."""
    payload = {
        "round": round_number,
        "status": "AI-GENERATED/NEEDS-REVIEW; dev-only",
        "manifests": {},
        "episodes": [],
    }
    for world, directory in (("crowd", crowd), ("empty", empty)):
        manifest = json.loads((directory / "manifest.json").read_text())
        # Commit provenance and source hashes, without host paths or raw traces.
        payload["manifests"][world] = {
            key: manifest[key]
            for key in ("head", "seeds", "scenarios", "arms", "horizon", "sha256")
        }
        if "reused_episode_sources" in manifest:
            payload["manifests"][world]["reused_episode_sources"] = manifest[
                "reused_episode_sources"
            ]
        payload["manifests"][world]["arm_flags"] = manifest.get(
            "arm_flags",
            {
                arm: {
                    "physical_static_exclusion_enabled": arm in {"static", "both"},
                    "platform_speed_candidates_enabled": arm in {"platform", "both"},
                    "sentinel_guard_coupled_to_static_flag": arm in {"static", "both"},
                }
                for arm in ARMS
            },
        )
        for path in sorted(directory.glob("*.json")):
            if path.name in {"manifest.json", "reuse-provenance.json"}:
                continue
            row = json.loads(path.read_text())
            compact = {
                key: row[key]
                for key in (
                    "scenario",
                    "seed",
                    "arm",
                    "outcome",
                    "duration_s",
                    "final_goal_distance_m",
                )
            }
            compact.update(
                world=world,
                execution_head=row.get("execution_head", manifest["head"]),
                **row["metrics"],
            )
            compact["next_valid_field_observed"] = row.get("next_valid_field_observed")
            decision = row["runtime"]["last_decision"]
            debug = decision["candidate_evaluator_debug"]
            compact["final_diagnostics"] = {
                "command": decision["selected_command"],
                "feasible_moving_count": debug["feasible_moving_count"],
                "stop_kind": debug["stop_kind"],
                "constraints": debug["constraints"],
                "progress_windows": decision["progress_windows"],
                "speed_safety": decision["speed_safety"],
            }
            payload["episodes"].append(compact)
    validate(payload)
    return payload


def summarize(payload: dict) -> dict:
    """Generate per-scenario and pooled tables plus paired acceptance witnesses."""
    validate(payload)
    cells, failures, gates = [], [], {}
    for world in ("crowd", "empty"):
        rows = [row for row in payload["episodes"] if row["world"] == world]
        lookup = {(row["scenario"], row["seed"], row["arm"]): row for row in rows}
        scenarios = payload["manifests"][world]["scenarios"]
        for scenario in ["ALL", *scenarios]:
            for arm in ARMS:
                subset = [
                    row
                    for row in rows
                    if row["arm"] == arm and (scenario == "ALL" or row["scenario"] == scenario)
                ]
                cells.append(
                    dict(
                        round=payload["round"],
                        world=world,
                        scenario=scenario,
                        arm=arm,
                        **aggregate(subset),
                    )
                )
        for row in rows:
            if row["arm"] != "off" and row["outcome"] != "success":
                off = lookup[(row["scenario"], row["seed"], "off")]
                if off["outcome"] == "success":
                    failures.append(
                        {
                            key: row[key]
                            for key in (
                                "world",
                                "scenario",
                                "seed",
                                "arm",
                                "outcome",
                                "final_goal_distance_m",
                            )
                        }
                    )
    for arm in ARMS[1:]:
        new_empty = [row for row in failures if row["world"] == "empty" and row["arm"] == arm]
        safe = all(
            cell["collision_wilson_low"]
            <= next(
                base["collision_wilson_high"]
                for base in cells
                if base["world"] == cell["world"]
                and base["scenario"] == cell["scenario"]
                and base["arm"] == "off"
            )
            for cell in cells
            if cell["arm"] == arm
        )
        gates[arm] = {
            "new_empty_failures": len(new_empty),
            "collision_ci_not_above_off": safe,
            "accepted": safe and not new_empty,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "round": payload["round"],
        "episode_payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "cells": cells,
        "new_failures": failures,
        "gates": gates,
    }


def csv_text(rows: list[dict]) -> str:
    """Use a normal header as the first line, allowing standard CSV readers."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def audit_braking_bound(directory: Path, native: Path) -> None:
    """Summarize cap bypasses in the 45 successful Round-2 station trajectories."""
    payload = json.loads((directory / "round2-episodes.json").read_text())
    validate(payload)
    selected = [
        row
        for row in payload["episodes"]
        if row["world"] == "crowd"
        and row["scenario"] == "classic_station_platform_medium"
        and row["arm"] in {"platform", "both"}
        and row["outcome"] == "success"
    ]
    results = []
    for episode in selected:
        filename = f"{episode['scenario']}__{episode['seed']}__{episode['arm']}__crowd.jsonl.gz"
        breaches = []
        with gzip.open(native / filename, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                decision = row["decision"]
                safety = decision.get("speed_safety", {})
                cap = safety.get("braking_cap")
                if (
                    decision.get("selected_source") == "admissible_speed"
                    and cap is not None
                    and row["command"][0] > cap + 1e-6
                ):
                    breaches.append(
                        {
                            "step": row["step"],
                            "command_m_s": row["command"][0],
                            "braking_cap_m_s": cap,
                            "pre_action_surface_gap_m": safety["min_surface_clearance"],
                        }
                    )
        results.append(
            {
                "scenario": episode["scenario"],
                "seed": episode["seed"],
                "arm": episode["arm"],
                "injected_bound_exceedance_steps": len(breaches),
                "first_bound_exceedance": breaches[0] if breaches else None,
            }
        )
    result = {
        "status": "AI-GENERATED/NEEDS-REVIEW; dev-only",
        "head": payload["manifests"]["crowd"]["head"],
        "meaning": "Requested injected commands above current-position braking cap; not actual plant-speed or contact claims",
        "episodes": results,
    }
    (directory / "round2-braking-bound-audit.json").write_text(json.dumps(result, indent=2) + "\n")


def outputs(directory: Path) -> dict[str, str]:
    """Derive reproducible tables from the committed compact input summaries."""
    results, summaries = {}, []
    for path in sorted(directory.glob("round*-episodes.json")):
        summary = summarize(json.loads(path.read_text()))
        summaries.append(summary)
        prefix = f"round{summary['round']}"
        results[f"{prefix}-summary.json"] = json.dumps(summary, indent=2) + "\n"
        results[f"{prefix}-results.csv"] = csv_text(summary["cells"])
    if len(summaries) == 2:
        pooled = [
            cell for summary in summaries for cell in summary["cells"] if cell["scenario"] == "ALL"
        ]
        results["round2-vs-round3.csv"] = csv_text(pooled)
        earlier = json.loads((directory / "round2-episodes.json").read_text())
        later = json.loads((directory / "round3-episodes.json").read_text())
        lookup = {
            (row["world"], row["scenario"], row["seed"], row["arm"]): row
            for row in earlier["episodes"]
        }
        transitions = []
        for row in later["episodes"]:
            key = (row["world"], row["scenario"], row["seed"], row["arm"])
            if lookup[key]["outcome"] == "success" and row["outcome"] != "success":
                transitions.append(
                    {
                        k: row[k]
                        for k in (
                            "world",
                            "scenario",
                            "seed",
                            "arm",
                            "outcome",
                            "final_goal_distance_m",
                            "final_diagnostics",
                        )
                    }
                )
        results["round3-new-failures-vs-round2.json"] = json.dumps(transitions, indent=2) + "\n"
    audit_path = directory / "round2-braking-bound-audit.json"
    if audit_path.exists():
        audited = []
        for row in json.loads(audit_path.read_text())["episodes"]:
            first = row["first_bound_exceedance"]
            audited.append(
                {
                    "scenario": row["scenario"],
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "injected_bound_exceedance_steps": row["injected_bound_exceedance_steps"],
                    "first_step": first["step"] if first else None,
                    "first_command_m_s": first["command_m_s"] if first else None,
                    "first_braking_cap_m_s": first["braking_cap_m_s"] if first else None,
                }
            )
        results["round2-braking-bound-audit.csv"] = csv_text(audited)
    return results


def main() -> None:
    """Import native summaries or verify/rebuild all committed result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--import-round", type=int, choices=(2, 3))
    parser.add_argument("--crowd", type=Path)
    parser.add_argument("--empty", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--audit-braking-bound",
        type=Path,
        help="Native Round-2 crowd folder; publish compact injected-cap audit, without raw traces",
    )
    args = parser.parse_args()
    if args.import_round is not None:
        if args.check or args.crowd is None or args.empty is None:
            parser.error("Import requires --crowd/--empty and cannot use --check")
        payload = import_runs(args.import_round, args.crowd, args.empty)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"round{args.import_round}-episodes.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    if args.audit_braking_bound is not None:
        if args.check:
            parser.error("Audit import cannot use --check")
        audit_braking_bound(args.output, args.audit_braking_bound)
    generated = outputs(args.output)
    if not generated:
        parser.error("No committed episode summaries found")
    for name, text in generated.items():
        path = args.output / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f"Out-of-date table: {path}")
        else:
            path.write_text(text)
    print(f"{'Verified' if args.check else 'Generated'} {len(generated)} CSV/JSON tables")


if __name__ == "__main__":
    main()
