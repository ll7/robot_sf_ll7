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


def comparison_markdown(summaries: list[dict]) -> str:
    """Render the public pooled comparison directly from the verified summaries."""
    lines = [
        "# HYBDIAG Round 2 versus Round 3",
        "",
        "AI-GENERATED/NEEDS-REVIEW. Development seeds only. Restoring the pedestrian",
        "braking bound removes the earlier platform gain; zero collisions do not",
        "prove unchanged pedestrian safety. Per-scenario intervals and all metrics",
        "are in the corresponding `round*-results.csv` files.",
        "",
        "| Round/world | Arm | S/C/T | Success Wilson 95% | Collision Wilson 95% | Timeout Wilson 95% | Freeze | Stopped % | No moving s | Min ped m | Near events; per 1,000 robot-s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        for cell in summary["cells"]:
            if cell["scenario"] != "ALL":
                continue
            intervals = [
                f"{100 * cell[f'{kind}_wilson_low']:.1f}–{100 * cell[f'{kind}_wilson_high']:.1f}%"
                for kind in ("success", "collision", "timeout")
            ]
            separation = cell["minimum_pedestrian_separation_m"]
            minimum = "—" if separation is None else f"{separation:.3f}"
            lines.append(
                f"| {cell['round']}/{cell['world']} | {cell['arm']} | "
                f"{cell['success']}/{cell['collision']}/{cell['timeout']} | "
                + " | ".join(intervals)
                + f" | {cell['freezing']}/{cell['episodes']} | {100 * cell['stopped_time_fraction']:.2f} | "
                f"{cell['no_feasible_moving_s']:.1f} | {minimum} | "
                f"{cell['near_miss_events']}; {cell['near_miss_events_per_1000_robot_seconds']:.2f} |"
            )
    lines.extend(["", "Each crowded arm has 300 episodes; each empty arm has 102.", ""])
    return "\n".join(lines)


def validate_wall_witness(witness: dict, final: dict, combined: dict) -> float:
    """Check physical-versus-comfort geometry and the paired combined success."""
    x, y = witness["position"]
    ax, ay, bx, by = witness["closest_wall"]
    dx, dy = bx - ax, by - ay
    fraction = min(1, max(0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    distance = math.hypot(x - ax - fraction * dx, y - ay - fraction * dy)
    if not math.isclose(distance, witness["center_distance_m"], abs_tol=1e-12):
        raise ValueError("Incorrect wall distance witness")
    if not witness["physical_radius_m"] < distance < witness["legacy_hard_radius_m"]:
        raise ValueError("Wall witness does not distinguish body from comfort")
    if combined["outcome"] != "success" or final["feasible_moving_count"] != 0:
        raise ValueError("Missing wall-margin counterexample")
    return distance


def classify_new_failures(directory: Path, earlier: dict, later: dict) -> dict:
    """Join every new failure to a compact, independently checkable cause witness.

    Unknown failures refuse publication until evidence is supplied. Restored-bound
    progress loss is not labeled genuine physical infeasibility: the comparison
    changes several safeguards and does not isolate a single causal intervention.
    """
    keys = ("world", "scenario", "seed", "arm")
    old = {tuple(row[k] for k in keys): row for row in earlier["episodes"]}
    current = {tuple(row[k] for k in keys): row for row in later["episodes"]}
    audited = {
        (row["scenario"], row["seed"], row["arm"]): row
        for row in json.loads((directory / "round2-braking-bound-audit.json").read_text())[
            "episodes"
        ]
    }
    walls = {
        (row["scenario"], row["seed"], row["arm"]): row
        for row in json.loads((directory / "round3-wall-witnesses.json").read_text())["episodes"]
    }
    classified = []
    for key, row in sorted(current.items()):
        if row["arm"] == "off" or row["outcome"] == "success":
            continue
        off = current[(row["world"], row["scenario"], row["seed"], "off")]
        comparisons = []
        if off["outcome"] == "success":
            comparisons.append("paired_off")
        if old[key]["outcome"] == "success":
            comparisons.append("same_arm_round2")
        if not comparisons:
            continue
        identity = (row["scenario"], row["seed"], row["arm"])
        record = {k: row[k] for k in keys}
        record.update(new_vs=comparisons, outcome=row["outcome"])
        final = row["final_diagnostics"]
        if identity in audited and row["world"] == "crowd":
            witness = audited[identity]
            first = witness["first_bound_exceedance"]
            if first is None:
                raise ValueError(f"Missing retained-bound progress evidence: {identity}")
            constraints = ", ".join(
                f"{c['constraint']} at {c['threshold']} ({c['rejected']} rejections)"
                for c in final["constraints"]
            )
            record.update(
                classification="progress shortfall under restored braking bound",
                evidence=(
                    f"Round 2 requested {first['command_m_s']:.6f} m/s above "
                    f"{first['braking_cap_m_s']:.6f} m/s braking cap at step {first['step']}; "
                    f"Round 3 times out with {final['feasible_moving_count']} feasible moving "
                    f"candidates at the final step and {row['final_goal_distance_m']:.3f} m remaining; "
                    f"no-moving time {row['no_feasible_moving_s']:.1f}/60 s; "
                    f"final constraints: {constraints or 'none'}. "
                    "Combined safeguards remove the former gain; no isolated causal or genuine-infeasibility proof."
                ),
                next_step="planner route/progress work within retained safety bounds",
                witness=witness,
            )
        elif identity in walls and row["world"] == "crowd":
            witness = walls[identity]
            combined = current[(row["world"], row["scenario"], row["seed"], "both")]
            distance = validate_wall_witness(witness, final, combined)
            record.update(
                classification="artificial geometric infeasibility",
                evidence=(
                    f"At rest wall-center distance {distance:.12f} m exceeds physical "
                    f"{witness['physical_radius_m']:.2f} m but is below legacy hard "
                    f"{witness['legacy_hard_radius_m']:.2f} m; no moving candidate; both arm succeeds."
                ),
                next_step="supplied physical-static flag; planner defect, not physical limit",
                witness=witness,
            )
        else:
            raise ValueError(f"Unclassified new failure: {key}")
        classified.append(record)
    return {
        "status": "AI-GENERATED/NEEDS-REVIEW; dev-only",
        "meaning": "Union of new failures versus paired off and same-arm Round 2; one cause/evidence line per episode",
        "episodes": classified,
    }


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


def audit_injected_commands(source: Path, world: str, arm: str) -> int:
    """Count selected added commands, refusing any current-position cap breach."""
    commands = 0
    for path in sorted(source.glob(f"*__{arm}__{world}.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                decision = row["decision"]
                if decision.get("selected_source") != "admissible_speed":
                    continue
                commands += 1
                cap = decision.get("speed_safety", {}).get("braking_cap")
                if cap is not None and row["command"][0] > cap + 1e-6:
                    raise ValueError(f"Injected command exceeds braking bound: {path.name}")
    return commands


def audit_native_controls(directory: Path, crowd: Path, empty: Path, previous: Path) -> None:
    """Prove native off identity and audit all selected injected braking commands.

    Previous contains round2-measure and round2-empty. Only per-episode hashes and
    aggregate command counts are published, never runtime trajectories.
    """
    result = {"status": "AI-GENERATED/NEEDS-REVIEW; dev-only", "off": [], "injected_bounds": {}}
    for world, source, old_folder in (
        ("crowd", crowd, "round2-measure"),
        ("empty", empty, "round2-empty"),
    ):
        for path in sorted(source.glob(f"*__off__{world}.jsonl.gz")):
            encoded = []
            for trace in (previous / old_folder / path.name, path):
                with gzip.open(trace, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                values = [
                    {k: row[k] for k in ("position", "end_position", "command", "collision")}
                    for row in rows
                ]
                encoded.append(json.dumps(values, sort_keys=True, separators=(",", ":")).encode())
            if encoded[0] != encoded[1]:
                raise ValueError(f"Default native trajectory changed: {path.name}")
            result["off"].append(
                {"episode": path.stem, "sha256": hashlib.sha256(encoded[1]).hexdigest()}
            )
        for arm in ("platform", "both"):
            result["injected_bounds"][f"{world}/{arm}"] = {
                "injected_commands": audit_injected_commands(source, world, arm),
                "bound_exceedances": 0,
            }
    if len(result["off"]) != 402:
        raise ValueError("Native controls require all 300 crowded and 102 empty off episodes")
    (directory / "round3-native-audit.json").write_text(json.dumps(result, indent=2) + "\n")


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
        results["round2-vs-round3.md"] = comparison_markdown(summaries)
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
        results["round3-failure-classifications.json"] = (
            json.dumps(classify_new_failures(directory, earlier, later), indent=2) + "\n"
        )
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


def import_requested_native(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Perform explicitly requested imports before generating verifiable tables."""
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
    if args.audit_native_controls is not None:
        if args.check or args.crowd is None or args.empty is None:
            parser.error("Native audit requires --crowd/--empty and cannot use --check")
        audit_native_controls(args.output, args.crowd, args.empty, args.audit_native_controls)


def main() -> None:
    """Import native summaries or verify/rebuild all committed result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--import-round", type=int, choices=(2, 3))
    parser.add_argument("--crowd", type=Path)
    parser.add_argument("--empty", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--audit-native-controls",
        type=Path,
        help="Prior artifact parent containing round2-measure/round2-empty; requires --crowd/--empty",
    )
    parser.add_argument(
        "--audit-braking-bound",
        type=Path,
        help="Native Round-2 crowd folder; publish compact injected-cap audit, without raw traces",
    )
    args = parser.parse_args()
    import_requested_native(args, parser)
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
