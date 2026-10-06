"""Regenerate the committed BIKEFIX CSV/JSON from source-pinned raw records and traces.

CSV bytes contain only a header and data; review and custody information live in
JSON sidecars. --output-root writes the same docs/validation filenames in an
independent tree so byte-for-byte regeneration can be checked without mutation.
"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from bicycle_probe_metrics import displacement_stuck_steps

from robot_sf.evidence.writers import sha256_file, write_json, write_review_sidecar

ARMS = ["DD", "T60-30", "T60-45", "T60-30-on", "T60-45-on"]
SCENARIOS = [
    "classic_cross_trap_high",
    "classic_station_platform_medium",
    "classic_t_intersection_medium",
    "francis2023_frontal_approach",
    "francis2023_narrow_hallway",
    "francis2023_robot_crowding",
]
STAT_FIELDS = [
    "n",
    "stuck",
    "zero_turn",
    *[
        item
        for outcome in ["success", "collision", "timeout"]
        for item in [outcome, outcome + "_rate", outcome + "_lo", outcome + "_hi"]
    ],
]
FAILURE_FIELDS = [
    "cohort",
    "cell",
    "planner",
    "arm",
    "probe",
    "scenario",
    "seed",
    "source_sha",
    "outcome",
    "zero_turn",
    "was_creeping_at_termination",
    "min_clearance_last_2s_m",
    "clearance_window_steps",
    "terminal_yaw_request_rad_s",
    "terminal_speed_m_s",
    "terminal_safety_interventions",
    "creep_steps_last_2s",
    "safety_creep_steps",
    "noise_creep_steps",
    "reverse_creep_steps",
    "stuck",
    "yaw_law_max_error",
    "empty_witness_steps",
    "alternative_success",
    "bicycle_path_witness_cell",
    "bicycle_path_witness_min_clearance_m",
    "classification",
    "reason",
]


def load_rows(directory):
    """Read and validate real runner outcomes, retaining a unique cell identity."""
    rows = [json.loads(p.read_text()) for p in sorted((directory / "results").glob("*.json"))]
    assert rows and len({r["cell"] for r in rows}) == len(rows)
    for row in rows:
        assert 1001 <= row["seed"] <= 1010, "STOP: non-dev raw seed"
        assert row["probe"] == 2 or row["seed"] <= 1005
        assert row["status"] != "error" and row.get("outcome") is not None, row["cell"]
        for key in ["route_complete", "collision_event", "timeout_event"]:
            assert key in row["outcome"], row["cell"]
        row["collision"] = bool(row["outcome"]["collision_event"])
        row["timeout"] = bool(row["outcome"]["timeout_event"])
        assert row["success"] == bool(row["outcome"]["route_complete"])
        if row["planner"] in {"ppo", "guarded_ppo", "sacadrl"}:
            checkpoint = row["metadata"]["planner_runtime"]["checkpoint_provenance"]
            assert checkpoint["load_succeeded"] and not checkpoint["fallback_triggered"]
    return rows


def load_trace(directory, row):
    """Read the saved motion, rather than an absolute path embedded in a CSV."""
    with gzip.open(directory / "traces" / f"{row['cell']}.json.gz", "rt") as handle:
        trace = json.load(handle)
    assert len(trace) == row["steps"], row["cell"]
    return trace


def creep_selected(step):
    """Recover actual creep selection for old traces that predate the explicit flag."""
    if "creeping" in step:
        return step["creeping"]
    projections = step["proj"]
    return bool(
        projections
        and projections[-1][1][0] > 0.0
        and any(
            abs(p[0][0]) < 1e-3 and p[1][0] > max(p[0][0], 0.0) and abs(p[0][1]) > 1e-6
            for p in projections
        )
    )


def measure_rows(directory, rows, *, instrumented):
    """Recalculate progress and creep counts from each raw trace."""
    for row in rows:
        trace = load_trace(directory, row)
        row["stuck"] = displacement_stuck_steps(trace, row["reset"]["pose"])
        row["zero_turn"] = sum(creep_selected(t) for t in trace)
        if instrumented:
            assert all("min_clearance_m" in t and "native_action" in t for t in trace)
            assert row["zero_turn"] == sum(t["creeping"] for t in trace)
            assert row["was_creeping_at_termination"] == bool(trace and trace[-1]["creeping"])
            row["safety_creep_steps"] = sum(
                bool(t["safety_interventions"]) and t["creeping"] for t in trace
            )
            row["path_min_clearance_m"] = min(
                t["min_clearance_m"] for t in trace if t["min_clearance_m"] is not None
            )


def row_key(row, arm=None):
    """Identify a paired planner, plant, scenario, and seed cell."""
    return row["planner"], arm or row["arm"], row["probe"], row["scenario"], row["seed"]


def safe_route(row):
    """Require route completion, no contact flag and no sampled disc penetration."""
    return bool(
        row and row["success"] and not row["collision"] and row["path_min_clearance_m"] >= -1e-6
    )


def validate_matrix(rows, roster):
    """Demand all requested axes, matched resets, and actual T60 limits."""
    assert len(roster) == 14 and len(rows) == 6650
    by = {row_key(r): r for r in rows}
    assert len(by) == len(rows)
    for planner in roster:
        for arm in ARMS:
            for probe, names, seeds in [
                (1, [str(b) for b in [0, 45, 90, 135, 180]], range(1001, 1006)),
                (2, SCENARIOS, range(1001, 1011)),
            ]:
                for name in names:
                    for seed in seeds:
                        row = by[planner, arm, probe, name, seed]
                        if arm != "DD":
                            assert row["reset"] == by[row_key(row, "DD")]["reset"]
                            cfg = row["robot_config"]
                            assert cfg["radius"] == 0.64 and cfg["wheelbase"] == 0.90
                            assert cfg["max_velocity"] == 1.34
                            assert cfg["max_steer"] == (0.52 if arm.startswith("T60-30") else 0.79)
                            assert cfg["max_accel"] == cfg["max_decel"] == 1.0
                            assert not cfg["allow_backwards"]
                            assert cfg["creep_speed"] == (0.1 if arm.endswith("-on") else 0.0)
        for arm in ["BI-off", "BI-on"]:
            assert sum(r["planner"] == planner and r["arm"] == arm for r in rows) == 25
    return by


def wilson(count, n):
    """Return descriptive 95% Wilson score bounds for an episode proportion."""
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (count / n + z * z / (2 * n)) / denominator
    half = z * math.sqrt(count / n * (1 - count / n) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def stats(rows):
    """Aggregate independent runner outcome flags and overlapping stall windows."""
    n = len(rows)
    assert n
    result = {
        "n": n,
        "stuck": sum(r["stuck"] for r in rows),
        "zero_turn": sum(r["zero_turn"] for r in rows),
    }
    for outcome in ["success", "collision", "timeout"]:
        count = sum(bool(r[outcome]) for r in rows)
        lo, hi = wilson(count, n)
        result.update(
            {outcome: count, outcome + "_rate": count / n, outcome + "_lo": lo, outcome + "_hi": hi}
        )
    return result


def failure_evidence(directory, row):
    """Use terminal motion and exact disc surface clearance to describe a failure."""
    trace = load_trace(directory, row)
    tail = trace[-20:]
    terminal = trace[-1]
    clearance = [t["min_clearance_m"] for t in tail if t["min_clearance_m"] is not None]
    assert clearance, row["cell"]
    cfg = row["robot_config"]
    law_error = max(
        abs(t["yaw"] - t["v"] * math.tan(t["action"][1]) / cfg["wheelbase"]) for t in trace
    )
    return {
        "outcome": "collision" if row["collision"] else "timeout",
        "zero_turn": row["zero_turn"],
        "was_creeping_at_termination": bool(terminal["creeping"]),
        "min_clearance_last_2s_m": min(clearance),
        "clearance_window_steps": len(tail),
        "terminal_yaw_request_rad_s": terminal["cmd"][1],
        "terminal_speed_m_s": terminal["v"],
        "terminal_safety_interventions": "+".join(terminal["safety_interventions"]) or "none",
        "creep_steps_last_2s": sum(t["creeping"] for t in tail),
        "safety_creep_steps": sum(bool(t["safety_interventions"]) and t["creeping"] for t in trace),
        "noise_creep_steps": sum(
            t["creeping"] and abs(t["cmd"][1]) < math.radians(1) for t in trace
        ),
        "reverse_creep_steps": sum(t["creeping"] and t["cmd"][0] < 0 for t in trace),
        "stuck": row["stuck"],
        "yaw_law_max_error": law_error,
    }


def classify(row, evidence, cohort, alternative, corrected_on, witness, path_witness):
    """Attribute only witnessed adapter defects or demonstrated physical reachability."""
    details = (
        f"{row['cell']}: {evidence['outcome']} at {row['steps'] * 0.1:.1f}s; "
        f"last-2s clearance {evidence['min_clearance_last_2s_m']:.6f}m, "
        f"creep {evidence['creep_steps_last_2s']}/{evidence['clearance_window_steps']} steps "
        f"(total {evidence['zero_turn']}, terminal {evidence['was_creeping_at_termination']}); "
        f"terminal yaw request {evidence['terminal_yaw_request_rad_s']:.6f}rad/s, "
        f"speed {evidence['terminal_speed_m_s']:.6f}m/s, "
        f"{evidence['stuck']} low-displacement windows; "
    )
    if evidence["safety_creep_steps"]:
        return (
            "adapter artefact",
            details
            + f"creep overrode a recorded safety intervention on {evidence['safety_creep_steps']} steps.",
        )
    if cohort == "review-44" and evidence["zero_turn"] and safe_route(alternative):
        return (
            "adapter artefact",
            details
            + "the same matched plant/reset succeeds after disabling implicit creep; the intervening change is adapter creep policy, not increased steering or reverse.",
        )
    if evidence["noise_creep_steps"] or evidence["reverse_creep_steps"]:
        defect = (
            f"{evidence['noise_creep_steps']} sub-degree creep steps and "
            f"{evidence['reverse_creep_steps']} reverse-to-forward creep steps; "
        )
        if safe_route(corrected_on):
            return "adapter artefact", details + defect + (
                "the matched controller/reset succeeds with corrected opt-in creep intent/sign gates."
            )
        return "unclear", details + defect + (
            f"the corrected opt-in controller outcome is {corrected_on['success'] if corrected_on else 'unavailable'} "
            "for success. The adapter defect is observed, but its contribution to failure cannot "
            "be separated from controller strategy by this counterfactual. "
            + (
                f"Matched path {path_witness['cell']} certifies initial reachability only."
                if path_witness
                else "No matched safe bicycle path witness is available."
            )
        )
    if evidence["yaw_law_max_error"] < 1e-5 and witness:
        return (
            "planner/controller error",
            details
            + f"the forward-only physical controller reaches this empty-world bearing in {witness['steps']} steps; the planner's strategy fails on a demonstrably reachable plant.",
        )
    if evidence["yaw_law_max_error"] < 1e-5 and safe_route(alternative):
        return (
            "planner/controller error",
            details
            + f"the same bicycle/reset reaches the goal without contact under the alternative creep policy ({alternative['arm']}); this is a feasible controller strategy, so a fundamental kinematic impossibility is not established.",
        )
    if evidence["yaw_law_max_error"] < 1e-5 and path_witness:
        return (
            "planner/controller error",
            details
            + f"matched bicycle trajectory {path_witness['cell']} reaches the goal without contact "
            "from identical robot/pedestrian reset states and the same plant limits. This certifies "
            "initial reachability; robot-dependent pedestrian reactions may diverge, so it does "
            "not certify an escape from this failed run's terminal state.",
        )
    if path_witness:
        return "unclear", details + (
            f"matched bicycle trajectory {path_witness['cell']} certifies initial reachability, "
            f"but this failed trace has yaw-law residual {evidence['yaw_law_max_error']:.9g}rad/s. "
            "Collision-time pose/velocity correction versus an adapter/controller defect remains "
            "unresolved; a safe initial path alone cannot attribute this motion discrepancy."
        )
    return "unclear", details + (
        f"yaw-law residual {evidence['yaw_law_max_error']:.9g}rad/s; "
        "DD succeeds, but no matched planner/creep setting supplies a collision-free bicycle path witness. "
        "The observed constrained turn/crowd interaction cannot distinguish a controller defect "
        "from a space/reverse limit; no geometric impossibility certificate is available."
    )


def failures(rows, directory, cohort, dd_by, current_by, reachability):
    """Classify every T60-only failure, with case-specific measured evidence."""
    result = []
    for row in rows:
        if (
            not row["arm"].startswith("T60")
            or row["success"]
            or not dd_by[row_key(row, "DD")]["success"]
        ):
            continue
        assert row["collision"] or row["timeout"], row["cell"]
        evidence = failure_evidence(directory, row)
        alt_arm = (
            row["arm"].removesuffix("-on")
            if row["arm"].endswith("-on") or cohort == "review-44"
            else row["arm"] + "-on"
        )
        alternative = current_by.get(row_key(row, alt_arm))
        corrected_on = current_by.get(row_key(row, row["arm"].removesuffix("-on") + "-on"))
        if alternative:
            assert row["reset"] == alternative["reset"]
        path_witness = next(
            (
                candidate
                for candidate in sorted(current_by.values(), key=lambda r: r["cell"])
                if candidate["arm"].startswith(row["arm"].removesuffix("-on"))
                and candidate["probe"] == row["probe"]
                and candidate["scenario"] == row["scenario"]
                and candidate["seed"] == row["seed"]
                and candidate["reset"] == row["reset"]
                and safe_route(candidate)
                and all(
                    candidate["robot_config"][key] == row["robot_config"][key]
                    for key in [
                        "radius",
                        "wheelbase",
                        "max_steer",
                        "max_velocity",
                        "max_accel",
                        "max_decel",
                        "allow_backwards",
                    ]
                )
            ),
            None,
        )
        witness = (
            next(
                (
                    w
                    for w in reachability
                    if w["steer"] == row["robot_config"]["max_steer"]
                    and str(w["bearing"]) == row["scenario"]
                ),
                None,
            )
            if row["probe"] == 1
            else None
        )
        label, reason = classify(
            row, evidence, cohort, alternative, corrected_on, witness, path_witness
        )
        result.append(
            {
                "cohort": cohort,
                **{
                    k: row[k]
                    for k in ["cell", "planner", "arm", "probe", "scenario", "seed", "source_sha"]
                },
                **evidence,
                "empty_witness_steps": witness["steps"] if witness else None,
                "alternative_success": safe_route(alternative),
                "bicycle_path_witness_cell": path_witness["cell"] if path_witness else None,
                "bicycle_path_witness_min_clearance_m": (
                    path_witness["path_min_clearance_m"] if path_witness else None
                ),
                "classification": label,
                "reason": reason,
            }
        )
    return result


def raw_identity(directory):
    """Hash sorted relative names and bytes of raw records/traces without local paths."""
    digest = hashlib.sha256()
    files = sorted(
        [directory / "manifest.json"]
        + [p for subdir in ["results", "traces", "records"] for p in (directory / subdir).glob("*")]
    )
    for path in files:
        digest.update(
            path.relative_to(directory).as_posix().encode()
            + b"\0"
            + bytes.fromhex(sha256_file(path))
        )
    return {"files": len(files), "sha256": digest.hexdigest()}


def emit_csv(output, name, fields, rows):
    """Preserve CSV syntax; attach the shared review marker via a stable sidecar."""
    path = output / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_review_sidecar(path, repo_root=output.parents[1])


def main():
    """Validate all axes and emit the exact committed artifact names and schemas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy-input", type=Path, required=True)
    parser.add_argument("--review-input", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    roster = [p["key"] for p in manifest["roster"]]
    current = load_rows(args.input)
    legacy = load_rows(args.legacy_input)
    review = load_rows(args.review_input)
    by = validate_matrix(current, roster)
    assert len(legacy) == 4620 and len(review) == 127
    legacy_by = {row_key(r): r for r in legacy}
    for row in review:
        prior = legacy_by[row_key(row)]
        assert all(row[k] == prior[k] for k in ["success", "steps", "outcome", "reset"])
    for directory, rows, instrumented in [
        (args.input, current, True),
        (args.legacy_input, legacy, False),
        (args.review_input, review, True),
    ]:
        measure_rows(directory, rows, instrumented=instrumented)
    assert all(r["safety_creep_steps"] == 0 for r in current), "creep overrode a safety veto"
    reach = json.loads(args.reachability.read_text())
    assert len(reach) == 10 and all(w["reached"] and w["steps"] < 600 for w in reach)
    comparisons = []
    per_scenario = []
    for planner in ["ALL", *roster]:
        for probe, names in [(1, [str(b) for b in manifest["bearings"]]), (2, SCENARIOS)]:
            for arm in ARMS:
                group = [
                    r
                    for r in current
                    if r["probe"] == probe
                    and r["arm"] == arm
                    and (planner == "ALL" or r["planner"] == planner)
                ]
                comparisons.append(dict(planner=planner, probe=probe, arm=arm, **stats(group)))
                for name in names:
                    per_scenario.append(
                        dict(
                            planner=planner,
                            probe=probe,
                            scenario=name,
                            arm=arm,
                            **stats([r for r in group if r["scenario"] == name]),
                        )
                    )
    classified = []
    for arms, cohort in [(ARMS[1:3], "fixed-off"), (ARMS[3:], "fixed-on")]:
        classified.extend(
            failures([r for r in current if r["arm"] in arms], args.input, cohort, by, by, reach)
        )
    classified.extend(failures(review, args.review_input, "review-44", legacy_by, by, reach))
    classified.sort(key=lambda r: (r["cohort"], r["cell"]))
    assert all(r["safety_creep_steps"] == 0 for r in classified if r["cohort"] != "review-44")
    output = args.output_root / "docs/validation"
    output.mkdir(parents=True, exist_ok=True)
    emit_csv(
        output,
        "bicycle_probe_comparison.csv",
        ["planner", "probe", "arm", *STAT_FIELDS],
        comparisons,
    )
    emit_csv(
        output,
        "bicycle_probe_per_scenario.csv",
        ["planner", "probe", "scenario", "arm", *STAT_FIELDS],
        per_scenario,
    )
    emit_csv(output, "bicycle_probe_failures.csv", FAILURE_FIELDS, classified)
    legacy_stats = {
        arm: stats(
            [
                r
                for r in (legacy if arm in {"DD-legacy", "BI-base", "BI-fixed"} else current)
                if r["arm"] == arm
            ]
        )
        for arm in ["DD-legacy", "BI-base", "BI-fixed", "BI-off", "BI-on"]
    }
    result = {
        "schema_version": "bicycle-probe-summary.v2",
        "rows": len(current),
        "roster": roster,
        "summary": comparisons,
        "legacy": legacy_stats,
        "classifications": {
            cohort: dict(
                collections.Counter(
                    r["classification"] for r in classified if r["cohort"] == cohort
                )
            )
            for cohort in ["review-44", "fixed-off", "fixed-on"]
        },
        "t60_only_failure_counts": dict(
            collections.Counter(r["classification"] for r in classified)
        ),
        "t60_only_by_arm_probe": dict(
            collections.Counter(f"{r['cohort']}/{r['arm']}/P{r['probe']}" for r in classified)
        ),
        "paired_reset_mismatches": [],
        "fixed_safety_creep_steps": sum(r["safety_creep_steps"] for r in current),
        "t_intersection": {
            "fixed": [
                r
                for r in per_scenario
                if r["planner"] == "ALL" and r["scenario"] == "classic_t_intersection_medium"
            ],
            "review_44": {
                arm: stats(
                    [
                        r
                        for r in legacy
                        if r["arm"] == arm and r["scenario"] == "classic_t_intersection_medium"
                    ]
                )
                for arm in ARMS[:3]
            },
        },
        "stuck_metric": {
            "window_s": 2.0,
            "net_displacement_below_m": 0.05,
            "stride_s": 0.1,
            "command_threshold": 1e-6,
            "command_rule": "at least one nonzero issued command in the full window; raw v/omega before projection, or native action components when native",
            "count_rule": "overlapping window ending steps, not unique seconds or yaw stalls",
        },
        "creep_policy": {
            "default_m_s": 0.0,
            "opt_in_m_s": 0.1,
            "min_yaw_request_rad_s": math.radians(1),
            "speed_gate": "0 <= v < .001 m/s",
            "safety_interventions_veto": True,
            "duration_gate": "none; a single .1s meaningful request moves <=.01m and the next stop immediately brakes",
        },
        "claim_boundary": "Diagnostic dev measurements only. Wilson intervals describe episode proportions; paired planners/scenarios sharing seeds are correlated, so these are not independent-seed ranking intervals. No release admission or trained-bicycle checkpoint claim.",
    }
    write_json(output / "bicycle_probe_summary.json", result)
    artifact_names = [
        "bicycle_probe_comparison.csv",
        "bicycle_probe_per_scenario.csv",
        "bicycle_probe_failures.csv",
        "bicycle_probe_summary.json",
    ]
    provenance = {
        "schema_version": "bicycle-probe-provenance.v2",
        "issue": 10093,
        "pr": 10100,
        "measurement_source_sha": manifest["source_sha"],
        "review_head": review[0]["source_sha"],
        "original_base_sha": next(r["source_sha"] for r in legacy if r["arm"] == "BI-base"),
        "source_files_sha256": manifest["source_files_sha256"],
        "summarizer_sha256": sha256_file(Path(__file__)),
        "metric_helper_sha256": sha256_file(Path(__file__).with_name("bicycle_probe_metrics.py")),
        "axes": {
            k: manifest[k]
            for k in [
                "probe1_seeds",
                "probe2_seeds",
                "bearings",
                "horizon",
                "dt",
                "template_sha256",
            ]
        },
        "raw_corpora": {
            name: raw_identity(directory)
            for name, directory in [
                ("fixed", args.input),
                ("original", args.legacy_input),
                ("review_replay", args.review_input),
            ]
        },
        "reachability_sha256": sha256_file(args.reachability),
        "public_artifact_sha256": {name: sha256_file(output / name) for name in artifact_names},
        "clearance_definition": "post-step minimum disc surface clearance to pedestrians and static map walls/bounds, using reset_spawn_clearance collision geometry; last20 available .1s samples",
        "zero_turn_definition": "count of steps on which creep was actually selected after projection and safety, not every zero-speed yaw request",
        "was_creeping_at_termination_definition": "creep was selected at the final executed step; actual terminal speed is reported separately",
        "classification_rule": "observed safety/noise/reverse creep defect with a resolving matched counterfactual is an adapter artefact; unresolved observed mapping defects retain case-specific uncertainty; otherwise a matched collision-free bicycle trajectory certifies initial controller reachability, with no unproved geometric limit attribution",
    }
    write_json(output / "bicycle_probe_provenance.json", provenance)
    print(
        json.dumps(
            {
                "rows": len(current),
                "legacy": legacy_stats,
                "classifications": result["classifications"],
                "aggregate": [r for r in comparisons if r["planner"] == "ALL"],
                "t_intersection": result["t_intersection"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
