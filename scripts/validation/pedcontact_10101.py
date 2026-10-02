"""PEDCONTACT dev-only CALFIT comparison with complete raw manifest custody.

Each Slurm array task runs one arm/seed bank through CALFIT's original runner.
The collector verifies every member before constructing a full-dev gate and
confidence intervals. This never changes release or acceptance policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from robot_sf.evidence.writers import write_json, write_text
from scripts.validation import calfit_search_10074 as search
from scripts.validation import pedestrian_validation_10074 as suite


def default_point(arm: str) -> dict[str, object]:
    """Return the profile reference at the milestone's shared .28m radius.

    Returns:
        Identity-bearing point with explicit measurement and optional laws.
    """
    point = search.candidate("calibrated_v2", 0.003, 0.375, 0.28, 2.0)
    point["pedcontact_measurement"] = True
    if arm == "on":
        point["pedestrian_contact_rule"] = "projection_v1"
        point["pedestrian_wall_rule"] = "bounded_edge_v1"
    elif arm != "off":
        raise ValueError("PEDCONTACT arm must be off or on")
    point.pop("id")
    point["id"] = hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:12]
    return point


def interval(values: list[float]) -> dict[str, object]:
    """Compute a two-sided 95% Student interval over observed dev episodes.

    Returns:
        Mean, sample count, spread and interval; missing observations stay missing.
    """
    arr = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(arr):
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else None
    half = (
        float(student_t.ppf(0.975, len(arr) - 1) * sd / np.sqrt(len(arr)))
        if sd is not None
        else None
    )
    mean = float(arr.mean())
    return {
        "n": len(arr),
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half] if half is not None else None,
    }


def collect(root: Path, out: Path) -> dict[str, object]:  # noqa: C901
    """Verify full dev off/on banks and retain every gate, violation and interval.

    Returns:
        Full comparison and explicit admission to the conditional fit.
    """
    config = suite.load_config(suite.DEFAULT_CONFIG)
    result = {"schema": "pedcontact.comparison.v1", "seeds": config["seeds"], "arms": {}}
    for arm in ("off", "on"):
        rows = []
        identities = []
        for seed in config["seeds"]:
            directory = root / arm / str(seed)
            search.verify_run(directory)
            identity = search.read_json(directory / "identity.json")
            if identity["seeds"] != [seed] or identity["candidate"] != default_point(arm):
                raise ValueError("PEDCONTACT bank identity mismatch")
            identities.append(identity)
            rows += [search.read_json(path) for path in sorted(directory.glob("case_*.json"))]
        if len({identity["source_sha"] for identity in identities}) != 1:
            raise ValueError("mixed producer heads in comparison")
        gate = suite.acceptance_gate(rows, config=config, require_complete=True)
        table = suite.protocol_summary(rows, config)
        per_case = {}
        fields = {
            "V1": "fitted_desired_speed_m_s",
            "V2": "speed_drop_m_s",
            "V3": "specific_flow_persons_m_s",
            "V4": "all_data_specific_flow_persons_m_s",
            "V5": "lateral_cm_to_edge_m",
            "V6": "onset_m",
        }
        for case, variant in sorted({(row["case"], row["variant"]) for row in rows}):
            bank = [row for row in rows if row["case"] == case and row["variant"] == variant]
            timing = [
                sum(r["step_time_s"] for r in row["step_runtime"])
                / sum(r["steps"] for r in row["step_runtime"])
                * 1000
                for row in bank
            ]
            per_case[f"{case}/{variant}"] = {
                "measurement": interval([row.get(fields[case]) for row in bank]),
                "runtime_ms_per_step": interval(timing),
                "overlap_pair_steps": sum(
                    row["pair_overlap"]["all"]["below_2r_count"] for row in bank
                ),
                "wall_penetration_ped_steps": sum(
                    row["wall_penetration_ped_steps"] for row in bank
                ),
                "passed_n": sum(bool(row.get("passed")) for row in bank),
                "attempted_n": len(bank),
                "maximum_projection_passes": max(
                    r["maximum_projection_passes"] for row in bank for r in row["step_runtime"]
                ),
            }
        for quantity in ("all_data_flow_persons_s", "steady_flow_persons_s"):
            slopes = []
            for seed in config["seeds"]:
                wide = sorted(
                    [row for row in rows if row["case"] == "V4" and row["seed"] == seed],
                    key=lambda row: float(row["variant"]),
                )
                if all(row.get(quantity) is not None for row in wide):
                    widths = np.asarray([float(row["variant"]) for row in wide])
                    flows = np.asarray([row[quantity] for row in wide])
                    slopes.append(float(widths @ flows / (widths @ widths)))
                else:
                    slopes.append(None)
            per_case[f"V4/{quantity} width slope"] = {"measurement": interval(slopes)}
        result["arms"][arm] = {
            "source_sha": identities[0]["source_sha"],
            "gate": gate,
            "per_case": per_case,
            "table": table,
            "overlap_pair_steps": sum(row["pair_overlap"]["all"]["below_2r_count"] for row in rows),
            "wall_penetration_ped_steps": sum(row["wall_penetration_ped_steps"] for row in rows),
            "runtime_ms_per_step": 1000
            * sum(r["step_time_s"] for row in rows for r in row["step_runtime"])
            / sum(r["steps"] for row in rows for r in row["step_runtime"]),
            "fallback_count": sum(
                r.get("fallback_count", 0) for row in rows for r in row["step_runtime"]
            ),
            "unresolved_count": sum(
                r.get("unresolved_count", 0) for row in rows for r in row["step_runtime"]
            ),
            "over_cap_samples": sum(
                r.get("over_cap_samples", 0) or 0 for row in rows for r in row["step_runtime"]
            ),
            "maximum_speed_m_s": max(
                r.get("maximum_speed_m_s", 0) for row in rows for r in row["step_runtime"]
            ),
            "runtime_scope": "All integration calls including five V6 no-interferer baselines; cold JIT included; use separate warmed probe for cost",
        }
    on = result["arms"]["on"]
    result["fit_admitted"] = (
        on["overlap_pair_steps"] == 0
        and on["wall_penetration_ped_steps"] == 0
        and all(
            bank["passed_n"] == bank["attempted_n"]
            for key, bank in on["per_case"].items()
            if key.startswith("V2/")
        )
    )
    result["runtime_on_off_ratio"] = (
        on["runtime_ms_per_step"] / result["arms"]["off"]["runtime_ms_per_step"]
    )
    write_json(out, result)
    text = "AI-GENERATED / NEEDS-REVIEW\n\nPEDCONTACT full-dev CALFIT comparison.\n\n"
    text += "Intervals cover observed dev-seed measurements, not censored episodes. V6 baselines are included in integration timing; cold JIT is included. Warmed cost is measured separately.\n\n"
    for arm, bank in result["arms"].items():
        text += f"{arm}: overlaps {bank['overlap_pair_steps']}; wall penetration ped-steps {bank['wall_penetration_ped_steps']}; gate {bank['gate']['exit_code']}.\n\n"
        text += "| Item | Value | 95% interval | Observed n | Status | Accepted interval |\n|---|---|---|---|---|---|\n"
        for check in bank["gate"]["checks"]:
            measurement = (
                bank["per_case"]
                .get(f"{check['case']}/{check['variant']}", {})
                .get("measurement", {})
            )
            text += f"| {check['case']}/{check['variant']} | {check['estimate']} | {measurement.get('ci95')} | {measurement.get('n', 0)} | {check['status']} | {check['tolerance_range']} |\n"
        text += "\n"
    write_text(out.with_suffix(".md"), text, issue_ref="#10101")
    return result


def main() -> None:
    """Acquire a seed bank or verify and collect both complete full-dev arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--arm", choices=("off", "on"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "run":
        if args.arm is None or args.seed is None or not 1001 <= args.seed <= 1030:
            parser.error("run requires an explicit off/on arm and dev1001..1030 seed")
        search.run_candidate(default_point(args.arm), [args.seed], args.out, workers=1)
    else:
        if args.root is None:
            parser.error("collect requires --root")
        collect(args.root, args.out)


if __name__ == "__main__":
    main()
