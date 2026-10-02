"""Author-delegated development tolerances, 2026-10-02; no release admission.

Reported spreads remain distinct from engineering acceptance ranges. Missing
literature spreads are explicit and cannot silently become invented SDs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NOTE = "author-delegated engineering tolerance, 2026-10-02"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/benchmarks/pedestrian_validation_0_0_9.json"
)
SOURCES = {
    "V1": "Moussaid et al. 2009, section 4(a), Fig. 1b; https://doi.org/10.1098/rspb.2009.0405",
    "V2": "Wilmut et al. 2015, Fig. 3; https://doi.org/10.1371/journal.pone.0124695",
    "V3": "Seyfried et al. 2009, Table 2; https://arxiv.org/abs/physics/0702004",
    "V4": "Liao et al. 2014, section 4; https://doi.org/10.1016/j.trpro.2014.09.005",
    "V5": "Gerin-Lajoie et al. 2008 pp.239-240, citing 2005; https://doi.org/10.1016/j.gaitpost.2007.03.015",
    "V6": "Huber et al. 2014, Table 1, 180-degree condition; https://doi.org/10.1371/journal.pone.0089589",
}
LIMITATION = "not reproducible with rigid discs (shoulder rotation)"


def feasible_apertures(radius_m: float, shoulder_m: float = 0.46) -> list[float]:
    """Return three widths from diameter plus .05 m to the source maximum."""
    lower, upper = 2 * radius_m + 0.05, 2.1 * shoulder_m
    if not np.isfinite([lower, upper]).all() or radius_m <= 0 or lower >= upper:
        raise ValueError(
            "three feasible aperture widths require a positive radius below source maximum"
        )
    return [round(float(width), 12) for width in np.linspace(lower, upper, 3)]


def _check(case, variant, quantity, values, target, bounds, rule, *, target_band=None):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    estimate = float(np.mean(finite)) if finite else None
    status = (
        "MISSING"
        if len(finite) != len(values) or not finite
        else (
            "UNSPECIFIED"
            if bounds is None
            else ("PASS" if bounds[0] - 1e-12 <= estimate <= bounds[1] + 1e-12 else "FAIL")
        )
    )
    distance = (
        None
        if estimate is None or bounds is None
        else max(bounds[0] - estimate, estimate - bounds[1], 0.0)
    )
    return {
        "case": case,
        "variant": variant,
        "quantity": quantity,
        "estimate": estimate,
        "target": target,
        "target_band": target_band,
        "difference": None if estimate is None else estimate - target,
        "tolerance_range": bounds,
        "distance_outside_tolerance": distance,
        "status": status,
        "observed_n": len(finite),
        "attempted_n": len(values),
        "source": SOURCES[case],
        "rule": rule,
        "tolerance_note": NOTE,
    }


REQUIRED = {
    "V1": ("fitted_desired_speed_m_s", "fitted_tau_s"),
    "V2": ("speed_drop_m_s",),
    "V3": ("specific_flow_persons_m_s",),
    "V4": ("all_data_specific_flow_persons_m_s", "steady_specific_flow_persons_m_s"),
    "V5": ("lateral_cm_to_edge_m",),
    "V6": ("onset_m",),
}


def _case_checks(case, variant, bank, config):
    """Build source-bound checks for one measurement bank.

    Returns:
        Individual checks with source and engineering tolerance.
    """
    checks = []
    required = REQUIRED
    values = [r.get(required[case][0]) for r in bank]
    if case == "V1":
        checks.append(
            _check(
                case,
                variant,
                required[case][0],
                values,
                1.29,
                [1.10, 1.48],
                "literature mean ± 1 SD (.19 m/s)",
            )
        )
    elif case == "V2":
        ratio = bank[0].get("aperture_shoulder_ratio", float(variant))
        alpha = np.clip((ratio - 0.9) / 0.4, 0.0, 1.0)
        band = [
            float((1 - alpha) * 0.40 + alpha * 0.12),
            float((1 - alpha) * 0.40 + alpha * 0.16),
        ]
        check = _check(
            case,
            variant,
            required[case][0],
            values,
            float(np.mean(band)),
            [band[0] * 0.5, band[1] * 1.5],
            "same positive sign; ±50% magnitude; .9-to-1.3 source-anchor interpolation below 1.3",
            target_band=band,
        )
        if check["estimate"] is not None and check["estimate"] <= 0:
            check["status"] = "FAIL"
        if any(not r.get("passed", True) for r in bank):
            check["status"] = "MISSING"
        checks.append(check)
    elif case == "V3":
        target = config["V3"]["published_specific_flow_persons_m_s"][
            config["V3"]["widths_m"].index(float(variant))
        ]
        checks.append(
            _check(
                case,
                variant,
                required[case][0],
                values,
                target,
                [0.8 * target, 1.2 * target],
                "±20% of literature flow",
            )
        )
    elif case == "V5":
        target = config["V5"]["published_lateral_m"]
        bounds = config["V5"].get("acceptance_range_m")
        if bounds is None and config["V5"].get("published_sd_m") is not None:
            sd = config["V5"]["published_sd_m"]
            bounds = [target - sd, target + sd]
        checks.append(
            _check(
                case,
                variant,
                required[case][0],
                values,
                target,
                bounds,
                "literature mean ± 1 reported SD; no SD verified in supplied target",
            )
        )
    elif case == "V6":
        i = config["V6"]["speeds_m_s"].index(float(variant))
        target = config["V6"]["published_onset_m"][i]
        ranges = config["V6"].get("acceptance_ranges_m")
        bounds = ranges[i] if ranges else None
        spread = config["V6"].get("published_onset_sd_m")
        if bounds is None and spread and spread[i] is not None:
            bounds = [target - spread[i], target + spread[i]]
        checks.append(
            _check(
                case,
                variant,
                required[case][0],
                values,
                target,
                bounds,
                "literature mean ± 1 reported SD; Table 1 reports means only",
            )
        )
    return checks


def _wide_checks(rows, config):
    """Fit complete per-seed width banks without substituting censored flow.

    Returns:
        All-data and stationary slope checks.
    """
    checks = []
    wide = [r for r in rows if r["case"] == "V4"]
    if wide:
        for key, target in [("all_data_flow_persons_s", 2.3), ("steady_flow_persons_s", 2.5)]:
            slopes = []
            for seed in sorted({r["seed"] for r in wide}):
                bank = sorted(
                    [r for r in wide if r["seed"] == seed], key=lambda r: float(r["variant"])
                )
                if {float(r["variant"]) for r in bank} != set(config["V4"]["widths_m"]) or any(
                    r.get(key) is None or not np.isfinite(r[key]) for r in bank
                ):
                    slopes.append(None)
                else:
                    widths = np.array([float(r["variant"]) for r in bank])
                    flows = np.array([r[key] for r in bank])
                    slopes.append(float(widths @ flows / (widths @ widths)))
            checks.append(
                _check(
                    "V4",
                    key + " width slope",
                    "persons/(m s)",
                    slopes,
                    target,
                    [0.8 * target, 1.2 * target],
                    "±20% of literature through-origin flow slope",
                )
            )
    return checks


def _original_shoulder_case(row):
    """Identify the explicitly excluded empirical shoulder-rotation condition.

    Returns:
        Whether this record belongs to the author-excluded empirical case.
    """
    return row["case"] == "V2" and np.isclose(
        float(row.get("aperture_shoulder_ratio", row["variant"])), 0.9, rtol=0, atol=1e-12
    )


def engineering_gate(rows, config=None, *, require_complete=False) -> dict[str, object]:
    """Evaluate population means, preserving every physical/censoring failure.

    Full acquisition admission additionally requires the complete declared grid;
    callers evaluating a single case receive an explicitly observed-cases scope.
    V5/V6 without reported spread stay unspecified until the author supplies it.

    Returns:
        Gate table with residuals, ranges, provenance and independent hard failures.
    """
    config = config or json.loads(DEFAULT_CONFIG.read_text())
    excluded = [r for r in rows if _original_shoulder_case(r)]
    active = [r for r in rows if not _original_shoulder_case(r)]
    required = REQUIRED
    missing = [
        f"{r['case']}/{r['variant']}/{r['seed']}"
        for r in active
        if any(r.get(key) is None or not np.isfinite(r[key]) for key in required.get(r["case"], ()))
    ]
    physical = [
        f"{r['case']}/{r['variant']}/{r['seed']}"
        for r in active
        if (
            r.get("wall_penetration_m", 0.0) > 0.0
            or r["pair_overlap"]["all"]["below_2r_count"] > 0
            or r["pair_overlap"]["all"].get("initial_overlapping_pairs", 0) > 0
        )
    ]
    checks = []
    for case, variant in sorted(
        {(r["case"], r["variant"]) for r in active if r["case"] in required}
    ):
        bank = [r for r in active if (r["case"], r["variant"]) == (case, variant)]
        checks.extend(_case_checks(case, variant, bank, config))
    checks.extend(_wide_checks(rows, config))
    if require_complete and rows:
        expected = {
            (case, seed, str(variant))
            for case, variants in {
                "V1": ["native"],
                "V2": feasible_apertures(rows[0]["radius_m"], config["V2"]["shoulder_proxy_m"]),
                "V3": config["V3"]["widths_m"],
                "V4": config["V4"]["widths_m"],
                "V5": ["diagnostic"],
                "V6": config["V6"]["speeds_m_s"],
            }.items()
            for variant in variants
            for seed in config["seeds"]
        }
        observed = [(r["case"], r["seed"], r["variant"]) for r in active if r["case"] in required]
        if len(observed) != len(expected) or set(observed) != expected:
            missing.append("complete declared V1-V6 dev grid")
    numerical = [f"{c['case']}/{c['variant']}" for c in checks if c["status"] == "FAIL"]
    unspecified = [f"{c['case']}/{c['variant']}" for c in checks if c["status"] == "UNSPECIFIED"]
    missing += [
        f"{c['case']}/{c['variant']}"
        for c in checks
        if c["status"] == "MISSING"
        and not any(item.startswith(f"{c['case']}/{c['variant']}/") for item in missing)
    ]
    code = (
        4
        if not rows
        else (
            3 if physical else (2 if missing else (1 if numerical else (5 if unspecified else 0)))
        )
    )
    return {
        "schema": "valsuite.gate.v2",
        "scope": "complete dev suite" if require_complete else "observed cases",
        "measurement_missing": missing,
        "physical_violations": physical,
        "numeric_failures": numerical,
        "unspecified_tolerances": unspecified,
        "checks": checks,
        "excluded_measurements": [
            {
                "case": r["case"],
                "variant": r["variant"],
                "seed": r["seed"],
                "reason": LIMITATION,
                "case_gated": False,
                "source": SOURCES["V2"],
            }
            for r in excluded
        ],
        "exit_code": code,
        "tolerance_note": NOTE,
        "model_limitations": [
            {"case": "V2", "shoulder_ratio": 0.9, "status": LIMITATION, "gate": "not applicable"}
        ],
        "release_admission": "engineering dev validation only; no release or paper admission",
    }
