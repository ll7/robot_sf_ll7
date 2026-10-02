"""Audit CALFIT estimators and admission without any simulation reset or step.

Synthetic implementation diagnostics never establish model or release acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robot_sf.evidence.writers import write_json
from robot_sf.research import pedestrian_validation as m
from robot_sf.research.pedestrian_acceptance import feasible_apertures
from scripts.validation import pedestrian_validation_10074 as suite


def estimator_controls() -> list[dict[str, object]]:
    """Return source-estimator receipts against independently calculated answers."""
    t = 0.35 + np.arange(201) * 0.1
    acceleration = m.acceleration_fit(1.29 * (1 - np.exp(-(t - 0.35) / 0.54)), t)
    t = np.arange(51) * 0.1
    aperture = m.aperture_drop(
        np.column_stack([4 + 1.05 * t - 0.05 * t**2, np.zeros(len(t))]), t, plane_m=8
    )  # Approach mean .95, passage speed .55: drop .40.
    t = np.arange(401) * 0.1
    # 60 crossings spaced 60/(59*1.9) seconds: 60/(t_last-t_first) = 1.9.
    x = 1.4 + t[:, None] - 1 - np.arange(60)[None, :] * (60 / (59 * 1.9))
    trace = np.stack([x, np.zeros_like(x)], axis=-1)
    # V3 uses the reused harness's finite-N estimator, unlike V4. Exercise that
    # exact production row path, replacing only its existing trajectory producer.
    with patch.object(
        suite,
        "protocol_simulate",
        return_value=(trace, np.ones((len(t) - 1, 60)), np.full(60, 1.29)),
    ):
        narrow = suite.run_task(("V3", 1001, "1.0", 0.25, "radius"))
    t = np.arange(141) * 0.1
    x = t[:, None] - 0.5 - np.arange(12)[None, :]
    wide = m.bottleneck_flow(
        np.stack([x, np.zeros_like(x)], axis=-1), t, width_m=2, plane_m=1, expected_n=12
    )  # All-data 12/(11*2); steady 11/(11*2).
    clearance = m.circumvention_clearance(
        np.array([[-1, 0.75], [0, 0.75], [1, 0.75]]),
        [0, 1, 2],
        centre_xy=(0, 0),
        obstacle_radius_m=0.25,
    )
    t = np.arange(101) * 0.1
    straight = np.column_stack([t, np.zeros(len(t))])
    turning = m.turning_onset(
        np.column_stack([t, np.clip(t - 3, 0, 1) * 0.4]),
        np.column_stack([10 - t, np.zeros(len(t))]),
        t,
        [straight] * 5,
    )  # Own PoMD x=5, first captured baseline-exceeding sample x=2.
    specifications = [
        ("V1", acceleration, "fitted_desired_speed_m_s", 1.29),
        ("V1", acceleration, "fitted_tau_s", 0.54),
        ("V2", aperture, "speed_drop_m_s", 0.4),
        ("V3", narrow, "specific_flow_persons_m_s", 1.9),
        ("V4", wide, "all_data_specific_flow_persons_m_s", 6 / 11),
        ("V4", wide, "steady_specific_flow_persons_m_s", 0.5),
        ("V5", clearance, "lateral_cm_to_edge_m", 0.5),
        ("V6", turning, "onset_m", 3.0),
    ]
    return [
        {
            "case": case,
            "quantity": key,
            "expected": expected,
            "observed": result[key],
            "known_answer_pass": bool(
                result[key] is not None and np.isclose(result[key], expected, rtol=0, atol=1e-8)
            ),
        }
        for case, result, key, expected in specifications
    ]


def ideal_gate_records() -> list[dict[str, object]]:
    """Return target-valued records isolating policy from trajectory feasibility.

    These hypothetical records are not simulated evidence. In particular they
    do not assert that the incompatible V2 disc geometry can produce them.
    """
    config = suite.load_config(suite.DEFAULT_CONFIG)
    values = [
        ("V1", "native", {"fitted_desired_speed_m_s": 1.29, "fitted_tau_s": 0.54}),
        ("V5", "diagnostic", {"lateral_cm_to_edge_m": 0.5}),
    ]
    for width in feasible_apertures(0.28):
        ratio = width / 0.46
        alpha = np.clip((ratio - 0.9) / 0.4, 0.0, 1.0)
        values.append(
            (
                "V2",
                str(width),
                {
                    "speed_drop_m_s": float((1 - alpha) * 0.4 + alpha * 0.14),
                    "aperture_shoulder_ratio": ratio,
                    "passed": True,
                },
            )
        )
    for width, target in zip(
        config["V3"]["widths_m"], config["V3"]["published_specific_flow_persons_m_s"], strict=True
    ):
        values.append(("V3", str(width), {"specific_flow_persons_m_s": target}))
    for width in config["V4"]["widths_m"]:
        values.append(
            (
                "V4",
                str(width),
                {
                    "all_data_specific_flow_persons_m_s": 2.3,
                    "steady_specific_flow_persons_m_s": 2.5,
                    "all_data_flow_persons_s": 2.3 * width,
                    "steady_flow_persons_s": 2.5 * width,
                },
            )
        )
    for speed, target in zip(
        config["V6"]["speeds_m_s"], config["V6"]["published_onset_m"], strict=True
    ):
        values.append(("V6", str(speed), {"onset_m": target}))
    return [
        {
            "case": case,
            "variant": variant,
            "seed": 1001,
            "radius_m": 0.28,
            "wall_penetration_m": 0.0,
            "pair_overlap": {"all": {"below_2r_count": 0}},
            **fields,
        }
        for case, variant, fields in values
    ]


def audit() -> dict[str, object]:
    """Return separate estimator, policy and geometry verdicts bound to source bytes."""
    config = suite.load_config(suite.DEFAULT_CONFIG)
    records, controls = ideal_gate_records(), estimator_controls()
    geometry = [
        {
            "radius_m": radius,
            "shoulder_ratio": ratio,
            "aperture_width_m": ratio * config["V2"]["shoulder_proxy_m"],
            "centre_slack_m": ratio * config["V2"]["shoulder_proxy_m"] - 2 * radius,
            "minimum_continuous_penetration_m": max(
                0.0, radius - ratio * config["V2"]["shoulder_proxy_m"] / 2
            ),
        }
        for radius in (0.25, 0.28, 0.30)
        for ratio in config["V2"]["ratios"]
    ]
    gate = suite.acceptance_gate(records)
    return {
        "schema": "calfit.preflight.v1",
        "evidence_tier": "synthetic implementation diagnostic; no model or release acceptance",
        "estimator_controls": controls,
        "all_estimators_known_answer_pass": all(c["known_answer_pass"] for c in controls),
        "ideal_gate_records": records,
        "ideal_gate": gate,
        "per_case_gate": {
            case: suite.acceptance_gate([r for r in records if r["case"] == case])
            for case in sorted({r["case"] for r in records})
        },
        "rigid_disc_aperture_geometry": geometry,
        "search_admissible": gate["exit_code"] == 0
        and all(c["known_answer_pass"] for c in controls)
        and all(width >= 2 * 0.28 + 0.05 for width in feasible_apertures(0.28)),
        "source_sha256": {
            str(path.relative_to(suite.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(suite.__file__),
                Path(m.__file__),
                Path(suite.reused.__file__),
                suite.DEFAULT_CONFIG,
            )
        },
        "experiment_episodes": 0,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a marked audit receipt; return 2 when the search contract is blocked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    return 0 if result["search_admissible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
