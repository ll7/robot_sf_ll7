#!/usr/bin/env python3
"""Diagnostic paired development tradeoffs; predictive braking is not a safety guarantee.

Consume native paired CSVs from run_predictive_braking_diagnostics.py --campaign.
Reject incomplete, unpaired or fallback/degraded inputs. Pool complete nearby
pedestrian-window counts rather than averaging episode violation percentages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("hybrid_v4_default", "hybrid_v4_predictive_braking")
REQUIRED_COLUMNS = (
    "scenario",
    "seed",
    "arm",
    "success",
    "collision",
    "duration_s",
    "near_miss_onsets",
    "near_miss_exposure_s",
    "bound_2s_windows",
    "bound_2s_violations",
    "fallback_count",
    "degraded_count",
    "prediction_speed_error_m_s",
    "dt_s",
)


def audit_prediction_windows(robot, positions, velocities, *, dt_s, speed_error_m_s):
    """Return (complete nearby pedestrian-windows, violated windows) for a 2 s tube.

    Start a window at every recorded frame for every pedestrian initially within
    2 m centre distance. At each later sample, compare observed position with
    start position + start velocity * elapsed time. A window violates the bound
    if any residual norm exceeds speed_error_m_s * elapsed + 1e-9 m. Incomplete
    terminal windows are excluded. Simulator row identities are required stable;
    respawns in those rows remain included as prediction discontinuities.
    """
    if (
        not math.isfinite(dt_s)
        or dt_s <= 0
        or not math.isfinite(speed_error_m_s)
        or speed_error_m_s < 0
    ):
        raise ValueError("Invalid prediction-window dt or speed error")
    width = round(2.0 / dt_s)
    if width < 1 or not math.isclose(width * dt_s, 2.0, abs_tol=1e-9):
        raise ValueError("dt_s must divide the complete 2 s window")
    robot, positions, velocities = (
        np.asarray(value, dtype=float) for value in (robot, positions, velocities)
    )
    if (
        positions.ndim != 3
        or positions.shape[-1] != 2
        or velocities.shape != positions.shape
        or robot.shape != (len(positions), 2)
    ):
        raise ValueError("Prediction audit requires aligned (frames, pedestrians, 2) traces")
    if any(not np.isfinite(value).all() for value in (robot, positions, velocities)):
        raise ValueError("Prediction audit requires finite traces")
    count = len(positions) - width
    if count <= 0:
        return 0, 0
    starts = positions[:count]
    nearby = np.linalg.norm(starts - robot[:count, None, :], axis=2) <= 2.0
    violated = np.zeros_like(nearby)
    for offset in range(1, width + 1):
        elapsed = offset * dt_s
        residual = positions[offset : offset + count] - starts - velocities[:count] * elapsed
        violated |= np.linalg.norm(residual, axis=2) > speed_error_m_s * elapsed + 1e-9
    return int(np.count_nonzero(nearby)), int(np.count_nonzero(nearby & violated))


def _number(row, key, *, integer=False):
    """Validate a finite, nonnegative metric without imputing missing observations."""
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key}") from exc
    if not math.isfinite(value) or value < 0 or (integer and value != int(value)):
        raise ValueError(f"Invalid {key}: {value}")
    return int(value) if integer else value


def _wilson(successes, total):
    """Return a descriptive Wilson 95% proportion interval, not paired significance."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_pairs(rows, *, expected_seeds, expected_scenarios=None):  # noqa: C901, PLR0912 -- one fail-closed input contract
    """Return one wide tradeoff row per scenario, requiring exactly the requested pairs."""
    expected = set(expected_seeds)
    if not expected or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 1001 <= seed <= 1030
        for seed in expected
    ):
        raise ValueError("Expected seeds must be development seeds 1001-1030")
    grouped = {}
    identity = set()
    settings = set()
    for raw in rows:
        missing = set(REQUIRED_COLUMNS) - raw.keys()
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        row = dict(raw)
        for key in (
            "seed",
            "success",
            "collision",
            "near_miss_onsets",
            "bound_2s_windows",
            "bound_2s_violations",
            "fallback_count",
            "degraded_count",
        ):
            row[key] = _number(row, key, integer=True)
        for key in ("duration_s", "near_miss_exposure_s", "prediction_speed_error_m_s", "dt_s"):
            row[key] = _number(row, key)
        if row["seed"] not in expected or row["arm"] not in ARMS or not row["scenario"]:
            raise ValueError("Unexpected scenario, arm or development seed")
        if (
            row["success"] > 1
            or row["collision"] > 1
            or row["bound_2s_violations"] > row["bound_2s_windows"]
        ):
            raise ValueError("Invalid outcome or bound-window count")
        if (
            row["duration_s"] <= 0
            or row["dt_s"] <= 0
            or row["near_miss_exposure_s"] > row["duration_s"] + 1e-9
        ):
            raise ValueError("Invalid duration or near-miss exposure")
        if row["fallback_count"] or row["degraded_count"]:
            raise ValueError("Fallback/degraded execution is not paired evidence")
        key = row["scenario"], row["seed"], row["arm"]
        if key in identity:
            raise ValueError(f"Duplicate paired episode: {key}")
        identity.add(key)
        settings.add((row["dt_s"], row["prediction_speed_error_m_s"]))
        grouped.setdefault(row["scenario"], {}).setdefault(row["arm"], {})[row["seed"]] = row
    if not grouped or len(settings) != 1:
        raise ValueError("Empty input or mixed prediction-audit contracts")
    if expected_scenarios is not None and set(grouped) != set(expected_scenarios):
        raise ValueError("Missing or unexpected paired scenarios")
    report = []
    for scenario, arms in sorted(grouped.items()):
        if set(arms) != set(ARMS) or any(set(arms[arm]) != expected for arm in ARMS):
            raise ValueError(f"Missing paired development episodes for {scenario}")
        output = {"scenario": scenario, "episodes_per_arm": len(expected)}
        for arm, prefix in zip(ARMS, ("default", "predictive"), strict=True):
            episodes = list(arms[arm].values())
            for metric in ("success", "collision"):
                total = sum(row[metric] for row in episodes)
                output[f"{prefix}_{metric}_rate"] = total / len(episodes)
                lo, hi = _wilson(total, len(episodes))
                output[f"{prefix}_{metric}_wilson95_low"] = lo
                output[f"{prefix}_{metric}_wilson95_high"] = hi
            for metric in (
                "near_miss_onsets",
                "near_miss_exposure_s",
                "bound_2s_windows",
                "bound_2s_violations",
            ):
                output[f"{prefix}_{metric}"] = sum(row[metric] for row in episodes)
            denominator = output[f"{prefix}_bound_2s_windows"]
            output[f"{prefix}_bound_2s_violation_rate"] = (
                output[f"{prefix}_bound_2s_violations"] / denominator if denominator else None
            )
        output["success_gain"] = output["predictive_success_rate"] - output["default_success_rate"]
        output["near_miss_onsets_delta"] = (
            output["predictive_near_miss_onsets"] - output["default_near_miss_onsets"]
        )
        output["near_miss_exposure_s_delta"] = (
            output["predictive_near_miss_exposure_s"] - output["default_near_miss_exposure_s"]
        )
        report.append(output)
    return report


def _load_inputs(paths, seeds):
    """Bind CSV inputs to native producer manifests, including scenario completeness."""
    rows, sources, scenarios, contracts = [], [], set(), set()
    for path in paths:
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "diagnostic-only"
            or manifest.get("bound_audit") != "2s_all_sampled_offsets_euclidean_nearby_2m_v1"
        ):
            raise ValueError("Missing native paired prediction-audit manifest")
        if set(manifest["seeds"]) != set(seeds) or manifest.get("arms") != list(ARMS):
            raise ValueError("Input manifest must declare the requested complete pair")
        contracts.add(
            json.dumps(
                {
                    key: manifest[key]
                    for key in (
                        "head",
                        "sha256",
                        "empty",
                        "horizon_override",
                        "prediction_speed_error_m_s",
                    )
                },
                sort_keys=True,
            )
        )
        scenarios.update(manifest["scenarios"])
        with path.open(newline="") as stream:
            input_rows = list(csv.DictReader(stream))
        summarize_pairs(input_rows, expected_seeds=seeds, expected_scenarios=manifest["scenarios"])
        rows.extend(input_rows)
        sources.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "producer_head": manifest["head"],
                "producer_sources": manifest["sha256"],
            }
        )
    if len(contracts) != 1:
        raise ValueError("Mixed producer head, configuration or run contract")
    return rows, sources, scenarios


def main():
    """Write CSV, a side-by-side Markdown tradeoff table and input byte provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1031)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, sources, scenarios = _load_inputs(args.episodes, args.seeds)
    report = summarize_pairs(rows, expected_seeds=args.seeds, expected_scenarios=scenarios)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    lines = [
        "# Predictive braking development pair",
        "",
        "Diagnostic-only. Predictive braking is not a safety guarantee. All pairs use development seeds.",
        "Rates are proportions. Bound rates pool complete, overlapping 2 s pedestrian-windows",
        "initially within 2 m centre distance; n/a means no eligible windows. Windows are correlated.",
        "Every sampled offset is checked against the Euclidean constant-velocity error tube;",
        "respawn discontinuities remain included. This is not continuous-time coverage or a directional-only audit.",
        "Near-miss exposure is episode time in near-miss state; onsets are false-to-true transitions.",
        "Wilson intervals in the CSV describe success/contact proportions, not paired significance.",
        "",
        "| Scenario | Success default / predictive | Success gain | Near-miss onsets default / predictive (cost) | Near-miss seconds default / predictive (cost) | Contact rate default / predictive | 2 s bound rate default / predictive (windows) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report:
        rates = [
            "n/a"
            if row[f"{p}_bound_2s_violation_rate"] is None
            else f"{row[f'{p}_bound_2s_violation_rate']:.6f}"
            for p in ("default", "predictive")
        ]
        lines.append(
            f"| {row['scenario']} | {row['default_success_rate']:.3f} / {row['predictive_success_rate']:.3f} | {row['success_gain']:+.3f} | "
            f"{row['default_near_miss_onsets']} / {row['predictive_near_miss_onsets']} ({row['near_miss_onsets_delta']:+}) | "
            f"{row['default_near_miss_exposure_s']:.1f} / {row['predictive_near_miss_exposure_s']:.1f} ({row['near_miss_exposure_s_delta']:+.1f}) | "
            f"{row['default_collision_rate']:.3f} / {row['predictive_collision_rate']:.3f} | "
            f"{rates[0]} / {rates[1]} ({row['default_bound_2s_windows']} / {row['predictive_bound_2s_windows']}) |"
        )
    (args.output / "paired.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "schema_version": "predictive_braking_pair.v1",
        "status": "diagnostic-only",
        "seeds": sorted(args.seeds),
        "arms": ARMS,
        "inputs": sources,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dt_s": float(rows[0]["dt_s"]),
        "prediction_speed_error_m_s": float(rows[0]["prediction_speed_error_m_s"]),
    }
    (args.output / "analysis.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
