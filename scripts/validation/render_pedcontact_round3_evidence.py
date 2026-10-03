"""Publish PEDCONTACT measurements from verified acquisition and test receipts.

Run with --evidence-root pointing at the preserved Round 3 artifact directory.
No environment is created or stepped by this renderer.
"""

import argparse
import json
from pathlib import Path

from robot_sf.evidence.writers import write_json, write_text

MARKER = "AI-GENERATED / NEEDS-REVIEW"
ARMS = ("off", "contact", "wall", "on")


def read(root, name):
    """Read a named receipt from the caller's explicit artifact root."""
    return json.loads((root / name).read_text())


def display(value):
    """Format a point or bound and its seed-mean confidence interval."""
    if not value or value.get("mean") is None:
        return "unavailable"
    mean = value["mean"]
    bounds = value.get("ci95", value.get("ci95_mean"))
    if bounds is None:
        bounds = [value.get("low"), value.get("high")]
    if bounds and all(x is not None for x in bounds):
        return f"{mean:.4g} [{bounds[0]:.4g}, {bounds[1]:.4g}]"
    return f"{mean:.4g}"


def publish_optional_receipts(root, output):
    """Copy completed robot and fit receipts without requiring unfinished ones."""
    for source, destination in (
        ("robot_gate_summary.json", "robot"),
        ("robot_failure_geometry.json", "robot_geometry"),
        ("fit/fit_result.json", "fit"),
    ):
        if (root / source).exists():
            payload = read(root, source)
            if isinstance(payload, list):
                payload = {"rows": payload}
            write_json(output / f"pedcontact_10101_round3_{destination}.json", payload)


def render(root, output):
    """Write versioned numerical receipts and a compact measurement table."""
    comparison = read(root, "step4_comparison.json")
    warm = read(root, "warm_congested.json")
    write_json(output / "pedcontact_10101_round3_step4.json", comparison)
    write_json(output / "pedcontact_10101_round3_warm.json", warm)
    lines = [
        "<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->",
        "# PEDCONTACT Round 3 measurements",
        MARKER,
        "",
        "Four arms; dev1001–1030; physical radius .28m; CALFIT default parameters. "
        "Seed-mean 95% Student intervals describe sampling uncertainty. V3 values "
        "from incomplete trials are completion-flow upper bounds, not point flows. "
        "V6 boundary onsets are right-censored lower bounds ≥3m, not exact onsets.",
        "",
        "| Item | Off | Contact only | Wall only | Both |",
        "|---|---|---|---|---|",
    ]
    items = comparison["arms"]["on"]["gate"]["checks"]
    for index, item in enumerate(items):
        cells = []
        for arm in ARMS:
            data = comparison["arms"][arm]
            check = data["gate"]["checks"][index]
            estimate = check.get("estimate")
            interval = check.get("observed_interval")
            if interval is None:
                interval = (
                    data.get("per_case", {})
                    .get(item["case"] + "/" + item["variant"], {})
                    .get("measurement")
                )
            if interval is None and item["case"] == "V4":
                interval = check.get("interval")
            cell = (
                display(interval)
                if interval
                else ("unavailable" if estimate is None else f"{estimate:.4g}")
            )
            if check.get("right_censored_n"):
                cell += f"; censored {check['right_censored_n']}/30"
            cells.append(cell + " " + check["status"])
        lines.append("| " + item["case"] + " " + item["variant"] + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The denominator is scoreable checks (measured estimates or decisive bounds). "
        "An upper bound overlapping the acceptance range is CENSORED and cannot PASS; "
        "an unavailable estimate is MISSING. Neither enters the scoreable denominator.",
        "",
        "| Arm | Passed / scoreable / potential | Overlap pair-steps | Wall penetration pedestrian-steps | ms/step |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        data = comparison["arms"][arm]
        gate = data["gate"]
        lines.append(
            f"| {arm} | {gate['passed_items']}/{gate['producible_items']}/{gate['potential_items']} "
            f"| {data['overlap_pair_steps']} | {data['wall_penetration_ped_steps']} "
            f"| {data['runtime_ms_per_step']:.4g} |"
        )
    lines += ["", "| Warm N | Off ms/step | Both ms/step | Ratio |", "|---|---|---|---|"]
    for n in (60, 150, 350):
        cases = {x["mode"]: x for x in warm["cases"] if x["n"] == n}
        off, on = cases["off"]["mean_ms"], cases["on"]["mean_ms"]
        lines.append(f"| {n} | {off:.4g} | {on:.4g} | {on / off:.3f} |")
    lines += [
        "",
        "Warm measurement uses the same real congested V4 frame after one warm-up "
        "step, then 20 timed steps. Full-suite runtime includes all measured scenarios "
        "and compilation/start-up effects; it must not be represented by the warm ratio.",
    ]
    publish_optional_receipts(root, output)
    write_text(output / "pedcontact_10101_round3_measurements.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    render(args.evidence_root, args.output_dir)
