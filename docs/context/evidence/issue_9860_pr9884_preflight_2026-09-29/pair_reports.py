#!/usr/bin/env python3
"""Pair the setup-only #9860 before/after reports and check the guard report."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_CHECKER_COMMIT = "4665cd13fc205761a4edb256a04d23d37ffbc235"
FIXED_CHECKER_COMMIT = "b626384207ca88422d3f0014d53785976d10fc0b"
REVIEWED_HEAD = "67b5a8a336c4bfc29c957c552e27d84675e72767"
INVARIANT_FIELDS = (
    "reset_clearance",
    "respawn_safety",
    "step1_collision",
    "relocated_pedestrian_rows",
    "robot_radius_m",
    "pedestrian_radius_m",
)


def sha256(data: bytes) -> str:
    """Hash exact bytes."""
    return hashlib.sha256(data).hexdigest()


def read_report(path: Path) -> tuple[dict[str, Any], str, str]:
    """Return a report, its raw digest, and its timing-independent digest."""
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    report = json.loads(data)
    stable = dict(report)
    stable.pop("runtime_s", None)
    stable_bytes = (json.dumps(stable, indent=2, sort_keys=True) + "\n").encode()
    return report, sha256(data), sha256(stable_bytes)


def pair_reports(
    before_path: Path,
    after_path: Path,
    guard_path: Path,
    *,
    closure_path: Path = HERE / "input_closure.json",
) -> dict[str, Any]:
    """Assert exact row identities and return the paired diagnostic receipt."""
    before, before_hash, before_stable_hash = read_report(before_path)
    after, after_hash, after_stable_hash = read_report(after_path)
    guard, guard_hash, guard_stable_hash = read_report(guard_path)
    assert before["release_inputs"] == after["release_inputs"]
    assert before["cell_count"] == after["cell_count"] == guard["cell_count"] == 1440
    assert before["scenario_count"] == after["scenario_count"] == guard["scenario_count"] == 48
    assert before["seed_count"] == after["seed_count"] == guard["seed_count"] == 30
    assert before["input_error"] is after["input_error"] is guard["input_error"] is None

    def keyed(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
        rows = {(row["scenario"], row["seed"]): row for row in report["rows"]}
        assert len(rows) == 1440
        return rows

    old, new = keyed(before), keyed(after)
    assert old.keys() == new.keys()
    transitions: dict[str, int] = {}
    changed: list[str] = []
    invariant_mismatches: list[dict[str, Any]] = []
    for scenario, seed in sorted(old):
        first, second = old[scenario, seed], new[scenario, seed]
        transition = f"{first['overall_status']}->{second['overall_status']}"
        transitions[transition] = transitions.get(transition, 0) + 1
        if first["overall_status"] != second["overall_status"]:
            assert first["footprint_reachability"]["status"] == "fail"
            assert second["footprint_reachability"]["status"] == "pass"
            assert second["footprint_reachability"].get("grid_status") == "fail"
            assert (
                second["footprint_reachability"].get("continuous_oracle", {}).get("status")
                == "pass"
            )
            changed.append(f"{scenario}:{seed}")
        for field in INVARIANT_FIELDS:
            if first.get(field) != second.get(field):
                invariant_mismatches.append({"scenario": scenario, "seed": seed, "field": field})
    assert transitions == {"blocked->valid": 75, "valid->valid": 1365}, transitions
    assert not invariant_mismatches, invariant_mismatches[:5]

    unsafe_nominal = [
        row
        for row in guard["rows"]
        if row["scenario"] != "francis2023_narrow_doorway"
        and row["footprint_reachability"].get("continuous_oracle", {}).get("reason")
        == "required_route_point_below_continuous_margin"
    ]
    doorway = [row for row in guard["rows"] if row["scenario"] == "francis2023_narrow_doorway"]
    assert guard["blocked_cell_count"] == 101
    assert len(unsafe_nominal) == 71 and len(doorway) == 30
    assert all(row["overall_status"] == "blocked" for row in unsafe_nominal + doorway)
    assert (
        sum(
            row["footprint_reachability"]["continuous_oracle"]["reason"]
            == "ordered_route_segment_disconnected_in_continuous_free_space"
            for row in doorway
        )
        == 28
    )
    assert (
        sum(
            row["footprint_reachability"]["continuous_oracle"]["reason"]
            == "required_route_point_below_continuous_margin"
            for row in doorway
        )
        == 2
    )

    return {
        "schema_version": "issue_9860_checker_pair.v2",
        "evidence_class": "preflight_diagnostic_only",
        "claim_boundary": "Static setup feasibility only; no planner action, episode outcome, or release admission.",
        "reviewed_head": REVIEWED_HEAD,
        "before_source_commit": BASE_CHECKER_COMMIT,
        "after_source_commit": FIXED_CHECKER_COMMIT,
        "diagnostic_overlay_sha256": sha256(
            (HERE / "goal_clearance_runtime.patch.gz").read_bytes()
        ),
        "inputs": before["release_inputs"],
        "input_closure_sha256": sha256(closure_path.read_bytes()),
        "before_report_sha256": before_hash,
        "after_report_sha256": after_hash,
        "unsafe_probe_report_sha256": guard_hash,
        "before_stable_sha256": before_stable_hash,
        "after_stable_sha256": after_stable_hash,
        "unsafe_probe_stable_sha256": guard_stable_hash,
        "unsafe_probe_manifest_sha256": guard["release_inputs"]["manifest_sha256"],
        "unsafe_probe_matrix_sha256": guard["release_inputs"]["scenario_matrix_sha256"],
        "unsafe_nominal_goals_blocked": len(unsafe_nominal),
        "historical_doorway_probe_cells_blocked": len(doorway),
        "before_status": before["status"],
        "after_status": after["status"],
        "before_blocked": before["blocked_cell_count"],
        "after_blocked": after["blocked_cell_count"],
        "transitions": transitions,
        "newly_blocked": 0,
        "recovered_cell_diagnostics": {
            "before_reachability": "fail",
            "after_reachability": "pass",
            "after_grid_status": "fail",
            "after_continuous_oracle": "pass",
        },
        "invariant_fields_compared": list(INVARIANT_FIELDS),
        "invariant_mismatches": invariant_mismatches,
        "changed_cells": changed,
    }


def write_pair(report: dict[str, Any], output_dir: Path) -> None:
    """Write the machine and human summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PR #9884 corrected setup preflight pair",
        "",
        "Diagnostic only: static setup feasibility; no planner action or outcome metric.",
        "",
        f"- Before checker: `{report['before_source_commit']}` (PR ancestor).",
        f"- After checker: `{report['after_source_commit']}` (PR fix commit).",
        f"- Reviewed head: `{report['reviewed_head']}`.",
        f"- Committed goal-clearance diagnostic overlay SHA-256: `{report['diagnostic_overlay_sha256']}`.",
        f"- Manifest SHA-256: `{report['inputs']['manifest_sha256']}`.",
        f"- Matrix SHA-256: `{report['inputs']['scenario_matrix_sha256']}`.",
        f"- Seed-set SHA-256: `{report['inputs']['seed_sets_sha256']}`.",
        f"- 79-file input closure SHA-256: `{report['input_closure_sha256']}`.",
        "- Identities: 48 scenarios × 30 seeds = 1,440 paired rows.",
        "- Before: 75 blocked; after: 0 blocked. Transitions: 75 blocked→valid, 1,365 valid→valid, zero newly blocked.",
        "- Each recovered cell retains grid failure and has a passing continuous oracle.",
        "- Reset clearance, respawn safety, step-1 collision diagnostic, relocation, and radius fields match for every identity.",
        "- Separate guard run: 71/71 unsafe nominal goals and 30/30 doorway cells blocked (28 disconnected, two unsafe goals).",
        f"- Guard manifest SHA-256: `{report['unsafe_probe_manifest_sha256']}`.",
        f"- Guard matrix SHA-256: `{report['unsafe_probe_matrix_sha256']}`.",
        "",
        "| Report | Raw JSON SHA-256 | Timing-independent SHA-256 |",
        "| --- | --- | --- |",
    ]
    for prefix, label in (("before", "Before"), ("after", "After"), ("unsafe_probe", "Guard")):
        lines.append(
            f"| {label} | `{report[f'{prefix}_report_sha256']}` | "
            f"`{report[f'{prefix}_stable_sha256']}` |"
        )
    lines.extend(
        (
            "",
            "Full reports, inputs, and the exact reproduction command are tracked here and in `README.md`.",
            "This physical diagnostic input is not the final DOI-free 0.0.8 campaign candidate or release admission.",
            "",
        )
    )
    (output_dir / "paired.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Read supplied reports, check claims, and write the pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=HERE / "before.json.gz")
    parser.add_argument("--after", type=Path, default=HERE / "after.json.gz")
    parser.add_argument("--guard", type=Path, default=HERE / "unsafe_probe.json.gz")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    report = pair_reports(args.before, args.after, args.guard)
    write_pair(report, args.output_dir)
    print(
        json.dumps(
            {
                "before_blocked": report["before_blocked"],
                "after_blocked": report["after_blocked"],
                "guard_blocked": report["unsafe_nominal_goals_blocked"]
                + report["historical_doorway_probe_cells_blocked"],
                "transitions": report["transitions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
