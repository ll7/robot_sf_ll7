#!/usr/bin/env python3
"""Rebuild the issue #3078 job-13521 diagnostic JSON and figures.

The builder consumes only the compact tracked Package A evidence bundle. It does
not read the private episode store, run a campaign, submit compute, or promote a
benchmark claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from robot_sf.evidence.writers import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO_ROOT / "docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16"
OUTPUT_NAMES = (
    "seed_rank_stability_diagnostic.json",
    "fig_seed_rank_stability.png",
    "fig_transfer_delta.png",
)
PLANNER_ORDER = ("goal", "social_force", "orca")
NATIVE_COLOR = "#4c72b0"
ADAPTER_COLOR = "#dd8452"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build_payload(bundle_dir: Path) -> dict[str, Any]:
    acceptance = _load_json(bundle_dir / "row_acceptance.json")
    registration = _load_json(bundle_dir / "registration.json")
    comparator = _load_json(bundle_dir / "no_eligible_comparator.json")
    transfer_rows = _load_csv(bundle_dir / "transfer_delta.csv")
    heldout_rows = _load_csv(bundle_dir / "heldout_family_table.csv")

    _require(acceptance.get("status") == "accepted", "row acceptance is not accepted")
    _require(acceptance.get("unique_identity_count") == 18, "expected 18 accepted identities")
    _require(acceptance.get("cell_count") == 6, "expected six held-out cells")
    _require(acceptance.get("seeds") == [111], "expected the sole evaluation seed 111")
    _require(
        acceptance.get("row_status_counts") == {"adapter": 12, "native": 6},
        "adapter/native accounting changed",
    )
    _require(
        comparator.get("status") == "no_eligible_comparator",
        "comparator receipt must remain no_eligible_comparator",
    )

    transfer_planners = [row["planner"] for row in transfer_rows]
    _require(
        len(transfer_planners) == len(set(transfer_planners)),
        "transfer table planner identities must be unique",
    )
    transfer_by_planner = {row["planner"]: row for row in transfer_rows}
    _require(
        set(transfer_by_planner) == set(PLANNER_ORDER),
        "transfer table planner identities changed",
    )

    planner_rows: list[dict[str, Any]] = []
    observed_status_counts: Counter[str] = Counter()
    for planner in PLANNER_ORDER:
        transfer = transfer_by_planner[planner]
        _require(not transfer["benchmark_set_mean_snqi"], f"{planner} benchmark mean is not empty")
        _require(not transfer["transfer_delta_snqi"], f"{planner} transfer delta is not empty")
        _require(
            transfer["claim_eligible"].lower() == "false",
            f"{planner} transfer row became claim eligible",
        )

        planner_heldout = [row for row in heldout_rows if row["planner"] == planner]
        _require(len(planner_heldout) == 2, f"{planner} must have two held-out family rows")
        statuses: set[str] = set()
        eligible_episode_count = 0
        family_means: list[float] = []
        for row in planner_heldout:
            counts = json.loads(row["row_status_counts"])
            _require(len(counts) == 1, f"{planner} row mixes evidence statuses")
            status, count = next(iter(counts.items()))
            statuses.add(status)
            observed_status_counts[status] += int(count)
            eligible_episode_count += int(row["eligible_episode_count"])
            family_means.append(float(row["mean_snqi"]))

        _require(len(statuses) == 1, f"{planner} evidence status differs by family")
        _require(eligible_episode_count == 6, f"{planner} must retain six eligible episodes")
        row_status = statuses.pop()
        expected_status = "native" if planner == "goal" else "adapter"
        _require(row_status == expected_status, f"{planner} must remain {expected_status}")

        mean_snqi = sum(family_means) / len(family_means)
        recorded_mean = float(transfer["heldout_family_mean_snqi"])
        _require(
            round(mean_snqi, 6) == round(recorded_mean, 6),
            f"{planner} held-out mean differs between compact tables",
        )
        planner_rows.append(
            {
                "heldout_family_mean_snqi": round(recorded_mean, 6),
                "planner": planner,
                "promotion_allowed": False,
                "rank_stability": "not_identifiable",
                "reason": "single evaluation seed (111)",
                "row_status": row_status,
                "seed_count": 1,
                "seeds": [111],
            }
        )

    _require(
        dict(observed_status_counts) == acceptance["row_status_counts"],
        "held-out table adapter/native counts differ from row acceptance",
    )
    _require(
        acceptance.get("fallback_degraded_rows") == 0,
        "unexpected fallback/degraded rows in accepted evidence",
    )
    _require(
        acceptance.get("synthetic_fixture_used") is False,
        "synthetic fixture usage must remain false",
    )

    source_store = registration["source_episode_store"]
    resolved_bundle = bundle_dir.resolve()
    try:
        generated_for = resolved_bundle.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        generated_for = str(resolved_bundle)
    return {
        "claim_boundary": (
            "Preliminary diagnostic evidence only. Seed/rank-stability and held-out "
            "transfer-delta are both not_identifiable. Adapter rows (social_force, orca) "
            "remain labeled adapter and are never relabeled native-only. No benchmark, "
            "ranking, paper/dissertation, or promotion claim."
        ),
        "diagnostic_scope": (
            "Real held-out-family pilot rows for job 13521; preliminary diagnostic evidence only."
        ),
        "forbidden_actions_confirmed": {
            "benchmark_campaign_run": False,
            "compute_submit": False,
            "paper_claim_edits": False,
            "ranking_claim_promotion": False,
        },
        "generated_for": generated_for,
        "headline_rank_stability_contract": {
            "caveats": [
                "Single evaluation seed (111); planner-rank stability cannot be estimated.",
                (
                    "This is a result of the diagnostic, not a gap to fill through substitution "
                    "or a new campaign (#6150 froze the comparator; #6156 Domain-Aware Approval)."
                ),
                "No benchmark, ranking, paper, or dissertation claim is promoted.",
            ],
            "claim_status": "not_identifiable_single_seed",
            "contract_scope": "issue_3078_seed_rank_stability_real_data",
            "label": "not_identifiable",
            "labels": ["not_identifiable"],
            "max_seed_count": 1,
            "metric_names": ["snqi"],
            "min_seed_budget": None,
            "missing_durable_roots": [],
            "pairwise": [],
            "promotion_allowed": False,
            "reason": (
                "Planner-rank stability is not identifiable from a single evaluation seed "
                "(111); stability requires multiple seeds per (cell, planner)."
            ),
            "seed_count": 1,
        },
        "heldout_transfer_delta_classification": {
            "baseline_table_empty": True,
            "claim_eligible": False,
            "claim_status": "not_identifiable_no_eligible_comparator",
            "comparator_receipt": "no_eligible_comparator.json",
            "label": "not_identifiable",
            "reason": (
                "No eligible benchmark-set comparator (frozen by #6150 / merged PR #6166); "
                "held-out transfer delta is empty with claim_eligible=false."
            ),
            "transfer_delta_snqi_empty": True,
        },
        "issue": 3078,
        "job_id": registration["job_id"],
        "planner_rank_stability": planner_rows,
        "provenance": {
            "cell_count": acceptance["cell_count"],
            "execution_commit": registration["execution_commit"],
            "fallback_degraded_rows": acceptance["fallback_degraded_rows"],
            "row_status_counts": acceptance["row_status_counts"],
            "rows_per_planner": registration["rows_per_planner"],
            "source_episode_store": source_store["uri"],
            "source_episode_store_sha256": source_store["sha256"],
            "synthetic_fixture_used": acceptance["synthetic_fixture_used"],
            "unique_identity_count": acceptance["unique_identity_count"],
        },
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "schema_version": "seed_sufficiency_analysis.v1",
        "seed_basis": {
            "note": "Sole evaluation seed is 111 across all 18 identities.",
            "planners": list(PLANNER_ORDER),
            "seed_count": 1,
            "seeds": [111],
        },
        "target_issue": 6156,
    }


def _save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def _write_seed_figure(path: Path, planner_rows: list[dict[str, Any]]) -> None:
    planners = [row["planner"] for row in planner_rows]
    values = [row["heldout_family_mean_snqi"] for row in planner_rows]
    colors = [
        NATIVE_COLOR if row["row_status"] == "native" else ADAPTER_COLOR for row in planner_rows
    ]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(planners, values, color=colors, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_ylabel("held-out-family mean SNQI (seed 111)")
    ax.set_title("Seed / rank stability: not_identifiable (single seed)")
    ax.set_ylim(min(values) - 0.12, 0.01)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax.text(
        0.98,
        0.97,
        "1 evaluation seed (111) -> rank stability not_identifiable\n"
        "no benchmark / ranking / paper claim",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.6"},
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.03,
            f"{value:.4f}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.legend(
        handles=[
            Patch(facecolor=NATIVE_COLOR, label="native (goal)"),
            Patch(facecolor=ADAPTER_COLOR, label="adapter (social_force, orca)"),
        ],
        loc="lower left",
        fontsize=8,
    )
    _save_figure(fig, path)


def _write_transfer_figure(path: Path, planner_rows: list[dict[str, Any]]) -> None:
    planners = [row["planner"] for row in planner_rows]
    values = [row["heldout_family_mean_snqi"] for row in planner_rows]
    colors = [
        NATIVE_COLOR if row["row_status"] == "native" else ADAPTER_COLOR for row in planner_rows
    ]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(planners, values, color=colors, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_ylabel("mean SNQI")
    ax.set_title("Held-out transfer delta: not_identifiable (no comparator)")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax.text(
        0.98,
        0.97,
        "no eligible benchmark-set comparator (#6150 / PR #6166)\n"
        "transfer delta = not_identifiable, claim_eligible=false",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.6"},
    )
    ax.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="0.6",
                hatch="///",
                label="benchmark-set: absent (no eligible comparator)",
            ),
            Patch(facecolor=NATIVE_COLOR, label="held-out: native (goal)"),
            Patch(facecolor=ADAPTER_COLOR, label="held-out: adapter (social_force, orca)"),
        ],
        loc="lower left",
        fontsize=8,
    )
    _save_figure(fig, path)


def build_outputs(bundle_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Build the diagnostic JSON and deterministic figures from tracked compact inputs."""
    payload = _build_payload(bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_path = output_dir / OUTPUT_NAMES[0]
    write_json(diagnostic_path, payload)
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        _write_seed_figure(output_dir / OUTPUT_NAMES[1], payload["planner_rank_stability"])
        _write_transfer_figure(output_dir / OUTPUT_NAMES[2], payload["planner_rank_stability"])
    return tuple(output_dir / name for name in OUTPUT_NAMES)


def check_outputs(bundle_dir: Path) -> list[str]:
    """Return tracked outputs whose bytes differ from a fresh deterministic build."""
    with tempfile.TemporaryDirectory(prefix="issue3078-job13521-") as tmp:
        generated = build_outputs(bundle_dir, Path(tmp))
        return [
            path.name
            for path in generated
            if path.read_bytes() != (bundle_dir / path.name).read_bytes()
        ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> int:
    """Run the deterministic builder or exact-byte check."""
    args = _build_parser().parse_args()
    if args.check:
        mismatches = check_outputs(args.bundle_dir)
        result = {
            "schema": "issue_3078_job_13521_diagnostic_build.v1",
            "ok": not mismatches,
            "checked_outputs": list(OUTPUT_NAMES),
            "mismatches": mismatches,
        }
        if args.json_output:
            print(json.dumps(result, sort_keys=True))
        elif mismatches:
            print(f"Diagnostic outputs differ: {', '.join(mismatches)}")
        else:
            print("Diagnostic outputs are byte-reproducible.")
        return 1 if mismatches else 0

    built = build_outputs(args.bundle_dir, args.output_dir)
    result = {"ok": True, "outputs": [str(path) for path in built]}
    print(json.dumps(result, sort_keys=True) if args.json_output else "\n".join(result["outputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
