#!/usr/bin/env python3
"""Calibrate scenario priors from simulation trace clusters.

Loads simulation trace exports, derives simple cluster features, clusters
traces deterministically, and emits a compact report plus scenario_prior.v1
candidate cards with lineage pointers.

Usage::

    uv run python scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py \\
        --trace-dir tests/fixtures/analysis_workbench/simulation_trace_export_v1 \\
        --output-dir docs/context/evidence/issue_2726_scenario_prior_trace_clusters
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import yaml

# parent 1 is scripts/analysis, parent 2 is scripts, parent 3 is the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# scenario-prior staging mode tags (Issue #2657). Kept in sync with the canonical staging tool.
MODE_PROXY = "proxy_schema_smoke"
MODE_DATASET_BACKED = "dataset_backed_prior"
# Canonical external-data subsystem (issue #3473): the SDD scenario-prior gate lives here now.
CANONICAL_EXTERNAL_DATA_SCRIPT = REPO_ROOT / "scripts" / "tools" / "manage_external_data.py"

CLAIM_BOUNDARY = (
    "repository_trace_grounded_not_real_world_calibrated: this report and generated prior "
    "cards are derived entirely from deterministic simulation trace clusters. They do not "
    "claim real-world validity, representativeness, or generalizability to real-world pedestrian "
    "behavior. Refer to issues #3161 and #2918 for real-world staging and calibration requirements."
)


def resolve_staging_mode() -> dict[str, Any]:
    """Resolve the scenario-prior staging mode from SDD staging state (Issue #2657).

    A missing or unvalidated SDD copy forces ``proxy_schema_smoke``; only a staged-and-validated
    SDD unlocks ``dataset_backed_prior``. This script consumes simulation traces (not SDD) and is
    therefore always at most ``proxy_schema_smoke``, but it surfaces the gate explicitly so a
    missing dataset can never be implied as dataset-backed evidence.
    """
    fallback = {
        "mode": MODE_PROXY,
        "dataset_backed": False,
        "reason": (
            "SDD staging gate unavailable; defaulting to proxy_schema_smoke. Trace-cluster priors "
            "are never dataset-backed regardless of SDD state."
        ),
    }
    if not CANONICAL_EXTERNAL_DATA_SCRIPT.is_file():
        return fallback
    try:
        spec = importlib.util.spec_from_file_location(
            "_sdd_staging_gate", CANONICAL_EXTERNAL_DATA_SCRIPT
        )
        if spec is None or spec.loader is None:
            return fallback
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gate = module.resolve_sdd_scenario_prior_mode()
    except Exception as exc:
        fallback["reason"] = f"SDD staging gate error ({exc}); defaulting to proxy_schema_smoke."
        return fallback
    # Trace-cluster priors are never dataset-backed; clamp to proxy regardless of SDD availability.
    sdd_backed = bool(gate.get("dataset_backed"))
    return {
        "mode": MODE_PROXY,
        "dataset_backed": False,
        "sdd_local_availability": "staged" if sdd_backed else "missing",
        "sdd_staging_reason": gate.get("reason"),
        "reason": (
            "Trace-cluster priors are proxy_schema_smoke by construction (simulation traces, not "
            "SDD). SDD staging state is surfaced for provenance only."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1",
        help="Directory to recursively search for trace JSON exports.",
    )
    parser.add_argument(
        "--trace-paths",
        type=pathlib.Path,
        nargs="*",
        default=None,
        help="Explicit list of trace paths to parse, overriding --trace-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "docs/context/evidence/issue_2726_scenario_prior_trace_clusters",
        help="Directory to write report and YAML registry.",
    )
    return parser.parse_args(argv)


def extract_features(  # noqa: C901, PLR0912, PLR0915
    trace_data: dict[str, Any], file_path: pathlib.Path
) -> dict[str, Any]:
    """Extract cluster features from a single simulation trace export."""
    source = trace_data.get("source") or {}
    scenario_id = source.get("scenario_id") or "unknown"
    episode_id = source.get("episode_id") or "unknown"
    planner_id = source.get("planner_id") or "unknown"
    trace_id = trace_data.get("trace_id") or "unknown"
    seed = source.get("seed")

    frames = trace_data.get("frames") or []
    if not frames:
        frames = trace_data.get("steps") or []

    # 1. Bottleneck width heuristic
    bottleneck_width = None
    if "bottleneck" in scenario_id:
        if "narrow" in scenario_id:
            bottleneck_width = 0.8
        elif "wide" in scenario_id:
            bottleneck_width = 2.5
        else:
            bottleneck_width = 1.5  # medium/default bottleneck width

    # 2. Pedestrian count
    ped_ids = set()
    for frame in frames:
        for ped in frame.get("pedestrians", []):
            ped_id = ped.get("id")
            if ped_id is not None:
                ped_ids.add(ped_id)
    pedestrian_count = len(ped_ids)

    # 3. Pedestrian density classification
    ped_density = None
    for key in ["dense_stress_metadata", "metadata", "simulation_config"]:
        meta = trace_data.get(key)
        if isinstance(meta, dict):
            ped_density = (
                meta.get("ped_density") or meta.get("pedestrian_density") or meta.get("density")
            )
            if ped_density is not None:
                break

    if ped_density is not None:
        try:
            ped_density_val = float(ped_density)
            if ped_density_val <= 0.025:
                density_label = "sparse"
            elif ped_density_val <= 0.075:
                density_label = "medium"
            else:
                density_label = "dense"
        except (ValueError, TypeError):
            density_label = str(ped_density)
    elif "dense" in scenario_id or pedestrian_count >= 3:
        density_label = "dense"
    elif "medium" in scenario_id or pedestrian_count == 2:
        density_label = "medium"
    else:
        density_label = "sparse"

    # 4. Signal/crossing metadata
    has_signal = False
    signal_states = set()
    green_steps = 0
    red_steps = 0
    signal_frames = 0

    for frame in frames:
        for ped in frame.get("pedestrians", []):
            sig = ped.get("signal_state")
            if sig and sig.get("available"):
                has_signal = True
                lbl = sig.get("label")
                if lbl:
                    signal_states.add(lbl)
                    signal_frames += 1
                    if lbl.lower() == "green":
                        green_steps += 1
                    elif lbl.lower() == "red":
                        red_steps += 1

    green_fraction = green_steps / signal_frames if signal_frames > 0 else 0.0

    # 5. Robot route progress
    robot_positions = []
    for frame in frames:
        r = frame.get("robot")
        if r and "position" in r:
            robot_positions.append(r["position"])

    displacement = 0.0
    total_distance = 0.0
    if len(robot_positions) >= 2:
        r_array = np.array(robot_positions)
        displacement = float(np.linalg.norm(r_array[-1] - r_array[0]))
        total_distance = float(
            sum(np.linalg.norm(r_array[i] - r_array[i - 1]) for i in range(1, len(r_array)))
        )

    # 6. Stop/yield events
    stop_steps = 0
    for frame in frames:
        r = frame.get("robot") or {}
        vel = r.get("velocity", [0.0, 0.0])
        speed = float(np.linalg.norm(vel))
        if speed < 0.05:
            stop_steps += 1

    stop_fraction = stop_steps / len(frames) if frames else 0.0
    planner_stop_events = sum(
        1
        for frame in frames
        if frame.get("planner", {}).get("event") in ("waiting_red", "stop", "yield", "waiting")
    )

    # 7. Topology-switch events
    topology_switches = 0
    prev_sig = None
    for frame in frames:
        planner = frame.get("planner") or {}
        sig = (
            planner.get("active_hypothesis")
            or planner.get("route_id")
            or planner.get("topology_signature")
        )
        if sig is not None:
            if isinstance(sig, (list, dict)):
                sig_key = str(sorted(sig) if isinstance(sig, list) else sorted(sig.items()))
            else:
                sig_key = str(sig)

            if prev_sig is not None and sig_key != prev_sig:
                topology_switches += 1
            prev_sig = sig_key

    event_switches = sum(
        1 for frame in frames if frame.get("planner", {}).get("event") == "topology-switch"
    )
    topology_switches = max(topology_switches, event_switches)

    # 8. Collision/near-miss outcome
    min_dist = float("inf")
    for frame in frames:
        r = frame.get("robot")
        if not r or "position" not in r:
            continue
        r_pos = np.array(r["position"])
        for ped in frame.get("pedestrians", []):
            if "position" not in ped:
                continue
            p_pos = np.array(ped["position"])
            dist = float(np.linalg.norm(r_pos - p_pos))
            min_dist = min(min_dist, dist)

    if min_dist < 0.35:
        outcome = "collision"
    elif min_dist < 0.8:
        outcome = "near-miss"
    else:
        outcome = "nominal"

    try:
        rel_path = file_path.relative_to(REPO_ROOT)
    except ValueError:
        rel_path = file_path

    return {
        "file_path": str(rel_path),
        "trace_id": trace_id,
        "scenario_id": scenario_id,
        "episode_id": episode_id,
        "planner_id": planner_id,
        "seed": seed,
        "features": {
            "bottleneck_width": bottleneck_width,
            "pedestrian_count": pedestrian_count,
            "pedestrian_density": density_label,
            "has_signal": has_signal,
            "signal_green_fraction": green_fraction,
            "robot_displacement_m": displacement,
            "robot_total_distance_m": total_distance,
            "stop_fraction": stop_fraction,
            "planner_stop_events": planner_stop_events,
            "topology_switches": topology_switches,
            "min_distance_m": min_dist if min_dist != float("inf") else None,
            "outcome": outcome,
        },
    }


def assign_to_cluster(features: dict[str, Any], scenario_id: str) -> str:
    """Map the extracted trace features to a deterministic cluster ID."""
    # 1. Scenario type
    if "bottleneck" in scenario_id:
        scenario_type = "bottleneck"
    elif "crossing" in scenario_id:
        scenario_type = "crossing"
    elif "occluded" in scenario_id:
        scenario_type = "occluded"
    else:
        scenario_type = "general"

    # 2. Density
    ped_count = features["pedestrian_count"]
    if ped_count >= 3:
        density = "dense"
    elif ped_count == 2:
        density = "medium"
    else:
        density = "sparse"

    # 3. Signal presence
    signal = "signalized" if features["has_signal"] else "unsignalized"

    # 4. Outcome
    outcome = features["outcome"]

    return f"{scenario_type}_{density}_{signal}_{outcome}"


def build_markdown_report(
    clusters: dict[str, list[dict[str, Any]]],
    total_processed: int,
) -> str:
    """Construct a Markdown report summarizing cluster stats and trace lineage."""
    lines = [
        "# Calibrate Scenario Priors from Simulation Trace Clusters",
        "",
        "## Claim Boundary & Evidence Status",
        "",
        f"- **Claim Boundary**: `{CLAIM_BOUNDARY}`",
        "- **Status**: `repository_trace_derived_proposal`",
        "- **Context References**: Refer to issue #3161 for real-world staging contracts and #2918 for calibration context.",
        "- **Axioms**: This analysis runs strictly on repository trace fixtures. No external dataset or real-world representativeness is claimed.",
        "",
        "## Summary",
        "",
        f"- **Total simulation traces processed**: {total_processed}",
        f"- **Total unique clusters identified**: {len(clusters)}",
        "",
        "## Cluster Table",
        "",
        "| Cluster ID | Scenarios | Density | Signal | Outcome | Traces | Mean Min Dist (m) | Mean Stop Frac |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for cluster_id, traces in sorted(clusters.items()):
        parts = cluster_id.split("_")
        scen_type, density, signal, outcome = parts[0], parts[1], parts[2], parts[3]

        min_dists = [
            t["features"]["min_distance_m"]
            for t in traces
            if t["features"]["min_distance_m"] is not None
        ]
        mean_dist = f"{np.mean(min_dists):.2f}" if min_dists else "N/A"

        stop_fracs = [t["features"]["stop_fraction"] for t in traces]
        mean_stop = f"{np.mean(stop_fracs):.2f}" if stop_fracs else "N/A"

        lines.append(
            f"| `{cluster_id}` | {scen_type} | {density} | {signal} | {outcome} | {len(traces)} | {mean_dist} | {mean_stop} |"
        )

    lines.extend(
        [
            "",
            "## Cluster Lineage Details",
            "",
            "For each cluster, the lineage pointers to source trace file paths, trace IDs, and episode IDs are detailed below.",
            "",
        ]
    )

    for cluster_id, traces in sorted(clusters.items()):
        lines.extend(
            [
                f"### Cluster: `{cluster_id}`",
                "",
                "- **Traces count**: " + str(len(traces)),
                "- **Source Trace Pointers**:",
            ]
        )
        for t in traces:
            lines.append(
                f"  - `{t['file_path']}` (trace_id: `{t['trace_id']}`, episode_id: `{t['episode_id']}`)"
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- These priors are repository-trace-grounded proposals, not real-world calibrated priors.",
            "- The input set is limited to committed simulation trace fixtures and may overrepresent synthetic edge cases.",
            "- Cluster assignment is deterministic and rule-based; it is intended for provenance and reviewability rather than statistical optimality.",
            "",
            "## Follow-Up Data Requirements",
            "",
            "- Stage real-world trajectory data under issue #3161 before making representativeness claims.",
            "- Use issue #2918 calibration context before promoting these proposal cards into calibrated priors.",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_prior_cards(
    clusters: dict[str, list[dict[str, Any]]],
    staging_mode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate scenario prior registry cards based on parsed clusters."""
    staging_mode = staging_mode or resolve_staging_mode()
    cards = []
    for cluster_id, traces in sorted(clusters.items()):
        parts = cluster_id.split("_")
        scenario_type = parts[0]
        density = parts[1]
        signal = parts[2]
        outcome = parts[3]

        source_traces_pointers = []
        for t in traces:
            trace_info = (
                f"{t['file_path']} (trace_id: {t['trace_id']}, episode_id: {t['episode_id']})"
            )
            source_traces_pointers.append(trace_info)

        min_dists = [
            t["features"]["min_distance_m"]
            for t in traces
            if t["features"]["min_distance_m"] is not None
        ]
        min_dist_range = (
            f"[{min(min_dists):.2f}, {max(min_dists):.2f}]" if min_dists else "not_applicable"
        )

        card = {
            "card_id": f"trace_cluster_{cluster_id}",
            "prior_family": f"Repository trace cluster for {scenario_type} scenarios with {density} pedestrian density ({signal}, {outcome})",
            "classification": "repository_trace_derived",
            "source_type": "repository_trace_derived",
            "dataset_license_status": "not_applicable_generated_synthetic",
            "source_traces": source_traces_pointers,
            "feature_extraction_method": (
                "Deterministic rule-based feature extraction and cluster assignment of "
                "repository simulation trace exports. No heavy clustering or machine learning applied."
            ),
            "parameter_bounds": {
                "representation": "scenario_prior.v1",
                "units": "position=m, velocity=m/s, time=s",
                "enforcement": "rule_based_clustering",
                "min_distance_range_m": min_dist_range,
                "trace_count": len(traces),
            },
            "excluded_populations": [
                "real-world datasets not staged in-repo (refer to #3161 for staging contracts)",
                "hardware-calibrated environments (refer to #2918 for calibration context)",
            ],
            "unsupported_claims": [
                "learned_prior_realism",
                "benchmark_usefulness",
                "cross_dataset_generalization",
                "planner_performance_improvement",
                "license_safe_redistribution_of_raw_data",
            ],
            "odd_conditions": [
                "repository-trace-grounded only; does not represent real-world statistics",
                "diagnostic-only cluster prior family",
            ],
            "related_surfaces": [
                "docs/context/evidence/issue_2726_scenario_prior_trace_clusters/report.md",
                "docs/context/evidence/issue_2726_scenario_prior_trace_clusters/report.json",
            ],
        }
        cards.append(card)

    registry = {
        "schema_version": "scenario-prior-card-registry.v1",
        "name": "calibrate_scenario_priors_from_traces_issue_2726",
        "issue": 2726,
        "status": "partial_initial_registry",
        "benchmark_evidence": False,
        "scenario_prior_mode": staging_mode["mode"],
        "dataset_backed": staging_mode["dataset_backed"],
        "sdd_staging_gate": staging_mode,
        "claim_boundary": CLAIM_BOUNDARY,
        "cards": cards,
    }
    return registry


def main(argv: list[str] | None = None) -> int:
    """Run CLI tool to parse, cluster, and output scenario prior evidence."""
    args = parse_args(argv)

    # 1. Resolve paths to process
    trace_files: list[pathlib.Path] = []
    if args.trace_paths:
        trace_files = [pathlib.Path(p) for p in args.trace_paths]
    # Recursively search trace-dir for json files
    elif args.trace_dir.exists():
        trace_files = sorted(args.trace_dir.glob("**/*.json"))

    print(f"Found {len(trace_files)} JSON candidate files to analyze.")

    # 2. Extract features
    results = []
    for tf in trace_files:
        try:
            with open(tf, encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation
            if not isinstance(data, dict):
                continue
            if "schema_version" not in data or "frames" not in data:
                # Skip files that don't match the simulation_trace_export schema
                continue

            features = extract_features(data, tf)
            results.append(features)
        except Exception as e:
            print(f"Skipping {tf} due to parsing error: {e}")

    # 3. Deterministic Clustering
    clusters: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        cluster_id = assign_to_cluster(r["features"], r["scenario_id"])
        clusters.setdefault(cluster_id, []).append(r)

    print(f"Traces grouped into {len(clusters)} clusters.")

    # 4. Resolve SDD staging mode gate (Issue #2657) and generate YAML registry cards
    staging_mode = resolve_staging_mode()
    print(
        f"scenario_prior_mode: {staging_mode['mode']} (dataset_backed={staging_mode['dataset_backed']})"
    )
    registry = generate_prior_cards(clusters, staging_mode)

    # 5. Build reports
    md_report = build_markdown_report(clusters, len(results))

    # 6. Ensure output directory exists and write artifacts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = args.output_dir / "scenario_prior_cards_issue_2726.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(registry, fh, default_flow_style=False, sort_keys=False)

    md_path = args.output_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_report)

    json_path = args.output_dir / "report.json"
    report_json_data = {
        "schema_version": "scenario-prior-calibration-report.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "repository_trace_derived_proposal",
        "scenario_prior_mode": staging_mode["mode"],
        "dataset_backed": staging_mode["dataset_backed"],
        "sdd_staging_gate": staging_mode,
        "trace_count": len(results),
        "cluster_count": len(clusters),
        "total_processed": len(results),
        "clusters_count": len(clusters),
        "limitations": [
            "Repository trace fixtures are synthetic or simulator-derived and are not real-world calibrated.",
            "Rule-based clustering is deterministic and reviewable, but not a statistical clustering claim.",
            "Candidate cards are proposal surfaces only and do not establish benchmark usefulness or planner performance improvement.",
        ],
        "follow_up_data_requirements": [
            "Stage real-world trajectory data under issue #3161 before making representativeness claims.",
            "Use issue #2918 calibration context before promoting these proposal cards into calibrated priors.",
        ],
        "scenario_prior_candidates": registry["cards"],
        "clusters": {
            cid: {
                "traces_count": len(traces),
                "traces": [
                    {
                        "file_path": t["file_path"],
                        "trace_id": t["trace_id"],
                        "episode_id": t["episode_id"],
                        "planner_id": t["planner_id"],
                        "features": t["features"],
                    }
                    for t in traces
                ],
            }
            for cid, traces in clusters.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_json_data, fh, indent=2, sort_keys=True)

    # Write README.md in output dir
    readme_path = args.output_dir / "README.md"
    readme_content = f"""# Issue #2726 Scenario Priors Calibration Report

## Scope

This directory contains evidence and report artifacts for calibrating scenario priors from trace clusters.
It processes simulation trace exports from the repository, extracts features, groups them deterministically,
and generates scenario prior candidate cards.

## Evidence Status

- `schema`: `scenario-prior-card-registry.v1`
- `claim_boundary`: `{CLAIM_BOUNDARY}`
- `validation_reference`: Refer to #3161 and #2918.

## Files

- [scenario_prior_cards_issue_2726.yaml](scenario_prior_cards_issue_2726.yaml): Generated prior cards
- [report.md](report.md): Human-readable Markdown summary report
- [report.json](report.json): Fully detailed cluster statistics and traces mapping JSON

## Reproducible Command

```bash
uv run python scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py \\
    --trace-dir tests/fixtures/analysis_workbench/simulation_trace_export_v1 \\
    --output-dir docs/context/evidence/issue_2726_scenario_prior_trace_clusters
```
"""
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(readme_content)

    print(f"Report and YAML cards successfully written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
