#!/usr/bin/env python3
"""Fail-closed curation readiness preflight for SDD-derived benchmark scenarios (issue #1126).

This is the *curation-step* gate, distinct from the SDD *staging* gate owned by
``scripts/tools/manage_external_data.py`` (issues #1497 / #2413) and the SDD *importer*
``scripts/tools/import_sdd_scenarios.py`` (issue #1091). It answers one question before any real
curation work begins:

    Is a licensed Stanford Drone Dataset (SDD) annotation source staged and trusted enough that a
    curated scenario may be promoted as *benchmark* evidence, or must the run stay strictly
    proxy/exploratory?

It does **not** download, ingest, or curate real SDD data, and it never writes scenario/map output.
It composes the canonical owners instead of re-deriving them:

- ``manage_external_data.resolve_sdd_scenario_prior_mode`` decides ``dataset_backed_prior`` vs
  ``proxy_schema_smoke`` from the canonical staging manifest + checksum policy.
- ``import_sdd_scenarios.load_sdd_points`` (the importer's own parser) probes whether a candidate
  annotation file satisfies the deterministic curation selection rule (enough usable tracks after
  ``lost``/label filtering), without producing any scenario artifacts.

Hard contract (mirrors AGENTS.md fail-closed benchmark policy and the issue's "synthetic/fixture
rows never promoted as benchmark evidence" rule):

- ``benchmark_promotion_allowed`` is True **only** when SDD is staged AND checksum-validated
  (``dataset_backed_prior``). A fixture or an unpinned/unvalidated staged copy can be *probed* for
  schema readiness, but its output stays ``proxy_schema_smoke`` and must not be promoted.
- When SDD is missing, the run is ``blocked_external_input``: curation must not proceed as benchmark
  evidence. ``--require-benchmark-ready`` makes that a non-zero exit so callers fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.tools import import_sdd_scenarios, manage_external_data

PREFLIGHT_SCHEMA = "sdd_curation_preflight.v1"
ISSUE = 1126

# Curation evidence states (kept strictly distinct so a proxy run can never read as benchmark).
EVIDENCE_BLOCKED = "blocked_external_input"
EVIDENCE_PROXY = "proxy_schema_smoke"
EVIDENCE_BENCHMARK_CANDIDATE = "benchmark_candidate"

# Output classification the downstream curation run may claim. The final benchmark_ready vs
# exploratory_only decision still requires a real importer + smoke run (out of this preflight's
# scope); this only states the ceiling the external-data gate currently permits.
OUTPUT_BLOCKED = "blocked"
OUTPUT_PROXY_ONLY = "proxy_only_exploratory"
OUTPUT_BENCHMARK_READY_CANDIDATE = "benchmark_ready_candidate"


def probe_annotation_file(
    path: Path,
    *,
    label: str,
    min_track_points: int,
    max_pedestrians: int,
) -> dict[str, Any]:
    """Probe a candidate SDD annotation file against the curation selection rule.

    Reuses the canonical importer parser so the probe matches exactly what curation would consume.
    No scenario/map output is written. The probe is intentionally agnostic to whether the file is a
    fixture or a staged real annotation; benchmark-promotion eligibility is decided separately by the
    staging gate, never by this probe.

    Returns:
        dict: ``exists``, parse/selection outcome, ``usable_label_points``, ``usable_track_count``
        (tracks meeting ``min_track_points``), ``selection_satisfiable``, and ``blockers``.
    """
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "label": label,
        "min_track_points": min_track_points,
        "max_pedestrians": max_pedestrians,
        "usable_label_points": 0,
        "usable_track_count": 0,
        "selection_satisfiable": False,
        "blockers": [],
    }
    if not report["exists"]:
        report["blockers"].append(f"annotation file not found: {path}")
        return report

    try:
        points = import_sdd_scenarios.load_sdd_points(path, label=label)
    except (ValueError, OSError) as exc:
        # Parse/format failures are a curation blocker, not a crash: fail closed with the reason.
        report["blockers"].append(f"annotation parse failed: {exc}")
        return report

    report["usable_label_points"] = len(points)
    track_lengths: dict[str, int] = {}
    for point in points:
        track_lengths[point.track_id] = track_lengths.get(point.track_id, 0) + 1
    usable_tracks = [tid for tid, count in track_lengths.items() if count >= min_track_points]
    report["usable_track_count"] = len(usable_tracks)
    report["selection_satisfiable"] = len(usable_tracks) >= 1
    if not report["selection_satisfiable"]:
        report["blockers"].append(
            f"no track has >= {min_track_points} usable '{label}' points after lost-filtering "
            f"(found {len(usable_tracks)} qualifying tracks)"
        )
    return report


def classify_curation_readiness(
    staging_gate: dict[str, Any],
    annotation_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify curation readiness from the canonical staging gate and an optional probe.

    Args:
        staging_gate: Output of ``manage_external_data.resolve_sdd_scenario_prior_mode``.
        annotation_probe: Optional output of :func:`probe_annotation_file`.

    Returns:
        dict: A fail-closed readiness report. ``benchmark_promotion_allowed`` is True only when the
        staging gate reports ``dataset_backed``.
    """
    dataset_backed = bool(staging_gate.get("dataset_backed", False))
    staging_mode = staging_gate.get("mode", manage_external_data.SDD_MODE_PROXY)

    blockers: list[str] = []
    if not dataset_backed:
        blockers.append(
            "SDD is not staged/validated as dataset-backed; curation output must not be promoted "
            "as benchmark evidence (forced to proxy_schema_smoke)."
        )

    probe_satisfiable = True
    if annotation_probe is not None:
        probe_satisfiable = bool(annotation_probe.get("selection_satisfiable", False))
        blockers.extend(annotation_probe.get("blockers", []))

    # Evidence status: dataset-backed + probe-clean is the only benchmark-candidate path.
    if dataset_backed:
        evidence_status = EVIDENCE_BENCHMARK_CANDIDATE if probe_satisfiable else EVIDENCE_BLOCKED
    elif staging_mode == manage_external_data.SDD_MODE_PROXY:
        evidence_status = EVIDENCE_PROXY
    else:
        evidence_status = EVIDENCE_BLOCKED

    benchmark_promotion_allowed = dataset_backed and probe_satisfiable
    # Curation may *run* (e.g. on a fixture for schema-smoke) whenever the probe selection rule is
    # satisfiable; promotion to benchmark evidence is the separately gated step.
    curation_runnable = annotation_probe is None or probe_satisfiable

    if not dataset_backed:
        output_classification = OUTPUT_BLOCKED if annotation_probe is None else OUTPUT_PROXY_ONLY
    elif benchmark_promotion_allowed:
        output_classification = OUTPUT_BENCHMARK_READY_CANDIDATE
    else:
        output_classification = OUTPUT_BLOCKED

    if benchmark_promotion_allowed:
        action = (
            "SDD is staged and validated and the candidate annotation satisfies the curation "
            "selection rule. You may run scripts/tools/import_sdd_scenarios.py and decide "
            "benchmark_ready vs exploratory_only after the smoke run."
        )
    elif dataset_backed:
        action = (
            "SDD is staged/validated but the candidate annotation does not satisfy the curation "
            "selection rule; resolve the probe blockers before curating."
        )
    else:
        action = (
            "SDD is not staged as dataset-backed. Stage licensed annotations via "
            "scripts/tools/manage_external_data.py (issue #1497) before curating. Any importer run "
            "now is proxy_schema_smoke only and must not be promoted as benchmark evidence."
        )

    report: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "issue": ISSUE,
        "staging_mode": staging_mode,
        "dataset_backed": dataset_backed,
        "availability": staging_gate.get("availability"),
        "staging_reason": staging_gate.get("reason"),
        "staging_dir": staging_gate.get("staging_dir"),
        "evidence_status": evidence_status,
        "benchmark_promotion_allowed": benchmark_promotion_allowed,
        "curation_runnable": curation_runnable,
        "output_classification": output_classification,
        "annotation_probe": annotation_probe,
        "blockers": blockers,
        "action": action,
    }
    return report


def run_preflight(
    *,
    manifest_path: Path | None,
    annotation: Path | None,
    label: str,
    min_track_points: int,
    max_pedestrians: int,
) -> dict[str, Any]:
    """Resolve the staging gate, optionally probe an annotation, and classify readiness."""
    staging_gate = manage_external_data.resolve_sdd_scenario_prior_mode(manifest_path=manifest_path)
    probe = None
    if annotation is not None:
        probe = probe_annotation_file(
            annotation,
            label=label,
            min_track_points=min_track_points,
            max_pedestrians=max_pedestrians,
        )
    return classify_curation_readiness(staging_gate, probe)


def _format_human(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of the readiness report."""
    lines = [
        f"SDD curation preflight (issue #{report['issue']})",
        f"  staging mode        : {report['staging_mode']}",
        f"  dataset_backed      : {report['dataset_backed']}",
        f"  evidence status     : {report['evidence_status']}",
        f"  benchmark promotion : {report['benchmark_promotion_allowed']}",
        f"  output class        : {report['output_classification']}",
    ]
    probe = report.get("annotation_probe")
    if probe is not None:
        lines.append(
            "  annotation probe    : "
            f"{probe['usable_track_count']} usable track(s), "
            f"{probe['usable_label_points']} '{probe['label']}' point(s), "
            f"satisfiable={probe['selection_satisfiable']}"
        )
    if report["blockers"]:
        lines.append("  blockers:")
        lines.extend(f"    - {item}" for item in report["blockers"])
    lines.append(f"  action: {report['action']}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the SDD staging manifest path (defaults to the canonical asset manifest).",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Optional candidate SDD annotation file to probe against the curation selection rule.",
    )
    parser.add_argument("--label", default="Pedestrian", help="SDD label to probe for.")
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--max-pedestrians", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument(
        "--require-benchmark-ready",
        action="store_true",
        help="Exit non-zero unless curation output may be promoted as benchmark evidence.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the curation readiness preflight and report.

    Returns:
        int: 0 by default (report-only). With ``--require-benchmark-ready``, returns 3 (fail closed)
        unless benchmark promotion is allowed.
    """
    args = parse_args(argv)
    if args.min_track_points <= 1:
        raise SystemExit("--min-track-points must be > 1")
    if args.max_pedestrians <= 0:
        raise SystemExit("--max-pedestrians must be > 0")
    report = run_preflight(
        manifest_path=args.manifest,
        annotation=args.annotation,
        label=args.label,
        min_track_points=args.min_track_points,
        max_pedestrians=args.max_pedestrians,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human(report))
    if args.require_benchmark_ready and not report["benchmark_promotion_allowed"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
