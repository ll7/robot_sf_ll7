#!/usr/bin/env python3
"""Run the issue #3079 Package B budget-matched adversarial sampler comparison.

Orchestrates the CPU-achievable portion of Package B:

1. preflight the committed manifest (fail-closed, no campaign execution);
2. run the 27-cell budget-matched sampler comparison in synthetic mode and write
   the durable report artifact;
3. validate the report through the Package-B report gate;
4. emit the confirmation sidecar (every cell censored, artifact-bound to the report)
   and validate it through the confirmation gate.

The script never submits Slurm jobs or runs benchmark episodes; the synthetic
evaluator makes the 27-cell comparison reproducible on CPU. Confirmed-failure
discovery (replay + independent-seed + mechanism attribution) is intentionally
left as a censored caveate until the artifacts exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_confirmation import (
    build_package_b_confirmation_sidecar,
    validate_package_b_confirmation,
)
from robot_sf.benchmark.adversarial_package_b_preflight import preflight_package_b_manifest
from robot_sf.benchmark.adversarial_package_b_report import validate_package_b_report
from scripts.tools.compare_adversarial_samplers import main as compare_main

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MANIFEST = Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml")


def _refine_manifest_paths(manifest_path: Path, *, repo_root: Path) -> Path:
    """Return an absolute manifest path resolved against the repository root."""
    return (manifest_path if manifest_path.is_absolute() else repo_root / manifest_path).resolve()


def _run_pipeline(manifest_path: Path, *, repo_root: Path) -> dict[str, object]:
    """Execute preflight, synthetic run, report gate, and confirmation sidecar/gate.

    Returns:
        A compact pipeline payload summarizing each stage's outcome.
    """
    manifest_path = _refine_manifest_paths(manifest_path, repo_root=repo_root)
    preflight = preflight_package_b_manifest(manifest_path, repo_root=repo_root)
    if not preflight.ready:
        return {
            "stage": "preflight",
            "ready": False,
            "blockers": list(preflight.blockers),
            "metadata": preflight.metadata,
        }

    output_artifacts = preflight.metadata.get("output_artifacts", {})
    report_json = repo_root / output_artifacts.get(
        "report_json", "output/adversarial/issue_3079_package_b/report.json"
    )

    compare_argv = [
        "--manifest",
        str(manifest_path),
        "--repo-root",
        str(repo_root),
        "--synthetic",
        "--out-json",
        str(report_json),
    ]
    if compare_main(compare_argv) != 0:
        raise RuntimeError("Package-B comparison runner returned a non-zero status")

    report_gate = validate_package_b_report(report_json)

    confirmation_path = report_json.parent / "confirmation.json"
    sidecar = build_package_b_confirmation_sidecar(report_json, confirmation_path=confirmation_path)
    sidecar.write()
    confirmation_gate = validate_package_b_confirmation(
        report_json,
        confirmation_path,
        artifact_root=repo_root,
    )

    return {
        "stage": "complete",
        "preflight_ready": preflight.ready,
        "report_json": report_json.relative_to(repo_root).as_posix()
        if report_json.is_relative_to(repo_root)
        else report_json.as_posix(),
        "confirmation_json": confirmation_path.relative_to(repo_root).as_posix()
        if confirmation_path.is_relative_to(repo_root)
        else confirmation_path.as_posix(),
        "report_gate": report_gate.to_payload(),
        "confirmation_gate": confirmation_gate.to_payload(),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the Package-B run CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Package-B manifest (defaults to the committed issue #3079 manifest).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve manifest-relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the compact pipeline JSON summary.",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Return a non-zero exit code when any gate is not ready (CPU-only checks).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Package-B pipeline and emit a compact summary."""
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    summary = _run_pipeline(args.manifest, repo_root=repo_root)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if not args.fail_closed:
        return 0
    if summary.get("stage") != "complete":
        return 2
    gates = summary.get("report_gate", {}), summary.get("confirmation_gate", {})
    if all(isinstance(gate, dict) and gate.get("ready") for gate in gates):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
