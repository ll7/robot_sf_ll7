#!/usr/bin/env python3
"""Freeze the issue #5355 factorial readiness receipt (CPU-only, no-submit).

Per the issue #5355 audit disposition, after the #5776 hierarchical paired-release
input gate lands on a green ``main`` this script freezes the factorial readiness
state: it runs the fail-closed campaign-readiness gate and the #5776 input-gate
evaluation, then writes a single deterministic receipt JSON into the evidence
directory. The receipt records achieved progress (input contract delivered) without
fabricating unavailable analysis data or authorizing any GPU/Slurm submission.

The receipt is the cheap-lane-achievable artifact; the authorized factorial
campaign RUN (compute) remains downstream work and is never performed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    BLOCKED_MISSING_SUCCESSOR_ROWS,
    evaluate_hierarchical_paired_release_inputs,
    load_hierarchical_paired_release_input_manifest,
)
from robot_sf.benchmark.prediction_mpc_factorial_preregistration import (
    DEFAULT_HIERARCHICAL_INPUT_MANIFEST_RELATIVE_PATH,
    assess_campaign_readiness,
)
from robot_sf.evidence.writers import write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_RELATIVE_PATH = "configs/research/prediction_mpc_factorial_v1.yaml"
DEFAULT_CONFIG = REPO_ROOT / DEFAULT_CONFIG_RELATIVE_PATH
RECEIPT_RELATIVE_PATH = (
    "docs/context/evidence/issue_5355_prediction_mpc_factorial_preregistration"
    "/issue_5355_factorial_readiness_receipt.json"
)


def _resolve_from_root(repo_root: Path, path: str | Path) -> Path:
    """Resolve relative overrides against the explicit checkout root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _portable_path(path: Path, *, repo_root: Path) -> str:
    """Return a repository-relative path when the input belongs to this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def freeze_receipt(
    repo_root: Path,
    *,
    config_path: Path | None = None,
    input_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Compute and return the frozen readiness receipt payload."""

    root = Path(repo_root).resolve()
    cfg = _resolve_from_root(root, config_path or DEFAULT_CONFIG_RELATIVE_PATH)
    input_manifest = _resolve_from_root(
        root, input_manifest_path or DEFAULT_HIERARCHICAL_INPUT_MANIFEST_RELATIVE_PATH
    )

    readiness = assess_campaign_readiness(cfg, registry_path=None)
    input_gate: dict[str, object] = {"status": "unknown", "present": False}
    if Path(input_manifest).is_file():
        manifest_display = _portable_path(Path(input_manifest), repo_root=root)
        try:
            manifest = load_hierarchical_paired_release_input_manifest(input_manifest)
            evaluation = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=root)
        except (ValueError, OSError) as exc:
            input_gate = {
                "present": True,
                "manifest": manifest_display,
                "status": "blocked_invalid_input_manifest",
                "error": str(exc),
            }
        else:
            input_gate = {
                "present": True,
                "manifest": manifest_display,
                "status": str(evaluation.get("status")),
                "blocking_prerequisites": evaluation.get("blocking_prerequisites", []),
                "claim_gate": evaluation.get("claim_gate", {}),
            }

    return {
        "schema_version": "robot_sf.issue_5355_factorial_readiness_receipt.v1",
        "issue": 5355,
        "frozen_by": "scripts/validation/freeze_issue_5355_factorial_readiness_receipt.py",
        "claim_boundary": (
            "CPU readiness receipt only; no campaign has run, no benchmark/paper/release "
            "claim is supported, and no GPU/Slurm submission is authorized."
        ),
        "ready": bool(readiness.get("ready", False)),
        "readiness_blockers": list(readiness.get("blockers", [])),
        "readiness_criteria": {
            name: criterion.get("ready", False)
            for name, criterion in readiness.get("criteria", {}).items()
        },
        "hierarchical_input_gate": input_gate,
        "input_gate_consistent_with_audit": (
            input_gate.get("status") == BLOCKED_MISSING_SUCCESSOR_ROWS
        ),
        "successor_slice_required": {
            "issue": 4364,
            "reason": "typed-ledger successor-release rows required before #5351 analysis runs",
        },
        "campaign_run_status": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    """Write the frozen readiness receipt and print a summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repo root.")
    parser.add_argument("--config", type=Path, default=None, help="Override factorial config.")
    parser.add_argument(
        "--input-manifest", type=Path, default=None, help="Override #5776 input manifest."
    )
    parser.add_argument("--out", type=Path, default=None, help="Override receipt output path.")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    receipt = freeze_receipt(root, config_path=args.config, input_manifest_path=args.input_manifest)

    out = args.out or (root / RECEIPT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, receipt)

    print(f"issue #5355 factorial readiness receipt -> {out}")
    print(f"  ready: {receipt['ready']}")
    print(f"  input_gate_status: {receipt['hierarchical_input_gate'].get('status')}")
    print(f"  campaign_run_status: {receipt['campaign_run_status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
