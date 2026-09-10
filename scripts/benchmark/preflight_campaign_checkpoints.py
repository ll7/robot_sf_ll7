"""Pre-``sbatch`` gate that verifies (and optionally stages) campaign arm checkpoints.

Run this before submitting a camera-ready benchmark campaign so a missing or corrupt arm
checkpoint fails in seconds on the submit node instead of ~14h into compute (issue #4613: the S30
campaign jobs 13296 and 13301 both failed identically on a missing PPO ``model_cache`` checkpoint).

Modes:

- default (``--check``): confirm every enabled arm's ``model_id`` / ``model_path`` checkpoint is
  present locally or has a durable remote source to stage from. Network-free. This mode is
  **not** submit-safe by itself: a ``stageable_remote`` arm passes here but the compute node would
  still discover a missing-cache failure if it has not actually been staged. The ``submit_safe``
  boolean in the JSON output reports whether the resolvability is sufficient for ``sbatch``.
- ``--stage``: enforced pre-submit staging -- actually download and checksum-verify each registry
  checkpoint into the durable cache so the compute node loads a validated file. After a successful
  ``--stage`` run with at least one checkpoint reference, ``submit_safe`` is ``true``. The ops
  ``sbatch`` wrapper must run this mode (or the public submit gate
  ``scripts/benchmark/submit_camera_ready_checkpoint_gate.sh``) before requeueing.

Optionally persist the preflight JSON report with ``--report-path`` so the requeue packet records
the per-arm staging status.

Exit codes are distinct so an sbatch wrapper can branch mechanically:

- ``0`` -- all arm checkpoints resolvable (``--stage`` also means staged + verified).
- ``2`` -- the campaign config file is missing or unreadable (cannot be evaluated).
- ``3`` -- one or more arm checkpoints are unresolvable (fail-closed; do not submit), or an
  optional ``expiring_resource`` contract blocks the run per its declared ``admission_policy``.

When the campaign manifest carries the optional expiring-resource contract (#8905), the gate also
runs the deterministic deadline feasibility check before touching checkpoints: a ``too_late`` or
``unknown`` verdict blocks when the contract's ``admission_policy`` is ``block`` (the default).
Manifests without the block keep their previous behavior, and the check never guesses a scheduler
start time. See ``docs/context/expiring_resource_deadlines.md``.

See ``docs/context/issue_4613_camera_ready_checkpoint_provisioning.md`` for the runbook.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight_from_config,
)
from robot_sf.benchmark.checkpoint_staging_receipt import CHECKPOINT_STAGING_RECEIPT_SCHEMA
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.models.registry import DEFAULT_REGISTRY_PATH

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_BLOCKED = 3


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Verify (and optionally stage) camera-ready campaign arm checkpoints "
        "before sbatch (issue #4613).",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a camera-ready campaign config YAML.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help=(
            "Download and checksum-verify each registry checkpoint into the durable cache "
            "(enforced pre-submit staging; produces submit_safe=true for non-empty coverage) "
            "instead of the cheap network-free resolvability check. The sbatch/submit wrapper "
            "must run this mode (or scripts/benchmark/submit_camera_ready_checkpoint_gate.sh) "
            "before requeueing."
        ),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional model-registry path override.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory override for staged downloads.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the preflight summary as JSON on stdout (includes submit_safe).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to persist the preflight summary JSON next to the submission "
        "packet/log root so the requeue record carries per-arm staging status.",
    )
    return parser


def expiring_resource_gate(config_path: Path) -> dict[str, Any] | None:
    """Evaluate the optional expiring-resource contract in *config_path*.

    Returns the deterministic feasibility report when the campaign manifest declares the
    ``expiring_resource`` block, otherwise ``None`` so historical manifests keep their behavior.
    Real manifests evaluate at the current UTC time; fixtures may pin ``as_of`` for reproducibility.
    """
    from scripts.validation.check_expiring_resource_feasibility import evaluate_manifest

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, Mapping) or "expiring_resource" not in payload:
        return None
    # Only an explicit ``as_of`` pins the clock for reproducible fixtures. Real manifests are
    # evaluated at the current UTC time; a stale ``generated_at`` must not make an expired window
    # look feasible.
    as_of = None if payload.get("as_of") else datetime.now(UTC)
    return evaluate_manifest(payload, as_of=as_of).to_dict()


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - gate plus checkpoint failure paths
    """Run the campaign checkpoint preflight CLI.

    Returns:
        int: Process exit code (see module docstring).
    """
    args = build_arg_parser().parse_args(argv)
    checkpoint_preflight_mode = "enforced_staged" if args.stage else "metadata_only"
    if not args.config.is_file():
        print(f"error: campaign config not found: {args.config}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    deadline = expiring_resource_gate(args.config)
    if deadline is not None and deadline["blocking"]:
        blocked = {
            "status": "blocked",
            "reason": "expiring_resource_deadline_gate",
            "expiring_resource": deadline,
        }
        print(
            f"error: expiring-resource deadline gate blocked ({deadline['verdict']}); "
            "do not submit.",
            file=sys.stderr,
        )
        if args.json:
            print(json.dumps(blocked, indent=2))
        if args.report_path is not None:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        return EXIT_BLOCKED
    try:
        summary = check_campaign_arm_checkpoints_preflight_from_config(
            args.config,
            stage=bool(args.stage),
            registry_path=args.registry_path,
            cache_dir=args.cache_dir,
        )
    except CampaignCheckpointPreflightError as exc:
        print(str(exc), file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "mode": checkpoint_preflight_mode,
                        "arms": list(exc.arms),
                    },
                    indent=2,
                )
            )
        if args.report_path is not None:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "mode": checkpoint_preflight_mode,
                        "stage": bool(args.stage),
                        "arms": list(exc.arms),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return EXIT_BLOCKED
    except (FileNotFoundError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: could not evaluate campaign config: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    mode = "staged" if args.stage else "resolvable"
    registry_sha256 = None
    if any(arm.get("kind") == "model_id" for arm in summary.get("arms", [])):
        registry_sha256 = sha256_file(Path(args.registry_path or DEFAULT_REGISTRY_PATH).resolve())
    payload = {
        "schema_version": CHECKPOINT_STAGING_RECEIPT_SCHEMA,
        "status": "ok",
        "mode": checkpoint_preflight_mode,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "campaign_config_path": str(args.config.resolve()),
        "campaign_config_sha256": sha256_file(args.config.resolve()),
        "checkpoint_registry_sha256": registry_sha256,
        **summary,
    }
    if deadline is not None:
        payload["expiring_resource"] = deadline
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        submit_safe_note = "submit_safe=true" if summary.get("submit_safe") else "submit_safe=FALSE"
        print(
            f"campaign checkpoint preflight passed: {summary['resolved']}/{summary['checked']} "
            f"arm checkpoint reference(s) {mode} ({submit_safe_note})."
        )
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.json:
            print(f"report: {args.report_path}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
