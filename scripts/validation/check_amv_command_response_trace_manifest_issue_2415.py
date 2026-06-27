"""CLI for the AMV command-response trace staging-manifest preflight (issue #2415).

Checks the metadata-only manifest that declares how real AMV command-response
actuation traces would be staged through the ``amv-calibration`` external-data
path (``scripts/tools/manage_external_data.py``) before a command-response
calibration of the synthetic actuation envelope can ingest them. It validates
provenance/license, the declared command/response/timing channels, and the
declared calibration targets (against the synthetic-actuation envelope
vocabulary), and -- when ``--probe-live-staging`` is passed -- reconciles each
trace's declared staging status against a live ``manage_external_data`` presence
probe.

It does NOT ingest any trace bundle, read raw command-response samples, run a
calibration, or make a hardware-calibrated realism claim. A calibration-ingest
is reported as allowed only once at least one trace bundle is staged and
manifest-clean. Per the maintainer decision on #2415 (2026-06-22) no realistic
real-data source is currently available, so the shipped manifest is
``blocked-external-input`` and this preflight fails closed.

Examples:
    uv run python scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py
    uv run python scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py \
        --probe-live-staging --require-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.synthetic_actuation import actuation_variability_fields
from robot_sf.research.amv_command_response_trace_manifest import (
    MANIFEST_STATUS_READY,
    AmvTraceManifestError,
    check_amv_trace_manifest,
    load_amv_trace_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs" / "research" / "amv_command_response_trace_manifest_issue_2415.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to an amv_command_response_trace_manifest.v1 file (JSON or YAML).",
    )
    parser.add_argument(
        "--probe-live-staging",
        action="store_true",
        help=(
            "Reconcile each trace's declared staging status against a live "
            "manage_external_data.check_asset presence probe (no download)."
        ),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless the manifest status is 'ready' (a trace is staged and clean).",
    )
    return parser.parse_args(argv)


def _probe_live_staging_status(asset_ids: set[str]) -> dict[str, str]:
    """Probe live staging presence for the given asset ids (no network access).

    Imported lazily so the manifest check works even if the external-data
    subsystem is unavailable; any per-asset probe failure is recorded as an
    ``error:<reason>`` status rather than aborting the whole check.

    Returns:
        Mapping of asset id to a live staging status string.
    """
    from scripts.tools.manage_external_data import ExternalDataError, check_asset

    statuses: dict[str, str] = {}
    for asset_id in sorted(asset_ids):
        try:
            statuses[asset_id] = str(check_asset(asset_id)["status"])
        except ExternalDataError as exc:
            statuses[asset_id] = f"error:{exc}"
    return statuses


def main(argv: list[str] | None = None) -> int:
    """Run the staging-manifest preflight and print a JSON report.

    Returns:
        Process exit code (0 on success, 2 on manifest error, 1 when
        ``--require-ready`` is set and the manifest is not ready).
    """
    args = _parse_args(argv)
    try:
        manifest = load_amv_trace_manifest(args.manifest)
    except (AmvTraceManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    live_status: dict[str, str] | None = None
    if args.probe_live_staging:
        asset_ids = {
            str(trace["asset_id"])
            for trace in manifest["traces"]
            if trace.get("asset_id") is not None
        }
        live_status = _probe_live_staging_status(asset_ids)

    try:
        report = check_amv_trace_manifest(
            manifest,
            allowed_calibration_targets=set(actuation_variability_fields()),
            live_staging_status=live_status,
            source=args.manifest,
        )
    except (AmvTraceManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.require_ready and report.manifest_status != MANIFEST_STATUS_READY:
        print(
            f"manifest status is {report.manifest_status!r}, expected 'ready'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
