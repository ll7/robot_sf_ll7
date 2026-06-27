"""CLI for the dataset-backed scenario-prior staging-contract checker (issue #3161).

Checks the metadata-only staging contract that declares how dataset-backed
scenario priors (SDD / SocNavBench ETH / AMV) would be staged and compared
against the authored + trace-derived baseline from #2919. It validates
provenance/license, the declared distribution fields (against the #2919
comparison parameter vocabulary), and the explicit external-data blockers, and
-- when ``--probe-live-staging`` is passed -- reconciles each dataset's declared
staging status against a live ``manage_external_data`` presence probe.

It does NOT ingest any dataset, read raw trajectories, run the comparison, or
make a realism claim. A dataset-backed comparison is reported as allowed only
once at least one dataset is staged and contract-clean.

Examples:
    uv run python scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py
    uv run python scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py \
        --probe-live-staging --require-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.research.scenario_prior_staging_contract import (
    CONTRACT_STATUS_READY,
    ScenarioPriorStagingContractError,
    check_scenario_prior_staging_contract,
    load_scenario_prior_staging_contract,
)
from scripts.analysis.compare_scenario_priors_issue_2919 import PARAMETER_GROUPS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "configs" / "research" / "scenario_prior_staging_contract_issue_3161.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Path to a scenario_prior_staging_contract.v1 file (JSON or YAML).",
    )
    parser.add_argument(
        "--probe-live-staging",
        action="store_true",
        help=(
            "Reconcile each dataset's declared staging status against a live "
            "manage_external_data.check_asset presence probe (no download)."
        ),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless the contract status is 'ready' (a dataset is staged and clean).",
    )
    return parser.parse_args(argv)


def _probe_live_staging_status(asset_ids: set[str]) -> dict[str, str]:
    """Probe live staging presence for the given asset ids (no network access).

    Imported lazily so the contract check works even if the external-data
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
    """Run the staging-contract check and print a JSON report.

    Returns:
        Process exit code (0 on success, 2 on contract error, 1 when
        ``--require-ready`` is set and the contract is not ready).
    """
    args = _parse_args(argv)
    try:
        contract = load_scenario_prior_staging_contract(args.contract)
    except (ScenarioPriorStagingContractError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    live_status: dict[str, str] | None = None
    if args.probe_live_staging:
        asset_ids = {
            str(dataset["asset_id"])
            for dataset in contract["datasets"]
            if dataset.get("asset_id") is not None
        }
        live_status = _probe_live_staging_status(asset_ids)

    try:
        report = check_scenario_prior_staging_contract(
            contract,
            allowed_distribution_groups=set(PARAMETER_GROUPS),
            live_staging_status=live_status,
            source=args.contract,
        )
    except (ScenarioPriorStagingContractError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.require_ready and report.contract_status != CONTRACT_STATUS_READY:
        print(
            f"contract status is {report.contract_status!r}, expected 'ready'",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
