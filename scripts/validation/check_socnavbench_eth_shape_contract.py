#!/usr/bin/env python3
"""Emit a SocNavBench ETH staged-data shape-contract audit.

This checker is intentionally local-data only: it never downloads, vendors, or
redistributes SocNavBench/S3DIS assets. It composes the public
``socnavbench_eth`` loader so the loader remains the shape-contract owner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.data.external import socnavbench_eth

REPORT_SCHEMA = "robot_sf_socnavbench_eth_shape_contract_audit.v1"
ISSUE = 4279


def _jsonable_path(path: Path) -> str:
    """Return expanded absolute path for stable local audit output."""
    return str(path.expanduser().resolve())


def build_report(root: Path | str | None = None) -> tuple[dict[str, Any], int]:
    """Build shape-contract report and matching process exit code.

    Returns exit code ``0`` only when staged assets pass the real loader
    contract. Missing or malformed local data exits ``2`` so automation can
    distinguish "not ready" from a successful staged-data audit.
    """
    layout = socnavbench_eth.expected_layout(root)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "issue": ISSUE,
        "asset_id": socnavbench_eth.ASSET_ID,
        "ok": False,
        "status": "missing",
        "claim_boundary": (
            "Local layout and cheap traversible shape only; not dataset-content "
            "correctness, benchmark readiness, or paper/dissertation evidence."
        ),
        "no_download_performed": True,
        "acquisition_doc": socnavbench_eth.ACQUISITION_DOC,
        "root": _jsonable_path(layout.root),
        "required_paths": {
            "mesh_dir": {
                "path": _jsonable_path(layout.mesh_dir),
                "exists": layout.mesh_dir.is_dir(),
                "kind": "directory",
            },
            "traversible_pickle": {
                "path": _jsonable_path(layout.traversible_pickle),
                "exists": layout.traversible_pickle.is_file(),
                "kind": "file",
            },
        },
    }

    try:
        contract = socnavbench_eth.load_shape_contract(root)
    except socnavbench_eth.SocNavBenchEthDataError as exc:
        report["error"] = str(exc)
        if any(path["exists"] for path in report["required_paths"].values()):
            report["status"] = "invalid"
        return report, 2

    report.update(
        {
            "ok": True,
            "status": "passed",
            "shape_contract": {
                "resolution": contract.resolution,
                "traversible_shape": list(contract.traversible_shape),
                "traversible_dtype": contract.traversible_dtype,
            },
        }
    )
    return report, 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Optional SocNavBench root. Defaults to the external-data registry "
            "path for socnavbench-s3dis-eth."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the audit JSON. Stdout is always written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit CLI."""
    args = _parse_args(argv)
    report, exit_code = build_report(args.root)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
