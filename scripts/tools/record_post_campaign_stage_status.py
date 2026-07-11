#!/usr/bin/env python3
"""Record campaign and post-campaign exit codes without conflating them.

This helper is intentionally metadata-only.  A launcher owns its process exit code;
the helper writes the separate campaign and reporting lanes that schedulers and
ledgers need to classify a completed campaign whose downstream report failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot-sf-post-campaign-stage-status.v1"


def _load_campaign_summary(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load a campaign summary and return its machine-readable load status."""
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid"
    if not isinstance(payload, dict):
        return None, "invalid"
    return payload, "loaded"


def build_stage_status(
    *,
    campaign_summary_path: Path,
    campaign_exit_code: int,
    stage_name: str,
    stage_exit_code: int,
) -> dict[str, Any]:
    """Build separate campaign and post-campaign status lanes.

    ``job_exit_code`` deliberately follows the campaign lane.  A reporting or
    analysis failure remains visible in ``post_campaign_stage`` but cannot relabel
    an already completed campaign as a failed scheduler job.
    """
    summary, summary_status = _load_campaign_summary(campaign_summary_path)
    if campaign_exit_code == 0 and summary_status != "loaded":
        raise ValueError(
            "campaign completed successfully (exit 0) but its summary could not be loaded "
            f"({summary_status}) from {campaign_summary_path}"
        )
    warnings = summary.get("warnings", []) if summary is not None else []
    if not isinstance(warnings, list):
        warnings = []
    soft_contract_warning = bool(
        summary is not None and summary.get("soft_contract_warning") is True
    )
    campaign_completed = campaign_exit_code == 0
    stage_completed = stage_exit_code == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": {
            "status": "completed" if campaign_completed else "failed",
            "exit_code": campaign_exit_code,
            "summary_json": str(campaign_summary_path),
            "summary_status": summary_status,
            "soft_contract_warning": soft_contract_warning,
            "warnings": warnings,
        },
        "post_campaign_stage": {
            "name": stage_name,
            "status": "completed" if stage_completed else "report_stage_failed",
            "exit_code": stage_exit_code,
        },
        "job_exit_code": campaign_exit_code,
        "claim_boundary": (
            "Campaign execution and post-campaign reporting are separate status lanes; "
            "a completed campaign is not benchmark evidence until its report is available "
            "and validated."
        ),
    }


def _exit_code(value: str) -> int:
    """Parse a portable process exit code."""
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("exit code must be between 0 and 255")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--campaign-exit-code", type=_exit_code, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--stage-exit-code", type=_exit_code, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the stage-status envelope without adopting either recorded exit code."""
    args = parse_args(argv)
    payload = build_stage_status(
        campaign_summary_path=args.campaign_summary,
        campaign_exit_code=args.campaign_exit_code,
        stage_name=args.stage_name,
        stage_exit_code=args.stage_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
