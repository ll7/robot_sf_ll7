#!/usr/bin/env python3
"""Run a chained post-campaign stage and record its exit as a separate lane.

Issue #5244 root-cause candidate: a post-campaign analysis step (for example an
SNQI tool such as ``recompute_snqi_weights.py``) chained after a completed camera-ready
campaign can surface its own nonzero exit — e.g. ``EXIT_OPTIONAL_DEPS_MISSING`` (5)
when an optional dependency such as matplotlib is missing on the compute node. Under
``set -e``/``set -euo pipefail`` a naive wrapper then propagates that code as the job
exit, orphaning an otherwise complete campaign.

This helper runs the chained stage as a subprocess and records its exit in the
``post_campaign_stage`` lane of the ``robot-sf-post-campaign-stage-status.v1`` envelope
without overwriting the campaign exit. The overall process exit follows the campaign
lane only; a failed reporting/analysis stage stays visible but never relabels a
completed campaign as a failed scheduler job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.tools.record_post_campaign_stage_status import build_stage_status

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _exit_code(value: str) -> int:
    """Parse a portable process exit code."""
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("exit code must be between 0 and 255")
    return parsed


def run_stage_command(command: Sequence[str]) -> tuple[int, str, str]:
    """Run a post-campaign stage command, returning (exit_code, stdout, stderr).

    The stage command is executed via ``subprocess.run`` so a nonzero exit is captured
    rather than propagating to the parent process under ``set -e``. ``FileNotFoundError``
    (the command itself is unavailable) is reported as the dedicated optional-deps-missing
    lane code shared with SNQI tooling.
    """
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
    except FileNotFoundError as exc:
        return 5, "", f"post-campaign stage command not found: {exc}"
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def build_post_campaign_stage_payload(
    *,
    campaign_summary_path: Path,
    campaign_exit_code: int,
    stage_name: str,
    stage_command: Sequence[str],
    output: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run a post-campaign stage and build its status envelope.

    Returns:
        The process exit code and the stage-status payload. The process exit code
        follows the campaign lane (``campaign_exit_code``); a failed stage is recorded
        separately and does not remap a completed campaign.
    """
    stage_exit_code, _, _ = run_stage_command(stage_command)
    payload = build_stage_status(
        campaign_summary_path=campaign_summary_path,
        campaign_exit_code=campaign_exit_code,
        stage_name=stage_name,
        stage_exit_code=stage_exit_code,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return campaign_exit_code, payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--campaign-exit-code", type=_exit_code, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument(
        "--stage-command",
        nargs=argparse.REMAINDER,
        required=True,
        help="Post-campaign stage command to run (e.g. a chained SNQI analysis step).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the chained stage and write its status envelope at the dispatch boundary."""
    args = _parse_args(list(argv) if argv is not None else None)
    # ``argparse.REMAINDER`` may keep a leading "--" separator when present.
    stage_command = [token for token in args.stage_command if token != "--"]
    if not stage_command:
        print("error: empty post-campaign stage command", file=sys.stderr)
        return 2
    process_exit_code, payload = build_post_campaign_stage_payload(
        campaign_summary_path=args.campaign_summary,
        campaign_exit_code=args.campaign_exit_code,
        stage_name=args.stage_name,
        stage_command=stage_command,
        output=args.output,
    )
    print(json.dumps(payload, sort_keys=True))
    if payload["post_campaign_stage"]["exit_code"] != 0:
        print(
            f"WARNING: post-campaign stage '{args.stage_name}' failed "
            f"(exit {payload['post_campaign_stage']['exit_code']}); "
            f"campaign exit remains {args.campaign_exit_code}.",
            file=sys.stderr,
        )
    return process_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
