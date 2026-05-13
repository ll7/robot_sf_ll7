"""Profile manual-control JSONL recording size and load throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.manual_control.profile import profile_manual_jsonl_recording


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Manual-control session JSONL produced by the manual recorder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination JSON path for the profile summary.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the recording profile command.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    profile = profile_manual_jsonl_recording(args.input)
    payload = profile.to_json_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "profiled "
        f"{profile.record_count} manual-control records across {profile.attempt_count} attempts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
