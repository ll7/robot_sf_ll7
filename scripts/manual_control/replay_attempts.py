"""Export grouped completed-attempt replay events from manual-control JSONL recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.manual_control.recording import load_manual_jsonl_records
from robot_sf.manual_control.replay import group_records_by_attempt, write_attempt_replay_json


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
        required=True,
        type=Path,
        help="Destination JSON path for grouped replay events.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the completed-attempt replay export command.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    records = load_manual_jsonl_records(args.input)
    replays = group_records_by_attempt(records)
    write_attempt_replay_json(replays, args.output)
    print(f"wrote {len(replays)} manual-control attempt replays to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
