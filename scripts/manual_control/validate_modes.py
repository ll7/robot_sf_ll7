"""Validate manual-control mode selection from CLI/config strings."""

from __future__ import annotations

import argparse
import json

from robot_sf.manual_control.config import ManualControlRuntimeConfig
from robot_sf.manual_control.modes import ManualControlMode, ManualViewMode


def build_parser() -> argparse.ArgumentParser:
    """Build the manual-control mode validation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-mode",
        default=ManualControlMode.KEYBOARD_HOLD.value,
        choices=[mode.value for mode in ManualControlMode],
        help="Manual-control steering mode.",
    )
    parser.add_argument(
        "--view-mode",
        default=ManualViewMode.FIXED_MAP.value,
        choices=[mode.value for mode in ManualViewMode],
        help="Manual-control view/camera mode.",
    )
    parser.add_argument(
        "--robot-action-space",
        default="differential_drive",
        help="Robot action-space identifier used for fail-closed mapper validation.",
    )
    return parser


def main() -> int:
    """Validate the selected manual-control modes and print metadata JSON."""
    args = build_parser().parse_args()
    config = ManualControlRuntimeConfig.from_strings(
        control_mode=args.control_mode,
        view_mode=args.view_mode,
        robot_action_space=args.robot_action_space,
    )
    print(json.dumps(config.to_json_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
