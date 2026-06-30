#!/usr/bin/env python3
"""Render deterministic TTC near-miss diagnostic decision packets.

The command is intentionally read-only and fixture-backed. It converts the
issue #3700 opt-in TTC diagnostic surface into review packets for issue #3808
without changing canonical near-miss metrics or promoting benchmark claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.near_miss_ttc import (
    DIAGNOSTIC_TTC_THRESHOLD_S,
    NearMissTtcDecisionPacket,
    build_ttc_near_miss_decision_packet,
    render_ttc_near_miss_decision_packet_markdown,
)

FixtureBuilder = Callable[[], EpisodeData]


def _make_episode(
    robot_pos: np.ndarray,
    peds_pos: np.ndarray,
    *,
    dt: float = 0.1,
    robot_vel: np.ndarray | None = None,
) -> EpisodeData:
    """Build a minimal synthetic episode for deterministic packet fixtures."""
    robot_pos = np.asarray(robot_pos, dtype=float)
    peds_pos = np.asarray(peds_pos, dtype=float)
    if robot_vel is None:
        if robot_pos.ndim == 2 and robot_pos.shape[0] >= 2:
            diffs = np.diff(robot_pos, axis=0) / dt
            robot_vel = np.vstack([diffs[:1], diffs])
        else:
            robot_vel = np.zeros_like(robot_pos)

    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=np.asarray(robot_vel, dtype=float),
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=np.zeros_like(peds_pos),
        goal=np.array([10.0, 0.0]),
        dt=dt,
    )


def _closing_fixture() -> EpisodeData:
    """Fast converging robot/pedestrian pair with TTC diagnostic signal."""
    n_steps = 6
    robot_pos = np.zeros((n_steps, 2))
    robot_pos[:, 0] = np.arange(n_steps) * 0.5
    peds_pos = np.zeros((n_steps, 1, 2))
    peds_pos[:, 0, 0] = 6.0 - np.arange(n_steps) * 0.5
    return _make_episode(robot_pos, peds_pos)


def _opening_fixture() -> EpisodeData:
    """Diverging robot/pedestrian pair with no approaching TTC pair."""
    n_steps = 6
    robot_pos = np.zeros((n_steps, 2))
    robot_pos[:, 0] = -np.arange(n_steps) * 0.2
    peds_pos = np.zeros((n_steps, 1, 2))
    peds_pos[:, 0, 0] = 3.0 + np.arange(n_steps) * 0.2
    return _make_episode(robot_pos, peds_pos)


def _missing_timing_fixture() -> EpisodeData:
    """Valid trajectory arrays with invalid timing, which must fail closed."""
    data = _closing_fixture()
    data.dt = float("nan")
    return data


def _unsupported_trajectory_fixture() -> EpisodeData:
    """Malformed pedestrian trajectory shape, which must fail closed."""
    n_steps = 6
    robot_pos = np.zeros((n_steps, 2))
    return _make_episode(robot_pos, np.zeros((n_steps, 2)))


FIXTURES: dict[str, FixtureBuilder] = {
    "closing": _closing_fixture,
    "opening": _opening_fixture,
    "missing-timing": _missing_timing_fixture,
    "unsupported-trajectory": _unsupported_trajectory_fixture,
}


def build_packets(
    *,
    fixture: str = "all",
    threshold_s: float = DIAGNOSTIC_TTC_THRESHOLD_S,
) -> dict[str, NearMissTtcDecisionPacket]:
    """Build packet(s) for one fixture or every deterministic fixture."""
    names = tuple(FIXTURES) if fixture == "all" else (fixture,)
    return {
        name: build_ttc_near_miss_decision_packet(FIXTURES[name](), t_thr=threshold_s)
        for name in names
    }


def render_packets_markdown(packets: dict[str, NearMissTtcDecisionPacket]) -> str:
    """Render all selected packets as one Markdown decision document."""
    sections: list[str] = []
    for name, packet in packets.items():
        sections.extend(
            [
                f"## Fixture: `{name}`",
                "",
                render_ttc_near_miss_decision_packet_markdown(packet),
            ]
        )
    return "\n\n".join(sections) + "\n"


def render_packets_json(packets: dict[str, NearMissTtcDecisionPacket]) -> str:
    """Render all selected packets as strict JSON."""
    payload = {
        "issue": 3808,
        "claim_boundary": (
            "diagnostic decision packet only; no canonical metric replacement, "
            "benchmark campaign, or paper-facing claim"
        ),
        "fixtures": {name: packet.to_dict() for name, packet in packets.items()},
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_packet_output(text: str, output_path: Path | None) -> None:
    """Write packet text to ``output_path`` or stdout for dry-run inspection."""
    if output_path is None:
        sys.stdout.write(text)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=("all", *FIXTURES.keys()),
        default="all",
        help="Synthetic fixture to render; default renders every issue #3808 fixture.",
    )
    parser.add_argument(
        "--threshold-s",
        type=float,
        default=DIAGNOSTIC_TTC_THRESHOLD_S,
        help="Diagnostic TTC threshold to inspect. This is not a calibrated benchmark value.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for the dry-run packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path for the rendered packet. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    packets = build_packets(fixture=args.fixture, threshold_s=args.threshold_s)
    if args.format == "json":
        rendered = render_packets_json(packets)
    else:
        rendered = render_packets_markdown(packets)
    write_packet_output(rendered, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
