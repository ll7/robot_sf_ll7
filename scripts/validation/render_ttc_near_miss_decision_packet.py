#!/usr/bin/env python3
"""Render deterministic TTC near-miss diagnostic packets for issue #3808.

This script is read-only and fixture-backed; it only supports local evidence
construction and does not execute benchmarks or claim improvements.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import sys

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
    """Build a minimal deterministic synthetic episode."""
    robot_pos = np.asarray(robot_pos, dtype=float)
    peds_pos = np.asarray(peds_pos, dtype=float)

    if robot_vel is None:
        if robot_pos.ndim == 2 and robot_pos.shape[0] >= 2:
            deltas = np.diff(robot_pos, axis=0) / dt
            robot_vel = np.vstack([deltas[:1], deltas])
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
    """Approaching pair should produce TTC near-miss evidence (`ok`).

    The robot approaches a static pedestrian and closes under the default TTC
    threshold.
    """
    robot_pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    peds_pos = np.array(
        [[[4.0, 0.0]], [[4.0, 0.0]], [[4.0, 0.0]], [[4.0, 0.0]]]
    )
    return _make_episode(robot_pos, peds_pos, dt=0.1)


def _opening_fixture() -> EpisodeData:
    """Opening separation should result in no-approaching-pairs status."""
    robot_pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    peds_pos = np.array(
        [[[-4.0, 0.0]], [[-3.0, 0.0]], [[-2.0, 0.0]], [[-1.0, 0.0]]]
    )
    return _make_episode(robot_pos, peds_pos, dt=0.1)


def _missing_timing_fixture() -> EpisodeData:
    """Malformed `dt` should produce unsupported-inputs."""
    return _make_episode(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[[4.0, 0.0]], [[4.5, 0.0]]]),
        dt=float("nan"),
    )


def _unsupported_trajectory_fixture() -> EpisodeData:
    """Malformed pedestrian trajectory shape should fail closed."""
    n_steps = 4
    return _make_episode(
        robot_pos=np.zeros((n_steps, 2)),
        # Missing pedestrian object axis, intentionally 2D instead of (T, K, 2).
        peds_pos=np.zeros((n_steps, 2)),
    )


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
    """Build one or all deterministic near-miss packets from fixtures."""
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
    """Render selected packets as strict JSON."""
    payload = {
        "issue": 3808,
        "claim_boundary": (
            "diagnostic decision packet only; no canonical metric replacement, "
            "no benchmark campaign, and no paper-facing claim."
        ),
        "fixtures": {name: packet.to_dict() for name, packet in packets.items()},
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=("all", *FIXTURES.keys()),
        default="all",
        help="Synthetic fixture to render. Default: all fixtures.",
    )
    parser.add_argument(
        "--threshold-s",
        type=float,
        default=DIAGNOSTIC_TTC_THRESHOLD_S,
        help="TTC diagnostic threshold used for this packet.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Dry-run packet output format.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    packets = build_packets(fixture=args.fixture, threshold_s=args.threshold_s)
    if args.format == "json":
        sys.stdout.write(render_packets_json(packets))
    else:
        sys.stdout.write(render_packets_markdown(packets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
