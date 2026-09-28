#!/usr/bin/env python3
"""Extract simulator operational quantities for external cost models — Issue #9350.

Reads one episode-record JSON file with trajectory arrays and emits the typed
simulator observables plus the field-availability matrix. This is a diagnostic
postprocessor only: it never prices, ranks planners, or claims economic
validity. A numeric cost total additionally requires service denominators and
external cost parameters (see ``robot_sf.benchmark.operational_quantities``);
trajectory-only input stays blocked by design.

Input JSON schema (arrays as nested lists)::

    {
        "robot_pos": [[x, y], ...],
        "robot_vel": [[vx, vy], ...],
        "dt": 0.1,
        "reached_goal_step": 42,
        "world_id": "w",
        "episode_id": "e",
        "agent_id": "a",
        "result_status": "goal_reached",
        "queue_time_s": null,
    }

Output: JSON record (``operational_quantities.v1``) written to stdout or
``--output``. Raw traces stay in ``output/``; only this compact record is
promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.operational_quantities import (
    extract_operational_observables,
    to_record,
)


def _load_episode(path: Path) -> EpisodeData:
    payload = json.loads(path.read_text())
    steps = len(payload["robot_pos"])
    zeros_2 = np.zeros((steps, 2), dtype=float)
    peds = np.zeros((steps, 0, 2), dtype=float)
    return EpisodeData(
        robot_pos=np.asarray(payload["robot_pos"], dtype=float),
        robot_vel=np.asarray(payload["robot_vel"], dtype=float),
        robot_acc=np.asarray(payload.get("robot_acc", zeros_2.tolist()), dtype=float),
        peds_pos=peds,
        ped_forces=peds.copy(),
        goal=np.asarray(payload.get("goal", [0.0, 0.0]), dtype=float).reshape(2),
        dt=float(payload["dt"]),
        reached_goal_step=payload.get("reached_goal_step"),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the operational-quantities postprocessor CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_json", type=Path, help="Episode-record JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--world-id", default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--result-status", default="unknown")
    parser.add_argument("--queue-time-s", type=float, default=None)
    args = parser.parse_args(argv)
    data = _load_episode(args.episode_json)
    observables = extract_operational_observables(
        data,
        world_id=args.world_id,
        episode_id=args.episode_id,
        agent_id=args.agent_id,
        result_status=args.result_status,
        queue_time_s=args.queue_time_s,
    )
    record = json.dumps(to_record(observables), indent=2)
    if args.output is None:
        print(record)
    else:
        args.output.write_text(record + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
