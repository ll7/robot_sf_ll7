#!/usr/bin/env python3
"""Run a frozen #5416 comparator through the geometry-aware native-command protocol.

The process receives live robot, goal, pedestrian, and static-map geometry over
stdin.  It reconstructs the same occupancy representation used by the existing
planner implementation, then calls that implementation directly.  It does not
load a scenario or use an adapter/fallback path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.planner.dwa import DWAPlannerAdapter, build_dwa_config
from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from robot_sf.planner.nmpc_social import (
    NMPCSocialPlannerAdapter,
    build_nmpc_social_config,
)
from robot_sf.planner.teb_commitment import (
    TEBCommitmentPlannerAdapter,
    build_teb_commitment_config,
)
from scripts.benchmark.sipp_native_command import (
    RequestError,
    _geometry_consumption,
    _load_config,
    _occupancy_observation,
)

_FROZEN_COMPARATORS = frozenset({"hybrid_rule_v0_minimal", "teb", "nmpc_social", "dwa"})


def _build_planner(planner_id: str, config: dict[str, Any]) -> Any:
    """Build exactly the existing implementation registered for ``planner_id``."""
    if planner_id not in _FROZEN_COMPARATORS:
        supported = ", ".join(sorted(_FROZEN_COMPARATORS))
        raise RequestError(f"planner_id must be one of: {supported}")
    if config.get("planner_variant") != planner_id:
        raise RequestError("config planner_variant does not match the declared planner_id")
    if planner_id == "hybrid_rule_v0_minimal":
        return HybridRuleLocalPlannerAdapter(build_hybrid_rule_local_planner_config(config))
    if planner_id == "teb":
        return TEBCommitmentPlannerAdapter(build_teb_commitment_config(config))
    if planner_id == "nmpc_social":
        return NMPCSocialPlannerAdapter(build_nmpc_social_config(config))
    return DWAPlannerAdapter(build_dwa_config(config))


def run(*, planner_id: str, config_path: Path) -> int:
    """Serve persistent native-command requests for one frozen comparator."""
    config = _load_config(config_path)
    planner = _build_planner(planner_id, config)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RequestError("request must be a JSON object")
            observation = _occupancy_observation(request, config)
            sim = request.get("sim")
            if isinstance(sim, dict):
                observation["sim"] = dict(sim)
            elif request.get("dt") is not None:
                observation["dt"] = request["dt"]
            command = planner.plan(observation)
            print(
                json.dumps(
                    {
                        "linear_velocity": float(command[0]),
                        "angular_velocity": float(command[1]),
                        "geometry_consumption": _geometry_consumption(observation),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(json.dumps({"error": str(exc), "status": "invalid_request"}), file=sys.stderr)
            return 2
    return 0


def main() -> int:
    """Parse the fixed comparator identity and tracked config path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-id", required=True, choices=sorted(_FROZEN_COMPARATORS))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return run(planner_id=args.planner_id, config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
