"""Derive one screening campaign config from the frozen 0.0.8 release campaign config.

Only five keys change: seed_policy (one fixed development seed), scenario_candidates
(the claim's frozen cell), planners (the claim's arms, copied verbatim from the release
config) export_publication_bundle (off; no publication), workers (1) and snqi_v2_spec
(off; see below). A trace-capable rerun also sets
record_simulation_step_trace. Everything else, including
the scenario matrix, authored horizon schedule, dt, metric settings and planner configs,
is the release config's own bytes as parsed.

    python make_config.py --release-config <yaml> --cell <id> --arms a b --seed <n> --out <yaml>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RETIRED = range(111, 141)


def main() -> int:
    """Write the derived config.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-config", required=True, type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arms", required=True, nargs="+")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--name", required=True)
    ap.add_argument("--trace", action="store_true", help="trace-capable rerun only")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.seed in RETIRED or not 1001 <= a.seed <= 1200:
        raise SystemExit("refusing: screening seeds must be development seeds 1001-1200")
    cfg = yaml.safe_load(a.release_config.read_text(encoding="utf-8"))
    by_key = {p["key"]: p for p in cfg["planners"]}
    missing = [arm for arm in a.arms if arm not in by_key]
    if missing:
        raise SystemExit(f"arms not in the release config: {missing}")
    cfg["name"] = a.name
    cfg["seed_policy"] = {"mode": "fixed-list", "seeds": [a.seed]}
    cfg["scenario_candidates"] = [a.cell]
    cfg["planners"] = [by_key[arm] for arm in a.arms]
    cfg["export_publication_bundle"] = False
    cfg["workers"] = 1
    # SNQI-v2 scoring is switched off. The release bound its acquired anchors at release
    # time for the full 48-cell schedule; the loader refuses a one-cell schedule and
    # development seeds 1001-1030. SNQI scores no screening measure or class predicate.
    cfg["snqi_v2_spec"] = None
    if a.trace:
        cfg["record_simulation_step_trace"] = True
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
