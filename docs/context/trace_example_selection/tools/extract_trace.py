"""Extract the accepted seed's step traces and check them against their own rows.

    python extract_trace.py --contract <claim>/contract.yaml.txt --runs <raw output dir> \
        --out-dir <claim>/trace

Writes, per arm, ``<arm>_seed<seed>_step_trace.json.gz`` (the row's
algorithm_metadata.simulation_step_trace plus row identity; gzip mtime 0, sorted keys)
and ``trace_check.json``: the trace rerun's match result against the screening row and,
per arm, row-versus-trace consistency facts (step count, minimum pedestrian surface
clearance, close-clearance step count against metrics.near_misses, terminal collision
flags). No motion is described.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exsel
from screen import arm_rows


def gz_bytes(payload: dict) -> bytes:
    """Deterministic gzip of sorted-key JSON.

    Returns:
        Compressed bytes.
    """
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as fh:
        fh.write(raw)
    return buf.getvalue()


def main() -> int:
    """Extract and check.

    Returns:
        0 when every arm's trace is consistent with its row, else 1.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    contract = yaml.safe_load(a.contract.read_text(encoding="utf-8"))
    claim = exsel.load_claim(contract)
    arms = [arm["arm"] for arm in claim["arms"]]
    rerun = json.loads((a.runs / "trace_rerun_result.json").read_text(encoding="utf-8"))
    seed, idx = rerun["seed"], rerun["screening_index"]
    prefix = contract["claim_id"].lower()
    rows = arm_rows(a.runs / f"{prefix}__trace__{idx:02d}__{seed}", arms)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    check = {
        "claim_id": contract["claim_id"],
        "seed": seed,
        "screening_index": idx,
        "trace_rerun": rerun,
        "arms": {},
    }
    ok = True
    for arm in arms:
        line, raw = rows[arm]
        trace = raw["algorithm_metadata"]["simulation_step_trace"]
        rec = exsel.project(raw)
        payload = {
            "claim_id": contract["claim_id"],
            "arm": arm,
            "seed": seed,
            "scenario_id": raw["scenario_id"],
            "git_hash": raw["git_hash"],
            "rerun_row_sha256": exsel.row_sha256(line),
            "simulation_step_trace": trace,
        }
        name = f"{arm}_seed{seed}_step_trace.json.gz"
        (a.out_dir / name).write_bytes(gz_bytes(payload))
        steps = trace["steps"]
        mins = [
            min(p["surface_clearance_m"] for p in s["pedestrians"])
            for s in steps
            if s.get("pedestrians")
        ]
        nmd = raw["metric_parameters"]["threshold_profile"]["near_miss_distance_m"]
        close = sum(1 for m in mins if 0 <= m < nmd)
        trace_min = min(mins) if mins else None
        row_min = rec["min_surface_clearance_m"]
        facts = {
            "file": name,
            "trace_schema": trace.get("schema_version"),
            "trace_dt_s": trace.get("dt"),
            "row_steps": raw["steps"],
            "trace_steps": len(steps),
            "row_min_surface_clearance_m": row_min,
            "trace_min_surface_clearance_m": trace_min,
            "near_miss_distance_m": nmd,
            "row_near_misses": raw["metrics"]["near_misses"],
            "trace_steps_with_clearance_below_near_miss_distance": close,
            "row_outcome": rec["outcome"],
            "row_collision_kind": rec["collision_kind"],
            "trace_last_step_collision": steps[-1].get("collision"),
        }
        consistent = (
            facts["row_steps"] == facts["trace_steps"]
            and (row_min is None) == (trace_min is None)
            and (row_min is None or abs(row_min - trace_min) <= 1e-9)
            and close == facts["row_near_misses"]
        )
        facts["consistent"] = consistent
        ok &= consistent
        check["arms"][arm] = facts
    (a.out_dir / "trace_check.json").write_text(
        json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({arm: check["arms"][arm]["consistent"] for arm in arms}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
