"""Screen one claim: run its arms on the committed screening order, one seed at a time.

Stops at the first accepted seed or when the budget is spent. A seed whose run exits
non-zero halts the claim (an unrun seed is not a rejected seed). Resumable: a seed whose
receipt shows rc 0 is evaluated from its stored rows instead of being rerun.

    python screen.py --contract <claim>/contract.yaml --wt <execution worktree> \
        --runs <raw output dir> [--trace-rerun]

With --trace-rerun it reruns the accepted seed for every arm with step traces and
compares the rerun rows with the screening rows on the contract's comparison fields.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exsel

RUN_ONE = Path(__file__).resolve().parent / "run_one.sh"


def arm_rows(run_dir: Path, arms: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Read the single episode row of every arm from one campaign output.

    Returns:
        Mapping arm to (serialized line, parsed row).
    """
    out = {}
    for arm in arms:
        files = sorted(
            glob.glob(str(run_dir / "out" / "*" / "runs" / f"{arm}__*" / "episodes.jsonl"))
        )
        if len(files) != 1:
            raise RuntimeError(
                f"{run_dir}: expected one episodes.jsonl for {arm}, found {len(files)}"
            )
        rows = exsel.read_rows(files[0])
        if len(rows) != 1:
            raise RuntimeError(f"{files[0]}: expected one row, found {len(rows)}")
        out[arm] = rows[0]
    return out


def run(
    wt: Path, out_root: Path, cid: str, cell: str, seed: int, arms: list[str], trace: bool
) -> int:
    """Run one campaign for (cell, seed, arms) unless a successful receipt exists.

    Returns:
        Runner exit code.
    """
    receipt = out_root / cid / "receipt.json"
    if receipt.exists() and json.loads(receipt.read_text())["rc"] == 0:
        return 0
    cmd = [str(RUN_ONE), str(wt), str(out_root), cid, cell, str(seed)]
    if trace:
        cmd.append("--trace")
    cmd.extend(arms)
    return subprocess.call(cmd)


def main() -> int:  # noqa: C901, PLR0915
    """Screen or trace-rerun one claim.

    Returns:
        0 on a terminal screening state, 1 on a run failure or rerun mismatch.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, type=Path)
    ap.add_argument("--wt", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--trace-rerun", action="store_true")
    a = ap.parse_args()
    contract = yaml.safe_load(a.contract.read_text(encoding="utf-8"))
    claim = exsel.load_claim(contract)
    arms = [arm["arm"] for arm in claim["arms"]]
    cell = claim["frozen"]["cell"]
    scr = claim["screening"]
    order = exsel.screening_order(scr["seed_band"], scr["selection_rng_seed"], scr["budget"])
    if order != [int(s) for s in contract["screening"]["order"]]:
        raise SystemExit("screening order differs from the contract")
    prefix = contract["claim_id"].lower()
    a.runs.mkdir(parents=True, exist_ok=True)
    state = a.runs / "screening_state.jsonl"

    if a.trace_rerun:
        accepted = None
        for line in state.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("accepted"):
                accepted = rec
        if accepted is None:
            print("no accepted seed; no trace rerun")
            return 0
        seed, idx = accepted["seed"], accepted["screening_index"]
        cid = f"{prefix}__trace__{idx:02d}__{seed}"
        rc = run(a.wt, a.runs, cid, cell, seed, arms, trace=True)
        if rc != 0:
            print(f"TRACE RUN FAILED rc={rc} seed={seed}")
            return 1
        screen = arm_rows(a.runs / f"{prefix}__screen__{idx:02d}__{seed}", arms)
        rerun = arm_rows(a.runs / cid, arms)
        fields = contract["trace_rerun"]["match_fields"]
        result = {"seed": seed, "screening_index": idx, "arms": {}}
        ok = True
        for arm in arms:
            p0, p1 = exsel.project(screen[arm][1]), exsel.project(rerun[arm][1])
            diff = {f: [p0[f], p1[f]] for f in fields if p0[f] != p1[f]}
            ok &= not diff
            result["arms"][arm] = {
                "match": not diff,
                "mismatches": diff,
                "step_trace_recorded": bool(
                    (rerun[arm][1].get("scenario_params") or {}).get("record_simulation_step_trace")
                ),
                "rerun_row_sha256": exsel.row_sha256(rerun[arm][0]),
            }
        result["trace_accepted"] = ok
        (a.runs / "trace_rerun_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if ok else 1

    done = {}
    if state.exists():
        for line in state.read_text().splitlines():
            rec = json.loads(line)
            done[rec["screening_index"]] = rec
    for idx, seed in enumerate(order, start=1):
        if idx in done:
            if done[idx]["accepted"]:
                print(f"ACCEPTED seed {seed} at screening index {idx} of {len(order)} (resumed)")
                return 0
            continue
        cid = f"{prefix}__screen__{idx:02d}__{seed}"
        rc = run(a.wt, a.runs, cid, cell, seed, arms, trace=False)
        if rc != 0:
            print(f"RUN FAILED rc={rc} at screening index {idx} seed {seed}; claim halted")
            return 1
        rows = arm_rows(a.runs / cid, arms)
        ev = exsel.evaluate({arm: rows[arm][1] for arm in arms}, claim)
        rec = {
            "screening_index": idx,
            "seed": seed,
            "campaign_id": cid,
            "accepted": ev["accepted"],
            "arms": {
                arm: {
                    "outcome": ev["per_arm"][arm]["record"]["outcome"],
                    "class": ev["per_arm"][arm]["class"],
                    "bands": ev["per_arm"][arm]["bands"],
                    "mode": ev["per_arm"][arm]["mode"],
                }
                for arm in arms
            },
        }
        with state.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)
        if ev["accepted"]:
            print(f"ACCEPTED seed {seed} at screening index {idx} of {len(order)}")
            return 0
    print(f"NO QUALIFYING EXAMPLE after {len(order)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
