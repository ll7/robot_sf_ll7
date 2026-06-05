#!/usr/bin/env python3
"""Simple trace activation extractor for issue #2306.
Reads a JSONL trace file and emits required activation fields as YAML-ish JSON.
This is intentionally small and defensive for local analysis runs.

Usage: python scripts/analysis/extract_activation_from_trace.py <trace.jsonl> [--row-id ID]
"""
import sys
import json
from pathlib import Path

def extract_activation(record: dict) -> dict:
    # Best-effort extraction from common trace keys; return None-able fields
    out = {
        "activation_count": None,
        "first_activation_step": None,
        "selected_command_source": None,
        "command_source_changed": None,
        "progress_delta_after_activation": None,
        "trajectory_delta": None,
        "terminal_outcome_changed": None,
    }
    # Example heuristics: look for 'activations' list or 'events' with type 'activation'
    acts = record.get("activations") or []
    if not isinstance(acts, list):
        acts = []
    out["activation_count"] = len(acts) if acts else 0
    if acts:
        out["first_activation_step"] = acts[0].get("step") if isinstance(acts[0], dict) else acts[0]
    # selected_command_source heuristic
    if "command_source" in record:
        out["selected_command_source"] = record.get("command_source")
    elif "selected_command" in record and isinstance(record["selected_command"], dict):
        out["selected_command_source"] = record["selected_command"].get("source")
    # command_source_changed: compare first/last
    try:
        sources = [e.get("source") for e in record.get("command_history", []) if isinstance(e, dict)]
        out["command_source_changed"] = (len(set(sources)) > 1) if sources else None
    except Exception:
        out["command_source_changed"] = None
    # progress delta heuristics
    try:
        prog = record.get("progress")
        if isinstance(prog, list) and out["first_activation_step"] is not None:
            step = int(out["first_activation_step"])
            after = prog[step:step+5]
            out["progress_delta_after_activation"] = (after[-1] - after[0]) if len(after) >= 2 else None
    except Exception:
        out["progress_delta_after_activation"] = None
    # trajectory_delta: naive L2 difference between pre/post small windows
    try:
        traj = record.get("trajectory")
        if isinstance(traj, list) and out["first_activation_step"] is not None:
            s = int(out["first_activation_step"])
            pre = traj[max(0, s-3):s]
            post = traj[s:s+3]
            def mean_pos(arr):
                pts = [p for p in arr if isinstance(p, (list, tuple)) and len(p) >= 2]
                if not pts:
                    return None
                x = sum(p[0] for p in pts)/len(pts)
                y = sum(p[1] for p in pts)/len(pts)
                return (x,y)
            prem = mean_pos(pre)
            postm = mean_pos(post)
            if prem and postm:
                dx = postm[0]-prem[0]
                dy = postm[1]-prem[1]
                out["trajectory_delta"] = (dx*dx+dy*dy)**0.5
    except Exception:
        out["trajectory_delta"] = None
    # terminal outcome changed: compare terminal fields if present
    if "terminal_outcome" in record:
        out["terminal_outcome_changed"] = record.get("terminal_outcome")
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Trace file not found: {path}")
        sys.exit(2)
    row_id_arg = None
    if len(sys.argv) >= 3:
        row_id_arg = sys.argv[2]
    results = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row_id_arg and str(rec.get("row_id", i)) != row_id_arg:
                continue
            out = extract_activation(rec)
            out["_row_index"] = i
            results.append(out)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
