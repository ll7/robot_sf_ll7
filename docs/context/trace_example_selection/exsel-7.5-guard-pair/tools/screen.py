"""Sequential EXSEL-7.5-GUARD-PAIR screening: one seed at a time, stop at first acceptance.

usage: screen.py <execution-worktree> <runs-dir> <ledger-dir>
Resumable: a seed whose complete run directory already holds both arms is re-read, not rerun.
"""
import csv, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exsel_screen as ex

wt, runs, ledger_dir = map(Path, sys.argv[1:4])
here = Path(__file__).parent
order = [int(x) for x in (here.parent / "screening_order.txt").read_text().split()]
assert len(order) == 40 and sorted(order) == list(range(1161, 1201))
inputs_rel = "docs/context/trace_example_selection/exsel-7.5-guard-pair/inputs"
ledger_dir.mkdir(parents=True, exist_ok=True)
rows_out, timing, result = [], [], None
for idx, seed in enumerate(order, 1):
    out = runs / f"s{seed}"
    done = out / "DONE"
    t0 = time.time()
    if not done.exists():
        if out.exists():
            raise SystemExit(f"incomplete run dir {out}: inspect before rerunning")
        cfg = f"{inputs_rel}/exsel75_screen_s{seed}.yaml"
        with open(ledger_dir / f"s{seed}.log", "w") as log:
            rc = subprocess.run([str(here / "run.sh"), cfg, str(out), f"exsel75_screen_s{seed}"],
                                env=dict(os.environ, EXSEL_WT=str(wt)), stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise SystemExit(f"runner failed for seed {seed} rc={rc}")
        done.write_text(f"{time.time() - t0:.1f}\n")
    loaded = ex.load_run(out)
    pair = {arm: loaded[(arm, seed)] for arm in ex.ARMS}
    assert len(loaded) == 2
    invalid = [a for a, r in pair.items() if r["mode"] != "adapter" or r["fallback_count"] != 0
               or r["fallback_or_degraded_marker"] is not None or r["scenario_id"] != ex.CELL
               or r["algo_ablation_disable_hard_guard"] != (a == "G0E1")]
    accepted = ex.verdicts(pair)
    for arm in ex.ARMS:
        r = dict(pair[arm], screening_index=idx)
        rows_out.append(r)
    timing.append({"seed": seed, "screening_index": idx, "wall_s": float(done.read_text())})
    print(f"[{idx:2d}/40] seed {seed}: " + "; ".join(
        f"{a} {pair[a]['outcome_class']}/{pair[a]['collision_kind']} steps {pair[a]['steps']} pass={pair[a]['arm_pass']}" for a in ex.ARMS)
        + f" -> accepted={accepted}", flush=True)
    if invalid:
        result = {"status": "BLOCKED_VALIDITY", "seed": seed, "screening_index": idx, "invalid_arms": invalid}
        break
    if accepted:
        result = {"status": "ACCEPTED", "seed": seed, "screening_index": idx}
        break
else:
    result = {"status": "NO_QUALIFYING_EXAMPLE", "seeds_screened": len(order)}
with open(ledger_dir / "screening_ledger.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=ex.LEDGER_FIELDS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows_out:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in ex.LEDGER_FIELDS})
result.update({"claim_id": ex.CLAIM_ID, "budget": 40, "seeds_screened": len(rows_out) // 2})
(ledger_dir / "screening_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(ledger_dir / "screening_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
print(json.dumps(result, sort_keys=True))
