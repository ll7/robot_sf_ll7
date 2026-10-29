"""Derive a campaign config from the guard/escape replication inputs.

Keeps every setting of inputs/full.yaml except: planners restricted to G0E1/G1E1,
scenario matrix restricted to one cell, seed list replaced, trace flags, name.
Arm configs G0E1.yaml/G1E1.yaml are copied byte-identically.
"""
import argparse, shutil, sys
from pathlib import Path
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--src-inputs", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--packet-rel", required=True)  # repo-relative dir for inputs
ap.add_argument("--name", required=True)
ap.add_argument("--cell", default="classic_cross_trap_high")
ap.add_argument("--seeds", required=True)  # comma list
ap.add_argument("--trace", action="store_true")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
src = Path(a.src_inputs); out = Path(a.repo) / a.packet_rel; out.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in a.seeds.split(",")]
for s in seeds:
    if 111 <= s <= 140 or not (1001 <= s <= 1200):
        sys.exit(f"refusing seed {s}")
for arm in ("G0E1", "G1E1"):
    shutil.copyfile(src / f"{arm}.yaml", out / f"{arm}.yaml")
scen = yaml.safe_load((src / "full_scenarios.yaml").read_text())
lst = scen["scenarios"] if isinstance(scen, dict) else scen
sel = [s for s in lst if s.get("name") == a.cell]
assert len(sel) == 1, a.cell
sel[0]["seeds"] = list(seeds)
payload = {"scenarios": sel} if isinstance(scen, dict) else sel
(out / f"{a.name}_scenarios.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
cfg = yaml.safe_load((src / "full.yaml").read_text())
cfg["name"] = a.name
cfg["scenario_matrix"] = f"{a.packet_rel}/{a.name}_scenarios.yaml"
cfg["seed_policy"] = {"mode": "fixed-list", "seeds": list(seeds)}
cfg["workers"] = a.workers
cfg["planners"] = [dict(p, algo_config=f"{a.packet_rel}/{p['key']}.yaml") for p in cfg["planners"] if p["key"] in ("G0E1", "G1E1")]
assert [p["key"] for p in cfg["planners"]] == ["G0E1", "G1E1"]
cfg["record_simulation_step_trace"] = bool(a.trace)
cfg["record_planner_decision_trace"] = False
(out / f"{a.name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out / f"{a.name}.yaml")
