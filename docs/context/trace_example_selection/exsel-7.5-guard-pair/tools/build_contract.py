"""Build contract.yaml: protocol sections copied verbatim (line ranges) + lane execution block."""
import hashlib, sys
from pathlib import Path
import yaml

proto_path, wt_inputs, order_path, out = map(Path, sys.argv[1:5])
lines = proto_path.read_text().splitlines(keepends=True)
def seg(a, b):  # 1-based inclusive
    return "".join(lines[a - 1:b])
assert lines[130].startswith("  - id: EXSEL-7.5-GUARD-PAIR") and lines[177].startswith("  - id: EXSEL-7.2")
assert lines[291].startswith("  ## EXSEL-7.5-GUARD-PAIR") and lines[319].startswith("  ## EXSEL-7.2")
order = [int(x) for x in order_path.read_text().split()]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = {p.name: sha(p) for p in sorted(wt_inputs.iterdir())}
head = f"""# EXSEL-7.5-GUARD-PAIR trace-example screening: execution contract. [AI-GENERATED]
#
# Committed before the first screening episode. Everything from `schema:` to the end
# of `acquisition_spec_markdown` is copied byte-for-byte from the frozen protocol
# ll7/diss docs/context/trace_example_selection_protocol.yaml at commit
# b95d41b524f1a12655ecf5722ee980d745861350 (decision EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28):
# lines 23-96 (identity, seeds, rules, outcome predicates, sources), the
# EXSEL-7.5-GUARD-PAIR claim entry (lines 131-177) and the acquisition spec's common
# steps plus its EXSEL-7.5-GUARD-PAIR section (lines 250-281, 292-319). That protocol is
# the preregistration; `execution:` at the end only records how this lane runs it.
"""
body = seg(23, 96) + "claims:\n" + seg(131, 177) + seg(250, 281) + seg(292, 319)
execution = {
  "execution": {
    "protocol_source": {"repository": "ll7/diss", "commit": "b95d41b524f1a12655ecf5722ee980d745861350",
                        "path": "docs/context/trace_example_selection_protocol.yaml",
                        "sha256": sha(proto_path)},
    "claim_id": "EXSEL-7.5-GUARD-PAIR",
    "evidence_tier": "development-seed diagnostic on an experimental rebuild of the 0.0.8 release code; new current-version execution; never release evidence",
    "claim_ceiling": "that of the guard/escape replication: within-controller development evidence (PROVISIONAL)",
    "repository": "ll7/robot_sf_ll7",
    "release_base_commit": "373dbfde4f39667cf9e8732dabe7df5118bdeab1",
    "execution_commit": "011c24045d6703a4a1e8b4428d6a9e931d220310",
    "execution_branch": "exscr2-guardpair-exec",
    "execution_tree": "7fa66da1a99db35c194dc99a2ab63ee68ecdf915",
    "execution_tree_note": "tree identical to the replication's execution commit 0035fba05c9b679e39f58ec286f5d57b96bb9a0a",
    "code_change": "acquisition-source.patch from packet acd878b4 applied unchanged (opt-in ablation_disable_hard_guard, default False); no other change",
    "rebuild_checks": "provenance/rebuild_checks.json (a-e all pass; e = provenance/parity_e.json)",
    "cell": "classic_cross_trap_high",
    "arms": {"G0E1": "inputs/G0E1.yaml (byte-identical to replication inputs/G0E1.yaml)",
             "G1E1": "inputs/G1E1.yaml (byte-identical to replication inputs/G1E1.yaml)"},
    "campaign_template": "replication inputs/full.yaml with planners restricted to G0E1, G1E1; scenario matrix restricted to the cell; seed_policy fixed-list [seed]; workers 2; name per seed; trace flags off for screening",
    "screening_configs": "inputs/exsel75_screen_s<seed>.yaml + inputs/exsel75_screen_s<seed>_scenarios.yaml, one campaign per seed",
    "input_sha256": inputs,
    "runner": "tools/run.sh <config> <output-root> <label> = OMP/OPENBLAS/MKL/NUMBA threads 1, CUDA_VISIBLE_DEVICES empty, SDL_VIDEODRIVER=dummy, MPLBACKEND=Agg, PYTHONPATH=<wt>:<wt>/fast-pysf, python scripts/tools/run_camera_ready_benchmark.py --config <config> --output-root <out> --label <label> --log-level INFO",
    "horizon": "authored schedule configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml (cell max_episode_steps 600), dt 0.1",
    "screening_order_source": "diss_worked_example_nomination.py --claims-protocol (exit 0, Frozen record: MATCH); recomputed independently as sorted(range(1161,1201), key=sha256('exsel-7.5-guard-pair-2026-10-28:<seed>').hexdigest())",
    "screening_order": order,
    "budget": 40,
    "stop_rule": "run seeds one at a time in screening_order; after each seed apply rules.acceptance; stop at the first accepted seed or after 40 seeds",
    "acceptance_as_applied": {
      "G0E1": "class {outcome: C, collision_kind: obstacle} using the guard-row projection of diss_worked_example_nomination.guard_record: outcome S if metrics.success; else C if outcome.collision_event; else T if outcome.timeout_event; collision_kind pedestrian if metrics.ped_collision_count > 0, else obstacle if metrics.obstacle_collision_count > 0; all G0E1 bands null, none checked",
      "G1E1": "class {outcome: S}; time_to_goal_s = round(steps * 0.1, 6) must lie in [33.2, 41.6] inclusive; clearance and exposure bands null, not checked",
      "seed": "accepted iff both arms pass",
      "secondary_reading": "metrics.wall_collisions is recorded per row; the release-row variant of the obstacle predicate (obstacle_collision_count > 0 or wall_collisions > 0) is reported as a secondary reading only",
    },
    "validity_guard": "every row must have mode == adapter, planner_runtime.fallback_count == 0 and no runtime fallback/degraded marker (robot_sf.benchmark.fallback_policy.runtime_fallback_or_degraded_marker, as in the replication project.py); a violating row is recorded in the ledger and stops the screening with the claim reported as BLOCKED; it is never skipped or replaced",
    "trace_rerun": "accepted seed only, both arms, same config with record_simulation_step_trace true and record_planner_decision_trace false; accept the trace iff each arm's rerun row equals its screening row on outcome class, termination_reason, steps, ped_collision_count, obstacle_collision_count, near_misses and min_ped_separation_m (exact); otherwise report as is (rules.never_replace)",
    "outcome_rule_for_thesis": {
      "ACCEPTED": "the accepted seed is the EXSEL-7.5-GUARD-PAIR trace example candidate; captions quote the chosen-cell base rates (joint pattern 27/30 replication pairs); whether a figure is made is decided on the dissertation side",
      "NO_QUALIFYING_EXAMPLE": "per rules.no_match the claim gets no trace example and Section 7.5 uses the aggregate guard/escape evidence alone",
      "TRACE_MISMATCH": "the rerun is reported as it is; the seed is not swapped (rules.never_replace); the trace is not admitted as the example without a dissertation-side decision",
    },
    "seed_hygiene": "screening seeds 1161-1200 only; parity seeds 1031-1035 (replication seeds, required by check e); never 111-140; no sealed release seed",
  }
}
text = head + body + yaml.safe_dump(execution, sort_keys=False, width=100)
doc = yaml.safe_load(text)
proto = yaml.safe_load(proto_path.read_text())
src_claim = [c for c in proto["claims"] if c["id"] == "EXSEL-7.5-GUARD-PAIR"][0]
assert doc["claims"] == [src_claim]
for k in ("schema", "frozen_on", "decision_id", "dev_seed_band", "band_percentiles", "min_matching_pairs", "rules", "sources"):
    assert doc[k] == proto[k], k
assert "## EXSEL-7.5-GUARD-PAIR" in doc["acquisition_spec_markdown"]
out.write_text(text)
print("ok", out)
