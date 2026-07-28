# Issue #6158: topology-parallel NMPC offline verdict
Diagnostic-only validation of the merged #6170 prototype (`robot_sf/planner/topology_parallel_nmpc.py`) for parent #5310. The prototype was executed **unchanged**; this validator only imports/calls it and reads back diagnostics.
## Verdict
**`label_only_or_objective_drift`** — gate 2 (material distinctness) or gate 3 (objective invariance) failed.
> ⚠️ **REAL-TIME BOUNDARY (prominent, independent of verdict):** the prototype's nominal `control_period_s` is **2.0 s (~20x the 100 ms real-time gate)**, so it is **offline-only and explicitly blocks downstream real-time use**. This is not a real-time qualification campaign; real-time/performance qualification stays in #5423. On this fixture, worst per-hypothesis solver p95 = 40.9 ms (under 100 ms, descriptive only; not a real-time qualification claim).
## Provenance
- Validated commit (`git rev-parse HEAD`): `b8b521220dff94bc427ff1260f378678055c8d34`
- Branch: `orchestrator/ll7-lease-6158-8860de392d30`
- Source PR: #6170 (merge commit `894bdfe71e9c2686ebe63e165f15c739d12f721c`)
- Config: `configs/algos/issue_5310_topology_parallel_nmpc.yaml`
## Exact commands
```
uv run pytest tests/planner/test_topology_parallel_nmpc.py tests/planner/test_nmpc_social.py -v
```
```
uv run python scripts/validation/check_issue_6158_topology_parallel_nmpc_offline_verdict.py --config configs/algos/issue_5310_topology_parallel_nmpc.yaml
```
```
uv run ruff check scripts/validation/ && uv run ruff format --check scripts/validation/
```
## Hardware context (fixed-CPU fixture, descriptive)
| field | value |
| --- | --- |
| platform_processor | `x86_64` |
| platform_machine | `x86_64` |
| platform_platform | `Linux-6.17.0-35-generic-x86_64-with-glibc2.39` |
| os_cpu_count | `20` |
| cpu_freq_mhz_sample | `unavailable` |
| python_version | `3.13.13` |
| numpy_version | `2.4.6` |
| scipy_version | `1.17.1` |
| cpu_pinning | `none (unpinned; descriptive only, not a controlled benchmark)` |

## Per-hypothesis solve latency (descriptive)
| hypothesis | p50 (ms) | p95 (ms) | max (ms) | n |
| --- | --- | --- | --- | --- |
| pass_left | 18.93 | 33.94 | 36.39 | 30 |
| yield_straight | 5.689 | 20.05 | 39.37 | 30 |
| pass_right | 35.73 | 40.91 | 43.28 | 30 |

End-to-end `plan()` wall-clock (measurement-safe deadline): p50=61.70633446890861 ms, p95=73.87282060226426 ms, max=74.85407800413668 ms (n=30).
End-to-end `plan()` wall-clock (real 2.0s deadline): p50=61.46642949897796 ms, p95=64.48990108910948 ms, max=65.69037795998156 ms; deadline fired 0 of 8 calls.

_Descriptive only on an unpinned CPU; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged._
## Gate-by-gate evidence
### gate_1_k1_legacy_parity — PASS
K=1 default command (0.9,-7.29807239e-07) vs legacy (0.9,-7.29807239e-07); |dv|=0.000e+00, |dw|=0.000e+00 (rtol=1e-06, atol=1e-06).
```json
{
  "topology_command": [
    0.9,
    -7.298072393551655e-07
  ],
  "legacy_command": [
    0.9,
    -7.298072393551655e-07
  ],
  "abs_delta": [
    0.0,
    0.0
  ],
  "rtol": 1e-06,
  "atol": 1e-06,
  "hypothesis_labels": [
    "default"
  ],
  "max_hypotheses": 1
}
```
### gate_2_material_distinctness — FAIL
best min pairwise material_separation across 6 conflict fixtures = 0.000186064 m (epsilon=0.001); gate passes only if at least one fixture separates feasible hypotheses. All fixtures collapsed to a single rollout -> topology identity fails (label-only).
```json
{
  "fixtures_tested": [
    "pedestrian_ahead_1p2",
    "pedestrian_close_0p5",
    "pedestrian_offset",
    "two_pedestrian_gate",
    "goal_offset_with_ped",
    "hard_wall_left_right_gaps"
  ],
  "epsilon_m": 0.001,
  "best_min_pairwise_separation_m": 0.00018606350715527403,
  "per_fixture": [
    {
      "fixture": "pedestrian_ahead_1p2",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 1.5592101923367186e-05,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 0.0003462401624407622
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 0.00033064806492490303
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 1.5592101923367186e-05
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.7874715063396169,
          "mean_y": -0.0024212445584959933,
          "span_x": 1.1248880322934551,
          "span_y": 0.006785913106531012,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.7874999999999993,
          "mean_y": -2.6977347472589836e-08,
          "span_x": 1.1249999999999987,
          "span_y": 5.041150866746945e-08,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.787486591380206,
          "mean_y": 0.0012250525934834218,
          "span_x": 1.1249205115394634,
          "span_y": 0.006743431723424951,
          "n_states": 6
        }
      }
    },
    {
      "fixture": "pedestrian_close_0p5",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 4.3745608233924324e-07,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 9.735127082719118e-05
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 8.144024845948835e-05
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 4.3745608233924324e-07
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.7874924609548964,
          "mean_y": 0.00038812232451622235,
          "span_x": 1.124955397083434,
          "span_y": 0.004514894041772958,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.7874999999999993,
          "mean_y": -2.7378698708169217e-08,
          "span_x": 1.1249999999999987,
          "span_y": 5.1161493941354035e-08,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.7874887454588776,
          "mean_y": 0.0010526206888845401,
          "span_x": 1.1249338442238543,
          "span_y": 0.006261936516120267,
          "n_states": 6
        }
      }
    },
    {
      "fixture": "pedestrian_offset",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 2.134114637979718e-05,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 6.335503175404234e-05
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 0.0002701239698273666
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 2.134114637979718e-05
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.7874836765866456,
          "mean_y": -0.0007663516121779622,
          "span_x": 1.1249205099512367,
          "span_y": 0.005948164548584899,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.7874999999999991,
          "mean_y": -3.821739018583512e-08,
          "span_x": 1.1249999999999978,
          "span_y": 7.15180749621957e-08,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.7874860281451063,
          "mean_y": 0.0010525462279931137,
          "span_x": 1.1249233259009743,
          "span_y": 0.0074480471846931975,
          "n_states": 6
        }
      }
    },
    {
      "fixture": "two_pedestrian_gate",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 9.423393415298936e-05,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 9.423393415298936e-05
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 0.0001737274428241798
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 0.0002679613661455612
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.7874906697990389,
          "mean_y": 0.00106066446213888,
          "span_x": 1.124949672036527,
          "span_y": 0.005629817856962316,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.7874999999999993,
          "mean_y": -3.123298153440145e-08,
          "span_x": 1.1249999999999987,
          "span_y": 5.867467731679452e-08,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.7874905460243474,
          "mean_y": 0.0002533209318209999,
          "span_x": 1.1249492173269378,
          "span_y": 0.005356288873019782,
          "n_states": 6
        }
      }
    },
    {
      "fixture": "goal_offset_with_ped",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 0.00018606350715527403,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 0.00018606350715527403
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 0.0015913418236559028
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 0.0003222806885182233
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.6945272337498625,
          "mean_y": 0.11905173245635063,
          "span_x": 0.9903152573700634,
          "span_y": 0.17988375932359904,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.6946694908949022,
          "mean_y": 0.11809056658512294,
          "span_x": 0.99046948518332,
          "span_y": 0.17821052014427846,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.6949463846481932,
          "mean_y": 0.11650871506734346,
          "span_x": 0.9907136970649153,
          "span_y": 0.17750228225988388,
          "n_states": 6
        }
      }
    },
    {
      "fixture": "hard_wall_left_right_gaps",
      "feasible_hypotheses": [
        "pass_left",
        "yield_straight",
        "pass_right"
      ],
      "min_pairwise_separation_m": 1.8915377488351905e-05,
      "pairwise_separations_m": [
        {
          "pair": [
            "pass_left",
            "yield_straight"
          ],
          "separation_m": 0.0010096006388823485
        },
        {
          "pair": [
            "pass_left",
            "pass_right"
          ],
          "separation_m": 0.001028515967872991
        },
        {
          "pair": [
            "yield_straight",
            "pass_right"
          ],
          "separation_m": 1.8915377488351905e-05
        }
      ],
      "rollout_signatures": {
        "pass_left": {
          "mean_x": 0.7874950213812805,
          "mean_y": 0.0022093504199595126,
          "span_x": 1.124995609602268,
          "span_y": 0.0019012071583989166,
          "n_states": 6
        },
        "yield_straight": {
          "mean_x": 0.7874999999998732,
          "mean_y": -4.0615195734946925e-07,
          "span_x": 1.1249999999996971,
          "span_y": 8.028925876145992e-07,
          "n_states": 6
        },
        "pass_right": {
          "mean_x": 0.7874967684778471,
          "mean_y": -0.0011627957231284385,
          "span_x": 1.1249914023444358,
          "span_y": 0.002858023952589533,
          "n_states": 6
        }
      }
    }
  ],
  "root_cause_note": "objective_preferred_turn == 0.0 for every hypothesis (gate 3), so the shared objective is identical; the only per-hypothesis difference is the initial-guess preferred_turn bias (+/-0.5 -> +/-0.1 rad/s w-seed via symmetry_break_bias=0.2). SLSQP converges to the unique optimum of the shared soft-penalty objective from every seed, so the rollouts collapse. The 'topology-parallel' mechanism is label-only under this configuration."
}
```
### gate_3_objective_invariance — PASS
objective_preferred_turn == 0.0 for every hypothesis; shared config={'horizon_steps': 6, 'solver_max_iterations': 32, 'solver_ftol': 0.001, 'warm_start': False}.
```json
{
  "per_hypothesis": {
    "pass_left": {
      "objective_preferred_turn": 0.0,
      "preferred_turn": 0.5,
      "solver_status": "0"
    },
    "yield_straight": {
      "objective_preferred_turn": 0.0,
      "preferred_turn": 0.0,
      "solver_status": "0"
    },
    "pass_right": {
      "objective_preferred_turn": 0.0,
      "preferred_turn": -0.5,
      "solver_status": "0"
    }
  },
  "shared_nmpc_config": {
    "horizon_steps": 6,
    "solver_max_iterations": 32,
    "solver_ftol": 0.001,
    "warm_start": false
  },
  "all_objective_preferred_turn_zero": true
}
```
### gate_4_selection_and_hysteresis — PASS
ordering=True, feasible_first/lowest_obj selection=True, hysteresis suppress(<2 ticks)=True, switch(>=2 ticks)=True.
```json
{
  "diagnostic_label_order": [
    "pass_left",
    "yield_straight",
    "pass_right"
  ],
  "expected_label_order": [
    "pass_left",
    "yield_straight",
    "pass_right"
  ],
  "feasible_first_selection_index": 2,
  "feasible_first_ranks": {
    "pass_left": 1,
    "yield_straight": -1,
    "pass_right": 0
  },
  "hysteresis": {
    "switch_hysteresis_ticks": 2,
    "tick1_selected": 0,
    "tick1_reasons": {
      "pass_left": "already_selected",
      "yield_straight": "",
      "pass_right": ""
    },
    "tick2_selected": 0,
    "tick2_reasons": {
      "pass_left": "hysteresis_hold",
      "yield_straight": "suppressed_by_hysteresis",
      "pass_right": ""
    },
    "tick3_selected": 1,
    "tick3_reasons": {
      "pass_left": "",
      "yield_straight": "new_best_selected",
      "pass_right": ""
    },
    "suppressed_before_threshold": true,
    "switched_at_or_after_threshold": true,
    "hypothesis_switches_recorded": 1
  }
}
```
### gate_5_fail_closed — PASS
infeasible->stop=True, deadline_exceeded->stop=True, solver_error_status->stop=True; exception_propagates=True (supplementary diagnostic).
```json
{
  "infeasible_status_command": [
    0.0,
    0.0
  ],
  "infeasible_solver_statuses": [
    "9",
    "9"
  ],
  "deadline_command": [
    0.0,
    0.0
  ],
  "deadline_solver_statuses": [
    "deadline_exceeded",
    "deadline_exceeded",
    "deadline_exceeded"
  ],
  "deadline_exceeded_flag": true,
  "solver_error_status_command": [
    0.0,
    0.0
  ],
  "exception_probe": {
    "exception_propagates": true,
    "exception_repr": "ValueError: synthetic objective failure",
    "note": "An exception inside the objective is NOT caught by the prototype; plan() propagates it. The prototype's documented fail-closed surfaces are deadline-overrun and infeasible/error-status (fallback_to_stop), both verified above. This is a robustness limitation, recorded transparently; it is not counted as a solver-error-status gate failure because scipy.optimize.minimize returns success=False on optimization errors rather than raising."
  }
}
```
### gate_6_registration_smoke — PASS
builder_ok=True, guard_reject_missing=True, guard_reject_false=True, registry_ok=True.
```json
{
  "guarded_build": {
    "algo_key": "topology_parallel_nmpc",
    "adapter_name": "TopologyParallelNMPCPlannerAdapter",
    "limitations": "experimental_topology_parallel_nmpc"
  },
  "guard_reject_missing": true,
  "guard_reject_false": true,
  "registry_build_from_issue_config": true
}
```
### gate_7_latency — PASS
per-hypothesis solver p95 (ms): pass_left=33.9, yield_straight=20.1, pass_right=40.9; worst p95=40.9 ms; exceeds_100ms=False.
```json
{
  "per_hypothesis_solver_runtime_ms": {
    "pass_left": {
      "p50_ms": 18.934933468699455,
      "p95_ms": 33.942097087856375,
      "max_ms": 36.38691082596779,
      "n": 30
    },
    "yield_straight": {
      "p50_ms": 5.6890781270340085,
      "p95_ms": 20.052773295901677,
      "max_ms": 39.37021689489484,
      "n": 30
    },
    "pass_right": {
      "p50_ms": 35.72503791656345,
      "p95_ms": 40.90983369387685,
      "max_ms": 43.27670601196587,
      "n": 30
    }
  },
  "plan_wall_clock_ms_measurement_safe_deadline": {
    "p50_ms": 61.70633446890861,
    "p95_ms": 73.87282060226426,
    "max_ms": 74.85407800413668,
    "n": 30
  },
  "plan_wall_clock_ms_real_2s_deadline": {
    "p50_ms": 61.46642949897796,
    "p95_ms": 64.48990108910948,
    "max_ms": 65.69037795998156,
    "n": 8
  },
  "real_deadline_fires_out_of_8": 0,
  "worst_hypothesis_p95_ms": 40.90983369387685,
  "latency_exceeds_100ms": false,
  "measurement_note": "Descriptive only on an unpinned CPU; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged."
}
```
### gate_8_pr_audit — PASS
PR #6170 @ 894bdfe71e9c: 11 files, +1238/-9 (net +1229); surfaces_match=True, totals_consistent=True.
```json
{
  "source_pr": 6170,
  "merge_commit": "894bdfe71e9c2686ebe63e165f15c739d12f721c",
  "files": [
    {
      "path": "CHANGELOG.md",
      "additions": 15,
      "deletions": 0
    },
    {
      "path": "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
      "additions": 41,
      "deletions": 0
    },
    {
      "path": "docs/context/issue_5310_state.yaml",
      "additions": 26,
      "deletions": 0
    },
    {
      "path": "robot_sf/benchmark/algorithm_metadata.py",
      "additions": 15,
      "deletions": 0
    },
    {
      "path": "robot_sf/benchmark/algorithm_readiness.py",
      "additions": 11,
      "deletions": 0
    },
    {
      "path": "robot_sf/benchmark/policy_builders.py",
      "additions": 35,
      "deletions": 0
    },
    {
      "path": "robot_sf/planner/__init__.py",
      "additions": 1,
      "deletions": 0
    },
    {
      "path": "robot_sf/planner/nmpc_social.py",
      "additions": 183,
      "deletions": 9
    },
    {
      "path": "robot_sf/planner/topology_parallel_nmpc.py",
      "additions": 447,
      "deletions": 0
    },
    {
      "path": "tests/planner/test_nmpc_social.py",
      "additions": 50,
      "deletions": 0
    },
    {
      "path": "tests/planner/test_topology_parallel_nmpc.py",
      "additions": 414,
      "deletions": 0
    }
  ],
  "total_additions": 1238,
  "total_deletions": 9,
  "net_lines": 1229,
  "audited_paths_match_implementation_surfaces": true,
  "totals_consistent": true,
  "prototype_present_at_head": true,
  "head_post_merge_note": "At current HEAD the prototype module received only docstring additions (PR #6282/#6299) after the #6170 merge; the nmpc_social seam received a docstring-only +5 change. An unrelated algorithm_metadata.py +81 hunk (issue #6190 predictive-foresight fallback provenance) is not part of the topology-parallel NMPC mechanism. The validated mechanism is behaviorally identical to the audited merge commit."
}
```
## PR #6170 changed-file / net-line audit
Merge commit `894bdfe71e9c2686ebe63e165f15c739d12f721c`.
| path | + | - | net |
| --- | ---: | ---: | ---: |
| CHANGELOG.md | +15 | -0 | +15 |
| configs/algos/issue_5310_topology_parallel_nmpc.yaml | +41 | -0 | +41 |
| docs/context/issue_5310_state.yaml | +26 | -0 | +26 |
| robot_sf/benchmark/algorithm_metadata.py | +15 | -0 | +15 |
| robot_sf/benchmark/algorithm_readiness.py | +11 | -0 | +11 |
| robot_sf/benchmark/policy_builders.py | +35 | -0 | +35 |
| robot_sf/planner/__init__.py | +1 | -0 | +1 |
| robot_sf/planner/nmpc_social.py | +183 | -9 | +174 |
| robot_sf/planner/topology_parallel_nmpc.py | +447 | -0 | +447 |
| tests/planner/test_nmpc_social.py | +50 | -0 | +50 |
| tests/planner/test_topology_parallel_nmpc.py | +414 | -0 | +414 |
| **total** | **+1238** | **-9** | **+1229** |

At current HEAD the prototype module received only docstring additions (PR #6282/#6299) after the #6170 merge; the nmpc_social seam received a docstring-only +5 change. An unrelated algorithm_metadata.py +81 hunk (issue #6190 predictive-foresight fallback provenance) is not part of the topology-parallel NMPC mechanism. The validated mechanism is behaviorally identical to the audited merge commit.

## Claim boundary
No real-time-suitability, safety, benchmark-superiority, default-planner-promotion, or #5423/STKP-eligibility claim. Diagnostic-only offline mechanism evidence.

## Machine-readable summary
```json
{
  "schema": "issue_6158_topology_parallel_nmpc_offline_verdict.v1",
  "issue": 6158,
  "parent_issue": 5310,
  "source_pr": 6170,
  "source_merge_commit": "894bdfe71e9c2686ebe63e165f15c739d12f721c",
  "validated_commit": "b8b521220dff94bc427ff1260f378678055c8d34",
  "branch": "orchestrator/ll7-lease-6158-8860de392d30",
  "verdict": "label_only_or_objective_drift",
  "verdict_rationale": "gate 2 (material distinctness) or gate 3 (objective invariance) failed.",
  "config": "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
  "commands": [
    "uv run pytest tests/planner/test_topology_parallel_nmpc.py tests/planner/test_nmpc_social.py -v",
    "uv run python scripts/validation/check_issue_6158_topology_parallel_nmpc_offline_verdict.py --config configs/algos/issue_5310_topology_parallel_nmpc.yaml",
    "uv run ruff check scripts/validation/ && uv run ruff format --check scripts/validation/"
  ],
  "hardware_context": {
    "platform_processor": "x86_64",
    "platform_machine": "x86_64",
    "platform_platform": "Linux-6.17.0-35-generic-x86_64-with-glibc2.39",
    "os_cpu_count": 20,
    "cpu_freq_mhz_sample": "unavailable",
    "python_version": "3.13.13",
    "numpy_version": "2.4.6",
    "scipy_version": "1.17.1",
    "cpu_pinning": "none (unpinned; descriptive only, not a controlled benchmark)"
  },
  "per_hypothesis_solver_latency_ms": {
    "pass_left": {
      "p50_ms": 18.934933468699455,
      "p95_ms": 33.942097087856375,
      "max_ms": 36.38691082596779,
      "n": 30
    },
    "yield_straight": {
      "p50_ms": 5.6890781270340085,
      "p95_ms": 20.052773295901677,
      "max_ms": 39.37021689489484,
      "n": 30
    },
    "pass_right": {
      "p50_ms": 35.72503791656345,
      "p95_ms": 40.90983369387685,
      "max_ms": 43.27670601196587,
      "n": 30
    }
  },
  "plan_wall_clock_ms_measurement_safe_deadline": {
    "p50_ms": 61.70633446890861,
    "p95_ms": 73.87282060226426,
    "max_ms": 74.85407800413668,
    "n": 30
  },
  "plan_wall_clock_ms_real_2s_deadline": {
    "p50_ms": 61.46642949897796,
    "p95_ms": 64.48990108910948,
    "max_ms": 65.69037795998156,
    "n": 8
  },
  "real_deadline_fires_out_of_8": 0,
  "worst_hypothesis_p95_ms": 40.90983369387685,
  "latency_exceeds_100ms": false,
  "control_period_s": 2.0,
  "real_time_blocking_notice": "NOT REAL-TIME QUALIFIED (prominent, independent of the per-solve number): the prototype's nominal control_period_s is 2.0 s, which is ~20x the 100 ms real-time gate, so the component is offline-only and explicitly blocks downstream real-time use. This is not a real-time qualification campaign; real-time/performance qualification stays in #5423. Per-hypothesis solver p95 was under 100 ms on this fixture, but this is descriptive only and does NOT establish real-time suitability.",
  "claim_boundary": "No real-time-suitability, safety, benchmark-superiority, default-planner-promotion, or #5423/STKP-eligibility claim. Diagnostic-only offline mechanism evidence.",
  "gates": [
    {
      "name": "gate_1_k1_legacy_parity",
      "passed": true,
      "detail": "K=1 default command (0.9,-7.29807239e-07) vs legacy (0.9,-7.29807239e-07); |dv|=0.000e+00, |dw|=0.000e+00 (rtol=1e-06, atol=1e-06).",
      "evidence": {
        "topology_command": [
          0.9,
          -7.298072393551655e-07
        ],
        "legacy_command": [
          0.9,
          -7.298072393551655e-07
        ],
        "abs_delta": [
          0.0,
          0.0
        ],
        "rtol": 1e-06,
        "atol": 1e-06,
        "hypothesis_labels": [
          "default"
        ],
        "max_hypotheses": 1
      }
    },
    {
      "name": "gate_2_material_distinctness",
      "passed": false,
      "detail": "best min pairwise material_separation across 6 conflict fixtures = 0.000186064 m (epsilon=0.001); gate passes only if at least one fixture separates feasible hypotheses. All fixtures collapsed to a single rollout -> topology identity fails (label-only).",
      "evidence": {
        "fixtures_tested": [
          "pedestrian_ahead_1p2",
          "pedestrian_close_0p5",
          "pedestrian_offset",
          "two_pedestrian_gate",
          "goal_offset_with_ped",
          "hard_wall_left_right_gaps"
        ],
        "epsilon_m": 0.001,
        "best_min_pairwise_separation_m": 0.00018606350715527403,
        "per_fixture": [
          {
            "fixture": "pedestrian_ahead_1p2",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 1.5592101923367186e-05,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 0.0003462401624407622
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 0.00033064806492490303
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 1.5592101923367186e-05
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.7874715063396169,
                "mean_y": -0.0024212445584959933,
                "span_x": 1.1248880322934551,
                "span_y": 0.006785913106531012,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.7874999999999993,
                "mean_y": -2.6977347472589836e-08,
                "span_x": 1.1249999999999987,
                "span_y": 5.041150866746945e-08,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.787486591380206,
                "mean_y": 0.0012250525934834218,
                "span_x": 1.1249205115394634,
                "span_y": 0.006743431723424951,
                "n_states": 6
              }
            }
          },
          {
            "fixture": "pedestrian_close_0p5",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 4.3745608233924324e-07,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 9.735127082719118e-05
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 8.144024845948835e-05
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 4.3745608233924324e-07
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.7874924609548964,
                "mean_y": 0.00038812232451622235,
                "span_x": 1.124955397083434,
                "span_y": 0.004514894041772958,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.7874999999999993,
                "mean_y": -2.7378698708169217e-08,
                "span_x": 1.1249999999999987,
                "span_y": 5.1161493941354035e-08,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.7874887454588776,
                "mean_y": 0.0010526206888845401,
                "span_x": 1.1249338442238543,
                "span_y": 0.006261936516120267,
                "n_states": 6
              }
            }
          },
          {
            "fixture": "pedestrian_offset",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 2.134114637979718e-05,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 6.335503175404234e-05
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 0.0002701239698273666
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 2.134114637979718e-05
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.7874836765866456,
                "mean_y": -0.0007663516121779622,
                "span_x": 1.1249205099512367,
                "span_y": 0.005948164548584899,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.7874999999999991,
                "mean_y": -3.821739018583512e-08,
                "span_x": 1.1249999999999978,
                "span_y": 7.15180749621957e-08,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.7874860281451063,
                "mean_y": 0.0010525462279931137,
                "span_x": 1.1249233259009743,
                "span_y": 0.0074480471846931975,
                "n_states": 6
              }
            }
          },
          {
            "fixture": "two_pedestrian_gate",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 9.423393415298936e-05,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 9.423393415298936e-05
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 0.0001737274428241798
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 0.0002679613661455612
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.7874906697990389,
                "mean_y": 0.00106066446213888,
                "span_x": 1.124949672036527,
                "span_y": 0.005629817856962316,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.7874999999999993,
                "mean_y": -3.123298153440145e-08,
                "span_x": 1.1249999999999987,
                "span_y": 5.867467731679452e-08,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.7874905460243474,
                "mean_y": 0.0002533209318209999,
                "span_x": 1.1249492173269378,
                "span_y": 0.005356288873019782,
                "n_states": 6
              }
            }
          },
          {
            "fixture": "goal_offset_with_ped",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 0.00018606350715527403,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 0.00018606350715527403
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 0.0015913418236559028
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 0.0003222806885182233
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.6945272337498625,
                "mean_y": 0.11905173245635063,
                "span_x": 0.9903152573700634,
                "span_y": 0.17988375932359904,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.6946694908949022,
                "mean_y": 0.11809056658512294,
                "span_x": 0.99046948518332,
                "span_y": 0.17821052014427846,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.6949463846481932,
                "mean_y": 0.11650871506734346,
                "span_x": 0.9907136970649153,
                "span_y": 0.17750228225988388,
                "n_states": 6
              }
            }
          },
          {
            "fixture": "hard_wall_left_right_gaps",
            "feasible_hypotheses": [
              "pass_left",
              "yield_straight",
              "pass_right"
            ],
            "min_pairwise_separation_m": 1.8915377488351905e-05,
            "pairwise_separations_m": [
              {
                "pair": [
                  "pass_left",
                  "yield_straight"
                ],
                "separation_m": 0.0010096006388823485
              },
              {
                "pair": [
                  "pass_left",
                  "pass_right"
                ],
                "separation_m": 0.001028515967872991
              },
              {
                "pair": [
                  "yield_straight",
                  "pass_right"
                ],
                "separation_m": 1.8915377488351905e-05
              }
            ],
            "rollout_signatures": {
              "pass_left": {
                "mean_x": 0.7874950213812805,
                "mean_y": 0.0022093504199595126,
                "span_x": 1.124995609602268,
                "span_y": 0.0019012071583989166,
                "n_states": 6
              },
              "yield_straight": {
                "mean_x": 0.7874999999998732,
                "mean_y": -4.0615195734946925e-07,
                "span_x": 1.1249999999996971,
                "span_y": 8.028925876145992e-07,
                "n_states": 6
              },
              "pass_right": {
                "mean_x": 0.7874967684778471,
                "mean_y": -0.0011627957231284385,
                "span_x": 1.1249914023444358,
                "span_y": 0.002858023952589533,
                "n_states": 6
              }
            }
          }
        ],
        "root_cause_note": "objective_preferred_turn == 0.0 for every hypothesis (gate 3), so the shared objective is identical; the only per-hypothesis difference is the initial-guess preferred_turn bias (+/-0.5 -> +/-0.1 rad/s w-seed via symmetry_break_bias=0.2). SLSQP converges to the unique optimum of the shared soft-penalty objective from every seed, so the rollouts collapse. The 'topology-parallel' mechanism is label-only under this configuration."
      }
    },
    {
      "name": "gate_3_objective_invariance",
      "passed": true,
      "detail": "objective_preferred_turn == 0.0 for every hypothesis; shared config={'horizon_steps': 6, 'solver_max_iterations': 32, 'solver_ftol': 0.001, 'warm_start': False}.",
      "evidence": {
        "per_hypothesis": {
          "pass_left": {
            "objective_preferred_turn": 0.0,
            "preferred_turn": 0.5,
            "solver_status": "0"
          },
          "yield_straight": {
            "objective_preferred_turn": 0.0,
            "preferred_turn": 0.0,
            "solver_status": "0"
          },
          "pass_right": {
            "objective_preferred_turn": 0.0,
            "preferred_turn": -0.5,
            "solver_status": "0"
          }
        },
        "shared_nmpc_config": {
          "horizon_steps": 6,
          "solver_max_iterations": 32,
          "solver_ftol": 0.001,
          "warm_start": false
        },
        "all_objective_preferred_turn_zero": true
      }
    },
    {
      "name": "gate_4_selection_and_hysteresis",
      "passed": true,
      "detail": "ordering=True, feasible_first/lowest_obj selection=True, hysteresis suppress(<2 ticks)=True, switch(>=2 ticks)=True.",
      "evidence": {
        "diagnostic_label_order": [
          "pass_left",
          "yield_straight",
          "pass_right"
        ],
        "expected_label_order": [
          "pass_left",
          "yield_straight",
          "pass_right"
        ],
        "feasible_first_selection_index": 2,
        "feasible_first_ranks": {
          "pass_left": 1,
          "yield_straight": -1,
          "pass_right": 0
        },
        "hysteresis": {
          "switch_hysteresis_ticks": 2,
          "tick1_selected": 0,
          "tick1_reasons": {
            "pass_left": "already_selected",
            "yield_straight": "",
            "pass_right": ""
          },
          "tick2_selected": 0,
          "tick2_reasons": {
            "pass_left": "hysteresis_hold",
            "yield_straight": "suppressed_by_hysteresis",
            "pass_right": ""
          },
          "tick3_selected": 1,
          "tick3_reasons": {
            "pass_left": "",
            "yield_straight": "new_best_selected",
            "pass_right": ""
          },
          "suppressed_before_threshold": true,
          "switched_at_or_after_threshold": true,
          "hypothesis_switches_recorded": 1
        }
      }
    },
    {
      "name": "gate_5_fail_closed",
      "passed": true,
      "detail": "infeasible->stop=True, deadline_exceeded->stop=True, solver_error_status->stop=True; exception_propagates=True (supplementary diagnostic).",
      "evidence": {
        "infeasible_status_command": [
          0.0,
          0.0
        ],
        "infeasible_solver_statuses": [
          "9",
          "9"
        ],
        "deadline_command": [
          0.0,
          0.0
        ],
        "deadline_solver_statuses": [
          "deadline_exceeded",
          "deadline_exceeded",
          "deadline_exceeded"
        ],
        "deadline_exceeded_flag": true,
        "solver_error_status_command": [
          0.0,
          0.0
        ],
        "exception_probe": {
          "exception_propagates": true,
          "exception_repr": "ValueError: synthetic objective failure",
          "note": "An exception inside the objective is NOT caught by the prototype; plan() propagates it. The prototype's documented fail-closed surfaces are deadline-overrun and infeasible/error-status (fallback_to_stop), both verified above. This is a robustness limitation, recorded transparently; it is not counted as a solver-error-status gate failure because scipy.optimize.minimize returns success=False on optimization errors rather than raising."
        }
      }
    },
    {
      "name": "gate_6_registration_smoke",
      "passed": true,
      "detail": "builder_ok=True, guard_reject_missing=True, guard_reject_false=True, registry_ok=True.",
      "evidence": {
        "guarded_build": {
          "algo_key": "topology_parallel_nmpc",
          "adapter_name": "TopologyParallelNMPCPlannerAdapter",
          "limitations": "experimental_topology_parallel_nmpc"
        },
        "guard_reject_missing": true,
        "guard_reject_false": true,
        "registry_build_from_issue_config": true
      }
    },
    {
      "name": "gate_7_latency",
      "passed": true,
      "detail": "per-hypothesis solver p95 (ms): pass_left=33.9, yield_straight=20.1, pass_right=40.9; worst p95=40.9 ms; exceeds_100ms=False.",
      "evidence": {
        "per_hypothesis_solver_runtime_ms": {
          "pass_left": {
            "p50_ms": 18.934933468699455,
            "p95_ms": 33.942097087856375,
            "max_ms": 36.38691082596779,
            "n": 30
          },
          "yield_straight": {
            "p50_ms": 5.6890781270340085,
            "p95_ms": 20.052773295901677,
            "max_ms": 39.37021689489484,
            "n": 30
          },
          "pass_right": {
            "p50_ms": 35.72503791656345,
            "p95_ms": 40.90983369387685,
            "max_ms": 43.27670601196587,
            "n": 30
          }
        },
        "plan_wall_clock_ms_measurement_safe_deadline": {
          "p50_ms": 61.70633446890861,
          "p95_ms": 73.87282060226426,
          "max_ms": 74.85407800413668,
          "n": 30
        },
        "plan_wall_clock_ms_real_2s_deadline": {
          "p50_ms": 61.46642949897796,
          "p95_ms": 64.48990108910948,
          "max_ms": 65.69037795998156,
          "n": 8
        },
        "real_deadline_fires_out_of_8": 0,
        "worst_hypothesis_p95_ms": 40.90983369387685,
        "latency_exceeds_100ms": false,
        "measurement_note": "Descriptive only on an unpinned CPU; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged."
      }
    },
    {
      "name": "gate_8_pr_audit",
      "passed": true,
      "detail": "PR #6170 @ 894bdfe71e9c: 11 files, +1238/-9 (net +1229); surfaces_match=True, totals_consistent=True.",
      "evidence": {
        "source_pr": 6170,
        "merge_commit": "894bdfe71e9c2686ebe63e165f15c739d12f721c",
        "files": [
          {
            "path": "CHANGELOG.md",
            "additions": 15,
            "deletions": 0
          },
          {
            "path": "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
            "additions": 41,
            "deletions": 0
          },
          {
            "path": "docs/context/issue_5310_state.yaml",
            "additions": 26,
            "deletions": 0
          },
          {
            "path": "robot_sf/benchmark/algorithm_metadata.py",
            "additions": 15,
            "deletions": 0
          },
          {
            "path": "robot_sf/benchmark/algorithm_readiness.py",
            "additions": 11,
            "deletions": 0
          },
          {
            "path": "robot_sf/benchmark/policy_builders.py",
            "additions": 35,
            "deletions": 0
          },
          {
            "path": "robot_sf/planner/__init__.py",
            "additions": 1,
            "deletions": 0
          },
          {
            "path": "robot_sf/planner/nmpc_social.py",
            "additions": 183,
            "deletions": 9
          },
          {
            "path": "robot_sf/planner/topology_parallel_nmpc.py",
            "additions": 447,
            "deletions": 0
          },
          {
            "path": "tests/planner/test_nmpc_social.py",
            "additions": 50,
            "deletions": 0
          },
          {
            "path": "tests/planner/test_topology_parallel_nmpc.py",
            "additions": 414,
            "deletions": 0
          }
        ],
        "total_additions": 1238,
        "total_deletions": 9,
        "net_lines": 1229,
        "audited_paths_match_implementation_surfaces": true,
        "totals_consistent": true,
        "prototype_present_at_head": true,
        "head_post_merge_note": "At current HEAD the prototype module received only docstring additions (PR #6282/#6299) after the #6170 merge; the nmpc_social seam received a docstring-only +5 change. An unrelated algorithm_metadata.py +81 hunk (issue #6190 predictive-foresight fallback provenance) is not part of the topology-parallel NMPC mechanism. The validated mechanism is behaviorally identical to the audited merge commit."
      }
    }
  ]
}
```
