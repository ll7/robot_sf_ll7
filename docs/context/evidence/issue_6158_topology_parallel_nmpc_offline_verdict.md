<!-- AI-GENERATED (robot_sf#6158) - NEEDS-REVIEW -->
# Issue #6158: topology-parallel NMPC offline verdict
Diagnostic-only validation of the merged #6170 prototype (`robot_sf/planner/topology_parallel_nmpc.py`) for parent #5310. The prototype was executed **unchanged**; this validator only imports/calls it and reads back diagnostics.
## Verdict
**`label_only_or_objective_drift`** — gate 2 (material distinctness) failed.
> ⚠️ **REAL-TIME BOUNDARY (prominent, independent of verdict):** the prototype's nominal `control_period_s` is **2.0 s (~20x the 100 ms real-time gate)**, so it is **offline-only and explicitly blocks downstream real-time use**. This is not a real-time qualification campaign; real-time/performance qualification stays in #5423. On this fixture, worst per-hypothesis solver p95 = 56.4 ms (under 100 ms, descriptive only; not a real-time qualification claim).
## Provenance
- Validated commit (`git rev-parse HEAD`): `af0d721fef61c744c1f792e9a62e1cceab9c1b36`
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
| platform_platform | `Linux-7.0.0-28-generic-x86_64-with-glibc2.39` |
| os_cpu_count | `20` |
| cpu_freq_mhz_sample | `unavailable` |
| python_version | `3.13.14` |
| numpy_version | `2.4.6` |
| scipy_version | `1.17.1` |
| cpu_pinning | `CPU 0 pinned for latency sampling; original affinity restored=True.` |

## Per-hypothesis solve latency (descriptive)
| hypothesis | p50 (ms) | p95 (ms) | max (ms) | n |
| --- | --- | --- | --- | --- |
| pass_left | 38.64 | 43.35 | 43.79 | 30 |
| yield_straight | 13.36 | 17.45 | 22.4 | 30 |
| pass_right | 53.21 | 56.4 | 56.94 | 30 |

End-to-end `plan()` wall-clock (measurement-safe deadline): p50=58.29334893496707 ms, p95=58.6876425310038 ms, max=58.827986009418964 ms (n=30).
End-to-end `plan()` wall-clock (real 2.0s deadline): p50=57.94054502621293 ms, p95=58.48700455389917 ms, max=58.52372199296951 ms; deadline fired 0 of 8 calls.

_Descriptive only on a single CPU-pinned fixture; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged._
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
best min pairwise material_separation across 6 conflict fixtures = 0.000186064 m (epsilon=0.001); gate passes only if at least one fixture separates every feasible hypothesis pair above epsilon. No fixture separated every feasible hypothesis pair above epsilon; topology identity is not established (label-only).
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
      "missing_rollout_pairs": [],
      "all_feasible_pairs_materially_distinct": false,
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
  "root_cause_note": "objective_preferred_turn == 0.0 for every hypothesis (gate 3), so the shared objective is identical; the only per-hypothesis difference is the initial-guess preferred_turn bias (+/-0.5 -> +/-0.1 rad/s w-seed via symmetry_break_bias=0.2). Across the tested seeds and fixtures, SLSQP left at least one feasible pair below the material-separation threshold on every fixture. Some individual pairs exceeded epsilon, but no fixture separated the full feasible hypothesis set under the shared soft-penalty objective; this diagnostic does not establish global uniqueness. The 'topology-parallel' mechanism is label-only under this configuration."
}
```
### gate_3_objective_invariance — PASS
objective_preferred_turn == 0.0 for every hypothesis; solver/bounds/constraints/options identical=True; shared config={'max_linear_speed': 0.9, 'max_angular_speed': 1.1, 'horizon_steps': 6, 'rollout_dt': 0.25, 'goal_tolerance': 0.25, 'waypoint_switch_distance': 0.75, 'path_goal_weight': 1.8, 'terminal_goal_weight': 4.5, 'progress_reward_weight': 2.0, 'heading_weight': 0.65, 'control_effort_weight': 0.06, 'smoothness_weight': 0.2, 'pedestrian_clearance_weight': 4.5, 'obstacle_clearance_weight': 4.2, 'occupancy_cost_weight': 1.2, 'collision_cost_kappa': 10.0, 'pedestrian_margin': 0.55, 'pedestrian_uncertainty_envelope_enabled': False, 'pedestrian_uncertainty_alpha_mps': 0.0, 'obstacle_margin': 0.45, 'desired_obstacle_clearance': 0.9, 'min_turn_speed_scale': 0.3, 'min_obstacle_speed_scale': 0.25, 'hard_obstacle_guard_enabled': False, 'hard_obstacle_clearance': 0.35, 'obstacle_threshold': 0.5, 'obstacle_search_cells': 12, 'avoidance_turn_bias_weight': 0.25, 'symmetry_break_bias': 0.2, 'solver_ftol': 0.001, 'solver_max_iterations': 32, 'warm_start': False, 'fallback_to_stop': True}.
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
    "max_linear_speed": 0.9,
    "max_angular_speed": 1.1,
    "horizon_steps": 6,
    "rollout_dt": 0.25,
    "goal_tolerance": 0.25,
    "waypoint_switch_distance": 0.75,
    "path_goal_weight": 1.8,
    "terminal_goal_weight": 4.5,
    "progress_reward_weight": 2.0,
    "heading_weight": 0.65,
    "control_effort_weight": 0.06,
    "smoothness_weight": 0.2,
    "pedestrian_clearance_weight": 4.5,
    "obstacle_clearance_weight": 4.2,
    "occupancy_cost_weight": 1.2,
    "collision_cost_kappa": 10.0,
    "pedestrian_margin": 0.55,
    "pedestrian_uncertainty_envelope_enabled": false,
    "pedestrian_uncertainty_alpha_mps": 0.0,
    "obstacle_margin": 0.45,
    "desired_obstacle_clearance": 0.9,
    "min_turn_speed_scale": 0.3,
    "min_obstacle_speed_scale": 0.25,
    "hard_obstacle_guard_enabled": false,
    "hard_obstacle_clearance": 0.35,
    "obstacle_threshold": 0.5,
    "obstacle_search_cells": 12,
    "avoidance_turn_bias_weight": 0.25,
    "symmetry_break_bias": 0.2,
    "solver_ftol": 0.001,
    "solver_max_iterations": 32,
    "warm_start": false,
    "fallback_to_stop": true
  },
  "all_objective_preferred_turn_zero": true,
  "solver_invocations": [
    {
      "method": "SLSQP",
      "bounds_lower": [
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1
      ],
      "bounds_upper": [
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1
      ],
      "constraints": [],
      "options": {
        "ftol": 0.001,
        "maxiter": 32,
        "disp": false
      }
    },
    {
      "method": "SLSQP",
      "bounds_lower": [
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1
      ],
      "bounds_upper": [
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1
      ],
      "constraints": [],
      "options": {
        "ftol": 0.001,
        "maxiter": 32,
        "disp": false
      }
    },
    {
      "method": "SLSQP",
      "bounds_lower": [
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1,
        0.0,
        -1.1
      ],
      "bounds_upper": [
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1,
        0.9,
        1.1
      ],
      "constraints": [],
      "options": {
        "ftol": 0.001,
        "maxiter": 32,
        "disp": false
      }
    }
  ],
  "solver_invocation_count": 3,
  "solver_configurations_identical": true
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
    "tick3_selected": 0,
    "tick3_reasons": {
      "pass_left": "already_selected",
      "yield_straight": "",
      "pass_right": ""
    },
    "tick4_selected": 1,
    "tick4_reasons": {
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
### gate_5_fail_closed — FAIL
infeasible->stop=True, deadline_exceeded->stop=True, solver_error_status->stop=True, solver_exception->stop=False.
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
    "note": "An exception inside the objective is NOT caught by the prototype; plan() propagates it instead of returning the fail-closed stop command. The #6158 gate requires fail-closed solver-error behavior, so this probe fails gate 5 even though deadline-overrun and infeasible/error-status fallbacks are correct."
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
per-hypothesis solver p95 (ms): pass_left=43.4, yield_straight=17.4, pass_right=56.4; worst p95=56.4 ms; exceeds_100ms=False; cpu_pinned=True.
```json
{
  "cpu_affinity_fixture": {
    "pinned": true,
    "measurement_cpu": 0,
    "original_cpu_set": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19
    ],
    "restored": true,
    "error": null
  },
  "per_hypothesis_solver_runtime_ms": {
    "pass_left": {
      "p50_ms": 38.63990353420377,
      "p95_ms": 43.35496834246442,
      "max_ms": 43.79475105088204,
      "n": 30
    },
    "yield_straight": {
      "p50_ms": 13.363074976950884,
      "p95_ms": 17.447194614214823,
      "max_ms": 22.404931019991636,
      "n": 30
    },
    "pass_right": {
      "p50_ms": 53.20639745332301,
      "p95_ms": 56.3979287922848,
      "max_ms": 56.9420310202986,
      "n": 30
    }
  },
  "plan_wall_clock_ms_measurement_safe_deadline": {
    "p50_ms": 58.29334893496707,
    "p95_ms": 58.6876425310038,
    "max_ms": 58.827986009418964,
    "n": 30
  },
  "plan_wall_clock_ms_real_2s_deadline": {
    "p50_ms": 57.94054502621293,
    "p95_ms": 58.48700455389917,
    "max_ms": 58.52372199296951,
    "n": 8
  },
  "real_deadline_fires_out_of_8": 0,
  "worst_hypothesis_p95_ms": 56.3979287922848,
  "latency_exceeds_100ms": false,
  "measurement_note": "Descriptive only on a single CPU-pinned fixture; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged."
}
```
### gate_8_pr_audit — PASS
PR #6170 @ 894bdfe71e9c: 11 files, +1238/-9 (net +1229); surfaces_match=True, totals_consistent=True, source_audit_matches_declared=True, current_pr_preserves_prototype=True.
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
  "source_audit": [
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
  "source_audit_matches_declared": true,
  "source_audit_parse_error": null,
  "current_pr_protected_path_deltas": [],
  "current_pr_preserves_prototype": true,
  "head_post_merge_note": "The source merge's first-parent diff matches the declared implementation packet, and this PR does not change the protected prototype/config/registry/test surfaces relative to origin/main."
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

The source merge's first-parent diff matches the declared implementation packet, and this PR does not change the protected prototype/config/registry/test surfaces relative to origin/main.

## Claim boundary
No real-time-suitability, safety, benchmark-superiority, default-planner-promotion, or #5423/STKP-eligibility claim. Diagnostic-only offline mechanism evidence.

## Machine-readable summary
```json
{
  "schema": "issue_6158_topology_parallel_nmpc_offline_verdict.v1",
  "review_marker": "AI-GENERATED NEEDS-REVIEW",
  "issue": 6158,
  "parent_issue": 5310,
  "source_pr": 6170,
  "source_merge_commit": "894bdfe71e9c2686ebe63e165f15c739d12f721c",
  "validated_commit": "af0d721fef61c744c1f792e9a62e1cceab9c1b36",
  "branch": "orchestrator/ll7-lease-6158-8860de392d30",
  "verdict": "label_only_or_objective_drift",
  "verdict_rationale": "gate 2 (material distinctness) failed.",
  "config": "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
  "commands": [
    "uv run pytest tests/planner/test_topology_parallel_nmpc.py tests/planner/test_nmpc_social.py -v",
    "uv run python scripts/validation/check_issue_6158_topology_parallel_nmpc_offline_verdict.py --config configs/algos/issue_5310_topology_parallel_nmpc.yaml",
    "uv run ruff check scripts/validation/ && uv run ruff format --check scripts/validation/"
  ],
  "hardware_context": {
    "platform_processor": "x86_64",
    "platform_machine": "x86_64",
    "platform_platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
    "os_cpu_count": 20,
    "cpu_freq_mhz_sample": "unavailable",
    "python_version": "3.13.14",
    "numpy_version": "2.4.6",
    "scipy_version": "1.17.1",
    "cpu_pinning": "CPU 0 pinned for latency sampling; original affinity restored=True."
  },
  "per_hypothesis_solver_latency_ms": {
    "pass_left": {
      "p50_ms": 38.63990353420377,
      "p95_ms": 43.35496834246442,
      "max_ms": 43.79475105088204,
      "n": 30
    },
    "yield_straight": {
      "p50_ms": 13.363074976950884,
      "p95_ms": 17.447194614214823,
      "max_ms": 22.404931019991636,
      "n": 30
    },
    "pass_right": {
      "p50_ms": 53.20639745332301,
      "p95_ms": 56.3979287922848,
      "max_ms": 56.9420310202986,
      "n": 30
    }
  },
  "plan_wall_clock_ms_measurement_safe_deadline": {
    "p50_ms": 58.29334893496707,
    "p95_ms": 58.6876425310038,
    "max_ms": 58.827986009418964,
    "n": 30
  },
  "plan_wall_clock_ms_real_2s_deadline": {
    "p50_ms": 57.94054502621293,
    "p95_ms": 58.48700455389917,
    "max_ms": 58.52372199296951,
    "n": 8
  },
  "real_deadline_fires_out_of_8": 0,
  "worst_hypothesis_p95_ms": 56.3979287922848,
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
      "detail": "best min pairwise material_separation across 6 conflict fixtures = 0.000186064 m (epsilon=0.001); gate passes only if at least one fixture separates every feasible hypothesis pair above epsilon. No fixture separated every feasible hypothesis pair above epsilon; topology identity is not established (label-only).",
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
            "missing_rollout_pairs": [],
            "all_feasible_pairs_materially_distinct": false,
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
        "root_cause_note": "objective_preferred_turn == 0.0 for every hypothesis (gate 3), so the shared objective is identical; the only per-hypothesis difference is the initial-guess preferred_turn bias (+/-0.5 -> +/-0.1 rad/s w-seed via symmetry_break_bias=0.2). Across the tested seeds and fixtures, SLSQP left at least one feasible pair below the material-separation threshold on every fixture. Some individual pairs exceeded epsilon, but no fixture separated the full feasible hypothesis set under the shared soft-penalty objective; this diagnostic does not establish global uniqueness. The 'topology-parallel' mechanism is label-only under this configuration."
      }
    },
    {
      "name": "gate_3_objective_invariance",
      "passed": true,
      "detail": "objective_preferred_turn == 0.0 for every hypothesis; solver/bounds/constraints/options identical=True; shared config={'max_linear_speed': 0.9, 'max_angular_speed': 1.1, 'horizon_steps': 6, 'rollout_dt': 0.25, 'goal_tolerance': 0.25, 'waypoint_switch_distance': 0.75, 'path_goal_weight': 1.8, 'terminal_goal_weight': 4.5, 'progress_reward_weight': 2.0, 'heading_weight': 0.65, 'control_effort_weight': 0.06, 'smoothness_weight': 0.2, 'pedestrian_clearance_weight': 4.5, 'obstacle_clearance_weight': 4.2, 'occupancy_cost_weight': 1.2, 'collision_cost_kappa': 10.0, 'pedestrian_margin': 0.55, 'pedestrian_uncertainty_envelope_enabled': False, 'pedestrian_uncertainty_alpha_mps': 0.0, 'obstacle_margin': 0.45, 'desired_obstacle_clearance': 0.9, 'min_turn_speed_scale': 0.3, 'min_obstacle_speed_scale': 0.25, 'hard_obstacle_guard_enabled': False, 'hard_obstacle_clearance': 0.35, 'obstacle_threshold': 0.5, 'obstacle_search_cells': 12, 'avoidance_turn_bias_weight': 0.25, 'symmetry_break_bias': 0.2, 'solver_ftol': 0.001, 'solver_max_iterations': 32, 'warm_start': False, 'fallback_to_stop': True}.",
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
          "max_linear_speed": 0.9,
          "max_angular_speed": 1.1,
          "horizon_steps": 6,
          "rollout_dt": 0.25,
          "goal_tolerance": 0.25,
          "waypoint_switch_distance": 0.75,
          "path_goal_weight": 1.8,
          "terminal_goal_weight": 4.5,
          "progress_reward_weight": 2.0,
          "heading_weight": 0.65,
          "control_effort_weight": 0.06,
          "smoothness_weight": 0.2,
          "pedestrian_clearance_weight": 4.5,
          "obstacle_clearance_weight": 4.2,
          "occupancy_cost_weight": 1.2,
          "collision_cost_kappa": 10.0,
          "pedestrian_margin": 0.55,
          "pedestrian_uncertainty_envelope_enabled": false,
          "pedestrian_uncertainty_alpha_mps": 0.0,
          "obstacle_margin": 0.45,
          "desired_obstacle_clearance": 0.9,
          "min_turn_speed_scale": 0.3,
          "min_obstacle_speed_scale": 0.25,
          "hard_obstacle_guard_enabled": false,
          "hard_obstacle_clearance": 0.35,
          "obstacle_threshold": 0.5,
          "obstacle_search_cells": 12,
          "avoidance_turn_bias_weight": 0.25,
          "symmetry_break_bias": 0.2,
          "solver_ftol": 0.001,
          "solver_max_iterations": 32,
          "warm_start": false,
          "fallback_to_stop": true
        },
        "all_objective_preferred_turn_zero": true,
        "solver_invocations": [
          {
            "method": "SLSQP",
            "bounds_lower": [
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1
            ],
            "bounds_upper": [
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1
            ],
            "constraints": [],
            "options": {
              "ftol": 0.001,
              "maxiter": 32,
              "disp": false
            }
          },
          {
            "method": "SLSQP",
            "bounds_lower": [
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1
            ],
            "bounds_upper": [
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1
            ],
            "constraints": [],
            "options": {
              "ftol": 0.001,
              "maxiter": 32,
              "disp": false
            }
          },
          {
            "method": "SLSQP",
            "bounds_lower": [
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1,
              0.0,
              -1.1
            ],
            "bounds_upper": [
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1,
              0.9,
              1.1
            ],
            "constraints": [],
            "options": {
              "ftol": 0.001,
              "maxiter": 32,
              "disp": false
            }
          }
        ],
        "solver_invocation_count": 3,
        "solver_configurations_identical": true
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
          "tick3_selected": 0,
          "tick3_reasons": {
            "pass_left": "already_selected",
            "yield_straight": "",
            "pass_right": ""
          },
          "tick4_selected": 1,
          "tick4_reasons": {
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
      "passed": false,
      "detail": "infeasible->stop=True, deadline_exceeded->stop=True, solver_error_status->stop=True, solver_exception->stop=False.",
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
          "note": "An exception inside the objective is NOT caught by the prototype; plan() propagates it instead of returning the fail-closed stop command. The #6158 gate requires fail-closed solver-error behavior, so this probe fails gate 5 even though deadline-overrun and infeasible/error-status fallbacks are correct."
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
      "detail": "per-hypothesis solver p95 (ms): pass_left=43.4, yield_straight=17.4, pass_right=56.4; worst p95=56.4 ms; exceeds_100ms=False; cpu_pinned=True.",
      "evidence": {
        "cpu_affinity_fixture": {
          "pinned": true,
          "measurement_cpu": 0,
          "original_cpu_set": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19
          ],
          "restored": true,
          "error": null
        },
        "per_hypothesis_solver_runtime_ms": {
          "pass_left": {
            "p50_ms": 38.63990353420377,
            "p95_ms": 43.35496834246442,
            "max_ms": 43.79475105088204,
            "n": 30
          },
          "yield_straight": {
            "p50_ms": 13.363074976950884,
            "p95_ms": 17.447194614214823,
            "max_ms": 22.404931019991636,
            "n": 30
          },
          "pass_right": {
            "p50_ms": 53.20639745332301,
            "p95_ms": 56.3979287922848,
            "max_ms": 56.9420310202986,
            "n": 30
          }
        },
        "plan_wall_clock_ms_measurement_safe_deadline": {
          "p50_ms": 58.29334893496707,
          "p95_ms": 58.6876425310038,
          "max_ms": 58.827986009418964,
          "n": 30
        },
        "plan_wall_clock_ms_real_2s_deadline": {
          "p50_ms": 57.94054502621293,
          "p95_ms": 58.48700455389917,
          "max_ms": 58.52372199296951,
          "n": 8
        },
        "real_deadline_fires_out_of_8": 0,
        "worst_hypothesis_p95_ms": 56.3979287922848,
        "latency_exceeds_100ms": false,
        "measurement_note": "Descriptive only on a single CPU-pinned fixture; not a controlled benchmark. max_runtime_s/control_period_s were raised to 300s during measurement so the runtime gate never truncates a solve; the shared NMPC config is unchanged."
      }
    },
    {
      "name": "gate_8_pr_audit",
      "passed": true,
      "detail": "PR #6170 @ 894bdfe71e9c: 11 files, +1238/-9 (net +1229); surfaces_match=True, totals_consistent=True, source_audit_matches_declared=True, current_pr_preserves_prototype=True.",
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
        "source_audit": [
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
        "source_audit_matches_declared": true,
        "source_audit_parse_error": null,
        "current_pr_protected_path_deltas": [],
        "current_pr_preserves_prototype": true,
        "head_post_merge_note": "The source merge's first-parent diff matches the declared implementation packet, and this PR does not change the protected prototype/config/registry/test surfaces relative to origin/main."
      }
    }
  ]
}
```
