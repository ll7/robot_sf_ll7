# Guarded PPO Baseline (`guarded_ppo`)

This document formalizes `guarded_ppo` as the current internal safety-aware planner-family
representative in `robot_sf_ll7`.

It should be read as a benchmark contract note, not as a claim of equivalence to stronger external
systems such as SoNIC. The current implementation is intentionally narrower: it keeps PPO as the
primary policy and applies a lightweight short-horizon safety veto plus local fallback when the PPO
action appears unsafe.

## Canonical profile

- Planner key: `guarded_ppo`
- Canonical benchmark config: `configs/algos/guarded_ppo_camera_ready.yaml`
- Primary policy model: `model_id: ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- Guard implementation: `robot_sf/planner/guarded_ppo.py`
- Fallback planner family: `RiskDWAPlannerAdapter`

This is the current internal safety-aware family representative because it is the lightest
benchmark-runnable way to add explicit short-horizon safety intervention to the strongest available
PPO baseline.

## Planner-facing input contract

`guarded_ppo` uses the same dict observation contract as the wrapped PPO policy. In benchmark runs,
that means a model-aligned dictionary observation keyed to the PPO input space, together with the
structured state fields needed by the safety guard:

- robot position / heading
- goal position
- pedestrian positions / velocities
- optional occupancy-grid payload for obstacle clearance checks

The PPO policy remains the primary action source. The guard reads the same observation and evaluates
the candidate action in a short local rollout.

## Intervention semantics

The guard applies a strict decision order:

1. If the route goal is already reached, command stop.
2. If the current near-field crowd density is low and the PPO action is safe, keep the PPO action.
3. If the PPO action is safe in the guarded rollout, keep the PPO action.
4. Otherwise, evaluate the configured `RiskDWAPlannerAdapter` fallback.
5. If fallback is also unsafe, evaluate stop.
6. If no safe option exists, choose the best-effort action with the greatest pedestrian clearance.

Safety checks currently include:

- hard pedestrian clearance
- first-step pedestrian clearance
- hard obstacle clearance
- minimum TTC threshold

This makes `guarded_ppo` a veto-and-recovery controller, not a jointly trained safety-aware policy.

## Fail-closed / fallback behavior

- If the PPO action is unsafe, `guarded_ppo` falls back to the local planner.
- If the fallback is also unsafe, it attempts a safe stop.
- If even stop is not classified as safe, it still returns the safest best-effort option available.

This means the planner does not fail hard on unsafe local decisions, but it also does not guarantee
formal safety. It is best understood as a practical short-horizon intervention layer.

## Kinematics and adapter status

- Benchmark execution mode: `mixed`
- Output contract: normalized to unicycle `v, omega`
- Differential-drive compatibility: yes, via the benchmark command normalization path

The wrapped PPO policy can internally use native mixed command semantics, but benchmark-facing
artifacts are emitted under the same normalized kinematics contract as the rest of the stack.

## Benchmark-readiness level

Current status:

- readiness tier: `experimental`
- benchmark status: runnable and documented
- paper-facing status: documentation-only support unless explicit promotion evidence is provided

`guarded_ppo` is suitable for controlled challenger benchmarks. It is **not** baseline-ready and
should not be presented as a frozen paper baseline unless it has explicit promotion evidence and a
documented canonical checkpoint.

## Conceptual comparison to SoNIC-style safety-aware navigation

Reference anchor:

- SoNIC: Yao et al. (2025), public repo noted in `docs/context/issue_603_alyassi_reference_set_2026-03-06.md`

What `guarded_ppo` currently captures:

- explicit safety-aware post-processing on top of a learned navigation policy
- local intervention when predicted near-field behavior is unsafe
- benchmark-runnable safety-aware family representative

What it still lacks compared with a SoNIC-style safety-aware system:

- no jointly trained uncertainty-aware safety objective
- no explicit uncertainty-calibrated prediction head in the policy loop
- no dedicated safety critic / auxiliary risk model integrated into learning
- no evidence yet that the intervention policy improves benchmark outcomes enough to justify
  paper-facing promotion

So the correct manuscript interpretation is:

- `guarded_ppo` provides **partial internal support** for a safety-aware learned family
- it is **not** a full SoNIC-equivalent implementation

## Current manuscript-facing recommendation

Use `guarded_ppo` as:

- `implemented but experimental` in the planner-family coverage matrix
- a documentation-backed internal safety-aware family representative

Do **not** use it as:

- baseline-ready support
- evidence of full safety-aware literature-family parity

Only upgrade this status after:

1. a canonical promoted guarded-PPO checkpoint exists
2. benchmark evidence shows stable value over the unguarded PPO baseline
3. the intervention semantics and failure modes are documented against actual evaluation artifacts
