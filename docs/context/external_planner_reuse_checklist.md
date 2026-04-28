# External Planner Reuse Checklist

Date: 2026-04-15

This checklist is a reusable, fail-fast path for reusing an external local planner or learned
navigation policy in `robot_sf_ll7`.

## 1. Verify provenance first

- confirm the upstream repo URL
- confirm the license
- confirm the exact checkpoint or release asset
- confirm a runnable source entrypoint or inference script
- record the canonical upstream files that define the contract

## 2. Reproduce the source harness before wrapping

- run the upstream test or inference path in the closest possible source environment
- record the first real blocker, not the intended blocker
- do not treat a wrapper smoke as source-harness proof
- if the source harness needs a side environment, keep that result separate from Robot SF

## 3. Capture the contract explicitly

- document the expected observation keys, tensor shapes, and preprocessing
- document the upstream action type and any kinematics assumptions
- document whether the planner is holonomic, unicycle, waypoint-based, or hybrid
- record any checkpoint-specific quirks, missing keys, or import shims

## 4. Build the Robot SF wrapper narrowly

- translate Robot SF observations into the upstream contract
- project the upstream action into Robot SF `unicycle_vw` only when needed
- fail closed on missing assets or incompatible kinematics
- avoid heuristic fallback unless the issue explicitly asks for it

## 5. Prove the wrapper with a smoke path

- run one real Robot SF scenario or minimal stateful smoke step
- prefer the actual upstream checkout and checkpoint over a fake stub
- assert finite outputs, bounded commands, and the expected metadata
- keep the smoke scenario small enough to run repeatedly

## 6. Record the verdict

- `integrate next` only if the source harness is reproducible and the wrapper is faithful enough
- `prototype only` if the wrapper works but source parity is still partial
- `assessment only` if the repo is only a reference or needs a larger reimplementation
- `reject` if the license, runtime, or contract mismatch is too large

## 7. Keep the follow-up explicit

- link the issue note, validation command, and smoke artifact
- record the remaining parity gaps
- state whether the next blocker is provenance, runtime, observation mapping, or action projection

## Use With

- source-harness probes
- model-only inference probes
- wrapper prototypes for CrowdNav/SoNIC-style planners
- future planner zoo intake and review notes
