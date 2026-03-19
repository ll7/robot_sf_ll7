# Issue 601 CrowdNav Family Feasibility Note

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#601` CrowdNav feasibility spike
- `robot_sf_ll7#624` planner quality audit workflow
- `amv_benchmark_paper#76` attention-based family roadmap

## Why this family matters now

The merged planner audit established that the current headline local suite is limited to `orca` and
`ppo`, while the repo still lacks a credible external attention-based family anchor. The corrected
camera-ready benchmark evidence in
`output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407/reports/planner_quality_audit.md`
therefore leaves a visible paper-facing gap: current weak results for local learned planners should
not be interpreted as evidence that attention-based crowd-navigation families are weak in general.

The CrowdNav family is the right next assessment target because it is the canonical historical
attention-based crowd-navigation benchmark line, and the local `output/repos/SoNIC-Social-Nav`
checkout provides a more modern, pretrained, testable descendant-style reference.

## Canonical source assets

### CrowdNav: historical family anchor

Use these files as the canonical source anchor for the original attention-based family:

- `output/repos/CrowdNav/README.md`
- `output/repos/CrowdNav/crowd_nav/test.py`
- `output/repos/CrowdNav/crowd_nav/configs/env.config`
- `output/repos/CrowdNav/crowd_nav/configs/policy.config`

Assessment:

- The repository is mature enough to study and is MIT licensed.
- It provides the canonical paper/repo lineage for attention-based crowd navigation.
- It ships the source test entrypoint and config files needed to define the intended simulator and
  policy contract.
- Local pretrained weights are not obviously bundled in this checkout, so it is better treated as
  the historical family anchor than as the immediate runnable spike target.

### SoNIC-Social-Nav: practical executable spike candidate

Use these files as the practical spike anchor for a future reproduction:

- `output/repos/SoNIC-Social-Nav/README.md`
- `output/repos/SoNIC-Social-Nav/test.py`
- `output/repos/SoNIC-Social-Nav/trained_models/SoNIC_GST/checkpoints/05207.pt`
- `output/repos/SoNIC-Social-Nav/trained_models/GST_predictor_rand/checkpoints/41665.pt`
- `output/repos/SoNIC-Social-Nav/trained_models/ORCA/checkpoints/05207.pt`
- `output/repos/SoNIC-Social-Nav/trained_models/SF/checkpoints/05207.pt`

Assessment:

- The repository includes local pretrained checkpoints and a source test path.
- It is the stronger practical spike candidate because executable assets are already present.
- It remains unsuitable for direct benchmark integration today because the released workflow is
  Docker/NVIDIA-heavy and training is not fully released.

### CrowdNav_DSRNN: family context only

Use `output/repos/CrowdNav_DSRNN/README.md` only as context for the broader CrowdNav family
evolution. It should not be the first target for reproduction.

## Integration shape decision

Decision:

- Canonical family anchor: `CrowdNav`
- Stronger practical spike candidate: `SoNIC-Social-Nav`
- Integration category: `prototype only`
- Preferred integration shape: `model-wrapping adapter`
- Fallback policy: `fail fast only`

Rationale:

- A direct planner adapter is the wrong abstraction because the source family is organized around a
  model-specific simulator, policy factory, and source evaluation harness.
- A future spike should wrap one concrete pretrained source policy and prove source-harness parity
  first.
- Any future Robot SF wrapper must refuse to fall back to heuristics, otherwise the result becomes
  implementation-contaminated and cannot support family-level claims.

## Observation and action contract translation

| Contract area | Source expectation | Robot SF supply/target | Judgment |
|---|---|---|---|
| observation | CrowdNav-family joint robot/human state or SoNIC config/model-driven observation stack | Robot SF structured state and planner-facing benchmark observations | direct compatibility: no |
| observation translation | source simulator controls normalization, temporal context, and agent-state packing | Robot SF would need an explicit adapter that rebuilds the expected source observation tensor/state | adapter required: yes |
| action | source policy outputs simulator-native control semantics, likely holonomic or otherwise source-native | Robot SF benchmark expects unicycle-compatible `unicycle_vw` | direct compatibility: no |
| action translation | source action set is interpreted inside the source simulator | Robot SF would need a post-policy action adapter before benchmark execution | post-policy adapter required: yes |

Required rule for any future wrapper spike:

- Validate the policy in the source harness first.
- Only attempt a Robot SF adapter after the source-harness behavior is reproducible.
- Keep claims at implementation level until both source-harness and wrapper parity are documented.

## Benchmark-readiness risks

| Risk area | CrowdNav | SoNIC-Social-Nav | Implication |
|---|---|---|---|
| dependency maturity | mature Python repo with source configs and test entrypoint | runnable assets present, but workflow is Docker/NVIDIA-heavy | SoNIC is easier to spike locally, but not cheaper to operationalize |
| training/model availability | code/configs visible, local pretrained weights not obvious | pretrained checkpoints bundled locally; training release incomplete | SoNIC is the stronger immediate inference candidate |
| reproducibility of weights | unclear from local checkout | stronger because checkpoints are present | future spike should start from bundled SoNIC checkpoints |
| kinematics limitations | source simulator semantics differ from Robot SF unicycle benchmark contract | same problem, plus model-specific observation stack | wrapper must remain explicit and fail-fast |
| fallback behavior expectations | no heuristic fallback acceptable | no heuristic fallback acceptable | future reproduction must abort on contract mismatch or missing assets |

## Final recommendation

Recommendation: `prototype only`

Interpretation boundary:

- Current Robot SF results remain implementation-level local evidence only.
- They should not be presented as CrowdNav-family or SoNIC-family performance.
- A future implementation issue should target a fail-fast source-harness reproduction for the
  CrowdNav/SoNIC family, likely starting from SoNIC inference because assets are available locally.

Not recommended in this issue:

- direct benchmark integration
- new training
- family-level benchmark claims
