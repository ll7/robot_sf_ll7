# Issue #1367 CrowdNav-Family Learned-Policy Verdict

Date: 2026-05-20

Related issues:

* <https://github.com/ll7/robot_sf_ll7/issues/1367>
* `#600` DS-RNN stretch follow-up
* `#601` CrowdNav family feasibility note
* `#627` SoNIC / GenSafeNav wrapper follow-up
* `#760` HEIGHT model-shortcoming investigation
* `#770` IGAT / ST2 attention-family assessment
* `#1355` external learned-policy candidate matrix
* `#1359` learned-policy reject/monitor registry
* `#1363` learned local-policy eligibility checklist
* `#1365` shared graph/social observation adapter
* `#1366` GenSafeNav / SoNIC conformal-contract assessment

## Goal

Refresh the CrowdNav-family learned-policy verdict using one shared Robot SF contract. The scope is
assessment only: compare CrowdNav / SARL, RGL, DS-RNN, CrowdNav++ / IGAT, HEIGHT, and
GenSafeNav / SoNIC, then select at most one first integration candidate or explicitly recommend no
new integration.

## Primary Sources Checked

External sources:

* CrowdNav / SARL: <https://github.com/vita-epfl/CrowdNav>
* RGL: <https://github.com/ChanganVR/RelationalGraphLearning> and
  <https://arxiv.org/abs/1909.13165>
* DS-RNN: <https://github.com/Shuijing725/CrowdNav_DSRNN> and
  <https://arxiv.org/abs/2011.04820>
* CrowdNav++ / IGAT: <https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph> and
  <https://arxiv.org/abs/2203.01821>
* HEIGHT: <https://github.com/Shuijing725/CrowdNav_HEIGHT> and
  <https://arxiv.org/abs/2411.12150>
* SoNIC / GenSafeNav: <https://github.com/tasl-lab/SoNIC-Social-Nav>,
  <https://github.com/tasl-lab/GenSafeNav>, <https://arxiv.org/abs/2407.17460>, and
  <https://arxiv.org/abs/2508.05634>

Repository evidence:

* `docs/context/issue_601_crowdnav_feasibility_note.md`
* `docs/context/issue_600_dsrnn_stretch_follow_up.md`
* `docs/context/issue_627_sonic_wrapper_followup.md`
* [`760_model_shortcoming_hypothesis.md`](../760_model_shortcoming_hypothesis.md)
* `docs/context/issue_770_igat_st2_attention_assessment.md`
* `docs/benchmark_planner_family_coverage.md`
* `robot_sf/planner/crowdnav_height.py`
* `robot_sf/planner/sonic_crowdnav.py`
* `robot_sf/benchmark/algorithm_metadata.py`

## Shared Contract

For Robot SF learned-local-policy consideration, a CrowdNav-family candidate must satisfy all of
these before it can move beyond an assessment or prototype label:

1. Source-side runtime is reproducible or the wrapper is explicitly labeled model-only.
2. Runtime observation fields are classifiable as deployment-observable, train-only,
   calibration-only, oracle-only, or forbidden.
3. Trajectory prediction, intention prediction, ACI/conformal uncertainty, or graph-history fields
   must be produced from current/past observations or frozen train/validation-only state.
4. Ground-truth future trajectories may be used only for source-side metrics or training labels, not
   for Robot SF benchmark policy input.
5. Source action semantics must be projected to Robot SF `unicycle_vw` with explicit metadata.
6. Missing assets, missing source environments, or unsupported kinematics must fail closed.

## Candidate Comparison

| Candidate | Source / license | Current Robot SF status | Observation / prediction surface | Action / adapter burden | Verdict |
| --- | --- | --- | --- | --- | --- |
| CrowdNav / SARL | MIT repository, canonical ICRA 2019 attention-family anchor | Historical external anchor only | Joint robot/human state with SARL attention; no obvious bundled pretrained weights in current source evidence | Source simulator action semantics; source-harness parity required | `assessment only` |
| RGL | Public source repository; source license not verified in this issue | External anchor only | Relational graph learning plus model-based multi-step lookahead; predicts human motion for planning | CrowdNav-style simulator and policy path; no Robot SF wrapper | `monitor / assessment only` |
| DS-RNN | MIT repository with example holonomic and unicycle weights | Roadmap/stretch note only | Structural recurrent graph state, temporal hidden state, partial-observation emphasis | Older Python/OpenAI Baselines stack; recurrent reset and hidden-state parity burden | `source-harness first` |
| CrowdNav++ / IGAT | MIT repository with trained models and test path | Not implemented; assessed in #770 | Gumbel Social Transformer / intention prediction over future pedestrian trajectories; may use `inferred`, `const_vel`, `truth`, or `none` prediction modes | Legacy CrowdNav stack plus additional trajectory-history adapter; no advantage over HEIGHT without an intent ablation | `prototype only / monitor` |
| HEIGHT | MIT repository with model checkpoints; in-repo experimental wrapper exists | `algo=crowdnav_height`, implemented but experimental | Heterogeneous spatio-temporal graph over humans, robot, and static obstacles; model-only wrapper reconstructs source dict inputs | Discrete delta-v/delta-theta actions accumulated statefully into `unicycle_vw`; known projection/domain mismatch risks | `existing experimental representative` |
| SoNIC / GenSafeNav | MIT repositories with checkpoints; model-only wrapper exists | `sonic_crowdnav`, `gensafenav_*`, guarded variants are experimental/prototype | Prediction and ACI/conformal uncertainty surfaces; #1366 keeps source-faithful claims source-side-reproduction first | Holonomic velocity projected to `unicycle_vw`; guarded aliases are explicit mixed-mode variants | `source-side reproduction first` |

## Decision

Verdict: `no new first integration from this family yet`.

For #1355, the single current family verdict should be:

* use `crowdnav_height` as the existing Robot SF attention-graph representative only when an
  experimental, model-only, adapter-heavy baseline is acceptable;
* do not start a second CrowdNav++ / IGAT, DS-RNN, RGL, or SARL wrapper until the shared graph/social
  observation adapter issue (#1365) and learned-policy checklist (#1363) land;
* keep GenSafeNav / SoNIC behind the #1366 source-side reproduction and calibration/leakage
  boundary before making source-faithful conformal or constrained-RL claims.

This is intentionally not a rejection of the literature family. It is a sequencing decision:
Robot SF already has one implemented experimental representative (`crowdnav_height`) and one
SoNIC-compatible model-only adapter family. Adding another wrapper before resolving source-harness,
graph-history, and prediction-field contracts would produce more adapter ambiguity than benchmark
signal.

## Why No New First Integration

CrowdNav / SARL is the historical anchor, but lacks a clear bundled trained-policy path in the
current evidence and is better covered by prior source-harness notes.

RGL adds model-based lookahead and human-motion prediction. That raises the same future-trajectory
classification burden as CrowdNav++ / IGAT without offering an in-repo wrapper or current source
proof.

DS-RNN has useful example weights and a permissive license, including a unicycle example, but it
adds recurrent hidden-state and reset-semantics burden on top of the usual CrowdNav observation
adapter.

CrowdNav++ / IGAT is scientifically interesting because it explicitly predicts pedestrian
intentions, but that is also the fairness risk: any Robot SF wrapper must prove whether predicted
future trajectories are inferred from past/current observations, train-only labels, or forbidden
ground-truth future. Since HEIGHT already represents the same author lineage and legacy stack, IGAT
should not be first unless the research question is specifically an intention-prediction ablation.

HEIGHT remains the current ceiling representative in Robot SF because it is already implemented and
metadata-covered, but it is not benchmark-promoted. The known issues are still material: upstream
training/domain mismatch, simplified Robot SF observation reconstruction, discrete action
projection, and poor prior Robot SF performance.

SoNIC / GenSafeNav has the most runnable model-only asset path, but #1366 keeps its conformal and
constrained-RL semantics at `source-side reproduction first`. The existing wrappers are useful
experimental adapter evidence, not source-faithful benchmark proof.

## Follow-Up Rules

Safe next steps:

* let #1355 cite this note as the family-level verdict instead of opening multiple first-wrapper
  tasks;
* let #1359 record the non-selected candidates as monitor/prototype entries;
* use #1365 to define a reusable graph/history observation adapter before revisiting DS-RNN,
  CrowdNav++ / IGAT, or RGL;
* use #1366 to gate any source-faithful SoNIC / GenSafeNav promotion.

Do not:

* treat `crowdnav_height` performance as the final quality verdict for the whole CrowdNav family;
* treat `gensafenav_*` model-only wrapper results as source-faithful conformal-uncertainty evidence;
* use `truth` future prediction modes, current-episode future labels, or evaluation-seed calibration
  as Robot SF policy inputs;
* open multiple CrowdNav-family implementation tasks before one shared adapter contract exists.

## Validation

Commands run for this assessment:

```bash
rg -n "CrowdNav|CrowdNav\\+\\+|HEIGHT|DS-RNN|DSRNN|RGL|SARL|ST2|Social-STGCNN|Gumbel|SoNIC|GenSafeNav|issue #?600|#600|#601|#760|#770" docs/context docs/benchmark_planner_family_coverage.md robot_sf tests scripts configs -g '!output/**'
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
uv run pytest tests/planner/test_crowdnav_height.py -q
uv run pytest tests/benchmark/test_algorithm_metadata_contract.py -k 'crowdnav_height or sonic_crowdnav or gensafenav or guarded' -q
uv run pytest tests/benchmark/test_map_runner_utils.py -k 'crowdnav_height or sonic_crowdnav or gensafenav' -q
```

Results: located prior family notes, current wrapper source, metadata, and benchmark-family matrix
entries; docs proof passed for the four changed files; whitespace diff check passed; targeted tests
passed with `5 passed, 3 skipped`, `7 passed, 20 deselected`, and `8 passed, 75 deselected`.

No source-harness smoke was run for this issue because the decision is docs-only consolidation. The
currently runnable GenSafeNav / SoNIC probe surface is tracked separately by #1366, so this issue
does not add another source-harness result. No Robot SF benchmark run is required for this verdict.
