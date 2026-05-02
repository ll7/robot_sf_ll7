# Issue #600 DSRNN Stretch Follow-up

Date: 2026-03-30
Related issues:

* `robot_sf_ll7#600` DSRNN stretch follow-up
* `robot_sf_ll7#601` CrowdNav family feasibility spike
* `robot_sf_ll7#599` Go-MPC feasibility assessment
* `robot_sf_ll7#604` Pred2Nav assessment
* `robot_sf_ll7#629` planner-zoo deep research

## Paper-Side Reference Context

This issue sits inside the broader paper-side planner reference set recorded in
`docs/context/issue_603_alyassi_reference_set_2026-03-06.md` .

Primary external anchors in that set:

* `CrowdNav`
  + citekey: `chenCrowdRobotInteractionCrowdaware2019`
  + repo: <https://github.com/vita-epfl/CrowdNav>
  + local clone: `output/repos/CrowdNav`
* `CrowdNav DSRNN`
  + citekey: `liuDecentralizedStructuralRNNRobot2025`
  + repo: <https://github.com/Shuijing725/CrowdNav_DSRNN.git>
  + local clone: `output/repos/CrowdNav_DSRNN`
* `Go-MPC`
  + citekey: `britoWhereGoNext2021`
  + repo: <https://github.com/tud-amr/go-mpc>
  + local clone: `output/repos/go-mpc`
* `Pred2Nav` (stretch)
  + citekey: `poddarCrowdMotionPrediction2023`
  + repo: <https://github.com/sriyash421/Pred2Nav.git>
  + local clone: `output/repos/Pred2Nav`

Safety-aware anchor note:

* Keep `yaoSoNICSafeSocial2025` as the current SoNIC paper anchor.
* The verified project/repo ladder is still missing.
* Reported `arXiv:2511.07820` should not be silently substituted for the current SoNIC anchor
  without verification.

Tracked context notes:

* `amv_benchmark_paper/context/external_repos/robot_sf_ll7_alyassi_planner_reference_set_2026-03-06.md`
* `docs/context/issue_603_alyassi_reference_set_2026-03-06.md`

## Why this remains stretch work

The repository already has a credible first attention-family anchor in `CrowdNav` and a separate
prediction-family backlog via `Go-MPC` and `Pred2Nav` . A DSRNN-style family should stay behind
those earlier spikes because it adds more than one new burden at once:

* graph-structured observation packing beyond the original CrowdNav joint-state adapter burden, 
* recurrent hidden-state handling across time, 
* an older runtime stack centered on Python 3.6 and OpenAI Baselines, 
* and a stronger need to prove source-harness parity before any Robot SF wrapper can be treated as
  family-faithful.

That makes issue 600 a sequencing and provenance note, not an implementation commitment.

## Canonical source anchors

Use these upstream assets as the canonical DSRNN-family anchor:

* Upstream repository: <https://github.com/Shuijing725/CrowdNav_DSRNN>
* Canonical local clone: `output/repos/CrowdNav_DSRNN`
* Paper/repo README: `output/repos/CrowdNav_DSRNN/README.md`
* Source test entrypoint: `output/repos/CrowdNav_DSRNN/test.py`
* Source config surface: `output/repos/CrowdNav_DSRNN/crowd_nav/configs/config.py`
* License: `output/repos/CrowdNav_DSRNN/LICENSE` (`MIT`)

Observed upstream facts from the checked-out clone:

* README positions the method as `Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning`.
* The repo advertises example checkpoints for both holonomic and unicycle robots.
* The tested upstream environment is Python 3.6 with OpenAI Baselines and `Python-RVO2`.
* The source test path imports its own simulator, vectorized env factory, and model stack rather
  than exposing a thin planner-only module.

## Dependency ordering

Issue 600 should remain behind these earlier steps:

1. `robot_sf_ll7#601`: establish the first attention-family feasibility note and source-harness
   strategy boundary.
2. `robot_sf_ll7#599` and `robot_sf_ll7#604`: finish the first prediction-family assessment path so
   DSRNN does not compete with higher-value breadth work.
3. Any future attention-family implementation issue: prove a source-harness run for one upstream
   policy before attempting a Robot SF wrapper.

If those dependencies are not met, the safe classification for DSRNN stays `assessment only` .

## Additional integration burden beyond the first attention spike

| Area                 | First attention-family spike ( `CrowdNav` / `SoNIC` )              | Extra DSRNN burden                                                                                 | Why it matters here                                                                      |
| -------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Observation contract | Joint robot/human state packing with source-specific normalization | Structural graph inputs plus temporal neighborhood state                                           | Robot SF would need explicit graph/history reconstruction, not just flat state remapping |
| Stateful execution   | Mostly policy inference with adapter-side state packing            | Recurrent hidden-state carry and reset semantics                                                   | Wrapper behavior becomes episode-stateful and easier to invalidate silently              |
| Runtime stack        | Already adapter-heavy and simulator-bound                          | Adds older Python 3.6 + OpenAI Baselines assumptions                                               | A side environment or isolated harness is likely required before any integration claim   |
| Action semantics     | Source simulator interprets model-native actions                   | Same simulator-native action boundary plus recurrent graph policy context                          | Post-policy projection alone is not enough evidence of family-faithful behavior          |
| Provenance risk      | Source-harness parity is already required                          | DSRNN is a descendant-style extension of CrowdNav, so wrapper drift is even harder to reason about | The repository should not present a local reimplementation as DSRNN-family support       |

## Recommended integration shape

Decision:

* Integration category: `assessment only`
* Preferred next proof: `source-harness reproduction first`
* Wrapper recommendation: `do not start until a source-harness run is proven in an isolated side environment`
* Fallback policy: `fail fast only`

Interpretation boundary:

* The current repository can safely cite DSRNN as a roadmap family anchor with a permissive
  upstream license and a visible source test path.
* The current repository cannot yet claim DSRNN-family benchmark support, prototype support, or
  wrapper readiness.
* Future work should only be promoted after source-harness evidence and a documented observation /
  action translation plan exist.

## Concrete next step when this leaves stretch status

Open a dedicated implementation issue only after the earlier attention/prediction spikes land. That
issue should be limited to:

1. create an isolated side environment matching the upstream runtime assumptions,
2. run the upstream `test.py` path with one bundled checkpoint, 
3. document the exact observation tensor, recurrent-state, and action contracts,
4. decide whether Robot SF can support a thin wrapper without turning the result into a local
   reimplementation.
