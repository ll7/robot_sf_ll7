# Issue 770 IGAT / ST2 Attention-Based RL Assessment

Date: 2026-04-12
Related issues:
- `robot_sf_ll7#770` Assess attention-based RL successors to HEIGHT
- `robot_sf_ll7#742` Awesome robot social navigation mining
- `robot_sf_ll7#629` Planner zoo deep research
- `robot_sf_ll7#601` CrowdNav family feasibility spike

## Goal

Determine whether IGAT (Intention Aware Robot Crowd Navigation with Attention-Based Interaction
Graph) or ST2 (Spatial-Temporal Transformer-style successors) can serve as a better benchmark
candidate than CrowdNav_HEIGHT, or whether HEIGHT remains the ceiling representative for this
family in the current benchmark stack.

## Candidates assessed

### IGAT — Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph

- **Upstream repo:** https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
- **License:** MIT
- **Pretrained checkpoints:** Yes, `trained_models/` folder in the upstream repo
- **Runnable test path:** Yes, `test.py` and visualization scripts
- **Python / dependency stack:** Python 3.6–3.8, PyTorch 1.12.1, OpenAI Baselines, Python-RVO2
- **Paper:** ICRA 2023 — intention prediction layered on an attention graph

**Observation contract:**
Robot pose / velocity / goal / radius plus per-pedestrian pose / velocity / radius. Uses a
Gumbel Social Transformer for intent prediction over the pedestrian set. Structured state packing
is the same CrowdNav-lineage contract that HEIGHT uses.

**Action contract:**
Continuous velocity control inheriting the CrowdNav lineage. Likely delta-v / delta-theta outputs
rather than direct `unicycle_vw`. Explicit projection to Robot SF `unicycle_vw` required.

**Integration tier decision:** `prototype only`

**Rationale:**
IGAT is a direct CrowdNav_HEIGHT sibling from the same author (Shuijing Liu). The architectural
difference is an intent-prediction module layered above the same interaction-graph backbone.
Integration burden, legacy Python stack, and adapter shape are identical to HEIGHT. There is no
reason to prefer IGAT over the existing `crowdnav_height` wrapper unless a source-harness
comparison with an intent-prediction ablation is specifically required. Do not add a second
wrapper without that concrete research motivation.

---

### ST2 — Spatial-Temporal Transformer

- **Upstream repo:** Not found publicly
- **License:** Unknown — behind IEEE paywall, no open-access or GitHub release identified
- **Pretrained checkpoints:** Not available
- **Runnable test path:** None
- **Python / dependency stack:** Unknown
- **Paper:** IEEE RA-L 2023

**Integration tier decision:** `do not pursue now`

**Rationale:**
No public implementation exists. The method cannot be assessed for observation contract,
adapter burden, or source-harness reproducibility. There is nothing to wrap.

---

## Comparison with HEIGHT

| Dimension | CrowdNav_HEIGHT | IGAT | ST2 |
|---|---|---|---|
| Repo available | Yes (MIT) | Yes (MIT) | No |
| Checkpoints available | Yes | Yes | No |
| Runnable test path | Yes | Yes | No |
| Legacy Python stack | Python 3.6–3.8 | Python 3.6–3.8 | Unknown |
| Observation contract | CrowdNav structured state | CrowdNav structured state + intent | Unknown |
| Action contract | delta-v/delta-theta | delta-v/delta-theta | Unknown |
| Adapter burden | Medium (existing wrapper) | Medium (would duplicate HEIGHT wrapper) | N/A |
| Benchmark surface result | Failed: success 0.24 → 0.12, collisions 0.11 → 0.57 | Untested; no reason to expect better | Untested |
| Integration tier | prototype only (implemented) | prototype only (not pursued) | do not pursue now |

## Decision

**Neither IGAT nor ST2 is a better benchmark candidate than HEIGHT for the current benchmark
surface.**

- IGAT is the same architectural tier, same legacy stack, same adapter shape, and same failure
  risk as HEIGHT. Adding a second wrapper buys no new information unless an explicit intent-prediction
  ablation is the research goal.
- ST2 cannot be evaluated without a public implementation.

**HEIGHT remains the ceiling representative** for this attention-based family in the current repo.
The open question from issue 770 — whether a newer attention method is actually better — cannot
be answered from IGAT (too similar) or ST2 (no code). If the paper-facing argument needs a
stronger attention-family anchor, the next honest step is either:

1. Run IGAT source-harness parity against HEIGHT on the same scenario set (same author, same family,
   controls for implementation quality), or
2. Wait for a public ST2 or equivalent implementation with checkpoints before assessing.

## Recommendation for `docs/benchmark_planner_family_coverage.md`

Add IGAT to the External Family Anchors table as `conceptually adjacent only` with a pointer to
this note. Do not add ST2 until a public repo exists.

## Observation and action contract translation (IGAT)

| Contract area | Source expectation | Robot SF supply/target | Judgment |
|---|---|---|---|
| observation | CrowdNav-lineage structured robot/human state + Gumbel Social Transformer intent prediction | Robot SF structured state (same fields, different packing path) | direct compatibility: no; adapter required |
| action | delta-v / delta-theta velocity control | Robot SF `unicycle_vw` | direct compatibility: no; post-policy adapter required |
| intent prediction module | requires pedestrian trajectory history or stochastic intent samples | Robot SF does not currently supply pedestrian trajectory history to planners | additional adapter surface: yes |

The intent prediction module adds a new adapter requirement beyond what the HEIGHT wrapper already
handles: pedestrian trajectory history or stochastic intent samples must be supplied. This is an
additional reason not to treat IGAT as a drop-in HEIGHT successor without explicit source-harness
work.
