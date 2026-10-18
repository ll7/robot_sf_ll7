# Guard/escape replication preregistration

This document is frozen before any replication episode. Registration proof is the
first pushed commit containing this file and the inputs, on the draft PR. A later
receipt records that commit, successful push completion UTC, remote readback and
first acquisition start UTC. No replication episode has run at registration.

## Hypotheses and design

Primary H_G: guard-on increases success in static constriction and portal.
Primary H_E: escape-on increases success in the same two groups.
Crowding control is the negative control for each mechanism. Report all six
success main effects and the interactions, with all original endpoints.
The fixed roster has 8 static-constriction, 7 portal and 3 crowding-control cells.
Use only fresh development seeds 1031–1060, all 4 arms paired within each complete
seed block: 2,160 episodes, 540 per arm. No training or held-out seeds are used.

## Decision rule (verbatim from the original review source note)

Pre-registered stop rule (written before the result): outcome = paired seed-block success difference (on minus off) per confinement group, for guard and for escape. EFFECT if |diff| >= 5 pp AND the interval excludes 0 in at least one confinement group -> keep three RQs with a feature answer scoped to that controller. NO EFFECT -> fold RQ3 into RQ2. MIXED, or the control group moves as much -> fold. No rescue reanalysis.

Interpretation frozen here follows the independent review: “guard and/or escape”
is OR across mechanisms; one qualifying mechanism and one nonqualifying mechanism
is not by itself MIXED. Apply the numerical rule to each factor separately.
A group qualifies when the absolute success main effect is at least 5.0 pp and
its pointwise 95% interval excludes zero. Report direction, including any effect
opposite to the positive hypotheses. A factor is MIXED if qualifying confinement
groups have opposite signs. For a factor, “control moves as much” means the
absolute control point estimate is at least as large as every qualifying
confinement point estimate. Such a factor cannot supply a confinement-specific
feature answer. Overall EFFECT requires at least one qualifying, non-MIXED
factor whose control does not move as much; otherwise NO EFFECT, or MIXED when
qualifying opposite directions prevent a consistent answer. Report both groups
individually; no outcome-selected subgroup, threshold, seed extension or rescue.
The rule's keep/fold consequence is a recommendation only, with no thesis edit
or evidence-admission action authorized by this packet.

Secondary pre-declared reading: an escape effect counts as “bounded below five
points” only if its 95% upper bound is below 5.0 pp, assessed separately in each
confinement group. Failure to qualify under the primary rule is not equivalence.

## Frozen execution and analysis

Execution source is 0035fba05c9b679e39f58ec286f5d57b96bb9a0a, as required by the
original README. Same controller, arm configs, scenario roster, authored budgets,
dt, adapter and runtime as the original. Four workers, CPU only; each numerical
library limited to one thread. Check load before starting. No smoke, identity or
pilot simulation; execute the complete registered run once. Existing original
behavior receipts support the unchanged runtime; they are not new replication runs.
Python 3.13.14 and NumPy 2.4.6, using the original acquisition environment.

Use the original analyzer unchanged (byte-identical copy in `analyze.py`), with
10,000 whole-seed-block draws, NumPy default_rng(1001), equal-cell weighting,
on-minus-off main effects averaged across the other factor, interaction
Y11−Y10−Y01+Y00, percentile 2.5/97.5 intervals. All intervals are pointwise and
conditional on the fixed cell roster. No p-values or multiplicity-adjusted claim.

The archived analyzer hard-codes seed labels 1001–1030. A deterministic input
adapter maps only analysis labels 1031–1060 to 1001–1030 by subtracting 30; it
changes no outcome, ordering, pairing, factor or numerical method. Raw and compact
producer evidence retain actual fresh seeds. Use a temporary analysis view;
retain its digest and mapping receipt. The output adapter corrects seed identity
metadata and removes the inherited original-review verdict. No original-data
rows enter the replication primary estimate. The unchanged original acquisition
analyzer is a crosscheck on the same label view. Frozen headline comparisons in
its portable CLI are original-study checks, so verify reproduction through
byte comparisons of reruns rather than its original headline --verify check.

`inputs/study.yaml` is byte-identical to the original except its primary seed
list. Retrospective packet fields, smoke and identity fields describe original
provenance, not newly authorized runs; this preregistration supersedes their
packet-role and registration-limit descriptions. Runtime input copies change only
seeds and packet/output paths (including campaign name). All four arm YAMLs are
byte-identical. Source note and review are linked above.

## Stopping, validity and preservation

Run all 2,160 episodes regardless of observed effects. No exclusions beyond the
original checks: reject missing/duplicate slots, invalid endpoints/termination or
budgets, source/factor mismatch, fallback/degraded episodes, wrong adapter or
nonregistered seeds. Any invalid acquisition blocks primary success evidence;
report it, without silent exclusion or additional-seed replacement.
Verify zero fallback/degraded episodes and differential_drive execution through
HybridRuleLocalPlannerAdapter, with the original runtime fallback parser.

Keep endpoint-complete compressed evidence, source line/file digests and configs
in Git using the original compact pattern. Keep the full run outside Git, with a
SHA-256 manifest; upload it once in-process to W&B, with no credential output.
Retain registration receipt, environment identity, command, timestamps and logs.

The requested pooled 60-seed estimate is secondary and is not a preregistered
primary analysis or a rescue. It uses all original and replication seed blocks,
the same estimands and 10,000-block-bootstrap method/default_rng(1001), with 60
blocks sampled per draw. It cannot alter the primary decision. No merge, release,
claim admission, frozen input mutation or dissertation editing.
