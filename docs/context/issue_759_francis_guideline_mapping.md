# Issue 759 Francis Guideline Mapping For Robot SF

Date: 2026-04-02
Related issues:
- `robot_sf_ll7#759` Distill Francis social-navigation evaluation guidelines into Robot SF benchmark contract checks
- `robot_sf_ll7#692` Scenario difficulty analysis and verified simple scenarios
- `robot_sf_ll7#750` Add interval-inclusive paper export handoff for Results statistical hardening
- `robot_sf_ll7#751` Run fixed-scenario multi-seed pilot and export paper-ready seed-variability evidence
- `robot_sf_ll7#691` Benchmark fallback policy

## Goal

Map Francis et al. section-by-section onto the current `robot_sf_ll7` benchmark contract so
maintainers, paper authors, and contributors can see where Robot SF already matches the guideline
intent, where coverage is only partial, and what remains intentionally out of scope.

This note is descriptive. It explains how the current benchmark maps to Francis; it does not adopt
the paper as a new repository policy document and it does not redesign the benchmark.

## Scope Boundary

This mapping covers the current Robot SF simulation benchmark surfaces:

- `docs/benchmark_spec.md`
- `docs/benchmark_camera_ready.md`
- `docs/benchmark_planner_family_coverage.md`
- the fail-closed benchmark policy and related benchmark-evidence issues

Out of scope for this note:

- human-subject evaluation
- real-world deployment studies
- subjective social acceptability studies
- redesigning scenario suites or metric implementations
- forcing Robot SF to satisfy Francis areas that the current repo does not claim to own

## Source

Primary source:

- Anthony Francis et al., *Principles and Guidelines for Evaluating Social Robot Navigation
  Algorithms*, ACM THRI 2025
  - paper page: <https://research.nvidia.com/publication/2025-02_principles-and-guidelines-evaluating-social-robot-navigation-algorithms>
  - PDF: <https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/25thri.pdf>

High-level paper structure used here:

- Section 3: principles of social navigation
- Section 4: scientific questions and evaluation lifecycle
- Section 5: taxonomy for benchmarks, datasets, and simulators
- Section 6: metrics
- Section 7: scenarios
- Section 8: benchmarks
- Section 9: datasets
- Section 10: simulators and interface unification

## Overall Verdict

Robot SF currently maps strongly to Francis on:

- reproducibility
- provenance and benchmark-boundary discipline
- explicit baseline and readiness framing
- fail-closed interpretation of degraded or unavailable planners

Robot SF maps only partially to Francis on:

- explicit metric-to-principle interpretation
- scenario-card style social rationale
- documented simulator and dataset taxonomy
- direct support for several social principles beyond safety/comfort proxies

Robot SF intentionally leaves the following Francis areas out of scope:

- human-subject evaluation
- real-world issue discovery and deployment studies
- benchmark claims that depend on subjective human feedback rather than the current artifact-driven
  simulation contract

## Section-By-Section Mapping

| Francis section | What the section asks evaluators to do | Robot SF mapping | Status | Evidence | Recommended action |
| --- | --- | --- | --- | --- | --- |
| Section 3: principles of social navigation | Evaluate social navigation against broad principles such as safety, comfort, legibility, politeness, social competency, agent understanding, proactivity, and contextual appropriateness. | Robot SF already treats benchmark claims conservatively and records strong safety/comfort-style proxies, but it does not claim that all Francis principles are directly measured. | `partial` | `docs/benchmark_spec.md`; `docs/benchmark_planner_family_coverage.md`; current metric surfaces in `docs/dev/issues/social-navigation-benchmark/metrics_spec.md` | Keep the benchmark descriptive about which principles are only indirectly proxied. Do not expand claims beyond what current metrics support. |
| Section 4: scientific questions and evaluation lifecycle | Be explicit about what scientific question a benchmark answers and where simulation fits in a broader lifecycle including field studies, lab studies, scenario development, and public benchmarking. | Robot SF is already explicit that it is a simulation benchmark with fixed scenario manifests, seeds, and planner comparisons. It is not framed as a human-study or field-deployment benchmark. | `partial` | `docs/benchmark_spec.md`; `docs/benchmark_camera_ready.md`; `docs/context/issue_692_scenario_difficulty_analysis.md` | Add a short reminder in this note and linked benchmark docs that Robot SF answers simulation-side comparability questions, not the full Francis lifecycle. |
| Section 5: taxonomy for benchmarks, datasets, and simulators | Use a clear vocabulary for what kind of benchmark, dataset, and simulator is being discussed and what factors differ across them. | Robot SF already distinguishes planner readiness, benchmark profiles, scenario manifests, and conceptual-adjacency vs implemented support. It is weaker on datasets/simulators taxonomy because those are mostly reference context, not active benchmark surfaces. | `partial` | `docs/benchmark_planner_family_coverage.md`; `docs/context/issue_742_awesome_robot_social_navigation_mining.md`; `docs/dev/benchmark_plan_2026-01-14.md` | No new issue by default. Keep dataset/simulator references descriptive unless the repo expands those surfaces materially. |
| Section 6: metrics | Define metrics clearly, distinguish objective vs subjective measures, record caveats, and avoid overstating what metrics prove. | Robot SF is strong here for simulation-side objective metrics: metric definitions, caveats, threshold provenance, schema validation, and aggregation checks are already explicit. Subjective human-evaluation metrics remain out of scope. | `satisfied` for objective simulation metrics, `out of scope` for subjective metrics | `docs/benchmark_spec.md`; `docs/dev/issues/social-navigation-benchmark/metrics_spec.md`; schema/provenance notes in the benchmark spec | Keep metric caveats close to the benchmark spec and continue separating direct evidence from interpretation. |
| Section 7: scenarios | Use scenarios intentionally, document what they represent, and avoid assuming a benchmark scenario set is complete just because it is reproducible. | Robot SF already has explicit scenario manifests and a scenario-difficulty analysis path, but it does not yet provide a Francis-style scenario-card mapping from each scenario family to social-navigation principles or target failure modes. | `partial` | `configs/scenarios/`; `docs/benchmark_spec.md`; `docs/context/issue_692_scenario_difficulty_analysis.md` | Treat scenario-principle mapping as a possible future note only if it becomes needed for review or publication. Do not open a new issue yet because `#692` already owns the current scenario-interpretation gap. |
| Section 8: benchmarks | Make benchmark scope, protocol, outputs, and interpretation rules explicit so algorithm comparisons are repeatable and fair. | This is a current strength of Robot SF: seed policy, profiles, artifacts, fallback policy, publication workflow, and campaign outputs are all documented and versioned. | `satisfied` | `docs/benchmark_spec.md`; `docs/benchmark_camera_ready.md`; `docs/context/issue_691_benchmark_fallback_policy.md` | Keep the benchmark-facing docs linked and conservative. No new issue justified from this section alone. |
| Section 9: datasets | Be explicit about dataset provenance, collection assumptions, and what datasets can or cannot say about social navigation. | Robot SF has some dataset-like artifact lineage for training and imitation workflows, but the main benchmark is not a dataset benchmark and does not currently position itself as one. External social-navigation datasets are mostly tracked as references. | `out of scope` for the main benchmark, `partial` for auxiliary training artifacts | `docs/imitation_learning_pipeline.md`; `docs/context/issue_742_awesome_robot_social_navigation_mining.md` | No new issue. Keep external dataset discussion descriptive and separate from benchmark claims. |
| Section 10: simulators and interface unification | Make simulator differences legible and ideally provide a common evaluation interface across simulator backends. | Robot SF already normalizes benchmark outputs and provenance inside its own stack, but it does not attempt broad cross-simulator interface unification beyond explicit adapter metadata and conceptual reference notes. | `partial` | `docs/benchmark_spec.md`; `docs/benchmark_camera_ready.md`; `docs/benchmark_planner_family_coverage.md` | Mention overlap with SocNavBench/BARN/DynaBARN/Arena-Rosnav as reference traditions, but do not open a simulator-unification issue unless the repo actually expands into multi-simulator execution. |

## Metric-To-Principle Mapping

This table is intentionally conservative. It records what the current benchmark can plausibly proxy,
not what it proves in a full human-centered sense.

| Francis principle | Current Robot SF proxy | Coverage | Main limitation |
| --- | --- | --- | --- |
| Safety | `success`, `collisions`, `near_misses`, `min_distance`, command-feasibility metadata, fail-closed fallback policy | `strong` | Still simulation-side safety only; does not measure human-perceived safety directly. |
| Comfort | force/comfort metrics, jerk, curvature, energy, `comfort_exposure`, force quantiles | `moderate to strong` | Comfort is approximated analytically; no subjective human comfort validation. |
| Legibility | indirect only through path shape, collision avoidance behavior, and scenario outcomes | `weak` | No direct metric for how understandable the robot's intent is to nearby humans. |
| Politeness | indirect only through clearance, force, and some near-miss behavior | `weak` | Current metrics do not directly capture yielding norms or social deference. |
| Social competency | indirect aggregate effect of safety + comfort + success across scenario families | `weak` | Too broad to infer from current metrics without overclaiming. |
| Agent understanding | none directly | `weak` | No explicit metric for how well a planner models or predicts human intent beyond internal algorithm design. |
| Proactivity | partially visible in time-to-goal, success, and some scenario outcomes | `weak to moderate` | Current metrics do not isolate proactive versus reactive behavior cleanly. |
| Contextual appropriateness | partially visible through scenario-conditioned outcomes and planner-family caveats | `weak` | No explicit principle-to-scenario contract that says what “appropriate” means in each context. |

## Reviewer Checklist

Use this checklist when reviewing a benchmark-facing change, report, or paper-ingestion claim against
the Francis-style guidance that the current Robot SF benchmark is actually able to support.

- Is the benchmark scope explicit about being simulation-side rather than human-study or field-study evidence?
- Are the scenario manifests and seed policy named explicitly?
- Are baseline categories and planner readiness boundaries explicit?
- Are metric definitions and caveats linked or restated accurately?
- Are fallback, degraded, or unavailable planner paths excluded from benchmark-success claims?
- Are provenance fields, config hashes, and run metadata recorded or linked?
- Are uncertainty/statistical surfaces present, or is the limitation stated explicitly?
- Does the text distinguish current implemented support from conceptual adjacency or external anchors?
- If a social principle is mentioned, is it backed by a direct metric, an indirect proxy, or only a qualitative interpretation?
- If scenario-level claims are made, do they rely on current benchmark evidence rather than assumed social realism?
- If a gap is identified, is it already owned by an open issue such as `#692`, `#750`, or `#751`?

This checklist is written so it can later be moved into a Codex skill or review workflow without
changing the underlying descriptive conclusions of this note.

## Overlap With Existing Benchmark Traditions

The current Robot SF benchmark already overlaps more strongly with reproducibility-oriented benchmark
traditions such as BARN, DynaBARN, Arena-Rosnav, and SocNavBench than with the human-study-heavy
parts of the Francis lifecycle:

- explicit protocol and artifacts
- baseline comparability
- seed-controlled execution
- structured outputs and provenance
- conservative scope boundaries

That overlap is useful context, but it does not replace the Francis-specific mapping above.

## Current Gaps That Are Already Owned Elsewhere

This audit does not justify reopening the following items under new wording:

- scenario difficulty and verified-simple interpretation:
  - `#692`
- publication-facing uncertainty/export handoff:
  - `#750`
- fixed-scenario seed-variability evidence:
  - `#751`

These issues already own several of the most plausible Francis-shaped follow-ups.

## Follow-Up Recommendation

Recommendation: no new issue is required immediately from this Francis mapping pass.

Why:

- the strongest benchmark-contract elements are already present,
- the most visible partial gaps are either intentionally out of scope or already owned by open
  benchmark-evidence issues,
- and the main value of this work is to make current coverage and limitations legible in one place.

If a future follow-up is needed, the best candidate would be a narrow scenario-card or
metric-to-principle traceability note, but that is not yet justified as a separate issue.
