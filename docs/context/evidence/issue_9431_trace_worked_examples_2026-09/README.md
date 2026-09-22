# Issue #9431 corrected 0.0.7 worked-example traces

Plain-language summary: these four tracked trace bundles are the dissertation's
bounded worked-example slice regenerated from the corrected 0.0.7 source. They
are not a population estimate, planner ranking, or causal ablation.

- source commit: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- benchmark release tag: `paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- trace job: Slurm `15716`
- trace campaign: `issue9431_worked_traces_0_0_7_20260922`
- input config SHA-256: `dcaf87a46182e1cb64283a7b2442b02ce5fec86d33e9d72df2bcc6a04ee19806`
- scenario/seed slice: `classic_head_on_corridor_medium` and
  `classic_group_crossing_medium`, seeds `22`, `23`, `24`
- durable package: `ll7/robot_sf/campaign-issue9431_trace_package_0_0_7_20260922_v2:v0`
- preservation receipt SHA-256:
  `c4baca06392978c505d26d298b8ba15d6b9d8d7ce2b30ce5660d6df79e86a382`
- package manifest SHA-256:
  `6c7bae4b052ecadd3eda989eaebd9bb7346c31575212224f1e71a4933905f055`
- package `SHA256SUMS` SHA-256:
  `6469a5f37102fc776704d88aad1b7cd5ab1ec52fcb083f62e37eae76df1808b3`

The head-on figure uses ORCA seed 24 (completion) and social-force seed 23
(timeout): they are two requested worked-example seeds, not a matched-seed
counterfactual. The group-crossing figure uses goal and social-force seed 22.
Every bundle retains the upstream exporter claim boundary and its own checksums.
