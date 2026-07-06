# Assurance Fragments

[Back to Documentation Index](./README.md)

Assurance fragments make a benchmark campaign's claim, argument, and evidence links explicit in
machine-readable form without changing the benchmark result.

Robot SF writes a Goal Structuring Notation (GSN)-flavored assurance fragment for camera-ready
campaign reports. The fragment is an audit artifact: it records which campaign files support each
claim, which assumptions remain unverified, and how release-gate outcomes are attached to argument
nodes. It does not promote diagnostic evidence to a benchmark, release, paper, or safety claim by
itself.

## Outputs

Camera-ready campaign export records these paths in `reports/campaign_summary.json`:

| Artifact | Purpose |
| --- | --- |
| `reports/assurance_fragment.json` | Schema-checked `assurance_fragment.v1` claim-argument-evidence tree. |
| `reports/assurance_fragment.md` | Human-readable Markdown rendering of the same tree. |
| `reports/assurance_fragment.svg` | Standalone visual rendering for review packets. |

The JSON schema lives at
`robot_sf/benchmark/schemas/assurance_fragment.schema.v1.json`. The builder and renderers live in
`robot_sf/benchmark/assurance_fragment.py`.

## Gate-To-Argument Mapping

Release gates become argument nodes only when the campaign export receives a release-gate report.
The mapping is intentionally mechanical so reviewers can trace from a campaign row to a checked
artifact.

| Input surface | Fragment node | Meaning |
| --- | --- | --- |
| Campaign metadata `scenario_matrix` and `scenario_matrix_hash` | `C_matrix` context | Names and hash-pins the scenario matrix used by the top claim. |
| Campaign metadata `git_hash` | `C_git` context | Records the source commit for the generated fragment. |
| Campaign `planner_rows[]` | `G_<planner_key>` goal | Creates one planner-level goal per planner row. |
| Campaign row `success_mean` and `benchmark_success` | `G_<planner_key>_success` goal | Records goal-reaching evidence from that planner row. |
| Campaign row `collisions_mean` | `G_<planner_key>_safety` goal | Records collision evidence from that planner row. |
| Campaign row `snqi_mean` when present | `G_<planner_key>_snqi` goal | Records the Social Navigation Quality Index score evidence for that planner row. |
| Campaign run `episodes_path` | `Sn_<planner_key>_episodes` solution | Hash-pins the episode log used by the planner metric goals. |
| Release-gate report provenance input | `Sn_release_gates` solution | Hash-pins the gate report input declared by the release-gate report. |
| Release-gate report `matrix_rows[]` for each matching `planner_key` | `G_<planner_key>_gates` goal | Records `safety_status`, `comfort_status`, and `overall_status`, then links that gate goal into `S_<planner_key>`. |

The top claim `G_root` stays campaign-scoped. It points to `S_campaign` plus context nodes, and
`S_campaign` decomposes the claim across planner-specific goals. Each `S_<planner_key>` strategy
links metric goals and, when available, the release-gate goal for the same planner key.

## Evidence Rules

Evidence leaves are `solution` nodes with `metadata.path` and `metadata.sha256`. The path is
repository-relative when the artifact is inside the repository; the digest is computed from the file
that existed when the fragment was exported. The current exporter records missing episode files with
an empty path or digest rather than inventing support. Treat such leaves as audit gaps, not as
supporting evidence.

Learned planners also receive Assuring Machine Learning for Autonomous Systems (AMLAS)-style
assumption nodes:

| Node | Default status boundary |
| --- | --- |
| `A_<planner_key>_train` | Training-data assumptions are stated, not verified by the campaign fragment. |
| `A_<planner_key>_deploy` | Deployment-context match is stated, not verified by the campaign fragment. |

Those assumptions keep learned-component assurance gaps visible even when the campaign metrics and
release gates are present.

## Validation Boundary

Use `validate_assurance_fragment()` to check the exported JSON against the v1 schema. A valid
fragment proves the claim-argument-evidence structure is well formed; it does not prove the
underlying benchmark should be interpreted as release evidence. Apply the normal benchmark evidence
rules from [Maintainer Values And Hard Contracts](./maintainer_values.md) and
[Benchmark Scenario And Model Governance](./benchmark_governance.md) before making benchmark,
release, or paper-facing claims.
