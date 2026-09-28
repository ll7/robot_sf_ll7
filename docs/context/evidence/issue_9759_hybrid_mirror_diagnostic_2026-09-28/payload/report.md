# Issue #9759 hybrid reflection matrix

Status: partial, diagnostic-only. The transformed scenes are counterfactual diagnostics and do not support official release metrics, arm ranking, performance, safety, causal, or universal trace-symmetry claims.

## Scope and provenance

The matrix used the 48 scenarios in configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml, seeds 111–140, a 600-step horizon, pinned hybrid v3 and v4 configs, and base, horizontal-reflection x, and vertical-reflection y axes. The plan contained 8,640 rows: 48 scenarios × 30 seeds × 2 arms × 3 axes.

Only the current producer output is counted. Producer commit: 4fc1a1d4ff793065de717b45d9769a3225653a6a; producer base: d0acb1c0d5e9e293165bafa4dbc95a4b34d9e255; input digest: 723ed478f69268aeb629fec7dc97c38e439909422666c9f335be45a2c2be9cc7; scenario-matrix digest: d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5. The v3 config digest is 905ec25e24b3d5cedee1508ac6ad20a09f913d368c99d4329c839bc359640935 and v4 config digest is 0a4bc7115294c71094d42b799d1d56c8ddb4884aca69e3cea3a63f26f521ba5d. Earlier output from a different producer revision was excluded.

## Coverage and exclusions

All 8,640 planned row identities are present with no duplicate or missing identities. Of these, 8,040 eligible rows have valid terminal outcomes; 600 rows are explicit preflight exclusions. Coverage is partial, not complete. Eligible status counts are success 7,067, collision 302, terminated 531, and max_steps 140.

Ten reflected scenario/axis preflight cells failed closed:

- classic_bottleneck_low, classic_bottleneck_medium, and classic_bottleneck_high, each on x and y: a post-loader pedestrian point at (21.801584, 42.135078) is outside [0, 40] × [0, 40].
- classic_overtaking_low and classic_overtaking_medium, each on x and y: a point at (60.0037192, 0.0032515661) is outside [0, 60] × [0, 27.69].

Each failed cell excludes 30 seeds × 2 arms, giving 600 rows. No coordinate clipping, transform fallback, or substitute run was used. All corresponding base rows remain present. There are 2,880 scenario/seed/arm pair groups. Each arm/axis has 1,290 valid paired comparisons; 150 comparisons per arm/axis are unavailable because of those explicit exclusions.

## Descriptive paired outcomes

Outcome agreement compares the complete outcome mappings. Step delta is reflected steps minus base steps; negative values mean fewer steps in the reflected episode.

| Arm | Axis | Comparable pairs | Outcome agreement | Disagreement | Mean step delta | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | x | 1,290 | 1,012 | 278 | -10.542 | -585 | 502 |
| v3 | y | 1,290 | 1,189 | 101 | 0.358 | -411 | 464 |
| v4 | x | 1,290 | 1,272 | 18 | -2.169 | -585 | 219 |
| v4 | y | 1,290 | 1,265 | 25 | 1.080 | -295 | 316 |

These aggregate results show nonzero reflected outcome disagreements for both arms, including v4. They do not identify the cause of those disagreements or attribute them to near-tie candidate selection. Trace symmetry is established only by the focused synthetic tests recorded in the implementation validation, not by these episode-level pairs. The v3 config and behavior were left unchanged.

## Integrity audit and limitation

An independent read-only audit and a local recomputation checked the 8,640 row identities, eligible terminal validation, exclusions, paired identities, provenance joins, stored summary, and pairs. There were no duplicate or missing identities, eligible error rows, or provenance mismatches; recomputed summary and paired output matched the stored artifacts. A fresh preflight replay matched map digests, statuses, failed cells, and overlap predicates. Nine robot-pedestrian clearance diagnostic floats differed on replay; the overlap predicates and eligibility decisions stayed the same, and eligibility uses the overlap boolean.

The full episodes, pairs, preflight data, run metadata, and logs remain in the private local run directory recorded in private_artifact_hashes.json. They are not committed. Hashes provide correspondence to the local artifacts but do not make those artifacts remotely retrievable. The run logs were not inspected for redaction, so the accompanying agent manifest leaves trace_redaction_checked false.

## Claim boundary

This is a partial, diagnostic-only counterfactual matrix. It records all unsupported reflected cells and completed eligible outcomes. It is not evidence that hybrid v4 is universally reflection-invariant, does not establish planner performance or safety, and is not official release benchmark evidence.
