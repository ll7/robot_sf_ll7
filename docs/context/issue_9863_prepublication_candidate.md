# DOI-free prepublication candidate (#9863)

## Goal and boundary

Provide a checksummed input for the 0.0.8 setup preflight before an author reserves a tag or DOI. The candidate is diagnostic: it cannot be passed to the canonical release loader, publication tooling, or Chapter 7 claim admission.

## Execution plan

1. Define a distinct candidate schema and loader that require an exact source commit, the 14-arm campaign config, 48 scenario identities, seeds 111–140, H600, and 20,160 planned episodes.
2. Require SHA-256 pins for the campaign config, scenario matrix and its includes, every effective SVG map, seed-set file, route certification, suite policy, every referenced planner config, and policy-search base configs selected by that manifest or a scenario algorithm override. Reject missing, extra, stale, or symlinked inputs.
3. Connect only the #9819 setup preflight to this loader. Keep `load_release_manifest` strict for published manifests. The preflight report records the candidate digest and source commit; its checker implementation is bound by that exact source commit. Recheck the complete pinned input closure after the run and invalidate the report on drift. The report remains diagnostic until all setup cells pass.
4. Test valid DOI-free loading, hash drift, omitted pins, roster/count drift, and publication-loader rejection. Review at the final head and run focused validation before delivery.

## Stop rule and custody

No nominal evaluation follows a partial, failed, fallback, or blocked preflight. Preserve the candidate and report with checksums in durable campaign custody; worktree-local `output/` is only scratch. The historical 0.0.7 inputs and publication coordinates remain unchanged. Promotion to a tagged manifest is a separate author-reserved step after candidate admission and paired release comparison.

After the corrected campaign config and its input files are committed at one exact source head, create a new ignored candidate path with:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark/build_prepublication_candidate.py \
  --campaign-config configs/benchmarks/<corrected-campaign>.yaml \
  --suite-policy configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1_suite_policy.yaml \
  --route-certification configs/benchmarks/route_clearance_certifications_v1.yaml \
  --candidate-id 0.0.8-prepublication-<source-short-sha> \
  --output output/release_candidate/<source-short-sha>.json
```

The builder requires a clean tracked checkout and refuses to overwrite an existing candidate. The loader checks the exact HEAD, ordered 14-arm roster against an explicit 0.0.8 mapping, and the 48 scenario identities against the byte-pinned 0.0.7 identity matrix. The four v4-named hybrid slots must bind their matching v4 release configs and v4 base config. The only currently approved algorithm switch in those configs is `francis2023_leave_group` to ORCA with its versioned ORCA base in the two scenario-adaptive arms; a scenario override back to historical v3 behavior is rejected even if its bytes are jointly pinned. It also checks seeds 111–140, H600/20,160 cells, template publication slots, and the SHA-256 closure of matrix includes, effective maps, top-level planner configs, policy-search base configs, and other referenced inputs. Reconcile the four successor keys, config paths, approved override roster, and tuning freeze with #9751's final reviewed head before admitting a physical candidate. External model checkpoint bytes and runtime model identity still require the separate release provenance gates; this setup candidate does not certify them. The distinct schema is rejected by `load_release_manifest`; a candidate cannot act as a tagged release manifest. Keep the printed candidate digest with the preflight report and copy both to durable custody.

Run the pinned setup preflight with the resulting path as `--manifest`, supplying unique JSON and Markdown output paths:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m robot_sf.benchmark.spawn_preflight \
  --manifest output/release_candidate/<source-short-sha>.json \
  --json-output output/release_candidate/<source-short-sha>.preflight.json \
  --markdown-output output/release_candidate/<source-short-sha>.preflight.md
```

The CLI exits nonzero for blocked or invalid inputs. A zero exit means only that the setup matrix passed its checks; the report is still diagnostic, and the remaining release gates decide whether nominal evaluation may start.
