# Goal

Close the implementation and audit portion of #9750 by checking every arm in the actual 0.0.7
release roster against its source method and the release robot geometry, then version physical
corrections for the 0.0.8 candidate without changing historical defaults.

# Scope

- In scope: source-bound 0.0.7 config audit, current-code correctness fixes, versioned 0.0.8
  configs, per-arm hand-built oracle tests, and linked audit documentation.
- Out of scope: changing 0.0.7 artifacts, executing evaluation episodes, freezing #9751's final
  hybrid roster, campaign submission, release tagging, or publication.
- Acceptance: one documented row per actual 0.0.7 arm, with reference, deviations/classification,
  resolved config identity and physical limits; targeted tests prove the corrected physical cases;
  defaults remain byte-compatible when new selectors are absent.

# Evidence sources

- #9750 and its current comments; #9668, #9751, and linked defect issues.
- Accepted 0.0.7 publication bundle SHA256
  `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`, source commit
  `07f7e8d43084de748915e1b1eb8b2a1603357c6e`, and the resolved campaign/release manifests inside it.
- Current source/config bytes, benchmark and observation contracts, and primary method references.

# Steps

1. Revalidate the 0.0.7 roster/configs and source behavior against the archived bundle and exact
   source commit.
2. Finish versioned radius-aware planner/runtime/config fixes and one deterministic oracle per arm;
   preserve historical defaults and learned checkpoint feature cadence.
3. Write the per-arm audit and link it through the context index/catalog.
4. Run focused planner/config tests, lint/format, then inspect the exact diff and readiness evidence.
5. Open a #9750 PR with evidence and findings; stop before merge, benchmark evaluation, or release.

# Decisions and risks

- The #9751 owner controls the final hybrid roster; any local hybrid configs are provisional
  successors pending that roster.
- Changing PPO learned-feature semantics would invalidate checkpoint provenance. Preserve its
  trained feature cadence and report any unresolved mismatch rather than silently changing inputs.
- Unit/oracle tests prove implementation integrity only, not benchmark or scientific validity.

# Validation route

- Targeted planner, clearance geometry, legacy serialization, and resolved candidate-config tests.
- `ruff check` and `ruff format --check` on changed Python files.
- Review the audit's every row against the archived release manifest and source/config bytes.
- No evaluation seeds are run until the SNQI anchors/calibration prerequisites are satisfied.

# Recovery / handoff

- All implementation is isolated in the linked #9750 worktree. Preserve the dirty diff and record
  any failed gate without retrying through a fallback. Resume from this plan and the issue-scoped
  audit note; campaign/release work requires its own validated plan and dependency readback.
