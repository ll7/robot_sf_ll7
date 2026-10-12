# Predictive sample identity development proof

Collection diagnostics only. No model is trained, no navigation quality is measured,
and no model, paper, release or held-out admission is claimed.

Both real collector CLIs ran three episodes on development seeds 1001, 1002 and 1003,
with 12 observations per episode and a four-step horizon. Each NPZ retains 24 samples.
`verification.json` records identity validation and a separate simulator-position oracle
for every active target. `*-source-frames.json` preserves that oracle, including all
simulator actor slots and respawn epochs. Padding and absent actors are masked.

`manifest.json` binds every artifact and the producer sources to SHA-256 and an exact
producer commit. This small bundle is committed to Git and preserved by the PR branch.
No historical dataset or frozen input is rewritten.

To repeat after preserving these files, invoke from the repository root:

```bash
PYTHONPATH=.:fast-pysf uv run python tests/fixtures/predictive_sample_identity/reproduce.py
```

The production collectors refuse overwriting an existing NPZ. Repeat into a fresh copy
of the fixture directory or preserve/remove its generated members first. Collection UUIDs
prevent different invocations from reusing an episode namespace; checksums are therefore
specific to each collected bundle.

New datasets use `sample_identity_schema: predictive_sample_identity_v1` alongside the
existing identity-v2 supervision contract. The seven `sample_*` arrays carry scenario,
episode, actual reset seed, source step, source actor, future actor and future step.
The base scenario ID binds collection settings and the resolved map geometry.
The mixer repeats/shuffles these arrays with their feature and target rows.
Training/mixing validates the marker and refuses missing or inconsistent marked fields.
Historical identity-v2 datasets are explicitly warned as lacking checkable sample
provenance; they are never relabelled. Marked and historical provenance cannot mix.

#10282 already repaired crossing/respawn joins. Open PR #10337 covers live actor joins;
this successor addresses persisted row provenance and guards observation/source alignment.
`base-regression.txt` records both production persistence regressions failing on pinned
main, rather than attributing the already repaired historical crossing to current main.
