<!-- AI-GENERATED (#9652) - NEEDS-REVIEW -->
# Source-bound observed-window corpus entry

**Round-7 disposition: admission FAIL; parking recommended.** The retained
corpus and persistence outputs are historical results from the superseded
gate, not valid current admissions. Continuation v1/v2 checkpoints lack
checkpoint-owned metric criteria and source/replay commit identities.
Admission and export of historical cases now refuse this evidence class.
#10331 ends at replay parity and is unchanged. See [round7_report.md](round7_report.md).

The historical seed-1001 admission is invalid under the required authority contract. Two native window replays match the source exactly; missing
parity proof is refused and duplicate admission preserves the immutable entry.
All ten evidence-registry locator findings are remediated. Existing debt remains
at 492 findings; no new finding was baselined. Raw rows have lane-host custody and
a byte-exact offline regeneration command. See the report for final checks.

Read [the stage report](report.md) for commands, provenance, required tests and
the historical main comparison; the round-7 report lists four current introduced failures.
[window_receipt.json](window_receipt.json) binds the retained source/checkpoint/
scenario/model and both native runs. [corpus.json](corpus/corpus.json) indexes
the first case; [replay_slice.json](replay_slice.json) exports existing runner
inputs. [admission_evidence.json](admission_evidence.json) records actual CLI
results, and [CHECKSUMS.sha256](CHECKSUMS.sha256) covers this packet.

This is an observed simulator regression case. Feasibility and nine perturbation
cells are unknown; separate robust promotion is rejected. No paper-facing
benchmark, release or safety claim is admitted. Original and replay-stage
reports are retained as historical records; the round-7 report reinstates the corpus
blocker. No simulation was run during admission.


## Configuration-owned input contract (round 6)

Admission requires `robot_sf.generated_continuation.v2`. The captured checkpoint
owns the complete immutable environment, planner and drive settings; the
canonical JSON digest must match its typed simulator snapshot's configuration
binding. Map bytes are bound separately; mutable map navigators and Python
factory addresses retain the existing snapshot exclusions. The captured input
object also owns seed, map, window bounds, actual timestep/horizon, observation
mode, recording options and the full observation-noise/tracking specifications.
An actual restore compares this entire input object before mutating state.

The native continuation metadata, result simulator settings, manifest simulator
settings and detailed trace each carry this same versioned object. Admission
reconstructs the source and candidate configuration using the production
scenario loader, config builder, policy resolver and drive binding. It rebuilds
the generated scenario using the existing materializer and recomputes complete
scenario/run identities using the producer's identity constructor. It compares
complete JSON input envelopes by SHA256, including redundant native declarations
and the entire trace step/time sequence. No projection removes runtime options.
The original source trace remains an independent oracle, with all channels
checked separately. A manifest self-hash or an asserted pass is not a proof.

The v1 reader remains for historical custody and replay compatibility; offline
admission refuses a v1 checkpoint because it does not own its configuration.
Adding a future input needs an explicit contract-version/producer update;
unknown, missing, or changed carrier keys fail whole-object equality. The
regression sweep mutates every captured input leaf independently in each of
four carriers, plus unknown/missing keys and the four round-6 reviewer attacks.
No independent perturbation result or robust-promotion claim is fabricated.
