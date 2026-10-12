# Source-bound observed-window corpus entry

The seed-1001 critical window is admitted once through the existing persistence
replay/event gate. Two native window replays match the source exactly; missing
parity proof is refused and duplicate admission preserves the immutable entry.
The final evidence-registry ratchet fails on config/custody locator bookkeeping;
the admission PR remains draft pending the maintainer disposition detailed in
the report. No registry baseline was increased.

Read [the stage report](report.md) for commands, provenance, required tests and
the main comparison (132 pre-existing failure/error IDs; zero introduced).
[window_receipt.json](window_receipt.json) binds the retained source/checkpoint/
scenario/model and both native runs. [corpus.json](corpus/corpus.json) indexes
the first case; [replay_slice.json](replay_slice.json) exports existing runner
inputs. [admission_evidence.json](admission_evidence.json) records actual CLI
results, and [CHECKSUMS.sha256](CHECKSUMS.sha256) covers this packet.

This is an observed simulator regression case. Feasibility and nine perturbation
cells are unknown; separate robust promotion is rejected. No paper-facing
benchmark, release or safety claim is admitted. Original and replay-stage
reports are retained as historical records; this report supersedes their corpus
blocker. No simulation was run during admission.
