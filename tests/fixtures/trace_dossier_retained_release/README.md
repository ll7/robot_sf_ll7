# Retained trace release fixture

`campaign.yaml` is the byte-identical campaign config from commit
`0b0214ced856eac77fa9a4c15b02921eabab1661`, the source commit recorded by the
retained #4848 trace. Its SHA-256 is
`280bff07464103b4a2448702c5eaca036f8a01f320639df521d2350f0e6287c4`.
The `c10df617a87c` matrix pin is historical preregistration identity, not the
current authored-input identity (OVTFIX2/D-085); comments must not change these
historical bytes.

`release.yaml` retains the pins and metadata of
`configs/benchmarks/releases/issue_7086_trace_dossier_diagnostic_v0_1.yaml`;
only its relative input paths point to this fixture and the original side inputs.
The exporter validates these files through its production manifest validator.
This fixture reads an existing trace; it does not run a campaign or simulation.
